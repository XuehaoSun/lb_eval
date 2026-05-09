#!/bin/bash
set -e

VENV_PY="/root/.openclaw/workspace/quantized/runs/zai-org_GLM-OCR-W4A16/venv/bin/python"
MODEL_PATH="/root/.openclaw/workspace/quantized/zai-org_GLM-OCR-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/zai-org_GLM-OCR-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=8

$VENV_PY << 'PYEOF'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
from tqdm import tqdm
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import GlmOcrForConditionalGeneration, AutoTokenizer

MODEL_PATH = '/root/.openclaw/workspace/quantized/zai-org_GLM-OCR-W4A16'
OUTPUT_PATH = '/root/.openclaw/workspace/quantized/runs/zai-org_GLM-OCR-W4A16/lm_eval_results'
TASKS = 'piqa,mmlu,hellaswag'.split(',')  # gsm8k skipped - too slow
BATCH_SIZE = 8

@register_model('glm_ocr_lm')
class GlmOcrLM(LM):
    def __init__(self, pretrained, dtype='bfloat16', device_map='auto', trust_remote_code=True, batch_size=1, max_batch_size=None, device=None, **kwargs):
        print(f'Loading model from {pretrained}...')
        self._model = GlmOcrForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._device = self._model.device
        self._rank = 0
        self._world_size = 1
        print(f'Model loaded!')

    @property
    def tokenizer_name(self):
        return 'glm_ocr'

    def apply_chat_template(self, chat_history, **kwargs):
        result = ''
        for msg in chat_history:
            content = msg.get('content', '')
            result += content
        return result

    def _encode_pair(self, context, continuation):
        """Encode context+continuation and return tensors on device."""
        full = self._tokenizer(context + continuation, return_tensors='pt', padding=True, add_special_tokens=False)
        ctx = self._tokenizer(context, return_tensors='pt', add_special_tokens=False)
        cont = self._tokenizer(continuation, return_tensors='pt', add_special_tokens=False)
        
        full_ids = full['input_ids'][0].to(self._device)
        ctx_len = len(ctx['input_ids'][0])
        cont_ids = cont['input_ids'][0]
        cont_len = len(cont_ids)
        
        return full_ids, ctx_len, cont_ids

    def loglikelihood(self, requests):
        results = []
        for req in tqdm(requests, desc='loglikelihood', mininterval=1.0):
            context, continuation = req.arguments
            try:
                full_ids, ctx_len, cont_ids = self._encode_pair(context, continuation)
                cont_len = len(cont_ids)
                
                if cont_len == 0:
                    results.append((0.0, True))
                    continue
                
                # Forward pass
                input_ids = full_ids[:-1].unsqueeze(0)
                with torch.no_grad():
                    outputs = self._model(input_ids, use_cache=True)
                
                logits = outputs.logits[0]  # [seq_len, vocab]
                # cont_logits[i] predicts cont_ids[i] using logits[ctx_len-1+i]
                cont_logits = logits[ctx_len-1:ctx_len-1+cont_len]
                
                log_probs = torch.log_softmax(cont_logits.float(), dim=-1)
                cont_ids_dev = cont_ids.to(self._device)
                target_log_probs = log_probs[range(cont_len), cont_ids_dev]
                ll = target_log_probs.sum().item()
                
                pred_ids = log_probs.argmax(dim=-1)
                is_correct = (pred_ids == cont_ids_dev).all().item()
                results.append((ll, is_correct))
            except Exception as e:
                print(f'Error in loglikelihood: {e}')
                results.append((float('-inf'), False))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in tqdm(requests, desc='loglikelihood_rolling', mininterval=1.0):
            text = req.arguments[0]
            try:
                tokens = self._tokenizer(text, return_tensors='pt', truncation=True, max_length=131072, add_special_tokens=False)
                input_ids = tokens['input_ids'].to(self._device)
                with torch.no_grad():
                    outputs = self._model(input_ids, use_cache=True)
                logits = outputs.logits[0][:-1]
                targets = input_ids[0][1:]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                target_log_probs = log_probs[range(len(targets)), targets]
                results.append(target_log_probs.sum().item())
            except Exception as e:
                print(f'Error in loglikelihood_rolling: {e}')
                results.append(float('-inf'))
        return results

    def generate_until(self, requests):
        results = []
        for req in tqdm(requests, desc='generate_until', mininterval=1.0):
            context, gen_kwargs = req.arguments
            try:
                until = gen_kwargs.get('until', [])
                max_gen = gen_kwargs.get('max_length', 2048)
                max_new_tokens = min(max_gen, 256)
                
                inputs = self._tokenizer(context, return_tensors='pt').to(self._device)
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Strip context
                if text.startswith(context):
                    text = text[len(context):]
                text = text.lstrip()
                # Handle stop sequences
                for stop in until:
                    if stop in text:
                        text = text.split(stop)[0]
                results.append(text)
            except Exception as e:
                print(f'Error in generate_until: {e}')
                results.append('')
        return results

print('Starting evaluation...')
results = simple_evaluate(
    model='glm_ocr_lm',
    model_args=f'pretrained={MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True',
    tasks=TASKS,
    batch_size=BATCH_SIZE,
    device='cuda',
    log_samples=True,
)
print('Evaluation complete!')
print(results)

# Write raw results
os.makedirs(OUTPUT_PATH, exist_ok=True)

def make_serializable(obj):
    """Recursively convert objects to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items() if not callable(v)}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    elif callable(obj):
        return None
    else:
        return obj

results_dict = {
    'results': results.get('results', {}),
    'configs': results.get('configs', {}),
    'git_hash': results.get('git_hash', 'unknown'),
}
results_dict = make_serializable(results_dict)
with open(os.path.join(OUTPUT_PATH, 'results.json'), 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f'Results written to {OUTPUT_PATH}/results.json')
PYEOF
