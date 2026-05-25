#!/bin/bash
# Stage A: Run lm_eval with HF backend for quantized AutoRound model
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/models/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Intel_Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound-W4A16-RTN/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=1

VENV_PY="/root/.openclaw/workspace/quantized/runs/Intel_Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound-W4A16-RTN/venv/bin/python"

mkdir -p "$OUTPUT_PATH"

cd /root/.openclaw/workspace/quantized/runs/Intel_Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound-W4A16-RTN

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$VENV_PY" << 'PYEOF'
import gc
gc.collect()
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import time
import json
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
import lm_eval
from lm_eval.models.huggingface import HFLM

model_path = "/root/.openclaw/workspace/quantized/models/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
output_path = "/root/.openclaw/workspace/quantized/runs/Intel_Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound-W4A16-RTN/lm_eval_results"

start_time = time.time()

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading model...")
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    model_path, config=config, trust_remote_code=True,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
)
gc.collect()

print("Moving to CUDA...")
model.thinker.to("cuda")
gc.collect()
print(f"Peak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

def model_forward(self, input_ids=None, attention_mask=None, **kwargs):
    return self.thinker(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
model.forward = model_forward.__get__(model, type(model))

print("Creating HFLM...")
lm = HFLM(
    pretrained=model, tokenizer=tokenizer,
    backend="causal", trust_remote_code=True,
)

print("Running evaluation on tasks: piqa, mmlu, hellaswag...")
results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["piqa", "mmlu", "hellaswag"],
    batch_size=1,
    device="cuda:0",
    gen_kwargs={"max_gen_toks": 2048},
    bootstrap_iters=0,
)

duration = time.time() - start_time
print(f"\nTotal evaluation time: {duration:.1f}s ({duration/3600:.1f} hours)")

print("\n=== Results ===")
for task_name, task_results in results["results"].items():
    print(f"\n{task_name}:")
    for metric, value in task_results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value}")

with open(f"{output_path}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written to {output_path}/results.json")
PYEOF