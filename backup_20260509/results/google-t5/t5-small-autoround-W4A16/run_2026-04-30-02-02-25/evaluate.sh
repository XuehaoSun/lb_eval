#!/bin/bash
# Stage A: Run lm_eval for quantized T5-small (W4A16) model with auto_round patch
# Tasks: piqa, mmlu, hellaswag, gsm8k
# Batch size: 8, Num GPUs: 1

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/google-t5_t5-small-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/google-t5_t5-small-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/google-t5_t5-small-W4A16/venv/bin/python"

# Tasks
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=8

echo "=== Stage A: lm_eval Execution ==="
echo "Model: ${MODEL_PATH}"
echo "Tasks: ${TASKS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output: ${OUTPUT_PATH}"
echo "Start time: $(date -Iseconds)"

# Create a Python script that applies the patches and runs lm_eval
cat > /tmp/run_lm_eval.py << 'PYEOF'
import torch
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/root/.openclaw/workspace/quantized/google-t5_t5-small-W4A16"
OUTPUT_PATH = "/root/.openclaw/workspace/quantized/runs/google-t5_t5-small-W4A16/lm_eval_results"

# Apply auto_round monkey patches FIRST
from auto_round.utils import monkey_patch
monkey_patch()

# Patch PreTrainedModel._initialize_weights to skip QuantLinear modules
import transformers.modeling_utils as modeling_utils
original_init_fn = modeling_utils.PreTrainedModel._initialize_weights

def patched_init_fn(self, module, is_remote_code):
    if module.__class__.__name__ == 'QuantLinear':
        module._is_hf_initialized = True
        return
    try:
        return original_init_fn(self, module, is_remote_code)
    except (AttributeError, RuntimeError) as e:
        if 'weight' in str(e) or 'qweight' in str(e) or 'QuantLinear' in str(e):
            module._is_hf_initialized = True
            return
        raise

modeling_utils.PreTrainedModel._initialize_weights = patched_init_fn

# Patch T5DenseActDense.forward to handle QuantLinear (no .weight attribute)
from transformers.models.t5 import modeling_t5

_original_dense_forward = modeling_t5.T5DenseActDense.forward

def patched_dense_forward(self, hidden_states):
    # Check if wi or wo are QuantLinear - if so, skip the dtype check
    if (self.wi.__class__.__name__ == 'QuantLinear' or 
        self.wo.__class__.__name__ == 'QuantLinear'):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    return _original_dense_forward(self, hidden_states)

modeling_t5.T5DenseActDense.forward = patched_dense_forward

print("Patches applied")

# Now run lm_eval via API
import lm_eval
from lm_eval import simple_evaluate

model_args = f"pretrained={MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True"

result = simple_evaluate(
    model='hf',
    model_args=model_args,
    tasks=['piqa', 'mmlu', 'hellaswag', 'gsm8k'],
    batch_size=8,
    device='cuda:0',
    log_samples=True,
)
print("Evaluation complete!")

# Save results to output_path
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Extract task results
results = result.get('results', {})
for task_name, task_results in results.items():
    task_dir = os.path.join(OUTPUT_PATH, f"results_{task_name}")
    os.makedirs(task_dir, exist_ok=True)
    result_file = os.path.join(task_dir, "results.json")
    with open(result_file, 'w') as f:
        json.dump(task_results, f, indent=2)
    print(f"Saved {task_name} results to {result_file}")

# Save overall results
overall_results = {
    'results': results,
    'config': result.get('configs', {}),
    'git_hash': result.get('git_hash', 'unknown'),
}
with open(os.path.join(OUTPUT_PATH, "results.json"), 'w') as f:
    json.dump(overall_results, f, indent=2)
print(f"Saved overall results to {OUTPUT_PATH}/results.json")

print("All results saved!")
PYEOF

"$VENV_PY" /tmp/run_lm_eval.py 2>&1

echo "End time: $(date -Iseconds)"
echo "=== Stage A Complete ==="
