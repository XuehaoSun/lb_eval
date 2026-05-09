#!/bin/bash
set -euo pipefail

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=1

LM_EVAL="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/lm-eval"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Patch huggingface.py to skip .to(device) when device_map is set
HF_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/lm_eval/models/huggingface.py"
if ! grep -q 'kwargs.get("device_map")' "$HF_PATH"; then
    sed -i 's/parallelize or autogptq or hasattr(self, "accelerator")/parallelize or autogptq or hasattr(self, "accelerator") or kwargs.get("device_map")/' "$HF_PATH"
fi

echo "=== Starting lm_eval evaluation ==="
"$LM_EVAL" run \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --device cuda
