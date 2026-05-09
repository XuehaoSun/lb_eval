#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/facebook_bart-large-cnn-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/facebook_bart-large-cnn-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=1
NUM_GPUS=1
DEVICE="cuda"

export CUDA_VISIBLE_DEVICES=0

VENV_PY="/root/.venv/bin/python"

echo "=== Starting lm_eval evaluation ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"

"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True,attn_implementation=sdpa" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE"

echo "=== lm_eval evaluation completed ==="