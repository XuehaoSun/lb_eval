#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/ByteDance-Seed_Seed-Coder-8B-Base-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/ByteDance-Seed_Seed-Coder-8B-Base-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/ByteDance-Seed_Seed-Coder-8B-Base-W4A16/venv/bin/python"

TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

echo "=== Stage A: Running lm_eval ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"
echo ""

"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda

echo ""
echo "=== Stage A complete ==="