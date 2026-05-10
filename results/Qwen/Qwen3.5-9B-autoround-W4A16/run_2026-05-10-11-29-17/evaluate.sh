#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-9B-W4A16/Qwen3.5-9B-w4g128"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=1

mkdir -p "$OUTPUT_PATH"

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-W4A16/venv/bin/lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda \
    --limit 500