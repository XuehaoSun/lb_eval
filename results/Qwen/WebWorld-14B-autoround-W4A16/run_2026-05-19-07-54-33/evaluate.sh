#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_WebWorld-14B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-14B-W4A16/lm_eval_results"
TASKS="piqa,hellaswag,mmlu"
BATCH_SIZE=8
NUM_GPUS=1

# HF backend evaluation
lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda