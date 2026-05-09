#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=8

export CUDA_VISIBLE_DEVICES=0

# Run lm_eval with HF backend (device_map=auto handles GPU placement)
lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=float16,device_map=auto,trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --gen_kwargs max_gen_toks=2048