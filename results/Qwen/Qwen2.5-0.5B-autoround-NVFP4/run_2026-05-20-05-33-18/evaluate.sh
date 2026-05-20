#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen2.5-0.5B-NVFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen2.5-0.5B-NVFP4/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8

export CUDA_VISIBLE_DEVICES=0

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --gen_kwargs max_gen_toks=2048