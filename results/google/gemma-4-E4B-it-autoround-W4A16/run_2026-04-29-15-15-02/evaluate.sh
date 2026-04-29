#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/google_gemma-4-E4B-it-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/google_gemma-4-E4B-it-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=8
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${NUM_GPUS},dtype=bfloat16,trust_remote_code=True,max_model_len=4096" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --device cuda
