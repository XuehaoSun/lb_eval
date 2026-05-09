#!/bin/bash
# Stage A: Run lm_eval for EbanLee/kobart-summary-v3 (W4A16 quantized)
# Tasks: piqa, mmlu, hellaswag, gsm8k
# Try with float32 to bypass quantization issues

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/EbanLee_kobart-summary-v3-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/EbanLee_kobart-summary-v3-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=1

export CUDA_VISIBLE_DEVICES=0

lm_eval run \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},torch_dtype=float32,device_map=auto,trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --device cuda
