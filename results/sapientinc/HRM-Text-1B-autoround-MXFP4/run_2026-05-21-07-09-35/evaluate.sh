#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/sapientinc_HRM-Text-1B-MXFP4"
OUTPUT_DIR="/root/.openclaw/workspace/quantized/runs/sapientinc_HRM-Text-1B-MXFP4/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

# Use HF backend for auto_round quantized model
lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_DIR}" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda