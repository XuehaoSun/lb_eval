#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
GEN_KWARGS="max_gen_toks=2048"

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --device cuda