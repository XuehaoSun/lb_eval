#!/usr/bin/env bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Jackrong_Gemopus-4-E4B-it-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Jackrong_Gemopus-4-E4B-it-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=8
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0
export VENV_PY="/root/.venv/bin/python"

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --device cuda