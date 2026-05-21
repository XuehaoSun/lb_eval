#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/FrontiersMind_Nandi-Mini-600M-Early-Checkpoint-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/FrontiersMind_Nandi-Mini-600M-Early-Checkpoint-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0

VENV_PY="/root/.venv/bin/python"

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --gen_kwargs "max_gen_toks=2048" \
    --device cuda