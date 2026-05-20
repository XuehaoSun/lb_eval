#!/bin/bash
set -euo pipefail

MODEL_PATH="/root/.openclaw/workspace/quantized/Zyphra_ZAYA1-8B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Zyphra_ZAYA1-8B-W4A16/lm_eval_results"
VENV_PY="/root/.venv/bin/python"

TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

$VENV_PY -m lm_eval \
    --model hf \
    --model_args \
        pretrained=$MODEL_PATH,\
dtype=bfloat16,\
device_map=auto,\
trust_remote_code=True \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda