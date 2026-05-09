#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/sugoitoolkit_Sugoi-14B-Ultra-HF-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/sugoitoolkit_Sugoi-14B-Ultra-HF-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE="1"

/root/.venv/bin/lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH}