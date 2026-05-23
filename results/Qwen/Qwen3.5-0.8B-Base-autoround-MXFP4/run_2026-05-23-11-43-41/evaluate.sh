#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-Base-MXFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-Base-MXFP4/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-Base-MXFP4/venv/bin/python"

# Use existing root venv Python
export PATH="/root/.venv/bin:$PATH"
VENV_PY="/root/.venv/bin/python"

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda