#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/jpacifico_Chocolatine-2-4B-Instruct-DPO-v2.1-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/jpacifico_Chocolatine-2-4B-Instruct-DPO-v2.1-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/jpacifico_Chocolatine-2-4B-Instruct-DPO-v2.1-W4A16/venv/bin/python"
TASKS="gsm8k"
BATCH_SIZE=4

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$VENV_PY -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --device cuda
