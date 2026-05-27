#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/gpt2-AutoRound-W4A16-RTN"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/openai-community_gpt2-W4A16-RTN/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=

/root/.venv/bin/lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --gen_kwargs max_gen_toks=2048 \
    --device cpu