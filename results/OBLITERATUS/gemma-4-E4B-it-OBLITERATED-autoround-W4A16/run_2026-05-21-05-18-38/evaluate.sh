#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/OBLITERATUS_gemma-4-E4B-it-OBLITERATED-W4A16/gemma-4-E4B-it-OBLITERATED-w4g128"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/OBLITERATUS_gemma-4-E4B-it-OBLITERATED-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH",dtype=bfloat16,device_map=auto,trust_remote_code=True \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda