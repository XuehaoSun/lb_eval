#!/bin/bash
set -e

MODEL_PATH="RedHatAI/Qwen3-8B-quantized.w4a16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/RedHatAI_Qwen3-8B-quantized.w4a16-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

/root/.openclaw/workspace/quantized/runs/RedHatAI_Qwen3-8B-quantized.w4a16-W4A16/venv/bin/lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --device cuda \
    --gen_kwargs max_gen_toks=2048