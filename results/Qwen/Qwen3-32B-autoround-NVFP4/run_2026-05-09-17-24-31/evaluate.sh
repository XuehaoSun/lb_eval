#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-NVFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-NVFP4/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=1
GEN_KWARGS="max_gen_toks=2048"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

VENV=/root/.venv
$VENV/bin/python -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,device_map=auto,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --gen_kwargs $GEN_KWARGS \
    --device cuda