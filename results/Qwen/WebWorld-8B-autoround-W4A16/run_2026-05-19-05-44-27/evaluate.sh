#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv/bin/python"

export CUDA_VISIBLE_DEVICES=0

# Run hellaswag with limit for quick estimate
"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL_PATH",dtype=bfloat16,device_map=auto,trust_remote_code=True,use_cache=False \
    --tasks hellaswag \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=256 \
    --limit 500 2>&1