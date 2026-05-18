#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-1.7B-Base-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-1.7B-Base-W4A16/lm_eval_results"
VENV_PY="/root/.venv/bin/python"

# Use HF backend with auto_round quantized model (W4A16)
$VENV_PY -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks piqa,mmlu,hellaswag \
    --batch_size 8 \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda