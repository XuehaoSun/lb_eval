#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-14B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-14B-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-14B-W4A16/venv/bin/python"

cd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-14B-W4A16

# Use HF backend with CUDA-compatible torch 2.5.0+cu124
exec $VENV_PY /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-14B-W4A16/lm_eval_wrapper.py \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True,parallelize=True" \
    --tasks piqa,mmlu,hellaswag \
    --batch_size 8 \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda