#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/google_gemma-4-E2B-it-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/google_gemma-4-E2B-it-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/google_gemma-4-E2B-it-W4A16/venv/bin/python"

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks piqa,hellaswag,mmlu \
    --batch_size 8 \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs "max_gen_toks=2048" \
    --device cuda