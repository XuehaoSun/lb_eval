#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/zed-industries_zeta-2.1-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/zed-industries_zeta-2.1-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/zed-industries_zeta-2.1-W4A16/venv/bin/python"

cd /root/.openclaw/workspace/quantized/runs/zed-industries_zeta-2.1-W4A16

"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks piqa,mmlu,hellaswag \
    --batch_size 8 \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda