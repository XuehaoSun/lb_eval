#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/deepset_roberta-large-squad2-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/deepset_roberta-large-squad2-W4A16/lm_eval_results"
DEVICE="cuda"

VENV_PY="/root/.openclaw/workspace/quantized/runs/deepset_roberta-large-squad2-W4A16/venv/bin/python"

export CUDA_VISIBLE_DEVICES=0

# Run gsm8k with limit=50 (slow at ~12s/sample; model is QA-type, not ideal for math generation)
"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks gsm8k \
    --batch_size 1 \
    --limit 50 \
    --output_path $OUTPUT_PATH \
    --device $DEVICE
