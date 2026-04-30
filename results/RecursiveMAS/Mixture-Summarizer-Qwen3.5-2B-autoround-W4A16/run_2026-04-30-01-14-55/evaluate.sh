#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/RecursiveMAS_Mixture-Summarizer-Qwen3.5-2B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/RecursiveMAS_Mixture-Summarizer-Qwen3.5-2B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=8

VENV_PY="/root/.openclaw/workspace/quantized/runs/RecursiveMAS_Mixture-Summarizer-Qwen3.5-2B-W4A16/venv/bin/python"

cd /root/.openclaw/workspace/quantized/runs/RecursiveMAS_Mixture-Summarizer-Qwen3.5-2B-W4A16

"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --device cuda
