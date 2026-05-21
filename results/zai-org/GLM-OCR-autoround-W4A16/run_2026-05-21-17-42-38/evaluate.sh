#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/zai-org_GLM-OCR-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/zai-org_GLM-OCR-W4A16/lm_eval_results"
VENV="/root/.openclaw/workspace/quantized/runs/zai-org_GLM-OCR-W4A16/venv"

export PATH="${VENV}/bin:$PATH"
export PYTHONPATH="${VENV}/lib/python3.12/site-packages:/root/.openclaw/workspace/quantized/runs/zai-org_GLM-OCR-W4A16:$PYTHONPATH"

TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

$VENV/bin/python /root/.openclaw/workspace/quantized/runs/zai-org_GLM-OCR-W4A16/run_eval.py