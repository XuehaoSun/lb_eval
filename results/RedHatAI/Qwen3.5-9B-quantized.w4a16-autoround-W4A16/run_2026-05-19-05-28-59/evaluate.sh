#!/bin/bash
# Stage A: Raw lm_eval execution for RedHatAI/Qwen3.5-9B-quantized.w4a16
# Backend: hf
# Tasks: piqa, mmlu, hellaswag
# Batch size: 8
# Num GPUs: 1

# NOTE: This model requires ~25GB free disk for evaluation.
# Current free space: ~9.7GB. Model alone is 14.3GB.
# The lm_eval framework also needs space for datasets cache.

set -e

VENV="/root/.openclaw/workspace/quantized/runs/RedHatAI_Qwen3.5-9B-quantized.w4a16-W4A16/venv"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
LM_EVAL="$VENV/bin/lm_eval"
MODEL="RedHatAI/Qwen3.5-9B-quantized.w4a16"
OUTPUT="/root/.openclaw/workspace/quantized/runs/RedHatAI_Qwen3.5-9B-quantized.w4a16-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH=8
GEN_TOKS=2048

mkdir -p "$OUTPUT"

$LM_EVAL run \
    --model hf \
    --model_args pretrained="$MODEL",dtype=bfloat16,trust_remote_code=True \
    --tasks $TASKS \
    --batch_size $BATCH \
    --gen_kwargs max_gen_toks=$GEN_TOKS \
    --output_path "$OUTPUT" \
    2>&1 | tee "$OUTPUT/eval.log"