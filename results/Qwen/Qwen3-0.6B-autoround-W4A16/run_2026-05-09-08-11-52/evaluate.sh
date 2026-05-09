#!/bin/bash
# Auto-Eval Stage A: lm_eval execution for Qwen/Qwen3-0.6B (W4A16, auto_round format)
# Tasks: piqa, mmlu, hellaswag | Batch size: 8 | Num GPUs: 1

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/lm_eval_results"
VENV_PY="/root/.venv/bin/python"

TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
MAX_GEN_TOKS=2048

echo "=== Stage A: lm_eval execution ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Output: $OUTPUT_PATH"

# Run lm_eval with HF backend
"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS" \
    --device cuda

echo "=== Stage A complete ==="
