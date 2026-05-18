#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-8B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-8B-W4A16/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-8B-W4A16/venv/bin/python"

TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

echo "=== Stage A: lm_eval execution ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"

mkdir -p "$OUTPUT_PATH"

export CUDA_VISIBLE_DEVICES=0

"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda

echo "=== Stage A complete ==="
