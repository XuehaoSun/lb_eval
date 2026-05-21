#!/bin/bash
# Auto-Eval Stage A: Run lm_eval for Intel/Qwen2.5-Omni-7B-int4-AutoRound (W4A16)
# Tasks: piqa, mmlu, hellaswag
# Backend: vLLM
# Batch size: 8, Num GPUs: 1, max_gen_toks: 2048

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/models/Intel_Qwen2.5-Omni-7B-int4-AutoRound"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Intel_Qwen2.5-Omni-7B-int4-AutoRound-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1
MAX_GEN_TOKS=2048
DEVICE="cuda"

VENV_PY="/root/.venv/bin/python"

echo "=== Stage A: lm_eval evaluation ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"

mkdir -p "$OUTPUT_PATH"

$VENV_PY -m lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=$MAX_GEN_TOKS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE"

echo "=== Stage A complete ==="