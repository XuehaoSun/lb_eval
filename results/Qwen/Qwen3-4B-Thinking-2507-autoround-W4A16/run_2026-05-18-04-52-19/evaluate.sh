#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-4B-Thinking-2507-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-4B-Thinking-2507-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=4

VENV_PY="/root/.venv/bin/python"

# Disable CUDA caching allocator warmup that triggers on old drivers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting lm_eval evaluation with HF backend ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"

# Run lm_eval with HF backend
# Use env var to disable warmup to bypass driver issue
TORCH_CUDA_ALLOC_CONF=expandable_segments "$VENV_PY" -m lm_eval run \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs "max_gen_toks=2048" \
    --device cuda

echo "=== lm_eval completed ==="