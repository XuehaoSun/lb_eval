#!/bin/bash
# Stage A: Raw lm_eval execution for OpenMOSS-Team/SciThinker-30B-W4A16 (W4A16 auto_round)
# All 4 tasks: piqa, mmlu, hellaswag, gsm8k
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/OpenMOSS-Team_SciThinker-30B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/OpenMOSS-Team_SciThinker-30B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

echo "=== Stage A: lm_eval execution (all tasks) ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_PATH"

mkdir -p "$OUTPUT_PATH"

PYTHON="/root/.venv/bin/python"

$PYTHON -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH"

echo "=== Stage A complete ==="