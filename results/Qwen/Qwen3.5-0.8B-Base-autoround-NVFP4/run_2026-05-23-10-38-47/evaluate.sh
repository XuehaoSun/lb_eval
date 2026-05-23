#!/bin/bash
# Auto-eval evaluate.sh for Qwen/Qwen3.5-0.8B-Base-NVFP4
# Scheme: NVFP4, export_format: auto_round (llm_compressor packing)
# Backend: HF with CUDA

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-Base-NVFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-Base-NVFP4/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

# Use root venv
VENV_PY="/root/.venv/bin/python"

echo "=== NVFP4 Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "GPUs: $NUM_GPUS"

# Check lm_eval is available
"$VENV_PY" -c "import lm_eval; print('lm_eval version:', lm_eval.__version__)" 2>/dev/null || {
    echo "Installing lm-eval..."
    uv pip install --python "$VENV_PY" "lm-eval[torch]" 2>&1 | tail -5
}

# Run evaluation
lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda