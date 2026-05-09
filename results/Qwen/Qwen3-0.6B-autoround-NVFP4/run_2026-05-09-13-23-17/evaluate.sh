#!/bin/bash
# Stage A: Raw lm_eval execution for Qwen/Qwen3-0.6B (NVFP4)
# Tasks: piqa, hellaswag, mmlu

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-NVFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-NVFP4/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-NVFP4/venv/bin/python"

# Tasks: piqa, mmlu, hellaswag
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

echo "=== Stage A: lm_eval ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"

# Create output dir
mkdir -p "$OUTPUT_PATH"

# Run lm_eval with HF backend
# NVFP4 (llm_compressor format) uses auto_round quantization - supported via HF
# max_gen_toks=2048 is required for all tasks
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-NVFP4/venv/bin/lm-eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda

echo "=== Stage A Complete ==="