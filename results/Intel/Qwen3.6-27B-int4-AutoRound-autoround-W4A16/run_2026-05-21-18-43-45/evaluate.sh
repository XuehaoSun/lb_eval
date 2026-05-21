#!/bin/bash
# Stage A: Raw lm_eval execution for Intel/Qwen3.6-27B-int4-AutoRound (W4A16)
# Backend: HF with CUDA

set -e

MODEL_ID="Intel/Qwen3.6-27B-int4-AutoRound"
OUTPUT_DIR="/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-AutoRound-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=1
NUM_GPUS=1

# Use system python with torch+cuda venv
PYTHON="/root/.venv/bin/python"

# Point to CUDA 12 libs for torch+cudnn
export LD_LIBRARY_PATH=/root/.venv/lib/python3.12/site-packages/torch/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Enable CUDA memory fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage A: lm_eval execution ==="
echo "Model: $MODEL_ID"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"
echo ""

# Run lm_eval with HF backend
# Use JSON format for model_args to handle max_memory properly
$PYTHON -m lm_eval \
    --model hf \
    --model_args "{\"pretrained\": \"$MODEL_ID\", \"torch_dtype\": \"bfloat16\", \"device_map\": \"auto\", \"max_memory\": {\"0\": \"14GB\", \"cpu\": \"200GB\"}, \"trust_remote_code\": true}" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --gen_kwargs "{\"max_gen_toks\": 2048}" \
    --output_path $OUTPUT_DIR \
    --device cuda

echo ""
echo "=== Stage A complete ==="
ls -la $OUTPUT_DIR/