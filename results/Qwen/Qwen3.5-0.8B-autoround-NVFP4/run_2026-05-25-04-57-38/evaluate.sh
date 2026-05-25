#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-NVFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-NVFP4/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-NVFP4/venv/bin/python"

TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
DEVICE="cuda"

echo "=== NVFP4 Auto-Round Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo ""

$VENV_PY -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True,low_cpu_mem_usage=True" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --gen_kwargs "max_gen_toks=2048" \
    --device $DEVICE

echo ""
echo "=== Evaluation Complete ==="