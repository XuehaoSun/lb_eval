#!/bin/bash
# Auto-Eval evaluate.sh for Qwen_WebWorld-32B-W4A16
# Backend: vLLM (W4A16 weight-only quantized model)
# max_model_len=128 - only safe setting for this 24GB GPU + 32B model combo
# Tasks: piqa, mmlu, hellaswag

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_WebWorld-32B-W4A16/WebWorld-32B-w4g128"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-32B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=1
GEN_TOKS=2048

VENV_PY="/root/.venv/bin/python"

echo "=== Evaluation Config ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_PATH"
echo "Start time: $(date)"
echo ""

mkdir -p "$OUTPUT_PATH"

echo "Starting vLLM evaluation at $(date)..."
CUDA_VISIBLE_DEVICES=0 "$VENV_PY" -m lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True,max_model_len=128,gpu_memory_utilization=0.85,enforce_eager=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot 0 \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs "max_gen_toks=${GEN_TOKS}" \
    --device cuda

echo ""
echo "=== lm_eval completed at $(date) ==="
ls -la "$OUTPUT_PATH/"