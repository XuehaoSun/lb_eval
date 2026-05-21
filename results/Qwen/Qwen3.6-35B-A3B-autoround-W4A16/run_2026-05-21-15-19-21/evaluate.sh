#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.6-35B-A3B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.6-35B-A3B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=1

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.6-35B-A3B-W4A16/venv/bin/python"

"$VENV_PY" -m lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16,max_gen_toks=2048,gpu_memory_utilization=0.85" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH" \
    --device cuda