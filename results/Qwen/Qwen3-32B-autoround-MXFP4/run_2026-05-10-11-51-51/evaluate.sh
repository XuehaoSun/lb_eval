#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-MXFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-MXFP4/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"

cd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-MXFP4

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-MXFP4/venv/bin/python"

"$VENV_PY" -m lm_eval run \
    --model hf \
    --model_args '{"pretrained":"'"${MODEL_PATH}"'","dtype":"float16","device_map":"auto","trust_remote_code":true,"offload_folder":"./offload","offload_buffers":true,"max_memory":"{0:500MB}"}' \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --output_path "${OUTPUT_PATH}" \
    --gen_kwargs max_gen_toks=256 \
    --device cuda