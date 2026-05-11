#!/bin/bash
# Auto-Eval for Qwen_Qwen3-32B-MXFP4 (MXFP4 scheme, auto_round format)
# Tasks: piqa, hellaswag, mmlu
# Backend: HF with CUDA

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-MXFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-MXFP4/lm_eval_results"
TASKS="piqa,hellaswag,mmlu"
BATCH_SIZE=1

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HF model args as JSON
# max_memory={"cuda:0":"22528MiB"} limits GPU memory to 22GiB to leave headroom for activations
MODEL_ARGS='{"pretrained":"/root/.openclaw/workspace/quantized/Qwen_Qwen3-32B-MXFP4","dtype":"bfloat16","device_map":"auto","trust_remote_code":true,"low_cpu_mem_usage":true,"max_memory":{"cuda:0":"22528MiB"}}'

lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --gen_kwargs "max_gen_toks=256"