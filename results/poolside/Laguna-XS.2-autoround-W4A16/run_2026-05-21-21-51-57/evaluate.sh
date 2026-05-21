#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/poolside_Laguna-XS.2-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/poolside_Laguna-XS.2-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8

export TORCH_CUDNN_V8_API_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

# Run with HF backend + SDPA
/root/.venv/bin/lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True,attn_implementation=sdpa" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --gen_kwargs "max_gen_toks=2048" \
    --device cuda