#!/bin/bash
set -e

VENV="/root/.openclaw/workspace/quantized/runs/ConicCat_Gemma4-GarnetV2-31B-W4A16/venv"
MODEL_PATH="/root/.openclaw/workspace/quantized/ConicCat_Gemma4-GarnetV2-31B-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/ConicCat_Gemma4-GarnetV2-31B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=1

VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 ${VENV}/bin/lm_eval run \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.9435,max_model_len=2592" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --device cuda
