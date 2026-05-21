#!/usr/bin/env bash
set -e

MODEL_PATH="Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Intel_Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

lm_eval run \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=${NUM_GPUS},max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path "${OUTPUT_PATH}" \
    --device cuda
