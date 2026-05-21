#!/bin/bash
set -e

OUTPUT_DIR="/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results"
MODEL_PATH="Intel/Qwen3.6-27B-int4-mlx-AutoRound"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

export PYTHONPATH="/root/.venv/lib/python3.12/site-packages:$PYTHONPATH"
export PATH="/root/.venv/bin:$PATH"

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,max_model_len=8192,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --gen_kwargs "max_gen_toks=2048" \
    --output_path "${OUTPUT_DIR}" \
    --device cuda
