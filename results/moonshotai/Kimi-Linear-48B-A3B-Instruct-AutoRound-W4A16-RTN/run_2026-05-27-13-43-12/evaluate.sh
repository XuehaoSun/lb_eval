#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Kimi-Linear-48B-A3B-Instruct-AutoRound-W4A16-RTN"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/moonshotai_Kimi-Linear-48B-A3B-Instruct-W4A16-RTN/lm_eval_results"
LOG_FILE="/root/.openclaw/workspace/quantized/runs/moonshotai_Kimi-Linear-48B-A3B-Instruct-W4A16-RTN/logs/eval_exec.log"

export PATH="${VENV}/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

mkdir -p "${OUTPUT_PATH}"

lm_eval run \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True,attn_implementation=eager,torch_compile=False" \
    --tasks piqa,mmlu,hellaswag \
    --batch_size 8 \
    --output_path "${OUTPUT_PATH}" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda \
    2>&1 | tee "${LOG_FILE}"
