#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-9B-NVFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-NVFP4/lm_eval_results"
BATCH_SIZE="1"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Run hellaswag
echo "=== Running hellaswag ===" | tee -a /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-NVFP4/logs/eval_exec.log
/root/.venv/bin/python -m lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True,low_cpu_mem_usage=True" \
    --tasks hellaswag \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    --gen_kwargs "max_gen_toks=2048" \
    --device cuda \
    2>&1 | tee -a /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-NVFP4/logs/eval_exec.log
echo "=== hellaswag done ===" | tee -a /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-NVFP4/logs/eval_exec.log