#!/bin/bash
# Auto-Eval evaluate.sh for LFM2.5-1.2B-Thinking-CodeX W4A16 AutoRound
# Stage A: raw lm_eval execution

set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Thinking-CodeX-AutoRound-W4A16-RTN"
OUTPUT_DIR="/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Thinking-CodeX-W4A16-RTN/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

# LFM2 requires eager attention mode due to SDPA shape mismatch (from quant_summary)
# Use trust_remote_code=True for custom Lfm2ForCausalLM architecture

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True,attn_implementation=eager" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_DIR} \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda