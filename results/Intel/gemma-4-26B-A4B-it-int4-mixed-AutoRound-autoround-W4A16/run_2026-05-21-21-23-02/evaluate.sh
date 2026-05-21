#!/bin/bash
set -e

MODEL_DIR="/root/.cache/huggingface/hub/models--Intel--gemma-4-26B-A4B-it-int4-mixed-AutoRound/snapshots/81939a2721231f6d1196e51ab4f4d265005b5d3a"
OUTPUT_DIR="/root/.openclaw/workspace/quantized/runs/Intel_gemma-4-26B-A4B-it-int4-mixed-AutoRound-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0

lm_eval --model hf \
    --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,device_map=auto" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --gen_kwargs "max_gen_toks=2048" \
    --output_path "$OUTPUT_DIR"