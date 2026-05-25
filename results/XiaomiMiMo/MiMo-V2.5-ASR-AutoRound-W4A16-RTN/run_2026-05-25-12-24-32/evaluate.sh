#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/MiMo-V2.5-ASR-AutoRound-W4A16-RTN"
OUTPUT_DIR="/root/.openclaw/workspace/quantized/runs/XiaomiMiMo_MiMo-V2.5-ASR-W4A16-RTN/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

VENV_PY="/root/.openclaw/workspace/quantized/runs/XiaomiMiMo_MiMo-V2.5-ASR-W4A16-RTN/venv/bin/python"

cd /root/.openclaw/workspace/quantized/runs/XiaomiMiMo_MiMo-V2.5-ASR-W4A16-RTN

"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda