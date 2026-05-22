#!/bin/bash
set -e

MODEL_PATH="RedHatAI/Qwen3.5-9B-quantized.w4a16"
OUTPUT_DIR="/root/.openclaw/workspace/quantized/runs/RedHatAI_Qwen3.5-9B-quantized.w4a16-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

LOG_DIR="/root/.openclaw/workspace/quantized/runs/RedHatAI_Qwen3.5-9B-quantized.w4a16-W4A16/logs"
LOG_FILE="${LOG_DIR}/eval_exec.log"
VENV="/root/.venv"

echo "=== Stage A: lm_eval raw evaluation ==="
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"

# Patch config.json to remove quantization_config so model loads as bf16
CONFIG_PATH="/root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-quantized.w4a16/snapshots/a398088c4228b0ae0c8c78df88fd1e4bf445f068/config.json"

"$VENV/bin/python" - << 'PYEOF'
import json
from pathlib import Path

config_path = Path("/root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-quantized.w4a16/snapshots/a398088c4228b0ae0c8c78df88fd1e4bf445f068/config.json")
with open(config_path) as f:
    config = json.load(f)

if "quantization_config" in config:
    del config["quantization_config"]
    print("Removed quantization_config from config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Saved updated config.json")
else:
    print("No quantization_config found")
PYEOF

echo "Running lm_eval..."

"$VENV/bin/lm_eval" \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --gen_kwargs "max_gen_toks=2048" \
    --output_path "$OUTPUT_DIR" \
    --device cuda