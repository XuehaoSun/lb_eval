#!/bin/bash
set -e
SCRIPT_DIR="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-32B-W4A16"
LOG_FILE="$SCRIPT_DIR/logs/eval_exec.log"
/root/.venv/bin/python "$SCRIPT_DIR/evaluate.py" 2>&1 | tee "$LOG_FILE"