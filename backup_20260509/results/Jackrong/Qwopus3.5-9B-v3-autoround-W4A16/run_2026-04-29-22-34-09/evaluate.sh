#!/bin/bash
set -e

# Run direct_eval.py which patches CUDA detection and uses lm_eval simple_evaluate
# on CPU. This bypasses the fork-based multiprocessing issue.
VENV_PY="/root/.openclaw/workspace/quantized/runs/Jackrong_Qwopus3.5-9B-v3-W4A16/venv/bin/python"

export PYTHONUNBUFFERED=1

"$VENV_PY" /root/.openclaw/workspace/quantized/runs/Jackrong_Qwopus3.5-9B-v3-W4A16/direct_eval.py 2>&1