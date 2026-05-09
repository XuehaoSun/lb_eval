#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1

MODEL_PATH="lvkaokao/Qwen3-0.6B-autoround-W4A16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/lvkaokao_Qwen3-0.6B-autoround-W4A16-W4A16/lm_eval_results"
RUN_DIR="/root/.openclaw/workspace/quantized/runs/lvkaokao_Qwen3-0.6B-autoround-W4A16-W4A16"
TASK="piqa"
BATCH_SIZE=8
NUM_GPUS=1

VENV_DIR="$RUN_DIR/venv"

# Remove existing venv
echo "[setup] Removing existing venv..."
rm -rf "$VENV_DIR"

# Create fresh venv with system-site-packages
echo "[setup] Creating venv with --system-site-packages"
python3 -m venv --system-site-packages "$VENV_DIR"

VENV_PY="$VENV_DIR/bin/python"

# Verify we get system torch (2.10.0+cu128) with working CUDA
echo "[check] System torch inherited in venv:"
"$VENV_PY" -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'cuda:', torch.version.cuda)"

# Bootstrap uv if needed
if ! "$VENV_PY" -c "import uv" 2>/dev/null; then
    echo "[setup] Installing uv"
    "$VENV_PY" -m pip install -U uv
fi

UV_BIN="$VENV_DIR/bin/uv"

# Create a constraints file to prevent torch/torchvision upgrade
CONSTRAINTS_FILE="$RUN_DIR/constraints.txt"
cat > "$CONSTRAINTS_FILE" << 'EOF'
torch==2.10.0
torchvision==0.25.0
numpy<2
EOF

# Install lm-eval with constraints to protect torch
echo "[setup] Installing lm-eval with torch protected by constraints..."
"$UV_BIN" pip install --python "$VENV_PY" --constraint "$CONSTRAINTS_FILE" "lm-eval" 2>&1 | tail -10

# Verify torch still correct after lm-eval install
echo "[check] Torch after lm-eval install (should still be 2.10.0+cu128):"
"$VENV_PY" -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'cuda:', torch.version.cuda)"

# Verify key packages
echo "[check] Package versions:"
"$VENV_PY" -c "import lm_eval; print('lm-eval:', lm_eval.__version__)"
"$VENV_PY" -c "import transformers; print('transformers:', transformers.__version__)"

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run with HF backend
echo "[eval] Running lm_eval with HF backend..."
"$VENV_DIR/bin/lm_eval" run \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --device cuda

echo "[eval] Parsing results..."

# Parse lm_eval results and write accuracy.json
"$VENV_PY" << 'PYEOF'
import json
from pathlib import Path
import glob

output_path = Path("/root/.openclaw/workspace/quantized/runs/lvkaokao_Qwen3-0.6B-autoround-W4A16-W4A16/lm_eval_results")

# Find the results file - lm_eval creates a subdir with model name and a timestamped file inside
model_subdirs = list(output_path.glob("lvkaokao__Qwen3-0.6B-autoround-W4A16"))
if model_subdirs:
    result_files = sorted(model_subdirs[0].glob("results_*.json"))
    if result_files:
        results_file = result_files[-1]  # Get the latest one
    else:
        print(f"ERROR: No results_*.json found in {model_subdirs[0]}")
        exit(1)
else:
    # Fallback: look for any results_*.json anywhere
    result_files = sorted(output_path.glob("**/results_*.json"))
    if result_files:
        results_file = result_files[-1]
    else:
        print(f"ERROR: No results file found in {output_path}")
        exit(1)

print(f"Reading results from: {results_file}")

with open(results_file) as f:
    results = json.load(f)

# Extract task results
# lm-eval uses keys like "acc,none" and "acc_stderr,none" (with filter suffix)
tasks = {}
raw_results = results.get("results", {})

for task_name, task_metrics in raw_results.items():
    # Try both plain keys and suffixed keys
    acc = task_metrics.get("acc") or task_metrics.get("acc,none")
    acc_stderr = task_metrics.get("acc_stderr") or task_metrics.get("acc_stderr,none")
    if acc is not None:
        tasks[task_name] = {
            "accuracy": acc,
            "accuracy_stderr": acc_stderr
        }

accuracy_json = {
    "model_id": "lvkaokao/Qwen3-0.6B-autoround-W4A16",
    "model_path": "lvkaokao/Qwen3-0.6B-autoround-W4A16",
    "scheme": "W4A16",
    "device": "cuda:0",
    "num_gpus": "1",
    "tasks": tasks,
    "status": "success",
    "duration_seconds": 0.0,
    "eval_framework": "lm_eval+hf",
    "errors": []
}

output_json_path = Path("/root/.openclaw/workspace/quantized/runs/lvkaokao_Qwen3-0.6B-autoround-W4A16-W4A16/accuracy.json")
with open(output_json_path, "w") as f:
    json.dump(accuracy_json, f, indent=2)

print(f"Written: {output_json_path}")
print(json.dumps(accuracy_json, indent=2))
PYEOF

echo "[done] Evaluation complete."