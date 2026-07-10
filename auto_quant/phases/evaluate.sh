#!/bin/bash
# Phase 3: Evaluation
# Runs lm_eval with either hf or vllm backend on the quantized model.
#
# Usage: evaluate.sh <model_path>
#
# Environment variables:
#   EVAL_BACKEND     — "hf" | "vllm" (default: hf)
#   EVAL_TASKS       — comma-separated lm_eval tasks
#   EVAL_BATCH_SIZE  — batch size (default: 8)
#   EVAL_OUTPUT_DIR  — output directory for eval results
#   NUM_GPUS         — number of GPUs (default: 1)

set -euo pipefail

MODEL_PATH="${1:-${QUANTIZED_MODEL_DIR:-}}"
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Usage: evaluate.sh <model_path>"
    exit 1
fi

EVAL_BACKEND="${EVAL_BACKEND:-hf}"
EVAL_TASKS="${EVAL_TASKS:-piqa,mmlu,hellaswag}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_OUTPUT_DIR:-./}/lm_eval_results}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "=== Phase 3: Evaluation ==="
echo "  backend=${EVAL_BACKEND}"
echo "  model=${MODEL_PATH}"
echo "  tasks=${EVAL_TASKS}"
echo "  batch_size=${EVAL_BATCH_SIZE}"
echo "  num_gpus=${NUM_GPUS}"

mkdir -p "${OUTPUT_DIR}"

if [ "$EVAL_BACKEND" == "hf" ]; then
    # ═══ HF Transformers backend ═══
    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True"
    if [ "$NUM_GPUS" -gt 1 ]; then
        MODEL_ARGS="${MODEL_ARGS},parallelize=True"
    fi

    echo "[evaluate] Running lm_eval with hf backend..."
    lm_eval \
        --model hf \
        --model_args "${MODEL_ARGS}" \
        --tasks ${EVAL_TASKS} \
        --batch_size ${EVAL_BATCH_SIZE} \
        --output_path "${OUTPUT_DIR}" \
        --log_samples \
        --seed 42 \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"

elif [ "$EVAL_BACKEND" == "vllm" ]; then
    # ═══ vLLM backend ═══
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    MODEL_ARGS="pretrained=${MODEL_PATH}"
    MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${NUM_GPUS}"
    MODEL_ARGS="${MODEL_ARGS},max_model_len=8192"
    MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=0.9"
    MODEL_ARGS="${MODEL_ARGS},dtype=bfloat16"
    MODEL_ARGS="${MODEL_ARGS},trust_remote_code=True"
    MODEL_ARGS="${MODEL_ARGS},add_bos_token=True"
    MODEL_ARGS="${MODEL_ARGS},enable_prefix_caching=False"

    echo "[evaluate] Running lm_eval with vllm backend..."
    lm_eval \
        --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks ${EVAL_TASKS} \
        --batch_size ${EVAL_BATCH_SIZE} \
        --output_path "${OUTPUT_DIR}" \
        --log_samples \
        --seed 42 \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
else
    echo "ERROR: Unknown EVAL_BACKEND=${EVAL_BACKEND}"
    exit 1
fi

# ═══ Parse results into accuracy.json ═══
echo "[evaluate] Parsing evaluation results..."
python3 - "${OUTPUT_DIR}" "${MODEL_PATH}" "${EVAL_TASKS}" "${EVAL_BACKEND}" "${NUM_GPUS}" <<'PYEOF'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
model_path = sys.argv[2]
eval_tasks = sys.argv[3]
eval_backend = sys.argv[4]
num_gpus = sys.argv[5]

# Find the lm_eval results JSON
results_files = sorted(output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
if not results_files:
    # Try alternate pattern
    results_files = sorted(output_dir.rglob("results.json"), key=lambda p: p.stat().st_mtime)

if not results_files:
    print("[evaluate] WARNING: No results JSON found in lm_eval output")
    accuracy = {
        "status": "failed",
        "errors": ["No results JSON found in lm_eval output directory"],
        "model_path": model_path,
        "tasks": {},
    }
else:
    latest = results_files[-1]
    with latest.open() as f:
        lm_results = json.load(f)

    # Extract per-task accuracy
    tasks = {}
    results_section = lm_results.get("results", {})
    for task_name, task_data in results_section.items():
        if isinstance(task_data, dict):
            # lm_eval uses "acc,none" or "acc_norm,none" keys
            acc = task_data.get("acc,none") or task_data.get("acc_norm,none") or task_data.get("acc")
            if acc is not None:
                tasks[task_name] = {"accuracy": round(float(acc), 6)}

    # Check for zero-accuracy tasks (indicates failure)
    has_zero = any(
        v.get("accuracy", -1) == 0.0
        for v in tasks.values()
    )

    accuracy = {
        "status": "failed" if has_zero else "success",
        "model_id": model_path.rsplit("/", 1)[-1] if "/" in model_path else model_path,
        "model_path": model_path,
        "eval_framework": f"lm_eval ({eval_backend})",
        "num_gpus": num_gpus,
        "eval_num_gpus": num_gpus,
        "tasks": tasks,
        "lm_eval_output_dir": str(output_dir),
        "errors": [],
    }
    if has_zero:
        zero_tasks = [k for k, v in tasks.items() if v.get("accuracy") == 0.0]
        accuracy["errors"] = [f"Zero accuracy on tasks: {zero_tasks}"]

# Write accuracy.json one level up (in RUN_OUTPUT_DIR)
accuracy_path = output_dir.parent / "accuracy.json"
with accuracy_path.open("w") as f:
    json.dump(accuracy, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"[evaluate] accuracy.json written to {accuracy_path}")
print(f"[evaluate] Status: {accuracy['status']}")
for task, data in accuracy.get("tasks", {}).items():
    print(f"  {task}: {data.get('accuracy', 'N/A')}")
PYEOF

echo ""
echo "=== Phase 3: DONE ==="
