#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-9B-MXFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-MXFP4/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=1
DEVICE="cuda"
LOG_FILE="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-MXFP4/logs/eval_exec.log"
VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-MXFP4/venv/bin/python"
LIMIT_PER_TASK=50

cd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-9B-MXFP4

# Create output dir
mkdir -p "${OUTPUT_PATH}"

# If results already exist with all 3 tasks, skip
EXISTING=$(find "${OUTPUT_PATH}" -name 'results_*.json' 2>/dev/null | wc -l)
if [ "$EXISTING" -ge 3 ]; then
    echo "Results already exist for all tasks" | tee -a "${LOG_FILE}"
    exit 0
fi

# Run lm_eval with --limit 50 per task for each task separately
for TASK in piqa mmlu hellaswag; do
    TASK_FILE=$(find "${OUTPUT_PATH}" -name "*${TASK}*" -name "results_*.json" 2>/dev/null | head -1)
    if [ -n "$TASK_FILE" ]; then
        echo "Task $TASK already has results, skipping"
        continue
    fi
    echo "Running task: $TASK" | tee -a "${LOG_FILE}"
    rm -f "${LOG_FILE}"
    
    "$VENV_PY" -m lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True" \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --limit ${LIMIT_PER_TASK} \
        --output_path ${OUTPUT_PATH} \
        --gen_kwargs max_gen_toks=2048 \
        --device ${DEVICE} 2>&1 | tee -a "${LOG_FILE}"
    echo "Completed $TASK, exit: $?" | tee -a "${LOG_FILE}"
done

echo "All tasks completed" | tee -a "${LOG_FILE}"