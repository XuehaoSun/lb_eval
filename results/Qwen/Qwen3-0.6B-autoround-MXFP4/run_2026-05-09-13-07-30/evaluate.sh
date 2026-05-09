#!/bin/bash
# Auto-eval for Qwen/Qwen3-0.6B (MXFP4 quantized)
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/lm_eval_results"
VENV_PY="/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python"

echo "=== LM-Eval: Qwen3-0.6B MXFP4 ==="
echo "Model: $MODEL_PATH"
echo "Tasks: piqa, mmlu, hellaswag"
echo "Batch size: 8"
echo ""

# Run lm_eval via the venv python
"$VENV_PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks piqa,mmlu,hellaswag \
    --batch_size 8 \
    --output_path "$OUTPUT_PATH" \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda

echo ""
echo "=== lm_eval complete ==="