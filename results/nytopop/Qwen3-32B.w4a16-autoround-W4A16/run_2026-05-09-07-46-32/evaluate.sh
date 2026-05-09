#!/bin/bash
set -euo pipefail

MODEL_PATH="nytopop/Qwen3-32B.w4a16"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/nytopop_Qwen3-32B.w4a16-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag,gsm8k"
BATCH_SIZE=1
NUM_GPUS=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=${NUM_GPUS},max_model_len=256,gpu_memory_utilization=0.15,max_gen_toks=256,enforce_eager=True,chunked_prefill_enabled=False" \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --device cuda
