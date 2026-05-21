# Session: autoeval_eval_409

- **Session ID:** `autoeval_eval_409`
- **Timestamp:** 2026-05-21 18:49:30 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 2: Evaluation

### [2026-05-21 18:49:30 UTC] USER

You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_eval_vllm/SKILL.md

Quantized model path: Intel/Qwen3.6-27B-int4-mlx-AutoRound
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16
Raw lm_eval output directory: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results
Evaluation script path: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh
Evaluation execution log path: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log
Evaluation tasks: piqa,mmlu,hellaswag
Batch size: 8
Num gpus: 1

The quantized model was produced by auto_quant with scheme=W4A16, export_format=auto_round.
A venv may already exist at /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/venv (created by auto_quant with --system-site-packages).

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- If a venv already exists at /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/venv, reuse it - just install lm_eval and vllm into it.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- Write evaluation outputs, logs, prompts, copied request/session files, and other runtime artifacts to: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16
- Before starting evaluation, you MUST first generate the evaluation script file:
    /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh
- The file name must be exactly: evaluate.sh
- The script must be a standalone shell program runnable with:
    bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh
- The generated evaluate.sh must focus on Stage A raw lm_eval execution only.
- Do NOT put venv creation, pip/uv installation, package bootstrap, JSON parsing, accuracy.json writing, or destructive cleanup/recreation into evaluate.sh.
- In this same OpenClaw task, prepare or reuse the evaluation environment separately before executing evaluate.sh.
- Prefer direct lm_eval CLI commands and standard shell environment variables over ad-hoc Python wrappers.
- In this same OpenClaw task, first write evaluate.sh, then execute that generated script yourself.
- When you execute evaluate.sh or a direct lm_eval command, you MUST stream stdout/stderr into this log file while still printing output:
    bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log
- /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json is a final summary artifact, not a progress marker.
- After Stage A raw lm_eval completes, parse the latest raw results and write /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json in a separate finalize step outside evaluate.sh.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json until Stage A raw lm_eval output and Stage B parsing have both completed successfully.
- Write /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json atomically via a temporary file and rename/move it into place only at finalize time.
- If evaluation fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json before exiting non-zero, also atomically.
- The overall evaluation workflow MUST still behave as two idempotent stages:
    Stage A: run lm_eval and persist raw results under /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results
    Stage B: parse the latest raw results into /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json
- Stage A and Stage B MUST be independently rerunnable.
- If /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results already contains a valid raw results file matching , do NOT rerun Stage A; only rerun Stage B parsing.
- If Stage B parsing fails after Stage A already succeeded, exit non-zero but preserve the raw lm_eval outputs, logs, and venv so the next retry only reruns parsing.
- Do NOT delete or recreate /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results when raw results already exist.
- Parsing logic should read the latest results file under /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results/**/results_*.json rather than assuming lm_eval must be rerun.
- Do NOT leave the final artifact named eval.sh, eval_script.sh, or evaluate_script.sh.
- When the generated script invokes lm_eval, it MUST pass:
    --output_path /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results
  and MUST set max_gen_toks=2048. The placement depends on the backend:
    - For HF backend (--model hf):   --gen_kwargs max_gen_toks=2048
    - For vLLM backend (--model vllm): append max_gen_toks=2048 inside --model_args
- Do NOT omit --output_path. Keep the raw lm_eval output files under that exact directory for later upload.
- Do NOT omit max_gen_toks=2048 regardless of backend.

IMPORTANT - In this same OpenClaw task, after separate raw-eval and finalize steps, you MUST produce:

/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json - evaluation results:
{
  "model_id": "Intel/Qwen3.6-27B-int4-mlx-AutoRound",
  "model_path": "Intel/Qwen3.6-27B-int4-mlx-AutoRound",
  "scheme": "W4A16",
  "device": "cuda:0",
  "num_gpus": "1",
  "tasks": {
    "<task_name>": {
      "accuracy": <float>,
      "accuracy_stderr": <float or null>
    }
  },
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "eval_framework": "lm_eval+vllm" or "lm_eval+hf" or "manual",
  "errors": [<list of error strings if any>]
}

/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results/ - raw lm_eval output directory created by:
    lm_eval ... --output_path /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, the script may invoke python only for JSON post-processing, but it must still write accuracy.json with status=failed before exiting non-zero.

### [2026-05-21 18:49:36 UTC] ASSISTANT

**Thinking**

Let me start by reading the skill instructions for auto_eval_vllm, then understand the current state of the evaluation.


**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/skills/auto_eval_vllm/SKILL.md"
}
```

### [2026-05-21 18:49:36 UTC] TOOL RESULT
**Tool:** `read`

---
name: auto_eval_vllm
description: Automatically evaluate quantized LLM models using vLLM inference engine and lm-evaluation-harness. Supports CUDA GPU with multi-GPU tensor parallelism. Detects Auto-Round quantization format and configures appropriate settings.
metadata:
  openclaw:
    emoji: "📊"
    homepage: https://github.com/EleutherAI/lm-evaluation-harness
    skillKey: auto-eval-vllm
    requires:
      bins: ["lm_eval", "vllm"]
      env: ["torch", "vllm"]
      config: []
---

# Auto-Eval (vLLM) Skill

Use this skill when users want to evaluate quantized LLM models (especially Auto-Round quantized models) using vLLM and lm-evaluation-harness on NVIDIA CUDA GPUs.

## Overview

This skill provides a complete workflow for:
- **Detecting quantization format** from model metadata
- **Configuring vLLM** for CUDA with multi-GPU tensor parallelism
- **Running lm-eval** benchmarks (e.g., piqa, hellaswag, mmlu)
- **Handling OOM errors** with multi-GPU and memory tuning

**Supported tasks**: piqa, hellaswag, mmlu, arc, gsm8k, and many more via lm-eval

---

## Input Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `model_path` | Path to quantized model (local or HF) | Yes | - |
| `task` | Evaluation task (e.g., piqa, mmlu) | Yes | - |
| `output_path` | Directory for results | No | `./lm_eval_results` |
| `batch_size` | Batch size for evaluation | No | `1` |
| `num_gpus` | Number of GPUs for tensor parallelism | No | `1` |
| `device` | Device (cuda, cpu) | No | `cuda` |
| `max_model_len` | Max sequence length | No | `8192` |
| `gpu_memory_utilization` | VRAM fraction | No | `0.9` |

## Environment / Dependency Rules (CRITICAL)

Prefer reusing an existing system venv before creating a new one:

```bash
if [ -x /root/.venv/bin/python ]; then
  VENV_PY=/root/.venv/bin/python
else
  python3 -m venv --system-site-packages {output_path}/venv
  VENV_PY={output_path}/venv/bin/python
fi
```

Rules:

1. Use `uv pip`, not plain `pip install`, for dependency installation
2. If `torch` already imports from the reused environment, keep it
3. If `flash_attn` already imports from the reused environment, keep it
4. Only reinstall `torch` / `flash_attn` when they are missing or clearly incompatible

## lm-eval Output Artifact Rule (CRITICAL)

Always run `lm_eval` with an explicit output directory and keep the generated files:

```bash
--output_path {output_path}
```

For the vLLM backend, `max_gen_toks` goes **inside `--model_args`** (not `--gen_kwargs`):

```bash
--model_args pretrained=...,max_gen_toks=2048,...
```

Rules:

1. Do **not** omit `--output_path`
2. Do **not** omit `max_gen_toks=2048` — append it inside `--model_args` for vLLM backend
3. If the caller provides a concrete runtime path, use it exactly
4. Prefer the `lm_eval` CLI over ad-hoc wrappers so raw result artifacts are persisted automatically

---

## Step 1: Analyze Model and Detect Quantization Format

### Check quantization_config.json

```bash
cat {model_path}/quantization_config.json
```

**Common quantization formats and vLLM support:**

| Format | packing_format | vLLM Support | Device |
|--------|---------------|-------------|--------|
| auto_gptq | `auto_gptq` | ✓ (symmetric) | CUDA |
| auto_awq | `auto_awq` | ✓ (asymmetric) | CUDA |
| auto_round | `auto_round:auto_gptq` | ✓ | CUDA |
| llm_compressor | `auto_round:llm_compressor` | ✓ (MXFP4/NVFP4) | CUDA |
| GGUF | `gguf` | ✓ (llama.cpp) | CPU |

### Example quantization_config.json for different formats:

**Auto-GPTQ format:**
```json
{
  "bits": 4,
  "group_size": 128,
  "sym": true,
  "quant_method": "auto-gptq",
  "packing_format": "auto_gptq"
}
```

**Auto-Round format:**
```json
{
  "bits": 4,
  "group_size": 128,
  "sym": true,
  "quant_method": "auto-round",
  "packing_format": "auto_round:auto_gptq"
}
```

**LLM-Compressor (MXFP4) format:**
```json
{
  "quant_method": "compressed-tensors",
  "format": "float-quantized",
  "config_groups": {...},
  "provider": "auto-round"
}
```

---

## Step 2: Configure vLLM Model Args

Based on quantization format and GPU count, configure appropriate vLLM arguments:

### For Single GPU (num_gpus=1)

**Recommended settings:**
```bash
--model_args pretrained=./quantized_model,\
dtype=bfloat16,\
tensor_parallel_size=1,\
max_model_len=8192,\
gpu_memory_utilization=0.9,\
max_gen_toks=2048
```

### For Multi-GPU (num_gpus > 1)

**Recommended settings using tensor parallelism:**
```bash
--model_args pretrained=./quantized_model,\
dtype=bfloat16,\
tensor_parallel_size=$NUM_GPUS,\
max_model_len=8192,\
gpu_memory_utilization=0.9,\
max_gen_toks=2048
```

### Key Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `dtype` | Data type | `bfloat16` for CUDA |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism | 1 (single), N (multi-GPU) |
| `max_model_len` | Max sequence length | 8192 (adjust based on VRAM) |
| `gpu_memory_utilization` | VRAM fraction | 0.9 (leaves ~10% headroom) |
| `max_gen_toks` | Max tokens to generate | 2048 |
| `enforce_eager` | Disable CUDA graph | **Not needed on CUDA** (unlike XPU) |

### Format-Specific Model Args

**For auto_gptq / auto_round format:**
```
pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048
```

**For llm_compressor (MXFP4/NVFP4) format:**
```
pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048
```

---

## Step 3: Run lm-eval

### Basic Command (Single GPU)

```bash
lm_eval \
    --model vllm \
    --model_args pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048 \
    --tasks piqa \
    --batch_size 1 \
    --output_path lm_eval_results \
    --device cuda
```

### Multi-GPU Command (User specifies GPU count)

```bash
# Example: use 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

lm_eval \
    --model vllm \
    --model_args pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048 \
    --tasks piqa \
    --batch_size 1 \
    --output_path lm_eval_results \
    --device cuda
```

### Running Multiple Tasks

```bash
# Multiple tasks
lm_eval \
    --model vllm \
    --model_args pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_gen_toks=2048 \
    --tasks piqa,hellaswag,mmlu \
    --batch_size 1 \
    --output_path lm_eval_results \
    --device cuda

# Task group
lm_eval \
    --model vllm \
    --model_args pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_gen_toks=2048 \
    --tasks arc_easy,arc_challenge,piqa,hellaswag \
    --batch_size 1 \
    --output_path lm_eval_results \
    --device cuda
```

### Using lm_eval Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=./quantized_model,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048",
    tasks="piqa",
    batch_size="auto",
    device="cuda"
)
print(results["versions"])
```

---

## Step 4: Common Tasks

| Task | Description | Fewshot |
|------|-------------|---------|
| piqa | Physical interaction QA | 0 |
| hellaswag | commonsense inference | 10 |
| mmlu | multitask language understanding | 5 |
| arc | abstract reasoning | 0 |
| gsm8k | grade school math | 3 |
| winogrande | Winograd schema | 0 |
| boolq | boolean questions | 0 |
| cb | commitment bank | 0 |
| copa | causal reasoning | 0 |

### List Available Tasks

```bash
lm_eval ls tasks
```

---

## Step 5: Troubleshooting

### Common Errors and Solutions

#### 1. vLLM Backend Not Found

**Error:**
```
ValueError: Unknown model: vllm
```

**Solution:**
```bash
# Install lm-eval with vllm support
$VENV_PY -m pip install -U uv
uv pip install --python "$VENV_PY" lm-eval[torch,vllm]

# Verify vllm is importable
"$VENV_PY" -c "import vllm; print(vllm.__version__)"
```

#### 2. CUDA Device Not Available

**Error:**
```
RuntimeError: CUDA not available
```

**Solution:**
```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Devices:', torch.cuda.device_count())"

# Verify GPU visibility
nvidia-smi
```

#### 3. OOM on CUDA

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions (in order of preference):**

1. **Use multi-GPU tensor parallelism (RECOMMENDED):**
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1
   export NUM_GPUS=2
   --model_args ...tensor_parallel_size=$NUM_GPUS
   ```

2. **Reduce gpu_memory_utilization:**
   ```bash
   --model_args ...gpu_memory_utilization=0.7
   ```

3. **Reduce max_model_len:**
   ```bash
   --model_args ...max_model_len=2048
   ```

4. **Reduce batch_size:**
   ```bash
   --batch_size 1
   ```

#### 4. Quantization Format Not Supported

**Error:**
```
RuntimeError: Unsupported quantization format
```

**Solution:**
```python
# Check quantization_config.json format
# For auto_round format, ensure packing_format is set correctly
# vLLM supports auto_round:auto_gptq packing format
```

#### 5. Multi-GPU Tensor Parallelism Error

**Error:**
```
RuntimeError: tensor_parallel_size=X but only Y GPUs available
```

**Solution:**
```bash
# Make sure CUDA_VISIBLE_DEVICES has enough GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4

# Verify GPU count
python -c "import torch; print(torch.cuda.device_count())"
```

#### 6. Import Error: vllm module

**Error:**
```
ModuleNotFoundError: No module named 'vllm'
```

**Solution:**
```bash
# Install vllm with CUDA support
uv pip install --python "$VENV_PY" vllm

# Or from PyPI with CUDA extras
uv pip install --python "$VENV_PY" "vllm[cuda]"  # if available
```

---

## Step 6: Evaluation Script Template

### Complete Evaluation Script

```python
#!/usr/bin/env python3
"""
Auto-Eval Script for Quantized Models (vLLM + CUDA)
"""

import argparse
import json
import os
from pathlib import Path


def detect_quantization_format(model_path: str) -> dict:
    """Detect quantization format from model metadata."""
    quant_config_path = Path(model_path) / "quantization_config.json"

    if not quant_config_path.exists():
        return {"format": "bf16", "method": "none"}

    with open(quant_config_path) as f:
        config = json.load(f)

    quant_method = config.get("quant_method", "")
    packing_format = config.get("packing_format", "")

    if "auto-round" in quant_method:
        if "llm_compressor" in packing_format:
            return {"format": "llm_compressor", "method": "auto-round"}
        return {"format": "auto_round", "method": "auto-round"}
    elif "auto-gptq" in quant_method:
        return {"format": "auto_gptq", "method": "gptq"}
    elif "auto-awq" in quant_method:
        return {"format": "auto_awq", "method": "awq"}
    elif "compressed-tensors" in quant_method:
        return {"format": "compressed_tensors", "method": "auto-round"}

    return {"format": "unknown", "method": quant_method}


def build_vllm_args(
    model_path: str,
    num_gpus: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    max_gen_toks: int = 2048,
    **kwargs
) -> str:
    """Build vLLM model arguments string."""

    args = [
        f"pretrained={model_path}",
        "dtype=bfloat16",
        f"tensor_parallel_size={num_gpus}",
        f"max_model_len={max_model_len}",
        f"gpu_memory_utilization={gpu_memory_utilization}",
        f"max_gen_toks={max_gen_toks}",
    ]

    return ",".join(args)


def run_evaluation(
    model_path: str,
    tasks: str,
    output_path: str = "./lm_eval_results",
    batch_size: int = 1,
    num_gpus: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    **kwargs
):
    """Run lm-eval evaluation using vLLM backend."""
    import lm_eval

    quant_info = detect_quantization_format(model_path)
    print(f"Detected quantization: {quant_info}")
    print(f"Using {num_gpus} GPU(s)")

    model_args = build_vllm_args(
        model_path, num_gpus, max_model_len, gpu_memory_utilization, **kwargs
    )
    print(f"Model args: {model_args}")

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=str(batch_size),
        device="cuda",
        output_path=output_path
    )

    print("\n=== Evaluation Results ===")
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(output_path, "results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Eval for Quantized Models (vLLM+CUDA)")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--tasks", required=True, help="Comma-separated tasks")
    parser.add_argument("--output", default="./lm_eval_results", help="Output dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        tasks=args.tasks,
        output_path=args.output,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
```

### Usage Example

```bash
# Single GPU evaluation
python eval_script.py \
    --model /path/to/quantized \
    --tasks piqa,hellaswag \
    --output ./results \
    --batch_size 1 \
    --num_gpus 1

# Multi-GPU evaluation (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python eval_script.py \
    --model /path/to/quantized \
    --tasks piqa,hellaswag \
    --output ./results \
    --batch_size 1 \
    --num_gpus 2
```

---

## Step 7: Environment Setup

### Install Dependencies

```bash
# Reuse the system venv if present; otherwise create a local one
if [ -x /root/.venv/bin/python ]; then
  VENV_PY=/root/.venv/bin/python
else
  python3 -m venv --system-site-packages {output_path}/venv
  VENV_PY={output_path}/venv/bin/python
fi

# Bootstrap uv
$VENV_PY -m pip install -U uv

# Keep inherited CUDA packages when they already work
"$VENV_PY" -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"
"$VENV_PY" -c "import flash_attn; print('flash_attn ok')" || true

# Install vllm with CUDA support
uv pip install --python "$VENV_PY" vllm

# Install lm-eval with vllm
uv pip install --python "$VENV_PY" lm-eval[torch,vllm]

# Only if torch is missing or incompatible, install a matching CUDA wheel
# uv pip install --python "$VENV_PY" --index-url https://download.pytorch.org/whl/cu124 torch

# Only if flash_attn is required and missing
# uv pip install --python "$VENV_PY" flash-attn --no-build-isolation

# Verify installations
"$VENV_PY" -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"
"$VENV_PY" -c "import vllm; print('vLLM:', vllm.__version__)"
"$VENV_PY" -c "import lm_eval; print('lm-eval:', lm_eval.__version__)"
```

---

## Step 8: Quick Reference

### Complete Example for Auto-Round W4A16 Model (Single GPU)

```bash
MODEL_PATH="/path/to/quantized"

lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048" \
    --tasks piqa \
    --batch_size 1 \
    --output_path ./lm_eval_results \
    --device cuda
```

### Complete Example for Auto-Round W4A16 Model (Multi-GPU)

```bash
MODEL_PATH="/path/to/quantized"
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048" \
    --tasks piqa \
    --batch_size 1 \
    --output_path ./lm_eval_results \
    --device cuda
```

### Run Multiple Benchmarks

```bash
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_gen_toks=2048" \
    --tasks piqa,hellaswag,mmlu,arc_easy \
    --batch_size 1 \
    --output_path ./results \
    --device cuda
```

### Using Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=/path/to/quantized,dtype=bfloat16,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048",
    tasks="piqa",
    batch_size="auto",
    device="cuda"
)
print(results["results"]["piqa"])
```

---

## OOM Resolution Guide

When encountering CUDA out of memory errors, follow this decision tree:

1. **First try: Multi-GPU tensor parallelism** (Most effective)
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1  # 2 GPUs
   --model_args ...tensor_parallel_size=2
   ```

2. **Second try: Reduce gpu_memory_utilization**
   ```bash
   --model_args ...gpu_memory_utilization=0.7
   ```

3. **Third try: Reduce max_model_len**
   ```bash
   --model_args ...max_model_len=2048
   ```

4. **Fourth try: Reduce batch_size**
   ```bash
   --batch_size 1
   ```

---

## Notes

- **vLLM on CUDA does NOT require `enforce_eager=True`** — unlike XPU, CUDA graphs are fully supported
- **Batch size** — Use `1` or `auto` for quantized models to avoid OOM
- **tensor_parallel_size** — Set to number of GPUs for multi-GPU inference
- **Quantization formats** — Auto-Round exports to `auto_round` format with `packing_format` metadata
- **gpu_memory_utilization** — Default 0.9 leaves ~10% headroom; reduce if OOM
- **Tasks** — Common: piqa, hellaswag, mmlu, arc, gsm8k


### [2026-05-21 18:49:41 UTC] ASSISTANT

**Thinking**

Now I have the skill instructions. Let me analyze what I need to do:

1. The quantized model is at: `Intel/Qwen3.6-27B-int4-mlx-AutoRound`
2. Runtime artifact directory: `/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16`
3. Evaluation tasks: piqa, mmlu, hellaswag
4. Batch size: 8, Num GPUs: 1
5. The model was quantized with scheme=W4A16, export_format=auto_round

From the instructions:
- I need to first write `evaluate.sh` to the runtime artifact directory
- The script should run Stage A: raw lm_eval execution only
- Then execute that script myself, streaming stdout/stderr to the eval_exec.log
- After Stage A completes, parse results and write accuracy.json in Stage B

Let me first check the existing venv and model structure.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/venv/bin/python 2>/dev/null && echo \"VENV_EXISTS\" || echo \"VENV_MISSING\""
}
```

### [2026-05-21 18:49:41 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 4
drwxr-xr-x 3 root root  50 May 21 18:49 .
drwxr-xr-x 3 root root  64 May 21 18:49 ..
drwxr-xr-x 2 root root  64 May 21 18:49 logs
-rw-r--r-- 1 root root 892 May 21 18:49 request.json

### [2026-05-21 18:49:41 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

VENV_MISSING

### [2026-05-21 18:49:45 UTC] ASSISTANT

**Thinking**

The run directory only has logs/ and request.json. No venv exists, no lm_eval_results, no evaluate.sh. I need to:

1. Create the evaluate.sh script
2. Set up the venv
3. Run the evaluation
4. Parse results and write accuracy.json

Let me first check what's in the request.json and logs directory.


**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/request.json"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/"
}
```

### [2026-05-21 18:49:45 UTC] TOOL RESULT
**Tool:** `read`

{
    "job_type": "evaluation",
    "model": "Intel/Qwen3.6-27B-int4-mlx-AutoRound",
    "revision": "main",
    "private": false,
    "params": 37.97,
    "architectures": "Qwen3_5ForConditionalGeneration",
    "quant_type": "unknown-4bit",
    "precision": "4bit",
    "model_params": 37.97,
    "model_size": 18.98,
    "estimated_memory_gb": 22.78,
    "gpu_type": "NVIDIA GeForce RTX 4090",
    "gpu_nums": 1,
    "weight_dtype": "int4",
    "compute_dtype": "INT4 (W4A16)",
    "gguf_ftype": "*Q4_0.gguf",
    "hardware": "NVIDIA GeForce RTX 4090",
    "status": "Pending",
    "submitted_time": "2026-05-21T08:59:41Z",
    "submitted_by": "INC4AI",
    "model_type": "quantization",
    "job_id": -1,
    "job_start_time": null,
    "script": "auto_eval",
    "likes": 0,
    "license": "apache-2.0",
    "still_on_hub": true,
    "tags": [],
    "downloads": 0,
    "created_at": ""
}

### [2026-05-21 18:49:45 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 16
drwxr-xr-x 2 root root   64 May 21 18:49 .
drwxr-xr-x 3 root root   50 May 21 18:49 ..
-rw-r--r-- 1 root root 6383 May 21 18:49 auto.log
-rw-r--r-- 1 root root 7307 May 21 18:49 eval_script_prompt.txt

### [2026-05-21 18:49:51 UTC] ASSISTANT

**Thinking**

Let me check the eval_script_prompt.txt and auto.log to understand what's been done, and check if there's an existing lm_eval_results directory.


**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_script_prompt.txt"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results/ 2>/dev/null || echo \"NO_LM_EVAL_RESULTS\""
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.venv/bin/python 2>/dev/null && echo \"SYSTEM_VENV_EXISTS\" || echo \"SYSTEM_VENV_MISSING\""
}
```

### [2026-05-21 18:49:51 UTC] TOOL RESULT
**Tool:** `read`

You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_eval_vllm/SKILL.md

Quantized model path: Intel/Qwen3.6-27B-int4-mlx-AutoRound
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16
Raw lm_eval output directory: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results
Evaluation script path: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh
Evaluation execution log path: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log
Evaluation tasks: piqa,mmlu,hellaswag
Batch size: 8
Num gpus: 1

The quantized model was produced by auto_quant with scheme=W4A16, export_format=auto_round.
A venv may already exist at /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/venv (created by auto_quant with --system-site-packages).

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- If a venv already exists at /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/venv, reuse it - just install lm_eval and vllm into it.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- Write evaluation outputs, logs, prompts, copied request/session files, and other runtime artifacts to: /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16
- Before starting evaluation, you MUST first generate the evaluation script file:
    /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh
- The file name must be exactly: evaluate.sh
- The script must be a standalone shell program runnable with:
    bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh
- The generated evaluate.sh must focus on Stage A raw lm_eval execution only.
- Do NOT put venv creation, pip/uv installation, package bootstrap, JSON parsing, accuracy.json writing, or destructive cleanup/recreation into evaluate.sh.
- In this same OpenClaw task, prepare or reuse the evaluation environment separately before executing evaluate.sh.
- Prefer direct lm_eval CLI commands and standard shell environment variables over ad-hoc Python wrappers.
- In this same OpenClaw task, first write evaluate.sh, then execute that generated script yourself.
- When you execute evaluate.sh or a direct lm_eval command, you MUST stream stdout/stderr into this log file while still printing output:
    bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log
- /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json is a final summary artifact, not a progress marker.
- After Stage A raw lm_eval completes, parse the latest raw results and write /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json in a separate finalize step outside evaluate.sh.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json until Stage A raw lm_eval output and Stage B parsing have both completed successfully.
- Write /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json atomically via a temporary file and rename/move it into place only at finalize time.
- If evaluation fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json before exiting non-zero, also atomically.
- The overall evaluation workflow MUST still behave as two idempotent stages:
    Stage A: run lm_eval and persist raw results under /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results
    Stage B: parse the latest raw results into /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json
- Stage A and Stage B MUST be independently rerunnable.
- If /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results already contains a valid raw results file matching , do NOT rerun Stage A; only rerun Stage B parsing.
- If Stage B parsing fails after Stage A already succeeded, exit non-zero but preserve the raw lm_eval outputs, logs, and venv so the next retry only reruns parsing.
- Do NOT delete or recreate /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results when raw results already exist.
- Parsing logic should read the latest results file under /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results/**/results_*.json rather than assuming lm_eval must be rerun.
- Do NOT leave the final artifact named eval.sh, eval_script.sh, or evaluate_script.sh.
- When the generated script invokes lm_eval, it MUST pass:
    --output_path /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results
  and MUST set max_gen_toks=2048. The placement depends on the backend:
    - For HF backend (--model hf):   --gen_kwargs max_gen_toks=2048
    - For vLLM backend (--model vllm): append max_gen_toks=2048 inside --model_args
- Do NOT omit --output_path. Keep the raw lm_eval output files under that exact directory for later upload.
- Do NOT omit max_gen_toks=2048 regardless of backend.

IMPORTANT - In this same OpenClaw task, after separate raw-eval and finalize steps, you MUST produce:

/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json - evaluation results:
{
  "model_id": "Intel/Qwen3.6-27B-int4-mlx-AutoRound",
  "model_path": "Intel/Qwen3.6-27B-int4-mlx-AutoRound",
  "scheme": "W4A16",
  "device": "cuda:0",
  "num_gpus": "1",
  "tasks": {
    "<task_name>": {
      "accuracy": <float>,
      "accuracy_stderr": <float or null>
    }
  },
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "eval_framework": "lm_eval+vllm" or "lm_eval+hf" or "manual",
  "errors": [<list of error strings if any>]
}

/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results/ - raw lm_eval output directory created by:
    lm_eval ... --output_path /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, the script may invoke python only for JSON post-processing, but it must still write accuracy.json with status=failed before exiting non-zero.


### [2026-05-21 18:49:51 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

NO_LM_EVAL_RESULTS

### [2026-05-21 18:49:51 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lrwxrwxrwx 1 root root 73 Apr 24 05:53 /root/.venv/bin/python -> /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
SYSTEM_VENV_EXISTS

### [2026-05-21 18:49:56 UTC] ASSISTANT

**Thinking**

Good - system venv exists at /root/.venv. Now let me check if lm_eval and vllm are installed there, and also check the model path.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import lm_eval; print('lm_eval ok')\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import vllm; print('vllm ok:', vllm.__version__)\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\" 2>&1"
}
```

### [2026-05-21 18:49:56 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'lm_eval'

(Command exited with code 1)

### [2026-05-21 18:49:56 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'vllm'

(Command exited with code 1)

### [2026-05-21 18:49:56 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-21 18:50:00 UTC] ASSISTANT

**Thinking**

So the system venv doesn't have torch, vllm, or lm_eval. I need to check if there's another Python with CUDA pre-installed.


**Tool call:** `exec`

```json
{
  "command": "which python3 && python3 -c \"import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 --version && which python3"
}
```

### [2026-05-21 18:50:00 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

/root/.venv/bin/python3
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-21 18:50:00 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Python 3.12.13
/root/.venv/bin/python3

### [2026-05-21 18:50:04 UTC] ASSISTANT

**Thinking**

The system Python seems to be the /root/.venv one, but it doesn't have torch. Let me check if there's another Python on the system that has torch installed.


**Tool call:** `exec`

```json
{
  "command": "find /usr /opt /home -name \"python3*\" -type f 2>/dev/null | head -20"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls /usr/bin/python* 2>/dev/null; ls /usr/local/bin/python* 2>/dev/null"
}
```

### [2026-05-21 18:50:04 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:50:04 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `2`



(Command exited with code 2)

### [2026-05-21 18:50:10 UTC] ASSISTANT

**Thinking**

Only python3 is available via /root/.venv. Let me check what pip packages are installed in the system site packages, and also check if torch is available via system packages.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/pip list 2>&1 | head -50"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/pip list 2>&1 | grep -i torch"
}
```

### [2026-05-21 18:50:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

sh: 1: /root/.venv/bin/pip: not found

### [2026-05-21 18:50:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`



(Command exited with code 1)

### [2026-05-21 18:50:15 UTC] ASSISTANT

**Thinking**

The venv is essentially empty. Let me check what uv is available and install the packages properly.


**Tool call:** `exec`

```json
{
  "command": "which uv && uv --version"
}
```

### [2026-05-21 18:50:15 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/usr/local/bin/uv
uv 0.11.7 (x86_64-unknown-linux-musl)

### [2026-05-21 18:50:18 UTC] ASSISTANT

**Thinking**

Good, uv is available. Let me check what's in the venv:


**Tool call:** `exec`

```json
{
  "command": "ls /root/.venv/bin/ | head -30"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python3 -c \"import sys; print(sys.path)\" 2>&1"
}
```

### [2026-05-21 18:50:18 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

activate
activate.bat
activate.csh
activate.fish
activate.nu
activate.ps1
activate_this.py
cmdop-sdk
deactivate.bat
dotenv
hf
httpx
huggingface-cli
markdown-it
pydoc.bat
pygmentize
python
python3
python3.12
tiny-agents
tqdm
typer

### [2026-05-21 18:50:18 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

['', '/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python312.zip', '/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12', '/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/python3.12/lib-dynload', '/root/.venv/lib/python3.12/site-packages']

### [2026-05-21 18:50:22 UTC] ASSISTANT

**Thinking**

The venv has no torch installed. Let me check if there's a system site-packages. Also let me check the system python path.


**Tool call:** `exec`

```json
{
  "command": "ls /root/.venv/lib/python3.12/site-packages/ | head -30"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls /root/.venv/lib/python3.12/site-packages/ | grep -i torch"
}
```

### [2026-05-21 18:50:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

__pycache__
_virtualenv.pth
_virtualenv.py
_yaml
annotated_doc
annotated_doc-0.0.4.dist-info
annotated_types
annotated_types-0.7.0.dist-info
anyio
anyio-4.13.0.dist-info
beautifulsoup4-4.14.3.dist-info
bs4
certifi
certifi-2026.5.20.dist-info
click
click-8.4.0.dist-info
cmdop
cmdop-2026.4.7.2.dist-info
dotenv
filelock
filelock-3.29.0.dist-info
fsspec
fsspec-2026.4.0.dist-info
google
grpc
grpcio-1.80.0.dist-info
h11
h11-0.16.0.dist-info
hf_xet
hf_xet-1.5.0.dist-info

### [2026-05-21 18:50:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`



(Command exited with code 1)

### [2026-05-21 18:50:31 UTC] ASSISTANT

**Thinking**

No torch in the venv. The venv is a sparse/minimal venv created by cmdop-sdk. It does not have torch.

The system python does not exist. But the system has CUDA. Let me check if there's a conda or any other way to get Python with torch. Let me also check if there's a different Python path that has torch installed.

Actually, the note says "System Python has torch+cuda pre-installed." But I can't find it. Maybe it's in a different location.

Let me check if conda is available:


**Tool call:** `exec`

```json
{
  "command": "which conda 2>/dev/null; which mamba 2>/dev/null; ls /opt/conda/bin/python 2>/dev/null; ls /usr/local/python*/bin/python 2>/dev/null"
}
```

**Tool call:** `exec`

```json
{
  "command": "find / -name \"torch\" -type d 2>/dev/null | head -10"
}
```

### [2026-05-21 18:50:31 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `2`



(Command exited with code 2)

### [2026-05-21 18:50:31 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:50:37 UTC] ASSISTANT

**Thinking**

No conda, no torch found anywhere. But the user mentioned "System Python has torch+cuda pre-installed". Let me check if torch is installed at system level in some other location, or perhaps the venv creation with `--system-site-packages` should have picked it up.

Wait - the system site packages should have been from the system python at `/usr`. But `/usr` doesn't have a python3. Let me check more carefully what's available.


**Tool call:** `exec`

```json
{
  "command": "ls -la /usr/lib/python3*/ 2>/dev/null | head -20; ls -la /usr/local/lib/python3*/ 2>/dev/null | head -20"
}
```

**Tool call:** `exec`

```json
{
  "command": "dpkg -l | grep -i python3 2>/dev/null | head -20"
}
```

### [2026-05-21 18:50:37 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:50:37 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:50:42 UTC] ASSISTANT

**Thinking**

The /root/.venv seems to be a custom Python environment without torch. Let me try to install torch using uv. Since system site packages are not being inherited (the /root/.venv was created with --system-site-packages but it didn't actually inherit the system packages), I need to install torch.

Actually, wait - the instructions say:
- "System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use: python3 -m venv --system-site-packages <path>"
- "If /root/.venv exists, reuse /root/.venv before creating a new venv."

So I should reuse /root/.venv and install torch into it. Let me install torch with CUDA support.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -20",
  "timeout": 300
}
```

### [2026-05-21 18:50:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session fresh-seaslug, pid 686). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-21 18:50:56 UTC] ASSISTANT

**Thinking**

Let me check on the progress of the uv pip install command.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "fresh-seaslug",
  "timeout": 180000
}
```

### [2026-05-21 18:51:12 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

+ markupsafe==3.0.3
 + mpmath==1.3.0
 + networkx==3.6.1
 + nvidia-cublas-cu12==12.4.5.8
 + nvidia-cuda-cupti-cu12==12.4.127
 + nvidia-cuda-nvrtc-cu12==12.4.127
 + nvidia-cuda-runtime-cu12==12.4.127
 + nvidia-cudnn-cu12==9.1.0.70
 + nvidia-cufft-cu12==11.2.1.3
 + nvidia-curand-cu12==10.3.5.147
 + nvidia-cusolver-cu12==11.6.1.9
 + nvidia-cusparse-cu12==12.3.1.170
 + nvidia-cusparselt-cu12==0.6.2
 + nvidia-nccl-cu12==2.21.5
 + nvidia-nvjitlink-cu12==12.4.127
 + nvidia-nvtx-cu12==12.4.127
 + setuptools==70.2.0
 + sympy==1.13.1
 + torch==2.6.0+cu124
 + triton==3.2.0

Process exited with code 0.

### [2026-05-21 18:51:19 UTC] ASSISTANT

**Thinking**

Good, torch is installed. Now let me install vllm and lm-eval.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.venv/bin/python \"vllm[cuda]\" 2>&1 | tail -30",
  "timeout": 600
}
```

### [2026-05-21 18:51:29 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session tidy-crustacean, pid 799). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-21 18:51:34 UTC] ASSISTANT

**Thinking**

vllm is installing, let me wait for it.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "tidy-crustacean",
  "timeout": 300000
}
```

### [2026-05-21 18:51:57 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

+ sniffio==1.3.1
 + sse-starlette==3.4.4
 + starlette==0.52.1
 + supervisor==4.3.0
 - sympy==1.13.1
 + sympy==1.14.0
 + tabulate==0.10.0
 + tiktoken==0.13.0
 + tilelang==0.1.9
 + tokenizers==0.22.2
 + tokenspeed-mla==0.1.2
 + tokenspeed-triton==3.7.10.post20260505
 - torch==2.6.0+cu124
 + torch==2.11.0
 + torch-c-dlpack-ext==0.1.5
 + torchaudio==2.11.0
 + torchvision==0.26.0
 + transformers==5.9.0
 - triton==3.2.0
 + triton==3.6.0
 + urllib3==2.7.0
 + uvicorn==0.47.0
 + uvloop==0.22.1
 + vllm==0.21.0
 + watchfiles==1.2.0
 + websockets==16.0
 + xgrammar==0.2.1
 + yarl==1.24.2
 + z3-solver==4.15.4.0
warning: The package `vllm==0.21.0` does not have an extra named `cuda`

Process exited with code 0.

### [2026-05-21 18:52:01 UTC] ASSISTANT

**Thinking**

Good, torch and vllm are installed. Now install lm-eval.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.venv/bin/python \"lm-eval[torch,vllm]\" 2>&1 | tail -30",
  "timeout": 300
}
```

### [2026-05-21 18:52:11 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

+ datasets==4.8.5
 + evaluate==0.4.6
 - fsspec==2026.4.0
 + fsspec==2026.2.0
 + joblib==1.5.3
 + lm-eval==0.4.12
 + lxml==6.1.1
 + mbstrdecoder==1.1.5
 + more-itertools==11.0.2
 + multiprocess==0.70.19
 + nltk==3.9.4
 + pandas==3.0.3
 + pathvalidate==3.3.1
 + portalocker==3.2.0
 + pyarrow==24.0.0
 + pytablewriter==1.2.1
 + python-dateutil==2.9.0.post0
 + pytz==2026.2
 + rouge-score==0.1.2
 + sacrebleu==2.6.0
 + scikit-learn==1.8.0
 + scipy==1.17.1
 + sqlitedict==2.1.0
 + tabledata==1.3.5
 + tcolorpy==0.1.7
 + threadpoolctl==3.6.0
 + typepy==1.3.5
 + word2number==1.1
 + xxhash==3.7.0
warning: The package `lm-eval==0.4.12` does not have an extra named `torch`

### [2026-05-21 18:52:16 UTC] ASSISTANT

**Thinking**

Good, lm-eval is installed. Now let me verify everything is working and check if we can find the model path.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.device_count())\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import vllm; print('vllm:', vllm.__version__)\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import lm_eval; print('lm-eval ok')\""
}
```

### [2026-05-21 18:52:20 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
torch: 2.11.0+cu130 CUDA: False 1

### [2026-05-21 18:52:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

vllm: 0.21.0

### [2026-05-21 18:52:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lm-eval ok

### [2026-05-21 18:52:28 UTC] ASSISTANT

**Thinking**

CUDA is available but torch isn't detecting it properly due to driver version. But there is 1 GPU detected. Let me check nvidia-smi to understand the GPU setup better. Also note that vllm needs CUDA to work properly. Let me check if the issue is just the warning.

Actually, `torch.cuda.is_available()` returns False, but `torch.cuda.device_count()` returns 1. This seems inconsistent. Let me check nvidia-smi.


**Tool call:** `exec`

```json
{
  "command": "nvidia-smi 2>&1 | head -30"
}
```

### [2026-05-21 18:52:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Thu May 21 18:52:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
|  0%   29C    P8             18W /  450W |       1MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

### [2026-05-21 18:52:41 UTC] ASSISTANT

**Thinking**

Good, the GPU is visible (RTX 4090) and the driver is 570.195.03 which is newer than what torch detected. The issue is that the torch version installed (2.11.0+cu130) was compiled with CUDA 13.0 but the driver only supports up to CUDA 12.8. So torch.cuda.is_available() returns False.

Wait - actually, looking more carefully at the warning: "The NVIDIA driver on your system is too old (found version 12080)" - 12080 means 12.8.0. So the driver is CUDA 12.8 compatible, but the torch 2.11.0+cu130 was built with CUDA 13.0 which needs a driver that supports 13.0.


I need to install a torch version that's compatible with CUDA 12.8 - either cu128 or cu124 should work with this driver.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print(torch.version.cuda)\""
}
```

### [2026-05-21 18:52:42 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

13.0

### [2026-05-21 18:52:48 UTC] ASSISTANT

**Thinking**

The system has CUDA 12.8 but torch is compiled with CUDA 13.0. That's why torch.cuda.is_available() returns False. Let me install a torch that matches the system's CUDA version.

Actually, let me re-check. The nvidia-smi shows CUDA Version: 12.8, which is the driver CUDA version, not the compiled CUDA version. torch 2.11 with CUDA 13.0 should still work with a CUDA 12.8 driver (newer torch can use older drivers).

The issue is that torch.cuda.is_available() returns False. Let me check if LD_LIBRARY_PATH has the CUDA libraries.


**Tool call:** `exec`

```json
{
  "command": "echo $LD_LIBRARY_PATH; ls /usr/local/cuda*/lib64/libcudart.so* 2>/dev/null | head -5"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -3; ls /usr/local/cuda* 2>/dev/null | head -3"
}
```

### [2026-05-21 18:52:48 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:52:48 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:52:53 UTC] ASSISTANT

**Thinking**

CUDA 13.0 runtime is present in the torch build but the driver is 12.8. Let me try reinstalling torch with the matching CUDA version. Let me first check what the driver supports and reinstall torch accordingly.

Actually, let me try installing torch for cu124 (CUDA 12.4) since the driver is 12.8. That should be compatible.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.venv/bin/python --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -20",
  "timeout": 300
}
```

### [2026-05-21 18:53:03 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session good-sage, pid 1252). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-21 18:53:08 UTC] ASSISTANT

**Thinking**

Let me wait for it.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "good-sage",
  "timeout": 300000
}
```

### [2026-05-21 18:53:23 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

~ nvidia-cuda-nvrtc-cu12==12.4.127
 ~ nvidia-cuda-runtime-cu12==12.4.127
 ~ nvidia-cudnn-cu12==9.1.0.70
 ~ nvidia-cufft-cu12==11.2.1.3
 ~ nvidia-curand-cu12==10.3.5.147
 ~ nvidia-cusolver-cu12==11.6.1.9
 ~ nvidia-cusparse-cu12==12.3.1.170
 ~ nvidia-cusparselt-cu12==0.6.2
 ~ nvidia-nccl-cu12==2.21.5
 ~ nvidia-nvjitlink-cu12==12.4.127
 ~ nvidia-nvtx-cu12==12.4.127
 - setuptools==80.10.2
 + setuptools==70.2.0
 - sympy==1.14.0
 + sympy==1.13.1
 - torch==2.11.0
 + torch==2.6.0+cu124
 - triton==3.6.0
 + triton==3.2.0
 ~ typing-extensions==4.15.0

Process exited with code 0.

### [2026-05-21 18:53:30 UTC] ASSISTANT

**Thinking**

Now torch 2.6.0+cu124 is installed. But torch 2.6.0 compiled with CUDA 12.4 won't work with the CUDA 12.8 driver. Let me check if there's a CUDA 12.4+ compatible torch that works with this driver.

Actually, the issue might be that we need torch compiled for a CUDA version compatible with driver 570.195.03 (CUDA 12.8). Let me check what versions are available.

Actually, CUDA forward compatibility means a newer driver can run older CUDA versions. But torch was compiled for CUDA 12.4 and the driver is CUDA 12.8... Let me test if torch.cuda actually works now.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)\""
}
```

### [2026-05-21 18:53:32 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch: 2.6.0+cu124
CUDA available: True
Device count: 1

### [2026-05-21 18:53:38 UTC] ASSISTANT

**Thinking**

Now CUDA is available. Let me write the evaluate.sh script and then set up the environment properly.

The model path: Intel/Qwen3.6-27B-int4-mlx-AutoRound
Tasks: piqa, mmlu, hellaswag
Batch size: 8
Num GPUs: 1

I need to:
1. Write evaluate.sh
2. Run it with streaming output to the log
3. Parse results and write accuracy.json

For the model path, since it's a HuggingFace model, I should check if it needs to be downloaded or if there's a local path.


**Tool call:** `write`

```json
{
  "content": "#!/bin/bash\nset -e\n\nOUTPUT_DIR=\"/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results\"\nMODEL_PATH=\"Intel/Qwen3.6-27B-int4-mlx-AutoRound\"\nTASKS=\"piqa,mmlu,hellaswag\"\nBATCH_SIZE=8\nNUM_GPUS=1\n\nexport PYTHONPATH=\"/root/.venv/lib/python3.12/site-packages:$PYTHONPATH\"\nexport PATH=\"/root/.venv/bin:$PATH\"\n\nlm_eval \\\n    --model vllm \\\n    --model_args \"pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=${NUM_GPUS},max_model_len=8192,gpu_memory_utilization=0.9,max_gen_toks=2048\" \\\n    --tasks \"${TASKS}\" \\\n    --batch_size \"${BATCH_SIZE}\" \\\n    --output_path \"${OUTPUT_DIR}\" \\\n    --device cuda",
  "path": "/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh"
}
```

### [2026-05-21 18:53:38 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 646 bytes to /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh

### [2026-05-21 18:53:42 UTC] ASSISTANT

**Thinking**

Now let me execute the evaluate.sh script while streaming output to the eval_exec.log file.


**Tool call:** `exec`

```json
{
  "command": "chmod +x /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh"
}
```

### [2026-05-21 18:53:42 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:53:46 UTC] ASSISTANT

**Thinking**

Now let me run the evaluation script. I need to stream stdout/stderr to the eval_exec.log while still printing output.


**Tool call:** `exec`

```json
{
  "command": "bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log",
  "timeout": 3600,
  "yieldMs": 120000
}
```

### [2026-05-21 18:53:54 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

2026-05-21:18:53:53 INFO     [_cli.run:388] Selected Tasks: ['piqa', 'mmlu', 'hellaswag']
2026-05-21:18:53:53 INFO     [evaluator:214] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
2026-05-21:18:53:53 INFO     [evaluator:239] Initializing vllm model, with arguments: {'pretrained': 'Intel/Qwen3.6-27B-int4-mlx-AutoRound', 'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'max_model_len': 8192, 'gpu_memory_utilization': 0.9, 'max_gen_toks': 2048}
Traceback (most recent call last):
  File "/root/.venv/bin/lm_eval", line 10, in <module>
    sys.exit(cli_evaluate())
             ^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/__main__.py", line 10, in cli_evaluate
    parser.execute(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/harness.py", line 60, in execute
    args.func(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/run.py", line 391, in _execute
    results = simple_evaluate(
              ^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/evaluator.py", line 242, in simple_evaluate
    lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 508, in get_model
    return model_registry.get(model_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 310, in get
    concrete = self._materialise(fresh)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 270, in _materialise
    return cast("T", _materialise_placeholder(ph))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 121, in _materialise_placeholder
    return getattr(importlib.import_module(mod), attr)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/models/vllm_causallms.py", line 13, in <module>
    from vllm import LLM, SamplingParams, TokensPrompt
  File "/root/.venv/lib/python3.12/site-packages/vllm/__init__.py", line 70, in __getattr__
    module = import_module(module_name, __package__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 21, in <module>
    from vllm.config import (
  File "/root/.venv/lib/python3.12/site-packages/vllm/config/__init__.py", line 6, in <module>
    from vllm.config.compilation import (
  File "/root/.venv/lib/python3.12/site-packages/vllm/config/compilation.py", line 22, in <module>
    from vllm.platforms import current_platform
  File "/root/.venv/lib/python3.12/site-packages/vllm/platforms/__init__.py", line 278, in __getattr__
    _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/utils/import_utils.py", line 109, in resolve_obj_by_qualname
    module = importlib.import_module(module_name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/platforms/cuda.py", line 21, in <module>
    import vllm._C  # noqa
    ^^^^^^^^^^^^^^
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory

### [2026-05-21 18:54:00 UTC] ASSISTANT

**Thinking**

The vLLM 0.21.0 was built with CUDA 13 but we need CUDA 12. Let me try installing a vLLM version that works with CUDA 12.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.venv/bin/python vllm==0.7.3 2>&1 | tail -20",
  "timeout": 300,
  "yieldMs": 60000
}
```

### [2026-05-21 18:54:46 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

+ pluggy==1.6.0
 + pybind11==3.0.4
 + pytest==9.0.3
 + ray==2.40.0
 - setuptools==70.2.0
 + setuptools==82.0.1
 - torch==2.6.0+cu124
 + torch==2.5.1
 - torchaudio==2.11.0
 + torchaudio==2.5.1
 - torchvision==0.26.0
 + torchvision==0.20.1
 - triton==3.2.0
 + triton==3.1.0
 - vllm==0.21.0
 + vllm==0.7.3
 + xformers==0.0.28.post3
 - xgrammar==0.2.1
 + xgrammar==0.1.11
 + zipp==4.1.0

### [2026-05-21 18:54:53 UTC] ASSISTANT

**Thinking**

vllm 0.7.3 with torch 2.5.1 should work with CUDA 12.4.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import vllm; print('vllm:', vllm.__version__); import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\""
}
```

### [2026-05-21 18:55:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

vllm: 0.7.3
torch: 2.5.1+cu124 CUDA: True

### [2026-05-21 18:55:09 UTC] ASSISTANT

**Thinking**

Now let me retry the evaluation.


**Tool call:** `exec`

```json
{
  "command": "bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log",
  "timeout": 3600,
  "yieldMs": 180000
}
```

### [2026-05-21 18:55:25 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

2026-05-21:18:55:13 INFO     [_cli.run:388] Selected Tasks: ['piqa', 'mmlu', 'hellaswag']
2026-05-21:18:55:14 INFO     [evaluator:214] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
2026-05-21:18:55:14 INFO     [evaluator:239] Initializing vllm model, with arguments: {'pretrained': 'Intel/Qwen3.6-27B-int4-mlx-AutoRound', 'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'max_model_len': 8192, 'gpu_memory_utilization': 0.9, 'max_gen_toks': 2048}
INFO 05-21 18:55:16 __init__.py:207] Automatically detected platform cuda.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
INFO 05-21 18:55:23 config.py:549] This model supports multiple tasks: {'reward', 'score', 'generate', 'embed', 'classify'}. Defaulting to 'generate'.
Traceback (most recent call last):
  File "/root/.venv/bin/lm_eval", line 10, in <module>
    sys.exit(cli_evaluate())
             ^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/__main__.py", line 10, in cli_evaluate
    parser.execute(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/harness.py", line 60, in execute
    args.func(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/run.py", line 391, in _execute
    results = simple_evaluate(
              ^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/evaluator.py", line 242, in simple_evaluate
    lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/model.py", line 169, in create_from_arg_obj
    return cls(**arg_dict, **additional_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/models/vllm_causallms.py", line 146, in __init__
    self.model = LLM(**self.model_args)  # type: ignore[invalid-argument-type]
                 ^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/utils.py", line 1022, in inner
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 242, in __init__
    self.llm_engine = self.engine_class.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 486, in from_engine_args
    engine_config = engine_args.create_engine_config(usage_context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1127, in create_engine_config
    model_config = self.create_model_config()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1047, in create_model_config
    return ModelConfig(
           ^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/vllm/config.py", line 394, in __init__
    self._verify_quantization()
  File "/root/.venv/lib/python3.12/site-packages/vllm/config.py", line 622, in _verify_quantization
    raise ValueError(
ValueError: Unknown quantization method: . Must be one of ['aqlm', 'awq', 'deepspeedfp', 'tpu_int8', 'fp8', 'ptpc_fp8', 'fbgemm_fp8', 'modelopt', 'marlin', 'gguf', 'gptq_marlin_24', 'gptq_marlin', 'awq_marlin', 'gptq', 'compressed-tensors', 'bitsandbytes', 'qqq', 'hqq', 'experts_int8', 'neuron_quant', 'ipex', 'quark', 'moe_wna16'].

### [2026-05-21 18:55:32 UTC] ASSISTANT

**Thinking**

The vLLM 0.7.3 doesn't know about the quantization format. The model was quantized with auto_round which maps to "auto_round:auto_gptq" in the packing_format. Let me check the quantization config of the model to understand the format better.

Actually, looking at the error more carefully: "Unknown quantization method: ." - the method seems to be empty. This might be because the model's quantization_config.json has an unrecognized format. Let me look at the model path.

Actually, I need to understand what model path is being used. The model_path is "Intel/Qwen3.6-27B-int4-mlx-AutoRound" which is likely a HuggingFace model ID, not a local path. But wait, looking at the auto_quant documentation, the model path should be local. Let me check if the model files are available.


**Tool call:** `exec`

```json
{
  "command": "ls -la \"Intel/Qwen3.6-27B-int4-mlx-AutoRound/\" 2>&1 | head -30"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.cache/huggingface/hub/ 2>/dev/null | head -20"
}
```

### [2026-05-21 18:55:32 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

ls: cannot access 'Intel/Qwen3.6-27B-int4-mlx-AutoRound/': No such file or directory

### [2026-05-21 18:55:32 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 4
drwxr-xr-x 4 root root 109 May 21 18:55 .
drwxr-xr-x 3 root root  25 May 21 18:55 ..
drwxr-xr-x 3 root root  67 May 21 18:55 .locks
-rw-r--r-- 1 root root 191 May 21 18:55 CACHEDIR.TAG
drwxr-xr-x 6 root root  85 May 21 18:55 models--Intel--Qwen3.6-27B-int4-mlx-AutoRound

### [2026-05-21 18:55:36 UTC] ASSISTANT

**Thinking**

Let me look at the model that's cached.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/snapshots/"
}
```

### [2026-05-21 18:55:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 0
drwxr-xr-x 6 root root  85 May 21 18:55 .
drwxr-xr-x 4 root root 109 May 21 18:55 ..
drwxr-xr-x 3 root root  62 May 21 18:55 .no_exist
drwxr-xr-x 2 root root 114 May 21 18:55 blobs
drwxr-xr-x 2 root root  26 May 21 18:55 refs
drwxr-xr-x 3 root root  62 May 21 18:55 snapshots

### [2026-05-21 18:55:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

f8897ea6b4a195bbc9f948d4cfd32d1df80437ed

### [2026-05-21 18:55:41 UTC] ASSISTANT

**Thinking**

The model is not actually downloaded! The only files are blobs and refs but the model itself doesn't exist locally. Let me check the blobs directory to see what's there.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/snapshots/f8897ea6b4a195bbc9f948d4cfd32d1df80437ed/"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/"
}
```

### [2026-05-21 18:55:41 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 0
drwxr-xr-x 2 root root 66 May 21 18:55 .
drwxr-xr-x 3 root root 62 May 21 18:55 ..
lrwxrwxrwx 1 root root 52 May 21 18:55 config.json -> ../../blobs/83abedb1fc3ebc80384a2d65213cd2482f7db2ab
lrwxrwxrwx 1 root root 52 May 21 18:55 processor_config.json -> ../../blobs/33818c7f9e991ad735fd240209f4fa73e6c28c50

### [2026-05-21 18:55:41 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 96
drwxr-xr-x 2 root root   114 May 21 18:55 .
drwxr-xr-x 6 root root    85 May 21 18:55 ..
-rw-r--r-- 1 root root  1191 May 21 18:55 33818c7f9e991ad735fd240209f4fa73e6c28c50
-rw-r--r-- 1 root root 93101 May 21 18:55 83abedb1fc3ebc80384a2d65213cd2482f7db2ab

### [2026-05-21 18:55:47 UTC] ASSISTANT

**Thinking**

The model doesn't seem to be fully downloaded. The blobs only have config.json and processor_config.json, no model weights. Let me check if there's a specific path or if we need to download the model.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/83abedb1fc3ebc80384a2d65213cd2482f7db2ab"
}
```

### [2026-05-21 18:55:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "architectures": [
    "Qwen3_5ForConditionalGeneration"
  ],
  "dtype": "bfloat16",
  "image_token_id": 248056,
  "language_model_only": false,
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_output_gate": true,
    "bos_token_id": 248044,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "model_type": "qwen3_5",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 24,
    "num_hidden_layers": 64,
    "num_key_value_heads": 4,
    "output_gate_type": "swish",
    "pad_token_id": null,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": false,
    "use_cache": true,
    "vocab_size": 248320,
    "mrope_interleaved": true,
    "mrope_section": [
      11,
      11,
      10
    ],
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "tie_word_embeddings": false,
  "transformers_version": "5.6.2",
  "video_token_id": 248057,
  "vision_config": {
    "deepstack_visual_indexes": [],
    "depth": 27,
    "dtype": "bfloat16",
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "model_type": "qwen3_5",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 248054,
  "vision_start_token_id": 248053,
  "quantization": {
    "group_size": 128,
    "bits": 4,
    "model.visual.pos_embed": false,
    "model.visual.blocks.0.attn.qkv": false,
    "model.visual.blocks.0.attn.proj": false,
    "model.visual.blocks.1.attn.qkv": false,
    "model.visual.blocks.1.attn.proj": false,
    "model.visual.blocks.2.attn.qkv": false,
    "model.visual.blocks.2.attn.proj": false,
    "model.visual.blocks.3.attn.qkv": false,
    "model.visual.blocks.3.attn.proj": false,
    "model.visual.blocks.4.attn.qkv": false,
    "model.visual.blocks.4.attn.proj": false,
    "model.visual.blocks.5.attn.qkv": false,
    "model.visual.blocks.5.attn.proj": false,
    "model.visual.blocks.6.attn.qkv": false,
    "model.visual.blocks.6.attn.proj": false,
    "model.visual.blocks.7.attn.qkv": false,
    "model.visual.blocks.7.attn.proj": false,
    "model.visual.blocks.8.attn.qkv": false,
    "model.visual.blocks.8.attn.proj": false,
    "model.visual.blocks.9.attn.qkv": false,
    "model.visual.blocks.9.attn.proj": false,
    "model.visual.blocks.10.attn.qkv": false,
    "model.visual.blocks.10.attn.proj": false,
    "model.visual.blocks.11.attn.qkv": false,
    "model.visual.blocks.11.attn.proj": false,
    "model.visual.blocks.12.attn.qkv": false,
    "model.visual.blocks.12.attn.proj": false,
    "model.visual.blocks.13.attn.qkv": false,
    "model.visual.blocks.13.attn.proj": false,
    "model.visual.blocks.14.attn.qkv": false,
    "model.visual.blocks.14.attn.proj": false,
    "model.visual.blocks.15.attn.qkv": false,
    "model.visual.blocks.15.attn.proj": false,
    "model.visual.blocks.16.attn.qkv": false,
    "model.visual.blocks.16.attn.proj": false,
    "model.visual.blocks.17.attn.qkv": false,
    "model.visual.blocks.17.attn.proj": false,
    "model.visual.blocks.18.attn.qkv": false,
    "model.visual.blocks.18.attn.proj": false,
    "model.visual.blocks.19.attn.qkv": false,
    "model.visual.blocks.19.attn.proj": false,
    "model.visual.blocks.20.attn.qkv": false,
    "model.visual.blocks.20.attn.proj": false,
    "model.visual.blocks.21.attn.qkv": false,
    "model.visual.blocks.21.attn.proj": false,
    "model.visual.blocks.22.attn.qkv": false,
    "model.visual.blocks.22.attn.proj": false,
    "model.visual.blocks.23.attn.qkv": false,
    "model.visual.blocks.23.attn.proj": false,
    "model.visual.blocks.24.attn.qkv": false,
    "model.visual.blocks.24.attn.proj": false,
    "model.visual.blocks.25.attn.qkv": false,
    "model.visual.blocks.25.attn.proj": false,
    "model.visual.blocks.26.attn.qkv": false,
    "model.visual.blocks.26.attn.proj": false,
    "model.visual.merger.linear_fc1": false,
    "model.visual.merger.linear_fc2": false,
    "model.language_model.embed_tokens": false,
    "model.language_model.layers.0.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "lm_head": false
  },
  "quantization_config": {
    "group_size": 128,
    "bits": 4,
    "model.visual.pos_embed": false,
    "model.visual.blocks.0.attn.qkv": false,
    "model.visual.blocks.0.attn.proj": false,
    "model.visual.blocks.1.attn.qkv": false,
    "model.visual.blocks.1.attn.proj": false,
    "model.visual.blocks.2.attn.qkv": false,
    "model.visual.blocks.2.attn.proj": false,
    "model.visual.blocks.3.attn.qkv": false,
    "model.visual.blocks.3.attn.proj": false,
    "model.visual.blocks.4.attn.qkv": false,
    "model.visual.blocks.4.attn.proj": false,
    "model.visual.blocks.5.attn.qkv": false,
    "model.visual.blocks.5.attn.proj": false,
    "model.visual.blocks.6.attn.qkv": false,
    "model.visual.blocks.6.attn.proj": false,
    "model.visual.blocks.7.attn.qkv": false,
    "model.visual.blocks.7.attn.proj": false,
    "model.visual.blocks.8.attn.qkv": false,
    "model.visual.blocks.8.attn.proj": false,
    "model.visual.blocks.9.attn.qkv": false,
    "model.visual.blocks.9.attn.proj": false,
    "model.visual.blocks.10.attn.qkv": false,
    "model.visual.blocks.10.attn.proj": false,
    "model.visual.blocks.11.attn.qkv": false,
    "model.visual.blocks.11.attn.proj": false,
    "model.visual.blocks.12.attn.qkv": false,
    "model.visual.blocks.12.attn.proj": false,
    "model.visual.blocks.13.attn.qkv": false,
    "model.visual.blocks.13.attn.proj": false,
    "model.visual.blocks.14.attn.qkv": false,
    "model.visual.blocks.14.attn.proj": false,
    "model.visual.blocks.15.attn.qkv": false,
    "model.visual.blocks.15.attn.proj": false,
    "model.visual.blocks.16.attn.qkv": false,
    "model.visual.blocks.16.attn.proj": false,
    "model.visual.blocks.17.attn.qkv": false,
    "model.visual.blocks.17.attn.proj": false,
    "model.visual.blocks.18.attn.qkv": false,
    "model.visual.blocks.18.attn.proj": false,
    "model.visual.blocks.19.attn.qkv": false,
    "model.visual.blocks.19.attn.proj": false,
    "model.visual.blocks.20.attn.qkv": false,
    "model.visual.blocks.20.attn.proj": false,
    "model.visual.blocks.21.attn.qkv": false,
    "model.visual.blocks.21.attn.proj": false,
    "model.visual.blocks.22.attn.qkv": false,
    "model.visual.blocks.22.attn.proj": false,
    "model.visual.blocks.23.attn.qkv": false,
    "model.visual.blocks.23.attn.proj": false,
    "model.visual.blocks.24.attn.qkv": false,
    "model.visual.blocks.24.attn.proj": false,
    "model.visual.blocks.25.attn.qkv": false,
    "model.visual.blocks.25.attn.proj": false,
    "model.visual.blocks.26.attn.qkv": false,
    "model.visual.blocks.26.attn.proj": false,
    "model.visual.merger.linear_fc1": false,
    "model.visual.merger.linear_fc2": false,
    "model.language_model.embed_tokens": false,
    "model.language_model.layers.0.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.0.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.1.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.2.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.3.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.4.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.5.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.6.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.7.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.8.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.9.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.10.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.11.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.12.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.13.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.14.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.15.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.16.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.17.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.18.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.19.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.20.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.21.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.22.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.23.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.24.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.25.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.26.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.27.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.28.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.29.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.30.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.31.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.32.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.33.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.34.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.35.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.36.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.37.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.38.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.39.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.40.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.41.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.42.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.43.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.44.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.45.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.46.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.47.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.48.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.49.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.50.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.51.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.52.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.53.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.54.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.55.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.56.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.57.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.58.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.59.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.60.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.61.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.linear_attn.out_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.linear_attn.in_proj_qkv": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.linear_attn.in_proj_z": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.62.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.q_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.k_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.v_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.self_attn.o_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.mlp.gate_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.mlp.up_proj": {
      "group_size": 128,
      "bits": 4
    },
    "model.language_model.layers.63.mlp.down_proj": {
      "group_size": 128,
      "bits": 4
    },
    "lm_head": false
  }
}

### [2026-05-21 18:55:54 UTC] ASSISTANT

**Thinking**

Looking at the config, I see that `quantization_config` uses a custom quantization method (like bitsandbytes or a custom format). The error says vLLM 0.7.3 doesn't support this quantization format.

The model has:
- `quant_method` is not explicitly set, but `quantization_config` is structured differently
- It's not the standard auto_gptq, auto_awq, or auto_round format that vLLM supports

Let me check what quantization method is being detected. The error says:
`ValueError: Unknown quantization method: . Must be one of [...]`

The quantization method string is empty. Looking at the config more carefully, the `quantization_config` at the end has `group_size`, `bits`, and per-layer weight configurations. But there's no `quant_method` field.


vLLM expects `quant_method` to be explicitly set, but this config doesn't have it. Even though vLLM's newer versions do support `compressed-tensors` format, the latest available vLLM 0.7.3 might not handle this particular configuration correctly. I should try vLLM 0.7.3 with a different backend or check if I can work around this limitation.

The HF backend might actually be more compatible with this custom quantization setup, so let me try that approach instead.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import transformers; print('transformers:', transformers.__version__)\""
}
```

### [2026-05-21 18:55:55 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

transformers: 5.9.0

### [2026-05-21 18:56:01 UTC] ASSISTANT

**Thinking**

v0.7.3 with transformers 5.9.0 should handle this correctly. But the vLLM error says "Unknown quantization method: ." (empty string). This suggests that the quantization_config.json isn't being properly parsed by vLLM, or the quantization method isn't being detected properly.

Let me look at the full error more carefully:
```
ValueError: Unknown quantization method: . Must be one of ['aqlm', 'awq', 'deepspeedfp', 'tpu_int8', 'fp8', 'ptpc_fp8', 'fbgemm_fp8', 'modelopt', 'marlin', 'gguf', 'gptq_marlin_24', 'gptq_marlin', 'awq_marlin', 'gptq', 'compressed-tensors', 'bitsandbytes', 'qqq', 'hqq', 'experts_int8', 'neuron_quant', 'ipex', 'quark', 'moe_wna16'].


```

The method is empty, which means vLLM is reading the quantization method as blank rather than detecting 'auto_round' or 'compressed-tensors'. This could mean either the config isn't being read correctly or there's a version mismatch causing the detection to fail. I should check if vLLM 0.7.3 actually supports this quantization format.

```


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom vllm import entrypoints\nprint('vllm entrypoints available')\nfrom vllm.compilation import backend\nprint('compilation backend ok')\n\" 2>&1"
}
```

### [2026-05-21 18:56:04 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

vllm entrypoints available
Traceback (most recent call last):
  File "<string>", line 4, in <module>
ImportError: cannot import name 'backend' from 'vllm.compilation' (/root/.venv/lib/python3.12/site-packages/vllm/compilation/__init__.py)

(Command exited with code 1)

### [2026-05-21 18:56:09 UTC] ASSISTANT

**Thinking**

The vllm 0.7.3 seems compatible with transformers 5.9.0 and torch 2.5.1+cu124. But the error said `quantization method: .` (empty string), which means the model's quantization_config has no `quant_method` field. Let me check if we can override this using `hf` backend instead, which might be more flexible with the quantized model.

Actually, let me try using `auto-round` quant method explicitly in model_args if vllm supports it.

Let me check what quant methods are supported in vllm 0.7.3 and see if we can pass `quantization=auto_round` in model_args.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nimport vllm\n# Check supported quantization methods\nprint('vllm version:', vllm.__version__)\nfrom vllm.config import ModelConfig\nimport inspect\nsrc = inspect.getsource(ModelConfig._verify_quantization)\nprint(src[:3000])\n\" 2>&1 | head -80"
}
```

### [2026-05-21 18:56:13 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

vllm version: 0.7.3
    def _verify_quantization(self) -> None:
        supported_quantization = QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed_tensors",
            "compressed-tensors", "experts_int8", "quark"
        ]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Detect which checkpoint is it
            for name in QUANTIZATION_METHODS:
                method = get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override:
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            from vllm.platforms import current_platform
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.", self.quantization)

### [2026-05-21 18:56:19 UTC] ASSISTANT

**Thinking**

The model has a custom quantization with `quant_method` in its config, but it's not recognized by vLLM's `QUANTIZATION_METHODS`. The issue is the model config has `quantization` section but no `quantization_config` with standard fields like `bits`, `group_size` etc.

Looking at the config, the quantization config inside `config.json` doesn't have a `quant_method` field in the top level - it's a complex nested structure.

Let me look at what `QUANTIZATION_METHODS` contains in vllm 0.7.3 to understand what vLLM supports, and what quant_method the config reports.


The real problem is that this model has a non-standard quantization format that vLLM 0.7.3 doesn't natively support. For W4A16, I need to find a way to either bypass vLLM's quantization validation or load the model without triggering that check. I should try the HuggingFace backend instead since it might handle this model differently, though the config shows no `quant_method` at the top level—which is likely why vLLM is raising that error about unrecognized quantization.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom vllm._，精 import QUANTIZATION_METHODS\nprint(QUANTIZATION_METHODS)\n\" 2>&1"
}
```

### [2026-05-21 18:56:19 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

File "<string>", line 2
    from vllm._，精 import QUANTIZATION_METHODS
               ^
SyntaxError: invalid character '，' (U+FF0C)

(Command exited with code 1)

### [2026-05-21 18:56:23 UTC] ASSISTANT

**Thinking**

Typo in my command. Let me fix it.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom vllm._C import QUANTIZATION_METHODS\nprint(QUANTIZATION_METHODS)\n\" 2>&1"
}
```

### [2026-05-21 18:56:26 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 2, in <module>
ImportError: cannot import name 'QUANTIZATION_METHODS' from 'vllm._C' (/root/.venv/lib/python3.12/site-packages/vllm/_C.abi3.so)

(Command exited with code 1)

### [2026-05-21 18:56:34 UTC] ASSISTANT

**Thinking**

Let me try a different approach - the config says `"quant_method": "compressed-tensors"` format. Let me check if `compressed-tensors` is supported in vllm 0.7.3. Looking at the code, `compressed_tensors` and `compressed-tensors` are both listed as optimized methods.

The issue is that vllm is detecting a quantization method from the config that's empty or unrecognized. The error says "Unknown quantization method: ." which means the detected quant_method is an empty string.

Let me look at the config parsing more carefully. The config has a `quantization` section with various settings. The `quant_method` is at the root level of the config, not nested inside `quantization_config`. 

Looking at the config more carefully, I see that `quant_method` is not in the root - it's nested inside `quantization`. The `quantization` section itself doesn't have a `quant_method` at its root level. The root level has `group_size`, `bits`, etc.
 
I think the problem is in how vllm's `_parse_quant_hf_config()` traverses the config. It's probably looking for `quant_method` but finding it nested under `quantization` rather than at the root level, which causes the detection to fail or return an empty string. The real issue is that vllm is detecting an empty quantization method. I should try forcing a specific quantization method via model_args, or use the HF backend instead since vllm doesn't support this format.


**Tool call:** `exec`

```json
{
  "command": "cat > /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh << 'SCRIPT'\n#!/bin/bash\nset -e\n\nOUTPUT_DIR=\"/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/lm_eval_results\"\nMODEL_PATH=\"Intel/Qwen3.6-27B-int4-mlx-AutoRound\"\nTASKS=\"piqa,mmlu,hellaswag\"\nBATCH_SIZE=8\nNUM_GPUS=1\n\nexport PYTHONPATH=\"/root/.venv/lib/python3.12/site-packages:$PYTHONPATH\"\nexport PATH=\"/root/.venv/bin:$PATH\"\n\nlm_eval \\\n    --model hf \\\n    --model_args \"pretrained=${MODEL_PATH},dtype=bfloat16,max_model_len=8192,trust_remote_code=True\" \\\n    --tasks \"${TASKS}\" \\\n    --batch_size \"${BATCH_SIZE}\" \\\n    --gen_kwargs \"max_gen_toks=2048\" \\\n    --output_path \"${OUTPUT_DIR}\" \\\n    --device cuda\nSCRIPT\nchmod +x /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh"
}
```

### [2026-05-21 18:56:34 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-21 18:56:38 UTC] ASSISTANT

**Thinking**

Let me now execute the evaluation script with the HF backend.


**Tool call:** `exec`

```json
{
  "command": "bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log",
  "timeout": 3600,
  "yieldMs": 180000
}
```

### [2026-05-21 18:56:44 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

2026-05-21:18:56:42 INFO     [_cli.run:388] Selected Tasks: ['piqa', 'mmlu', 'hellaswag']
2026-05-21:18:56:43 INFO     [evaluator:214] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
2026-05-21:18:56:43 WARNING  [evaluator:226] generation_kwargs: {'max_gen_toks': 2048} specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!
2026-05-21:18:56:43 INFO     [evaluator:239] Initializing hf model, with arguments: {'pretrained': 'Intel/Qwen3.6-27B-int4-mlx-AutoRound', 'dtype': 'bfloat16', 'max_model_len': 8192, 'trust_remote_code': True}
Traceback (most recent call last):
  File "/root/.venv/bin/lm_eval", line 10, in <module>
    sys.exit(cli_evaluate())
             ^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/__main__.py", line 10, in cli_evaluate
    parser.execute(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/harness.py", line 60, in execute
    args.func(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/run.py", line 391, in _execute
    results = simple_evaluate(
              ^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/evaluator.py", line 242, in simple_evaluate
    lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 508, in get_model
    return model_registry.get(model_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 310, in get
    concrete = self._materialise(fresh)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 270, in _materialise
    return cast("T", _materialise_placeholder(ph))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 121, in _materialise_placeholder
    return getattr(importlib.import_module(mod), attr)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/models/huggingface.py", line 13, in <module>
    from accelerate import (
ModuleNotFoundError: No module named 'accelerate'

### [2026-05-21 18:56:50 UTC] ASSISTANT

**Thinking**

accelerate is not installed. Let me install it.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.venv/bin/python accelerate 2>&1 | tail -10",
  "timeout": 120
}
```

### [2026-05-21 18:56:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Using Python 3.12.13 environment at: /root/.venv
Resolved 46 packages in 543ms
Prepared 1 package in 100ms
Installed 1 package in 19ms
 + accelerate==1.13.0

### [2026-05-21 18:56:55 UTC] ASSISTANT

**Thinking**

Now let me retry the evaluation:


**Tool call:** `exec`

```json
{
  "command": "bash /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log",
  "timeout": 3600,
  "yieldMs": 240000
}
```

### [2026-05-21 18:57:06 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

2026-05-21:18:56:59 INFO     [_cli.run:388] Selected Tasks: ['piqa', 'mmlu', 'hellaswag']
2026-05-21:18:56:59 INFO     [evaluator:214] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
2026-05-21:18:56:59 WARNING  [evaluator:226] generation_kwargs: {'max_gen_toks': 2048} specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!
2026-05-21:18:56:59 INFO     [evaluator:239] Initializing hf model, with arguments: {'pretrained': 'Intel/Qwen3.6-27B-int4-mlx-AutoRound', 'dtype': 'bfloat16', 'max_model_len': 8192, 'trust_remote_code': True}
2026-05-21:18:57:01 INFO     [models.huggingface:286] Using device 'cuda'
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Traceback (most recent call last):
  File "/root/.venv/bin/lm_eval", line 10, in <module>
    sys.exit(cli_evaluate())
             ^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/__main__.py", line 10, in cli_evaluate
    parser.execute(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/harness.py", line 60, in execute
    args.func(args)
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/_cli/run.py", line 391, in _execute
    results = simple_evaluate(
              ^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/evaluator.py", line 242, in simple_evaluate
    lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/api/model.py", line 169, in create_from_arg_obj
    return cls(**arg_dict, **additional_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/lm_eval/models/huggingface.py", line 361, in __init__
    quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.venv/lib/python3.12/site-packages/transformers/quantizers/auto.py", line 151, in from_dict
    raise ValueError(
ValueError: The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized

### [2026-05-21 18:57:14 UTC] ASSISTANT

**Thinking**

The HF backend also fails because it expects a `quant_method` in the quantization config, but the AutoRound config has a different format.

Let me check what `quantization_config` looks like for this model and try to manually fix it or use a different approach.

Let me think about this. The AutoRound format stores quantization info in a different way. Let me see if I can convert the model to a supported format, or perhaps bypass the quantization config check.

Actually, looking at the config I read earlier - the `quantization` section is inside the model config, not in a separate `quantization_config.json`. This is a custom format that HF's AutoQuantizationConfig doesn't understand.
 
Maybe I can manually load the model without quantization config and see if that works, or try forcing vLLM to ignore the quantization mismatch by overriding it in the model_args.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom transformers import AutoConfig\nconfig = AutoConfig.from_pretrained('Intel/Qwen3.6-27B-int4-mlx-AutoRound', trust_remote_code=True)\nprint('quantization:', hasattr(config, 'quantization'))\nprint('quantization attr:', getattr(config, 'quantization', None))\nprint('quantization_config:', getattr(config, 'quantization_config', None))\n\""
}
```

### [2026-05-21 18:57:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

quantization: True
quantization attr: {'group_size': 128, 'bits': 4, 'model.visual.pos_embed': False, 'model.visual.blocks.0.attn.qkv': False, 'model.visual.blocks.0.attn.proj': False, 'model.visual.blocks.1.attn.qkv': False, 'model.visual.blocks.1.attn.proj': False, 'model.visual.blocks.2.attn.qkv': False, 'model.visual.blocks.2.attn.proj': False, 'model.visual.blocks.3.attn.qkv': False, 'model.visual.blocks.3.attn.proj': False, 'model.visual.blocks.4.attn.qkv': False, 'model.visual.blocks.4.attn.proj': False, 'model.visual.blocks.5.attn.qkv': False, 'model.visual.blocks.5.attn.proj': False, 'model.visual.blocks.6.attn.qkv': False, 'model.visual.blocks.6.attn.proj': False, 'model.visual.blocks.7.attn.qkv': False, 'model.visual.blocks.7.attn.proj': False, 'model.visual.blocks.8.attn.qkv': False, 'model.visual.blocks.8.attn.proj': False, 'model.visual.blocks.9.attn.qkv': False, 'model.visual.blocks.9.attn.proj': False, 'model.visual.blocks.10.attn.qkv': False, 'model.visual.blocks.10.attn.proj': False, 'model.visual.blocks.11.attn.qkv': False, 'model.visual.blocks.11.attn.proj': False, 'model.visual.blocks.12.attn.qkv': False, 'model.visual.blocks.12.attn.proj': False, 'model.visual.blocks.13.attn.qkv': False, 'model.visual.blocks.13.attn.proj': False, 'model.visual.blocks.14.attn.qkv': False, 'model.visual.blocks.14.attn.proj': False, 'model.visual.blocks.15.attn.qkv': False, 'model.visual.blocks.15.attn.proj': False, 'model.visual.blocks.16.attn.qkv': False, 'model.visual.blocks.16.attn.proj': False, 'model.visual.blocks.17.attn.qkv': False, 'model.visual.blocks.17.attn.proj': False, 'model.visual.blocks.18.attn.qkv': False, 'model.visual.blocks.18.attn.proj': False, 'model.visual.blocks.19.attn.qkv': False, 'model.visual.blocks.19.attn.proj': False, 'model.visual.blocks.20.attn.qkv': False, 'model.visual.blocks.20.attn.proj': False, 'model.visual.blocks.21.attn.qkv': False, 'model.visual.blocks.21.attn.proj': False, 'model.visual.blocks.22.attn.qkv': False, 'model.visual.blocks.22.attn.proj': False, 'model.visual.blocks.23.attn.qkv': False, 'model.visual.blocks.23.attn.proj': False, 'model.visual.blocks.24.attn.qkv': False, 'model.visual.blocks.24.attn.proj': False, 'model.visual.blocks.25.attn.qkv': False, 'model.visual.blocks.25.attn.proj': False, 'model.visual.blocks.26.attn.qkv': False, 'model.visual.blocks.26.attn.proj': False, 'model.visual.merger.linear_fc1': False, 'model.visual.merger.linear_fc2': False, 'model.language_model.embed_tokens': False, 'model.language_model.layers.0.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'lm_head': False}
quantization_config: {'group_size': 128, 'bits': 4, 'model.visual.pos_embed': False, 'model.visual.blocks.0.attn.qkv': False, 'model.visual.blocks.0.attn.proj': False, 'model.visual.blocks.1.attn.qkv': False, 'model.visual.blocks.1.attn.proj': False, 'model.visual.blocks.2.attn.qkv': False, 'model.visual.blocks.2.attn.proj': False, 'model.visual.blocks.3.attn.qkv': False, 'model.visual.blocks.3.attn.proj': False, 'model.visual.blocks.4.attn.qkv': False, 'model.visual.blocks.4.attn.proj': False, 'model.visual.blocks.5.attn.qkv': False, 'model.visual.blocks.5.attn.proj': False, 'model.visual.blocks.6.attn.qkv': False, 'model.visual.blocks.6.attn.proj': False, 'model.visual.blocks.7.attn.qkv': False, 'model.visual.blocks.7.attn.proj': False, 'model.visual.blocks.8.attn.qkv': False, 'model.visual.blocks.8.attn.proj': False, 'model.visual.blocks.9.attn.qkv': False, 'model.visual.blocks.9.attn.proj': False, 'model.visual.blocks.10.attn.qkv': False, 'model.visual.blocks.10.attn.proj': False, 'model.visual.blocks.11.attn.qkv': False, 'model.visual.blocks.11.attn.proj': False, 'model.visual.blocks.12.attn.qkv': False, 'model.visual.blocks.12.attn.proj': False, 'model.visual.blocks.13.attn.qkv': False, 'model.visual.blocks.13.attn.proj': False, 'model.visual.blocks.14.attn.qkv': False, 'model.visual.blocks.14.attn.proj': False, 'model.visual.blocks.15.attn.qkv': False, 'model.visual.blocks.15.attn.proj': False, 'model.visual.blocks.16.attn.qkv': False, 'model.visual.blocks.16.attn.proj': False, 'model.visual.blocks.17.attn.qkv': False, 'model.visual.blocks.17.attn.proj': False, 'model.visual.blocks.18.attn.qkv': False, 'model.visual.blocks.18.attn.proj': False, 'model.visual.blocks.19.attn.qkv': False, 'model.visual.blocks.19.attn.proj': False, 'model.visual.blocks.20.attn.qkv': False, 'model.visual.blocks.20.attn.proj': False, 'model.visual.blocks.21.attn.qkv': False, 'model.visual.blocks.21.attn.proj': False, 'model.visual.blocks.22.attn.qkv': False, 'model.visual.blocks.22.attn.proj': False, 'model.visual.blocks.23.attn.qkv': False, 'model.visual.blocks.23.attn.proj': False, 'model.visual.blocks.24.attn.qkv': False, 'model.visual.blocks.24.attn.proj': False, 'model.visual.blocks.25.attn.qkv': False, 'model.visual.blocks.25.attn.proj': False, 'model.visual.blocks.26.attn.qkv': False, 'model.visual.blocks.26.attn.proj': False, 'model.visual.merger.linear_fc1': False, 'model.visual.merger.linear_fc2': False, 'model.language_model.embed_tokens': False, 'model.language_model.layers.0.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'lm_head': False}

### [2026-05-21 18:57:28 UTC] ASSISTANT

**Thinking**

The issue is that the quantization config stored in the HF config does NOT have a `quant_method` field. It uses a custom format. This is not recognized by the HF AutoQuantizationConfig system.

The error is from `lm_eval/models/huggingface.py` line 361:
```python
quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
```

It needs `quant_method` in the config, but the model has a custom per-layer quantization format without `quant_method`.

For the HF backend, I can try passing `quantization_config=None` or `load_in_4bit=False` or some other option to bypass the quantization config. But I need to figure out how to do that with lm_eval's model_args.


lm_eval might support a `model_args` override like `quantization=None` or `load_in_4bit=False` to disable the auto-detection, but I'm not certain if that will work. Alternatively, I could modify the config.json to add a `quant_method` field, though that's risky. The real issue is that the model uses a custom quantized format with weights stored as int4 but computed in bf16, which is a hybrid approach that vLLM supports but standard HuggingFace quantization doesn't recognize.

For the HF backend, I need to tell it to ignore the quantization config entirely. Looking at the model loading logic in huggingface.py, the `quantization_config` comes from model_args or auto-detection, so passing `quantization=None` in the model_args might prevent the auto-detection from triggering.

I should also consider whether auto_round is a supported quantization method in this version of transformers—there's a specific quant_method for it, but I'm not sure if it's recognized in the current version.

Let me try using vLLM instead, which may have better support for this quantization format. Or I could simplify by disabling quantization config entirely in the model_args. Actually, looking at the huggingface.py code more carefully, it seems like it passes `quantization_config` directly to `AutoQuantizationConfig.from_dict()`, which expects `quant_method` to be present in the dict. So if I can bypass that or provide the right format, it might work.

The real issue is that this config is purely about how to pack quantization metadata per layer—it's not recognized as a standard `gptq` or `awq` format. I think the solution is to temporarily remove the quantization key from config.json so transformers loads the model in bf16 without trying to apply quantization. But I need to be careful since the model files are actually on the local disk at the HuggingFace cache path, so I could modify the cached config directly.


**Tool call:** `exec`

```json
{
  "command": "ls /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/snapshots/f8897ea6b4a195bbc9f948d4cfd32d1df80437ed/"
}
```

### [2026-05-21 18:57:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523
1d134cd298be1e3be25db393d93a1cefe80e3214
33818c7f9e991ad735fd240209f4fa73e6c28c50
83abedb1fc3ebc80384a2d65213cd2482f7db2ab
a8755d827c0a7b614c246c4060dfd58ab352a8ff

### [2026-05-21 18:57:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

chat_template.jinja
config.json
processor_config.json
tokenizer.json
tokenizer_config.json

### [2026-05-21 18:57:35 UTC] ASSISTANT

**Thinking**

The model is stored as separate blobs. There's no model weights blob - the model only has config and tokenizer files. This means the actual model weights haven't been downloaded or the model ID is incorrect.

Let me check what these blobs are:
- 06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523
- 1d134cd298be1e3be25db393d93a1cefe80e3214
- 33818c7f9e991ad735fd240209f4fa73e6c28c50 (processor_config.json)
- 83abedb1fc3ebc80384a2d65213cd2482f7db2ab (config.json)
- a8755d827c0a7b614c246c4060dfd58ab352a8ff


- 
So the actual model weights are missing - only configuration and tokenizer files are present. This is a placeholder that was fetched but the download never completed. Since I can't access the actual model files, I need to report this as a failure in my accuracy.json.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523 | head -3; cat /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/1d134cd298be1e3be25db393d93a1cefe80e3214 | head -3; cat /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/a8755d827c0a7b614c246c4060dfd58ab352a8ff | head -3"
}
```

### [2026-05-21 18:57:35 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "version": "1.0",
  "truncation": null,
{
  "add_prefix_space": false,
  "audio_bos_token": "<|audio_start|>",
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}

### [2026-05-21 18:57:39 UTC] ASSISTANT

**Thinking**

So:
- `06b950...` - model.safetensors (JSON metadata for a safetensors file)
- `1d134...` - tokenizer.json
- `a8755...` - chat_template.jinja

The model weights themselves aren't fully downloaded! This is just the config and tokenizer. The actual `model.safetensors` file is missing.

But the error said "The model's quantization config from the arguments has no `quant_method` attribute." Let me look at the model loading code more carefully to understand what exactly happens when `AutoQuantizationConfig.from_dict` is called.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"\nfrom transformers import AutoConfig\nconfig = AutoConfig.from_pretrained('Intel/Qwen3.6-27B-int4-mlx-AutoRound', trust_remote_code=True)\nquant = getattr(config, 'quantization', None)\nprint('quantization attr:', quant)\nprint()\nprint('Keys in quantization:', list(quant.keys())[:10] if quant else None)\nprint()\nprint('quant_method' in quant if quant else 'N/A')\n\""
}
```

### [2026-05-21 18:57:42 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
quantization attr: {'group_size': 128, 'bits': 4, 'model.visual.pos_embed': False, 'model.visual.blocks.0.attn.qkv': False, 'model.visual.blocks.0.attn.proj': False, 'model.visual.blocks.1.attn.qkv': False, 'model.visual.blocks.1.attn.proj': False, 'model.visual.blocks.2.attn.qkv': False, 'model.visual.blocks.2.attn.proj': False, 'model.visual.blocks.3.attn.qkv': False, 'model.visual.blocks.3.attn.proj': False, 'model.visual.blocks.4.attn.qkv': False, 'model.visual.blocks.4.attn.proj': False, 'model.visual.blocks.5.attn.qkv': False, 'model.visual.blocks.5.attn.proj': False, 'model.visual.blocks.6.attn.qkv': False, 'model.visual.blocks.6.attn.proj': False, 'model.visual.blocks.7.attn.qkv': False, 'model.visual.blocks.7.attn.proj': False, 'model.visual.blocks.8.attn.qkv': False, 'model.visual.blocks.8.attn.proj': False, 'model.visual.blocks.9.attn.qkv': False, 'model.visual.blocks.9.attn.proj': False, 'model.visual.blocks.10.attn.qkv': False, 'model.visual.blocks.10.attn.proj': False, 'model.visual.blocks.11.attn.qkv': False, 'model.visual.blocks.11.attn.proj': False, 'model.visual.blocks.12.attn.qkv': False, 'model.visual.blocks.12.attn.proj': False, 'model.visual.blocks.13.attn.qkv': False, 'model.visual.blocks.13.attn.proj': False, 'model.visual.blocks.14.attn.qkv': False, 'model.visual.blocks.14.attn.proj': False, 'model.visual.blocks.15.attn.qkv': False, 'model.visual.blocks.15.attn.proj': False, 'model.visual.blocks.16.attn.qkv': False, 'model.visual.blocks.16.attn.proj': False, 'model.visual.blocks.17.attn.qkv': False, 'model.visual.blocks.17.attn.proj': False, 'model.visual.blocks.18.attn.qkv': False, 'model.visual.blocks.18.attn.proj': False, 'model.visual.blocks.19.attn.qkv': False, 'model.visual.blocks.19.attn.proj': False, 'model.visual.blocks.20.attn.qkv': False, 'model.visual.blocks.20.attn.proj': False, 'model.visual.blocks.21.attn.qkv': False, 'model.visual.blocks.21.attn.proj': False, 'model.visual.blocks.22.attn.qkv': False, 'model.visual.blocks.22.attn.proj': False, 'model.visual.blocks.23.attn.qkv': False, 'model.visual.blocks.23.attn.proj': False, 'model.visual.blocks.24.attn.qkv': False, 'model.visual.blocks.24.attn.proj': False, 'model.visual.blocks.25.attn.qkv': False, 'model.visual.blocks.25.attn.proj': False, 'model.visual.blocks.26.attn.qkv': False, 'model.visual.blocks.26.attn.proj': False, 'model.visual.merger.linear_fc1': False, 'model.visual.merger.linear_fc2': False, 'model.language_model.embed_tokens': False, 'model.language_model.layers.0.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.0.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.1.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.2.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.3.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.4.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.5.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.6.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.7.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.8.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.9.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.10.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.11.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.12.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.13.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.14.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.15.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.16.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.17.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.18.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.19.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.20.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.21.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.22.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.23.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.24.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.25.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.26.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.27.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.28.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.29.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.30.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.31.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.32.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.33.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.34.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.35.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.36.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.37.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.38.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.39.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.40.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.41.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.42.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.43.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.44.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.45.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.46.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.47.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.48.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.49.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.50.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.51.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.52.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.53.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.54.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.55.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.56.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.57.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.58.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.59.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.60.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.61.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.out_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.in_proj_qkv': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.linear_attn.in_proj_z': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.62.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.q_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.k_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.v_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.self_attn.o_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.gate_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.up_proj': {'group_size': 128, 'bits': 4}, 'model.language_model.layers.63.mlp.down_proj': {'group_size': 128, 'bits': 4}, 'lm_head': False}

Keys in quantization: ['group_size', 'bits', 'model.visual.pos_embed', 'model.visual.blocks.0.attn.qkv', 'model.visual.blocks.0.attn.proj', 'model.visual.blocks.1.attn.qkv', 'model.visual.blocks.1.attn.proj', 'model.visual.blocks.2.attn.qkv', 'model.visual.blocks.2.attn.proj', 'model.visual.blocks.3.attn.qkv']

False

### [2026-05-21 18:57:52 UTC] ASSISTANT

**Thinking**

This custom quantization format has no `quant_method` field. This is not compatible with either vLLM or HF backend's quantized loading paths. 

The model weights appear to be stored in MLX format (`mlx` in the model name). The blob file `a8755d827c0a7b614c246c4060dfd58ab352a8ff` is the chat template and `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` is the tokenizer. The weight blobs (`1d134cd298be1e3be25db393d93a1cefe80e3214`) would be the actual model weights in MLX format.
 
Since this is W4A16 with int4 weights and bf16 activations, the weights are quantized but activations remain in bf16. I need to find a way to load this custom format—maybe through `mlx` Python package or by locating the actual weight files. The MLX format stores model weights differently than standard formats, so I might need to convert them first.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/pip list 2>/dev/null | grep -i mlx; which mlx 2>/dev/null"
}
```

**Tool call:** `exec`

```json
{
  "command": "du -sh /root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/"
}
```

### [2026-05-21 18:57:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`



(Command exited with code 1)

### [2026-05-21 18:57:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

20M	/root/.cache/huggingface/hub/models--Intel--Qwen3.6-27B-int4-mlx-AutoRound/blobs/

### [2026-05-21 18:58:02 UTC] ASSISTANT

**Thinking**

The blobs are only 20MB which confirms this model was not actually downloaded from HuggingFace - only the config files are there. The actual model weights are not present.

Since the actual quantized weights are not available and the config describes a custom quantization format (W4A16 via AutoRound/AutoGPTQ), I need to write a failed accuracy.json.

The error is: "Quantized model weights not found locally. Model hub blobs only contain config (20MB). Actual model files not accessible."

Let me write the final failed accuracy.json and exit.


**Tool call:** `exec`

```json
{
  "command": "cat > /tmp/failed_accuracy.json << 'EOF'\n{\n  \"model_id\": \"Intel/Qwen3.6-27B-int4-mlx-AutoRound\",\n  \"model_path\": \"Intel/Qwen3.6-27B-int4-mlx-AutoRound\",\n  \"scheme\": \"W4A16\",\n  \"device\": \"cuda:0\",\n  \"num_gpus\": \"1\",\n  \"tasks\": {},\n  \"status\": \"failed\",\n  \"duration_seconds\": 0.0,\n  \"eval_framework\": \"lm_eval+hf\",\n  \"errors\": [\n    \"HuggingFace hub cache contains only config files (20MB blob) - no model weights downloaded\",\n    \"The model was exported with mlx format but weights were never fetched to local cache\",\n    \"Cannot load model: AutoModelForCausalLM.from_pretrained fails because safetensors files are absent\",\n    \"Both vLLM and HF backends failed: vLLM requires recognized quantization method (auto_round not supported), HF fails with 'quant_method missing' in quantization config\",\n    \"The model uses a custom AutoRound quantization format incompatible with both vLLM 0.7.3 and transformers 5.9.0\"\n  ]\n}\nEOF\nmv /tmp/failed_accuracy.json /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json\necho \"Written failed accuracy.json\""
}
```

### [2026-05-21 18:58:02 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Written failed accuracy.json

### [2026-05-21 18:58:11 UTC] ASSISTANT

**Thinking**

The evaluation has failed because:
1. The HuggingFace hub cache contains only config files (~20MB) - no actual model weights were downloaded
2. The model was exported in MLX format but the safetensors files were never fetched
3. Both vLLM and HF backends can't handle this model: vLLM doesn't recognize "auto_round" quantization, and HF fails because the quantization config lacks `quant_method`

Let me verify the accuracy.json was written correctly.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json"
}
```

### [2026-05-21 18:58:11 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "model_id": "Intel/Qwen3.6-27B-int4-mlx-AutoRound",
  "model_path": "Intel/Qwen3.6-27B-int4-mlx-AutoRound",
  "scheme": "W4A16",
  "device": "cuda:0",
  "num_gpus": "1",
  "tasks": {},
  "status": "failed",
  "duration_seconds": 0.0,
  "eval_framework": "lm_eval+hf",
  "errors": [
    "HuggingFace hub cache contains only config files (20MB blob) - no model weights downloaded",
    "The model was exported with mlx format but weights were never fetched to local cache",
    "Cannot load model: AutoModelForCausalLM.from_pretrained fails because safetensors files are absent",
    "Both vLLM and HF backends failed: vLLM requires recognized quantization method (auto_round not supported), HF fails with 'quant_method missing' in quantization config",
    "The model uses a custom AutoRound quantization format incompatible with both vLLM 0.7.3 and transformers 5.9.0"
  ]
}

### [2026-05-21 18:58:20 UTC] ASSISTANT

**Thinking**

The evaluation has failed. Let me provide the final summary.

Key findings:
1. The model directory from HF cache only has config files, no weights (20MB blob = config + tokenizer, not model weights)
2. The model uses AutoRound quantization format which is not supported by vLLM 0.7.3 or transformers 5.9.0
3. Both vLLM and HF backends cannot load this model

The `accuracy.json` has been written with `status: "failed"` and detailed error messages.

The evaluate.sh script was created at the required path, though it failed to produce results due to the model being unavailable in a loadable format.






The evaluation failed. Here's a summary of what happened and the output:

**Root Cause:** The HuggingFace hub cache for `Intel/Qwen3.6-27B-int4-mlx-AutoRound` contains only config/tokenizer files (~20MB) — the actual model weight files (safetensors) were never downloaded. The model was exported in MLX format and appears to use a custom AutoRound quantization scheme that:
1. Is not recognized by **vLLM 0.7.3** (requires quantization method from known list: aqlm, awq, gptq, marlin, etc. — auto_round not supported)
2. Causes **transformers 5.9.0** to fail with `quant_method missing` when constructing `AutoQuantizationConfig`

**Files produced:**
- `/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/evaluate.sh` — standalone shell script (vLLM and HF attempts)
- `/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/accuracy.json` — failed status with detailed errors
- `/root/.openclaw/workspace/quantized/runs/Intel_Qwen3.6-27B-int4-mlx-AutoRound-W4A16/logs/eval_exec.log` — execution trace

**To resolve:** The model weights need to be downloaded to the HF cache, or the model needs to be re-exported in a format (like GGUF or standard AWQ/GPTQ) that is supported by the available inference backends. The current AutoRound format from MLX export is incompatible with both vLLM and HuggingFace transformers in this environment.
