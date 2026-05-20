# Session: autoeval_eval_294

- **Session ID:** `autoeval_eval_294`
- **Timestamp:** 2026-05-20 13:48:06 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 2: Evaluation

### [2026-05-20 13:48:06 UTC] USER

You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_eval/SKILL.md

Quantized model path: /root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16
Raw lm_eval output directory: /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results
Evaluation script path: /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh
Evaluation execution log path: /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/logs/eval_exec.log
Evaluation tasks: piqa,mmlu,hellaswag
Batch size: 8
Num gpus: 1

The quantized model was produced by auto_quant with scheme=W4A16, export_format=auto_round.
A venv may already exist at /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv (created by auto_quant with --system-site-packages).

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- If a venv already exists at /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv, reuse it - just install lm_eval and vllm into it.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- Write evaluation outputs, logs, prompts, copied request/session files, and other runtime artifacts to: /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16
- Before starting evaluation, you MUST first generate the evaluation script file:
    /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh
- The file name must be exactly: evaluate.sh
- The script must be a standalone shell program runnable with:
    bash /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh
- The generated evaluate.sh must focus on Stage A raw lm_eval execution only.
- Do NOT put venv creation, pip/uv installation, package bootstrap, JSON parsing, accuracy.json writing, or destructive cleanup/recreation into evaluate.sh.
- In this same OpenClaw task, prepare or reuse the evaluation environment separately before executing evaluate.sh.
- Prefer direct lm_eval CLI commands and standard shell environment variables over ad-hoc Python wrappers.
- In this same OpenClaw task, first write evaluate.sh, then execute that generated script yourself.
- When you execute evaluate.sh or a direct lm_eval command, you MUST stream stdout/stderr into this log file while still printing output:
    bash /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/logs/eval_exec.log
- /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json is a final summary artifact, not a progress marker.
- After Stage A raw lm_eval completes, parse the latest raw results and write /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json in a separate finalize step outside evaluate.sh.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json until Stage A raw lm_eval output and Stage B parsing have both completed successfully.
- Write /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json atomically via a temporary file and rename/move it into place only at finalize time.
- If evaluation fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json before exiting non-zero, also atomically.
- The overall evaluation workflow MUST still behave as two idempotent stages:
    Stage A: run lm_eval and persist raw results under /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results
    Stage B: parse the latest raw results into /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json
- Stage A and Stage B MUST be independently rerunnable.
- If /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results already contains a valid raw results file matching , do NOT rerun Stage A; only rerun Stage B parsing.
- If Stage B parsing fails after Stage A already succeeded, exit non-zero but preserve the raw lm_eval outputs, logs, and venv so the next retry only reruns parsing.
- Do NOT delete or recreate /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results when raw results already exist.
- Parsing logic should read the latest results file under /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/**/results_*.json rather than assuming lm_eval must be rerun.
- Do NOT leave the final artifact named eval.sh, eval_script.sh, or evaluate_script.sh.
- When the generated script invokes lm_eval, it MUST pass:
    --output_path /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results
  and MUST set max_gen_toks=2048. The placement depends on the backend:
    - For HF backend (--model hf):   --gen_kwargs max_gen_toks=2048
    - For vLLM backend (--model vllm): append max_gen_toks=2048 inside --model_args
- Do NOT omit --output_path. Keep the raw lm_eval output files under that exact directory for later upload.
- Do NOT omit max_gen_toks=2048 regardless of backend.

IMPORTANT - In this same OpenClaw task, after separate raw-eval and finalize steps, you MUST produce:

/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json - evaluation results:
{
  "model_id": "Zyphra/ZR1-1.5B",
  "model_path": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16",
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

/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/ - raw lm_eval output directory created by:
    lm_eval ... --output_path /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, the script may invoke python only for JSON post-processing, but it must still write accuracy.json with status=failed before exiting non-zero.

### [2026-05-20 13:48:11 UTC] ASSISTANT

**Thinking**

Let me start by reading the skill instructions to understand what I need to do.


**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/skills/auto_eval/SKILL.md"
}
```

### [2026-05-20 13:48:11 UTC] TOOL RESULT
**Tool:** `read`

---
name: auto_eval
description: Automatically evaluate quantized LLM models using HuggingFace (HF) inference backend and lm-evaluation-harness. Supports CUDA device with multi-GPU tensor parallelism. Detects Auto-Round quantization format and configures appropriate settings.
metadata:
  openclaw:
    emoji: "📊"
    homepage: https://github.com/EleutherAI/lm-evaluation-harness
    skillKey: auto-eval
    requires:
      bins: ["lm_eval"]
      env: ["torch", "transformers"]
      config: []
---

# Auto-Eval Skill

Use this skill when users want to evaluate quantized LLM models (especially Auto-Round quantized models) using HuggingFace backend and lm-evaluation-harness on CUDA GPUs.

## Overview

This skill provides a complete workflow for:
- **Detecting quantization format** from model metadata
- **Configuring HF backend** for CUDA with multi-GPU support
- **Running lm-eval** benchmarks (e.g., piqa, hellaswag, mmlu)
- **Handling OOM errors** with multi-GPU suggestions

**Supported tasks**: piqa, hellaswag, mmlu, arc, gsm8k, and many more via lm-eval

---

## Input Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `model_path` | Path to quantized model (local or HF) | Yes | - |
| `task` | Evaluation task (e.g., piqa, mmlu) | Yes | - |
| `output_path` | Directory for results | No | `./lm_eval_results` |
| `batch_size` | Batch size for evaluation | No | `1` |
| `num_gpus` | Number of GPUs to use for inference | No | `1` |
| `max_model_len` | Max sequence length | No | `8192` |
| `trust_remote_code` | Allow custom model code | No | `True` |
| `max_gen_toks` | Max tokens to generate (via `--gen_kwargs`) | No | `2048` |

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
--gen_kwargs max_gen_toks=2048
```

Rules:

1. Do **not** omit `--output_path`
2. Do **not** omit `--gen_kwargs max_gen_toks=2048` — this is required for all evaluation runs
3. If the caller provides a concrete runtime path, use it exactly
4. Prefer the `lm_eval` CLI over ad-hoc wrappers so raw result artifacts are persisted automatically

---

## Step 1: Analyze Model and Detect Quantization Format

### Check quantization_config.json

```bash
# For Auto-Round quantized models
cat {model_path}/quantization_config.json
```

**Common quantization formats and HF support:**

| Format | packing_format | HF Support | Device |
|--------|---------------|-------------|--------|
| auto_gptq | `auto_gptq` | ✓ | CUDA |
| auto_awq | `auto_awq` | ✓ | CUDA |
| auto_round | `auto_round:auto_gptq` | ✓ | CUDA |
| llm_compressor | `auto_round:llm_compressor` | ✓ | CUDA |
| GGUF | `gguf` | ✓ (llama.cpp) | CPU |

### Example quantization_config.json for different formats:

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

## Step 2: Configure HF Model Args

Based on quantization format and GPU count, configure appropriate HF arguments:

### For Single GPU (num_gpus=1)

**Recommended settings:**
```bash
--model_args pretrained=./quantized_model,\
dtype=bfloat16,\
device_map=auto,\
trust_remote_code=True
```

### For Multi-GPU (num_gpus > 1)

**Recommended settings using device_map + parallelized:**
```bash
--model_args pretrained=./quantized_model,\
dtype=bfloat16,\
device_map=auto,\
parallelize=True,\
trust_remote_code=True
```

**Alternative: tensor_parallel via model_args (for vLLM backend):**
```bash
--model_args pretrained=./quantized_model,\
dtype=bfloat16,\
tensor_parallel_size=$NUM_GPUS,\
trust_remote_code=True
```

### Key Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `dtype` | Data type | `bfloat16` for CUDA |
| `device_map` | Device placement | `auto` for automatic |
| `parallelize` | Multi-GPU parallelization | `True` when num_gpus > 1 |
| `trust_remote_code` | Allow custom model code | `True` |
| `max_model_len` | Max sequence length | 8192 (adjust based on VRAM) |

### Format-Specific Model Args

**For auto_gptq format:**
```bash
--model_args pretrained=./quantized_model,dtype=bfloat16,device_map=auto,trust_remote_code=True
```

**For auto_round format:**
```bash
--model_args pretrained=./quantized_model,dtype=bfloat16,device_map=auto,trust_remote_code=True
```

**For llm_compressor (MXFP4/NVFP4) format:**
```bash
--model_args pretrained=./quantized_model,dtype=bfloat16,device_map=auto,trust_remote_code=True
```

---

## Step 3: Run lm-eval

### Basic Command (Single GPU)

```bash
lm_eval \
    --model hf \
    --model_args pretrained=./quantized_model,dtype=bfloat16,device_map=auto,trust_remote_code=True \
    --tasks piqa \
    --batch_size 1 \
    --output_path lm_eval_results \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda
```

### Multi-GPU Command (User specifies GPU count)

```bash
# Example: use 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

lm_eval \
    --model hf \
    --model_args pretrained=./quantized_model,dtype=bfloat16,device_map=auto,parallelize=True,trust_remote_code=True \
    --tasks piqa \
    --batch_size 1 \
    --output_path lm_eval_results \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda
```

### Running Multiple Tasks

```bash
# Multiple tasks
lm_eval \
    --model hf \
    --model_args pretrained=./quantized_model,dtype=bfloat16,device_map=auto,trust_remote_code=True \
    --tasks piqa,hellaswag,mmlu \
    --batch_size 1 \
    --output_path lm_eval_results \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda

# Task group
lm_eval \
    --model hf \
    --model_args pretrained=./quantized_model,dtype=bfloat16,device_map=auto,trust_remote_code=True \
    --tasks arc_easy,arc_challenge,piqa,hellaswag \
    --batch_size 1 \
    --output_path lm_eval_results \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda
```

### Using lm_eval Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=./quantized_model,dtype=bfloat16,device_map=auto,trust_remote_code=True",
    tasks="piqa",
    batch_size="auto",
    device="cuda",
    gen_kwargs={"max_gen_toks": 2048}
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

#### 1. HF Backend Not Found

**Error:**
```
ValueError: Unknown model: hf
```

**Solution:**
```bash
# Install lm-eval with HF support
$VENV_PY -m pip install -U uv
uv pip install --python "$VENV_PY" lm-eval[torch]

# Or ensure transformers is importable
"$VENV_PY" -c "import transformers; print(transformers.__version__)"
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

1. **Use multi-GPU (RECOMMENDED):**
   ```bash
   # Set CUDA_VISIBLE_DEVICES to use multiple GPUs
   export CUDA_VISIBLE_DEVICES=0,1
   export NUM_GPUS=2
   
   # Use parallelize=True in model_args
   --model_args ...parallelize=True
   ```

2. **Reduce max_model_len:**
   ```bash
   --model_args ...max_model_len=2048
   ```

3. **Reduce batch_size:**
   ```bash
   --batch_size 1
   ```

4. **Reduce gpu_memory_utilization (if using device_map):**
   ```bash
   --model_args ...offload_folder="./offload"
   ```

#### 4. Quantization Format Not Supported

**Error:**
```
RuntimeError: Unsupported quantization format
```

**Solution:**
```python
# Check quantization_config.json format
# For unsupported formats, try with trust_remote_code=True
--model_args ...trust_remote_code=True
```

#### 5. Import Error: transformers module

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Install transformers
uv pip install --python "$VENV_PY" transformers

# Install with accelerator for better GPU support
uv pip install --python "$VENV_PY" transformers accelerate
```

#### 6. Multi-GPU Parallelization Error

**Error:**
```
RuntimeError: parallelize=True requires multiple GPUs
```

**Solution:**
```bash
# Make sure CUDA_VISIBLE_DEVICES has multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4

# Verify GPU count
python -c "import torch; print(torch.cuda.device_count())"
```

---

## Step 6: Evaluation Script Template

### Complete Evaluation Script

```python
#!/usr/bin/env python3
"""
Auto-Eval Script for Quantized Models (CUDA + HF Backend)
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
    
    # Detect format
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


def build_hf_args(model_path: str, num_gpus: int = 1, **kwargs) -> str:
    """Build HF model arguments string."""
    
    # Base args
    args = [
        f"pretrained={model_path}",
        "dtype=bfloat16",
        "trust_remote_code=True",
    ]
    
    # Multi-GPU settings
    if num_gpus > 1:
        args.append("device_map=auto")
        args.append("parallelize=True")
    else:
        args.append("device_map=auto")
    
    # Add user overrides
    max_model_len = kwargs.get("max_model_len", 8192)
    
    return ",".join(args)


def run_evaluation(
    model_path: str,
    tasks: str,
    output_path: str = "./lm_eval_results",
    batch_size: int = 1,
    num_gpus: int = 1,
    **kwargs
):
    """Run lm-eval evaluation using HF backend."""
    import lm_eval
    
    # Detect format
    quant_info = detect_quantization_format(model_path)
    print(f"Detected quantization: {quant_info}")
    print(f"Using {num_gpus} GPU(s)")
    
    # Build model args
    model_args = build_hf_args(model_path, num_gpus, **kwargs)
    print(f"Model args: {model_args}")
    
    # Run evaluation
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=str(batch_size),
        device="cuda",
        output_path=output_path,
        gen_kwargs={"max_gen_toks": 2048}
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    for task, metrics in results["results"].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    result_file = os.path.join(output_path, "results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Eval for Quantized Models (CUDA+HF)")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--tasks", required=True, help="Comma-separated tasks")
    parser.add_argument("--output", default="./lm_eval_results", help="Output dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--max_model_len", type=int, default=8192)
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model,
        tasks=args.tasks,
        output_path=args.output,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len
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
export CUDA_VISIBLE_DEVICES=0,1
python eval_script.py \
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

# Install transformers and accelerate
uv pip install --python "$VENV_PY" transformers accelerate

# Install lm-eval with torch
uv pip install --python "$VENV_PY" lm-eval[torch]

# Only if torch is missing or incompatible, install a matching CUDA wheel
# uv pip install --python "$VENV_PY" --index-url https://download.pytorch.org/whl/cu124 torch

# Only if flash_attn is required and missing
# uv pip install --python "$VENV_PY" flash-attn --no-build-isolation

# Verify installations
"$VENV_PY" -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"
"$VENV_PY" -c "import transformers; print('Transformers:', transformers.__version__)"
"$VENV_PY" -c "import lm_eval; print('lm-eval:', lm_eval.__version__)"
```

---

## Step 8: Quick Reference

### Complete Example for Auto-Round W4A16 Model (Single GPU)

```bash
# Model path
MODEL_PATH="/path/to/quantized"

# Run piqa evaluation
lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks piqa \
    --batch_size 1 \
    --output_path ./lm_eval_results \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda
```

### Complete Example for Auto-Round W4A16 Model (Multi-GPU)

```bash
# Model path
MODEL_PATH="/path/to/quantized"

# Use 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

# Run piqa evaluation with multi-GPU
lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,parallelize=True,trust_remote_code=True" \
    --tasks piqa \
    --batch_size 1 \
    --output_path ./lm_eval_results \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda
```

### Run Multiple Benchmarks

```bash
lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks piqa,hellaswag,mmlu,arc_easy \
    --batch_size 1 \
    --output_path ./results \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda
```

### Using Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=/path/to/quantized,dtype=bfloat16,device_map=auto,trust_remote_code=True",
    tasks="piqa",
    batch_size="auto",
    device="cuda",
    gen_kwargs={"max_gen_toks": 2048}
)
print(results["results"]["piqa"])
```

---

## OOM Resolution Guide

When encountering CUDA out of memory errors, follow this decision tree:

1. **First try: Multi-GPU** (Most effective)
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1  # 2 GPUs
   --model_args ...parallelize=True
   ```

2. **Second try: Reduce max_model_len**
   ```bash
   --model_args ...max_model_len=2048
   ```

3. **Third try: Reduce batch_size**
   ```bash
   --batch_size 1
   ```

4. **Fourth try: Enable CPU offload**
   ```bash
   --model_args ...offload_folder="./offload"
   ```

---

## Notes

- **Multi-GPU requires `parallelize=True`** - Enables model parallelism across GPUs
- **Batch size** - Use `1` or `auto` for quantized models to avoid OOM
- **Quantization formats** - Auto-Round exports to `auto_round` format with `packing_format` metadata
- **device_map=auto** - Automatically places layers on available GPUs
- **trust_remote_code=True** - Required for custom model architectures


### [2026-05-20 13:48:27 UTC] ASSISTANT

**Thinking**

Let me analyze the task at hand. I need to evaluate a quantized LLM model with the following details:

- Model path: /root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16
- Scheme: W4A16 (weight-only quantization, 4-bit weights, 16-bit activations)
- Export format: auto_round
- Tasks: piqa, mmlu, hellaswag
- Batch size: 8
- Num GPUs: 1
- Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16
- Raw lm_eval output: /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results

Key requirements:
1. Use the auto_eval skill
2. The model is quantized with auto_round scheme (W4A16)
3. Run lm_eval with HF backend
4. Output path must be /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results
5. max_gen_toks=2048 must be set
6. Tasks: piqa, mmlu, hellaswag
7. Batch size: 8
8. Num gpus: 1
9. Write evaluate.sh first, then execute it
10. Stream output to eval_exec.log
11. After evaluation, parse results and write accuracy.json

Let me first check the model structure to understand what we're dealing with.




I'll evaluate the quantized model following the auto_eval skill. Let me start by checking the model structure and existing files.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ 2>/dev/null | head -30"
}
```

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/quantization_config.json 2>/dev/null"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/ 2>/dev/null"
}
```

### [2026-05-20 13:48:27 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 4
drwxr-xr-x 3 root root   37 May 20 13:44 .
drwxr-xr-x 4 root root   59 May 20 13:40 ..
drwxr-xr-x 2 root root 4096 May 20 13:44 ZR1-1.5B-w4g128

### [2026-05-20 13:48:27 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`



(Command exited with code 1)

### [2026-05-20 13:48:27 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 148
drwxr-xr-x 4 root root    154 May 20 13:48 .
drwxr-xr-x 3 root root     43 May 20 13:40 ..
drwxr-xr-x 2 root root    118 May 20 13:48 logs
-rw-r--r-- 1 root root   1572 May 20 13:47 quant_summary.json
-rw-r--r-- 1 root root   1174 May 20 13:41 quantize.py
-rw-r--r-- 1 root root   1068 May 20 13:40 request.json
-rw-r--r-- 1 root root 138990 May 20 13:48 session_quant_294.jsonl
drwxr-xr-x 6 root root    132 May 20 13:41 venv

### [2026-05-20 13:48:44 UTC] ASSISTANT

**Thinking**

I see the actual model directory is `/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128`. Let me check the structure and quantization config there.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/"
}
```

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/quantization_config.json 2>/dev/null"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/ 2>/dev/null || echo \"No lm_eval_results yet\""
}
```

### [2026-05-20 13:48:44 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 1587956
drwxr-xr-x 2 root root      4096 May 20 13:44 .
drwxr-xr-x 3 root root        37 May 20 13:44 ..
-rw-r--r-- 1 root root      2237 May 20 13:44 chat_template.jinja
-rw-r--r-- 1 root root      1716 May 20 13:44 config.json
-rw-r--r-- 1 root root       180 May 20 13:44 generation_config.json
-rw-r--r-- 1 root root 680876600 May 20 13:44 model-00001-of-00002.safetensors
-rw-r--r-- 1 root root 933676568 May 20 13:44 model-00002-of-00002.safetensors
-rw-r--r-- 1 root root     59813 May 20 13:44 model.safetensors.index.json
-rw-r--r-- 1 root root       273 May 20 13:44 quantization_config.json
-rw-r--r-- 1 root root  11422774 May 20 13:44 tokenizer.json
-rw-r--r-- 1 root root       459 May 20 13:44 tokenizer_config.json

### [2026-05-20 13:48:44 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "bits": 4,
  "data_type": "int",
  "group_size": 128,
  "sym": true,
  "enable_quanted_input": false,
  "iters": 0,
  "autoround_version": "0.13.0",
  "block_name_to_quantize": "model.layers",
  "quant_method": "auto-round",
  "packing_format": "auto_round:auto_gptq"
}

### [2026-05-20 13:48:44 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

No lm_eval_results yet

### [2026-05-20 13:48:51 UTC] ASSISTANT

**Thinking**

Good. The model is in `/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/`. It's auto-round format.

Now I need to:
1. Create the evaluate.sh script
2. Set up the environment
3. Run the evaluation
4. Parse results and write accuracy.json

Let me first check the venv and what's available.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/quant_summary.json"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/ | head -20"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python -c \"import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count())\" 2>/dev/null || echo \"torch not available\""
}
```

### [2026-05-20 13:48:51 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "model_id": "Zyphra/ZR1-1.5B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16",
  "status": "success",
  "duration_seconds": 100.0,
  "original_size_mb": 3072.0,
  "quantized_size_mb": 1638.0,
  "compression_ratio": 1.88,
  "errors": [],
  "solutions": [],
  "output_files": [
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/chat_template.jinja",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/config.json",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/generation_config.json",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/model-00001-of-00002.safetensors",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/model-00002-of-00002.safetensors",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/model.safetensors.index.json",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/quantization_config.json",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/tokenizer.json",
    "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128/tokenizer_config.json"
  ]
}

### [2026-05-20 13:48:51 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 58220
drwxr-xr-x 2 root root     4096 May 20 13:42 .
drwxr-xr-x 6 root root      132 May 20 13:41 ..
-rw-r--r-- 1 root root     9033 May 20 13:41 Activate.ps1
-rwxr-xr-x 1 root root      378 May 20 13:42 accelerate
-rwxr-xr-x 1 root root      370 May 20 13:42 accelerate-config
-rwxr-xr-x 1 root root      372 May 20 13:42 accelerate-estimate-memory
-rwxr-xr-x 1 root root      370 May 20 13:42 accelerate-launch
-rwxr-xr-x 1 root root      369 May 20 13:42 accelerate-merge-weights
-rw-r--r-- 1 root root     2254 May 20 13:41 activate
-rw-r--r-- 1 root root      963 May 20 13:41 activate.csh
-rw-r--r-- 1 root root     2238 May 20 13:41 activate.fish
-rwxr-xr-x 1 root root      361 May 20 13:42 auto-round
-rwxr-xr-x 1 root root      371 May 20 13:42 auto-round-best
-rwxr-xr-x 1 root root      371 May 20 13:42 auto-round-eval
-rwxr-xr-x 1 root root      373 May 20 13:42 auto-round-light
-rwxr-xr-x 1 root root      371 May 20 13:42 auto-round-mllm
-rwxr-xr-x 1 root root      377 May 20 13:42 auto-round-opt-rtn
-rwxr-xr-x 1 root root      369 May 20 13:42 auto-round-rtn
-rwxr-xr-x 1 root root      361 May 20 13:42 auto_round

### [2026-05-20 13:48:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

CUDA: True 1

### [2026-05-20 13:48:56 UTC] ASSISTANT

**Thinking**

Good - torch + CUDA works with the venv. Now I need to check if lm_eval is installed in the venv.


**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python -c \"import lm_eval; print('lm_eval ok')\" 2>/dev/null || echo \"lm_eval not installed\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python -c \"import transformers; print('transformers:', transformers.__version__)\" 2>/dev/null || echo \"transformers not installed\""
}
```

### [2026-05-20 13:48:56 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lm_eval not installed

### [2026-05-20 13:48:56 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

transformers: 5.8.1

### [2026-05-20 13:49:00 UTC] ASSISTANT

**Thinking**

lm_eval is not installed but transformers is. I need to install lm_eval in the existing venv. Let me install it using uv pip.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python \"lm-eval[torch]\" 2>&1",
  "timeout": 120
}
```

### [2026-05-20 13:49:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Using Python 3.12.13 environment at: quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv
Resolved 72 packages in 1.20s
   Building sqlitedict==2.1.0
   Building word2number==1.1
   Building rouge-score==0.1.2
Downloading scikit-learn (8.5MiB)
Downloading nltk (1.5MiB)
Downloading lm-eval (8.5MiB)
Downloading lxml (5.0MiB)
Downloading scipy (33.6MiB)
      Built sqlitedict==2.1.0
      Built word2number==1.1
      Built rouge-score==0.1.2
 Downloaded nltk
 Downloaded lxml
 Downloaded scikit-learn
 Downloaded scipy
 Downloaded lm-eval
Prepared 26 packages in 1.46s
Installed 26 packages in 3.02s
 + absl-py==2.4.0
 + chardet==6.0.0.post1
 + colorama==0.4.6
 + dataproperty==1.1.1
 + evaluate==0.4.6
 + joblib==1.5.3
 + lm-eval==0.4.12
 + lxml==6.1.1
 + mbstrdecoder==1.1.5
 + more-itertools==11.0.2
 + nltk==3.9.4
 + pathvalidate==3.3.1
 + portalocker==3.2.0
 + pytablewriter==1.2.1
 + pytz==2026.2
 + rouge-score==0.1.2
 + sacrebleu==2.6.0
 + scikit-learn==1.8.0
 + scipy==1.17.1
 + sqlitedict==2.1.0
 + tabledata==1.3.5
 + tabulate==0.10.0
 + tcolorpy==0.1.7
 + threadpoolctl==3.6.0
 + typepy==1.3.5
 + word2number==1.1
warning: The package `lm-eval==0.4.12` does not have an extra named `torch`

### [2026-05-20 13:49:13 UTC] ASSISTANT

**Thinking**

Good, lm_eval is installed now. Let me verify it works and then create the evaluate.sh script.

The model path is: `/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128`
Tasks: piqa, mmlu, hellaswag
Batch size: 8
Num gpus: 1
max_gen_toks: 2048

Let me first verify lm_eval works, then write the evaluate.sh script.


**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python -c \"import lm_eval; print('lm_eval ok')\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python -c \"from lm_eval.api.model import LM\" 2>&1 | head -5"
}
```

### [2026-05-20 13:49:13 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lm_eval ok

### [2026-05-20 13:49:14 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-20 13:49:28 UTC] ASSISTANT

**Thinking**

Good, lm_eval is installed. Now let me write the evaluate.sh script and then execute it.

Model path: /root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128
Tasks: piqa, mmlu, hellaswag
Batch size: 8
Num GPUs: 1

The model is auto_round format with auto_gptq packing. It uses device_map=auto for single GPU.

Let me create the evaluate.sh script:


**Tool call:** `write`

```json
{
  "content": "#!/bin/bash\nset -e\n\nMODEL_PATH=\"/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128\"\nOUTPUT_PATH=\"/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results\"\nTASKS=\"piqa,mmlu,hellaswag\"\nBATCH_SIZE=8\nNUM_GPUS=1\n\nVENV_PY=\"/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python\"\n\nexport CUDA_VISIBLE_DEVICES=0\n\n$VENV_PY -m lm_eval --model hf \\\n    --model_args \"pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True\" \\\n    --tasks $TASKS \\\n    --batch_size $BATCH_SIZE \\\n    --output_path $OUTPUT_PATH \\\n    --gen_kwargs max_gen_toks=2048 \\\n    --device cuda",
  "path": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh"
}
```

### [2026-05-20 13:49:28 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 644 bytes to /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh

### [2026-05-20 13:49:37 UTC] ASSISTANT

**Thinking**

Now I need to run the evaluate.sh script and stream output to the log file.


**Tool call:** `exec`

```json
{
  "command": "bash /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/logs/eval_exec.log",
  "timeout": 1200
}
```

### [2026-05-20 13:49:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session salty-gulf, pid 1552). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-20 13:49:51 UTC] ASSISTANT

**Thinking**

The command is still running. Let me poll it with a timeout to check on progress.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "salty-gulf",
  "timeout": 600000
}
```

### [2026-05-20 13:51:51 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

2026-05-20:13:49:43 INFO     [_cli.run:388] Selected Tasks: ['piqa', 'mmlu', 'hellaswag']
2026-05-20:13:49:44 INFO     [evaluator:214] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
2026-05-20:13:49:44 WARNING  [evaluator:226] generation_kwargs: {'max_gen_toks': 2048} specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!
2026-05-20:13:49:44 INFO     [evaluator:239] Initializing hf model, with arguments: {'pretrained': '/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128', 'dtype': 'bfloat16', 'device_map': 'auto', 'trust_remote_code': True}
2026-05-20:13:49:47 INFO     [models.huggingface:286] Using device 'cuda'
2026-05-20:13:49:48 INFO     [models.huggingface:585] Model parallel was set to False.
/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/auto.py:262: UserWarning: You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading already has a `quantization_config` attribute. The `quantization_config` from the model will be used.However, loading attributes (e.g. ['backend']) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored.
  warnings.warn(warning_msg)
[33;1m2026-05-20 13:49:48 WARNING modeling_utils.py L4555: `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.[0m
[33;1m2026-05-20 13:49:48 WARNING backend.py L1176: Better backend is found, please install all the following requirements to enable it.[0m
[33;1m2026-05-20 13:49:48 WARNING backend.py L1176: `pip install "torch>=2.8.0" "auto-round-lib"`[0m
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Generating train split:   0%|          | 0/16113 [00:00<?, ? examples/s]
Generating train split: 100%|██████████| 16113/16113 [00:00<00:00, 1127771.26 examples/s]

Generating validation split:   0%|          | 0/1838 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 1838/1838 [00:00<00:00, 419020.04 examples/s]

Generating test split:   0%|          | 0/3084 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 3084/3084 [00:00<00:00, 952591.03 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 66629.13 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13346.06 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6432.98 examples/s]

Generating test split:   0%|          | 0/135 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 135/135 [00:00<00:00, 89636.07 examples/s]

Generating validation split:   0%|          | 0/14 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 14/14 [00:00<00:00, 19804.47 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6415.27 examples/s]

Generating test split:   0%|          | 0/152 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 152/152 [00:00<00:00, 94100.99 examples/s]

Generating validation split:   0%|          | 0/16 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 16/16 [00:00<00:00, 22024.57 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6775.94 examples/s]

Generating test split:   0%|          | 0/144 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 144/144 [00:00<00:00, 102595.51 examples/s]

Generating validation split:   0%|          | 0/16 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 16/16 [00:00<00:00, 21995.69 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6358.86 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 28585.18 examples/s]

Generating validation split:   0%|          | 0/8 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 8/8 [00:00<00:00, 5329.48 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 2687.97 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 63947.31 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 11082.72 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 5961.21 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 65088.52 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13629.94 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6387.91 examples/s]

Generating test split:   0%|          | 0/102 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 102/102 [00:00<00:00, 73584.28 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 15613.31 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6470.69 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 63559.69 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13307.57 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6476.69 examples/s]

Generating test split:   0%|          | 0/235 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 235/235 [00:00<00:00, 158517.44 examples/s]

Generating validation split:   0%|          | 0/26 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 26/26 [00:00<00:00, 36290.15 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7128.32 examples/s]

Generating test split:   0%|          | 0/145 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 145/145 [00:00<00:00, 92865.18 examples/s]

Generating validation split:   0%|          | 0/16 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 16/16 [00:00<00:00, 21163.31 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6884.94 examples/s]

Generating test split:   0%|          | 0/378 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 378/378 [00:00<00:00, 211986.48 examples/s]

Generating validation split:   0%|          | 0/41 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 41/41 [00:00<00:00, 37133.76 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6387.91 examples/s]

Generating test split:   0%|          | 0/310 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 310/310 [00:00<00:00, 172421.99 examples/s]

Generating validation split:   0%|          | 0/32 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 32/32 [00:00<00:00, 30202.01 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 4975.45 examples/s]

Generating test split:   0%|          | 0/203 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 203/203 [00:00<00:00, 131193.18 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 26855.26 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6395.71 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 50821.57 examples/s]

Generating validation split:   0%|          | 0/9 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 9/9 [00:00<00:00, 10430.71 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6202.76 examples/s]

Generating test split:   0%|          | 0/270 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 270/270 [00:00<00:00, 173903.88 examples/s]

Generating validation split:   0%|          | 0/29 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 29/29 [00:00<00:00, 33619.35 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6626.07 examples/s]

Generating test split:   0%|          | 0/151 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 151/151 [00:00<00:00, 86403.81 examples/s]

Generating validation split:   0%|          | 0/17 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 17/17 [00:00<00:00, 20897.76 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6533.18 examples/s]

Generating test split:   0%|          | 0/216 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 216/216 [00:00<00:00, 122477.99 examples/s]

Generating validation split:   0%|          | 0/23 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 23/23 [00:00<00:00, 28642.81 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6324.34 examples/s]

Generating test split:   0%|          | 0/112 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 112/112 [00:00<00:00, 76408.92 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 14355.12 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6582.40 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 65362.38 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13867.55 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6506.83 examples/s]

Generating test split:   0%|          | 0/265 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 265/265 [00:00<00:00, 164738.49 examples/s]

Generating validation split:   0%|          | 0/29 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 29/29 [00:00<00:00, 34594.66 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6535.22 examples/s]

Generating test split:   0%|          | 0/173 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 173/173 [00:00<00:00, 105774.72 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 27635.43 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7002.18 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 57354.08 examples/s]

Generating validation split:   0%|          | 0/10 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 10/10 [00:00<00:00, 11878.52 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6241.52 examples/s]

Generating test split:   0%|          | 0/223 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 223/223 [00:00<00:00, 138056.06 examples/s]

Generating validation split:   0%|          | 0/23 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 23/23 [00:00<00:00, 29987.25 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7094.56 examples/s]

Generating test split:   0%|          | 0/103 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 103/103 [00:00<00:00, 66269.87 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13768.23 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6693.75 examples/s]

Generating test split:   0%|          | 0/234 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 234/234 [00:00<00:00, 146051.66 examples/s]

Generating validation split:   0%|          | 0/25 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 25/25 [00:00<00:00, 28378.24 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6460.73 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 65917.08 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13784.69 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6466.70 examples/s]

Generating test split:   0%|          | 0/783 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 783/783 [00:00<00:00, 248610.15 examples/s]

Generating validation split:   0%|          | 0/86 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 86/86 [00:00<00:00, 97252.67 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6813.36 examples/s]

Generating test split:   0%|          | 0/306 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 306/306 [00:00<00:00, 164630.20 examples/s]

Generating validation split:   0%|          | 0/33 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 33/33 [00:00<00:00, 39750.73 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6632.36 examples/s]

Generating test split:   0%|          | 0/282 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 282/282 [00:00<00:00, 164140.12 examples/s]

Generating validation split:   0%|          | 0/31 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 31/31 [00:00<00:00, 36740.16 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6749.77 examples/s]

Generating test split:   0%|          | 0/272 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 272/272 [00:00<00:00, 128895.12 examples/s]

Generating validation split:   0%|          | 0/31 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 31/31 [00:00<00:00, 38708.97 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6488.71 examples/s]

Generating test split:   0%|          | 0/166 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 166/166 [00:00<00:00, 113600.01 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 22968.50 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6282.66 examples/s]

Generating test split:   0%|          | 0/114 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 114/114 [00:00<00:00, 71783.61 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 15563.28 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6496.75 examples/s]

Generating test split:   0%|          | 0/198 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 198/198 [00:00<00:00, 122236.12 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 25760.66 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6254.55 examples/s]

Generating test split:   0%|          | 0/193 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 193/193 [00:00<00:00, 118278.88 examples/s]

Generating validation split:   0%|          | 0/21 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 21/21 [00:00<00:00, 27001.96 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7020.93 examples/s]

Generating test split:   0%|          | 0/390 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 390/390 [00:00<00:00, 237206.87 examples/s]

Generating validation split:   0%|          | 0/43 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 43/43 [00:00<00:00, 52307.16 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6898.53 examples/s]

Generating test split:   0%|          | 0/238 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 238/238 [00:00<00:00, 144380.15 examples/s]

Generating validation split:   0%|          | 0/26 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 26/26 [00:00<00:00, 30317.46 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6184.46 examples/s]

Generating test split:   0%|          | 0/545 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 545/545 [00:00<00:00, 294043.69 examples/s]

Generating validation split:   0%|          | 0/60 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 60/60 [00:00<00:00, 70631.00 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6376.26 examples/s]

Generating test split:   0%|          | 0/131 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 131/131 [00:00<00:00, 84883.95 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 12787.51 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 5589.42 examples/s]

Generating test split:   0%|          | 0/612 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 612/612 [00:00<00:00, 198600.70 examples/s]

Generating validation split:   0%|          | 0/69 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 69/69 [00:00<00:00, 70074.33 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6202.76 examples/s]

Generating test split:   0%|          | 0/110 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 110/110 [00:00<00:00, 71753.26 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 14921.92 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6226.70 examples/s]

Generating test split:   0%|          | 0/245 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 245/245 [00:00<00:00, 121194.07 examples/s]

Generating validation split:   0%|          | 0/27 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 27/27 [00:00<00:00, 31085.98 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6615.62 examples/s]

Generating test split:   0%|          | 0/201 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 201/201 [00:00<00:00, 125904.29 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 26485.27 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6036.71 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 67759.35 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13968.31 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6043.67 examples/s]

Generating test split:   0%|          | 0/126 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 126/126 [00:00<00:00, 81317.48 examples/s]

Generating validation split:   0%|          | 0/14 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 14/14 [00:00<00:00, 16791.61 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6563.86 examples/s]

Generating test split:   0%|          | 0/165 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 165/165 [00:00<00:00, 86702.60 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 23237.14 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6252.69 examples/s]

Generating test split:   0%|          | 0/204 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 204/204 [00:00<00:00, 90247.65 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 24865.18 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6370.45 examples/s]

Generating test split:   0%|          | 0/237 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 237/237 [00:00<00:00, 29982.81 examples/s]

Generating validation split:   0%|          | 0/26 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 26/26 [00:00<00:00, 24385.49 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6403.52 examples/s]

Generating test split:   0%|          | 0/121 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 121/121 [00:00<00:00, 75387.82 examples/s]

Generating validation split:   0%|          | 0/13 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 13/13 [00:00<00:00, 16938.79 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6119.50 examples/s]

Generating test split:   0%|          | 0/108 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 108/108 [00:00<00:00, 53367.68 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 12202.42 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6574.14 examples/s]

Generating test split:   0%|          | 0/163 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 163/163 [00:00<00:00, 109021.14 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 22290.37 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6345.39 examples/s]

Generating test split:   0%|          | 0/346 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 346/346 [00:00<00:00, 191783.95 examples/s]

Generating validation split:   0%|          | 0/38 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 38/38 [00:00<00:00, 45878.97 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6698.03 examples/s]

Generating test split:   0%|          | 0/895 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 895/895 [00:00<00:00, 447585.80 examples/s]

Generating validation split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 100/100 [00:00<00:00, 115928.80 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6603.12 examples/s]

Process still running.

### [2026-05-20 13:52:06 UTC] ASSISTANT

**Thinking**

The evaluation is still running. Let me wait a bit more.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "salty-gulf",
  "timeout": 300000
}
```

### [2026-05-20 13:54:06 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

Generating test split:   0%|          | 0/311 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 311/311 [00:00<00:00, 183052.00 examples/s]

Generating validation split:   0%|          | 0/34 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 34/34 [00:00<00:00, 40057.96 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6527.08 examples/s]

Generating test split:   0%|          | 0/324 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 324/324 [00:00<00:00, 201954.90 examples/s]

Generating validation split:   0%|          | 0/35 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 35/35 [00:00<00:00, 43227.51 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6527.08 examples/s]

Generating test split:   0%|          | 0/1534 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 1534/1534 [00:00<00:00, 286085.48 examples/s]

Generating validation split:   0%|          | 0/170 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 170/170 [00:00<00:00, 153406.13 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6335.81 examples/s]

Generating test split:   0%|          | 0/171 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 171/171 [00:00<00:00, 114792.89 examples/s]

Generating validation split:   0%|          | 0/19 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 19/19 [00:00<00:00, 24237.16 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6582.40 examples/s]

Generating train split:   0%|          | 0/39905 [00:00<?, ? examples/s]
Generating train split: 100%|██████████| 39905/39905 [00:00<00:00, 516199.94 examples/s]

Generating test split:   0%|          | 0/10003 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 10003/10003 [00:00<00:00, 277489.78 examples/s]

Generating validation split:   0%|          | 0/10042 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 10042/10042 [00:00<00:00, 607001.12 examples/s]

Map:   0%|          | 0/39905 [00:00<?, ? examples/s]
Map: 100%|██████████| 39905/39905 [00:02<00:00, 14893.56 examples/s]

Map:   0%|          | 0/10042 [00:00<?, ? examples/s]
Map: 100%|██████████| 10042/10042 [00:00<00:00, 11089.03 examples/s]
2026-05-20:13:52:07 INFO     [evaluator_utils:446] Selected tasks:
2026-05-20:13:52:07 INFO     [evaluator_utils:462] Group: mmlu
2026-05-20:13:52:07 INFO     [evaluator_utils:462]   Group: mmlu_stem
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_abstract_algebra (mmlu/default/mmlu_abstract_algebra.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_anatomy (mmlu/default/mmlu_anatomy.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_astronomy (mmlu/default/mmlu_astronomy.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_college_biology (mmlu/default/mmlu_college_biology.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_college_chemistry (mmlu/default/mmlu_college_chemistry.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_college_computer_science (mmlu/default/mmlu_college_computer_science.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_college_mathematics (mmlu/default/mmlu_college_mathematics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_college_physics (mmlu/default/mmlu_college_physics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_computer_security (mmlu/default/mmlu_computer_security.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_conceptual_physics (mmlu/default/mmlu_conceptual_physics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_electrical_engineering (mmlu/default/mmlu_electrical_engineering.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_elementary_mathematics (mmlu/default/mmlu_elementary_mathematics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_biology (mmlu/default/mmlu_high_school_biology.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_chemistry (mmlu/default/mmlu_high_school_chemistry.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_computer_science (mmlu/default/mmlu_high_school_computer_science.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_mathematics (mmlu/default/mmlu_high_school_mathematics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_physics (mmlu/default/mmlu_high_school_physics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_statistics (mmlu/default/mmlu_high_school_statistics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_machine_learning (mmlu/default/mmlu_machine_learning.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:462]   Group: mmlu_other
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_business_ethics (mmlu/default/mmlu_business_ethics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_clinical_knowledge (mmlu/default/mmlu_clinical_knowledge.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_college_medicine (mmlu/default/mmlu_college_medicine.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_global_facts (mmlu/default/mmlu_global_facts.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_human_aging (mmlu/default/mmlu_human_aging.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_management (mmlu/default/mmlu_management.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_marketing (mmlu/default/mmlu_marketing.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_medical_genetics (mmlu/default/mmlu_medical_genetics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_miscellaneous (mmlu/default/mmlu_miscellaneous.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_nutrition (mmlu/default/mmlu_nutrition.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_professional_accounting (mmlu/default/mmlu_professional_accounting.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_professional_medicine (mmlu/default/mmlu_professional_medicine.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_virology (mmlu/default/mmlu_virology.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:462]   Group: mmlu_social_sciences
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_econometrics (mmlu/default/mmlu_econometrics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_geography (mmlu/default/mmlu_high_school_geography.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_government_and_politics (mmlu/default/mmlu_high_school_government_and_politics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_macroeconomics (mmlu/default/mmlu_high_school_macroeconomics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_microeconomics (mmlu/default/mmlu_high_school_microeconomics.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_psychology (mmlu/default/mmlu_high_school_psychology.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_human_sexuality (mmlu/default/mmlu_human_sexuality.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_professional_psychology (mmlu/default/mmlu_professional_psychology.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_public_relations (mmlu/default/mmlu_public_relations.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_security_studies (mmlu/default/mmlu_security_studies.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_sociology (mmlu/default/mmlu_sociology.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_us_foreign_policy (mmlu/default/mmlu_us_foreign_policy.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:462]   Group: mmlu_humanities
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_formal_logic (mmlu/default/mmlu_formal_logic.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_european_history (mmlu/default/mmlu_high_school_european_history.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_us_history (mmlu/default/mmlu_high_school_us_history.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_high_school_world_history (mmlu/default/mmlu_high_school_world_history.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_international_law (mmlu/default/mmlu_international_law.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_jurisprudence (mmlu/default/mmlu_jurisprudence.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_logical_fallacies (mmlu/default/mmlu_logical_fallacies.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_moral_disputes (mmlu/default/mmlu_moral_disputes.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_moral_scenarios (mmlu/default/mmlu_moral_scenarios.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_philosophy (mmlu/default/mmlu_philosophy.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_prehistory (mmlu/default/mmlu_prehistory.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_professional_law (mmlu/default/mmlu_professional_law.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:470]     Task: mmlu_world_religions (mmlu/default/mmlu_world_religions.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:480] Task: hellaswag (hellaswag/hellaswag.yaml)
2026-05-20:13:52:07 INFO     [evaluator_utils:480] Task: piqa (piqa/piqa.yaml)
2026-05-20:13:52:07 INFO     [api.task:312] Building contexts for piqa on rank 0...

  0%|          | 0/1838 [00:00<?, ?it/s]
100%|██████████| 1838/1838 [00:00<00:00, 1984.14it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_abstract_algebra on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1892.92it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_anatomy on rank 0...

  0%|          | 0/135 [00:00<?, ?it/s]
100%|██████████| 135/135 [00:00<00:00, 1680.85it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_astronomy on rank 0...

  0%|          | 0/152 [00:00<?, ?it/s]
100%|██████████| 152/152 [00:00<00:00, 1703.28it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_college_biology on rank 0...

  0%|          | 0/144 [00:00<?, ?it/s]
100%|██████████| 144/144 [00:00<00:00, 1646.80it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_college_chemistry on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1892.72it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_college_computer_science on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1618.54it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_college_mathematics on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1628.62it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_college_physics on rank 0...

  0%|          | 0/102 [00:00<?, ?it/s]
100%|██████████| 102/102 [00:00<00:00, 1831.15it/s]
2026-05-20:13:52:08 INFO     [api.task:312] Building contexts for mmlu_computer_security on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1584.29it/s]
2026-05-20:13:52:09 INFO     [api.task:312] Building contexts for mmlu_conceptual_physics on rank 0...

  0%|          | 0/235 [00:00<?, ?it/s]
100%|██████████| 235/235 [00:00<00:00, 1743.27it/s]
2026-05-20:13:52:09 INFO     [api.task:312] Building contexts for mmlu_electrical_engineering on rank 0...

  0%|          | 0/145 [00:00<?, ?it/s]
100%|██████████| 145/145 [00:00<00:00, 1665.80it/s]
2026-05-20:13:52:09 INFO     [api.task:312] Building contexts for mmlu_elementary_mathematics on rank 0...

  0%|          | 0/378 [00:00<?, ?it/s]
100%|██████████| 378/378 [00:00<00:00, 1746.74it/s]
2026-05-20:13:52:09 INFO     [api.task:312] Building contexts for mmlu_high_school_biology on rank 0...

  0%|          | 0/310 [00:00<?, ?it/s]
100%|██████████| 310/310 [00:00<00:00, 1675.72it/s]
2026-05-20:13:52:09 INFO     [api.task:312] Building contexts for mmlu_high_school_chemistry on rank 0...

  0%|          | 0/203 [00:00<?, ?it/s]
100%|██████████| 203/203 [00:00<00:00, 1673.13it/s]
2026-05-20:13:52:09 INFO     [api.task:312] Building contexts for mmlu_high_school_computer_science on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1707.73it/s]
2026-05-20:13:52:09 INFO     [api.task:312] Building contexts for mmlu_high_school_mathematics on rank 0...

  0%|          | 0/270 [00:00<?, ?it/s]
100%|██████████| 270/270 [00:00<00:00, 1602.31it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_high_school_physics on rank 0...

  0%|          | 0/151 [00:00<?, ?it/s]
100%|██████████| 151/151 [00:00<00:00, 1703.74it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_high_school_statistics on rank 0...

  0%|          | 0/216 [00:00<?, ?it/s]
100%|██████████| 216/216 [00:00<00:00, 1768.75it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_machine_learning on rank 0...

  0%|          | 0/112 [00:00<?, ?it/s]
100%|██████████| 112/112 [00:00<00:00, 1605.76it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_business_ethics on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1904.89it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_clinical_knowledge on rank 0...

  0%|          | 0/265 [00:00<?, ?it/s]
100%|██████████| 265/265 [00:00<00:00, 1640.97it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_college_medicine on rank 0...

  0%|          | 0/173 [00:00<?, ?it/s]
100%|██████████| 173/173 [00:00<00:00, 1697.92it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_global_facts on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1551.31it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_human_aging on rank 0...

  0%|          | 0/223 [00:00<?, ?it/s]
100%|██████████| 223/223 [00:00<00:00, 1637.71it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_management on rank 0...

  0%|          | 0/103 [00:00<?, ?it/s]
100%|██████████| 103/103 [00:00<00:00, 1532.30it/s]
2026-05-20:13:52:10 INFO     [api.task:312] Building contexts for mmlu_marketing on rank 0...

  0%|          | 0/234 [00:00<?, ?it/s]
100%|██████████| 234/234 [00:00<00:00, 1765.91it/s]
2026-05-20:13:52:11 INFO     [api.task:312] Building contexts for mmlu_medical_genetics on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1653.07it/s]
2026-05-20:13:52:11 INFO     [api.task:312] Building contexts for mmlu_miscellaneous on rank 0...

  0%|          | 0/783 [00:00<?, ?it/s]
100%|██████████| 783/783 [00:00<00:00, 1716.70it/s]
2026-05-20:13:52:11 INFO     [api.task:312] Building contexts for mmlu_nutrition on rank 0...

  0%|          | 0/306 [00:00<?, ?it/s]
100%|██████████| 306/306 [00:00<00:00, 1715.75it/s]
2026-05-20:13:52:11 INFO     [api.task:312] Building contexts for mmlu_professional_accounting on rank 0...

  0%|          | 0/282 [00:00<?, ?it/s]
100%|██████████| 282/282 [00:00<00:00, 1668.92it/s]
2026-05-20:13:52:11 INFO     [api.task:312] Building contexts for mmlu_professional_medicine on rank 0...

  0%|          | 0/272 [00:00<?, ?it/s]
100%|██████████| 272/272 [00:00<00:00, 1676.32it/s]
2026-05-20:13:52:12 INFO     [api.task:312] Building contexts for mmlu_virology on rank 0...

  0%|          | 0/166 [00:00<?, ?it/s]
100%|██████████| 166/166 [00:00<00:00, 1712.50it/s]
2026-05-20:13:52:12 INFO     [api.task:312] Building contexts for mmlu_econometrics on rank 0...

  0%|          | 0/114 [00:00<?, ?it/s]
100%|██████████| 114/114 [00:00<00:00, 1881.45it/s]
2026-05-20:13:52:12 INFO     [api.task:312] Building contexts for mmlu_high_school_geography on rank 0...

  0%|          | 0/198 [00:00<?, ?it/s]
100%|██████████| 198/198 [00:00<00:00, 1677.28it/s]
2026-05-20:13:52:12 INFO     [api.task:312] Building contexts for mmlu_high_school_government_and_politics on rank 0...

  0%|          | 0/193 [00:00<?, ?it/s]
100%|██████████| 193/193 [00:00<00:00, 1671.91it/s]
2026-05-20:13:52:12 INFO     [api.task:312] Building contexts for mmlu_high_school_macroeconomics on rank 0...

  0%|          | 0/390 [00:00<?, ?it/s]
100%|██████████| 390/390 [00:00<00:00, 1726.64it/s]
2026-05-20:13:52:12 INFO     [api.task:312] Building contexts for mmlu_high_school_microeconomics on rank 0...

  0%|          | 0/238 [00:00<?, ?it/s]
100%|██████████| 238/238 [00:00<00:00, 1697.35it/s]
2026-05-20:13:52:12 INFO     [api.task:312] Building contexts for mmlu_high_school_psychology on rank 0...

  0%|          | 0/545 [00:00<?, ?it/s]
100%|██████████| 545/545 [00:00<00:00, 1702.96it/s]
2026-05-20:13:52:13 INFO     [api.task:312] Building contexts for mmlu_human_sexuality on rank 0...

  0%|          | 0/131 [00:00<?, ?it/s]
100%|██████████| 131/131 [00:00<00:00, 1687.87it/s]
2026-05-20:13:52:13 INFO     [api.task:312] Building contexts for mmlu_professional_psychology on rank 0...

  0%|          | 0/612 [00:00<?, ?it/s]
100%|██████████| 612/612 [00:00<00:00, 1723.18it/s]
2026-05-20:13:52:13 INFO     [api.task:312] Building contexts for mmlu_public_relations on rank 0...

  0%|          | 0/110 [00:00<?, ?it/s]
100%|██████████| 110/110 [00:00<00:00, 1589.18it/s]
2026-05-20:13:52:13 INFO     [api.task:312] Building contexts for mmlu_security_studies on rank 0...

  0%|          | 0/245 [00:00<?, ?it/s]
100%|██████████| 245/245 [00:00<00:00, 1741.08it/s]
2026-05-20:13:52:13 INFO     [api.task:312] Building contexts for mmlu_sociology on rank 0...

  0%|          | 0/201 [00:00<?, ?it/s]
100%|██████████| 201/201 [00:00<00:00, 1644.12it/s]
2026-05-20:13:52:14 INFO     [api.task:312] Building contexts for mmlu_us_foreign_policy on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1845.21it/s]
2026-05-20:13:52:14 INFO     [api.task:312] Building contexts for mmlu_formal_logic on rank 0...

  0%|          | 0/126 [00:00<?, ?it/s]
100%|██████████| 126/126 [00:00<00:00, 1659.60it/s]
2026-05-20:13:52:14 INFO     [api.task:312] Building contexts for mmlu_high_school_european_history on rank 0...

  0%|          | 0/165 [00:00<?, ?it/s]
100%|██████████| 165/165 [00:00<00:00, 251.62it/s]
2026-05-20:13:52:14 INFO     [api.task:312] Building contexts for mmlu_high_school_us_history on rank 0...

  0%|          | 0/204 [00:00<?, ?it/s]
100%|██████████| 204/204 [00:00<00:00, 1709.64it/s]
2026-05-20:13:52:14 INFO     [api.task:312] Building contexts for mmlu_high_school_world_history on rank 0...

  0%|          | 0/237 [00:00<?, ?it/s]
100%|██████████| 237/237 [00:00<00:00, 1756.64it/s]
2026-05-20:13:52:15 INFO     [api.task:312] Building contexts for mmlu_international_law on rank 0...

  0%|          | 0/121 [00:00<?, ?it/s]
100%|██████████| 121/121 [00:00<00:00, 1649.38it/s]
2026-05-20:13:52:15 INFO     [api.task:312] Building contexts for mmlu_jurisprudence on rank 0...

  0%|          | 0/108 [00:00<?, ?it/s]
100%|██████████| 108/108 [00:00<00:00, 1627.40it/s]
2026-05-20:13:52:15 INFO     [api.task:312] Building contexts for mmlu_logical_fallacies on rank 0...

  0%|          | 0/163 [00:00<?, ?it/s]
100%|██████████| 163/163 [00:00<00:00, 1694.78it/s]
2026-05-20:13:52:15 INFO     [api.task:312] Building contexts for mmlu_moral_disputes on rank 0...

  0%|          | 0/346 [00:00<?, ?it/s]
100%|██████████| 346/346 [00:00<00:00, 1757.83it/s]
2026-05-20:13:52:15 INFO     [api.task:312] Building contexts for mmlu_moral_scenarios on rank 0...

  0%|          | 0/895 [00:00<?, ?it/s]
100%|██████████| 895/895 [00:00<00:00, 1744.76it/s]
2026-05-20:13:52:16 INFO     [api.task:312] Building contexts for mmlu_philosophy on rank 0...

  0%|          | 0/311 [00:00<?, ?it/s]
100%|██████████| 311/311 [00:00<00:00, 1721.09it/s]
2026-05-20:13:52:16 INFO     [api.task:312] Building contexts for mmlu_prehistory on rank 0...

  0%|          | 0/324 [00:00<?, ?it/s]
100%|██████████| 324/324 [00:00<00:00, 1723.59it/s]
2026-05-20:13:52:16 INFO     [api.task:312] Building contexts for mmlu_professional_law on rank 0...

  0%|          | 0/1534 [00:00<?, ?it/s]
100%|██████████| 1534/1534 [00:00<00:00, 1703.11it/s]
2026-05-20:13:52:17 INFO     [api.task:312] Building contexts for mmlu_world_religions on rank 0...

  0%|          | 0/171 [00:00<?, ?it/s]
100%|██████████| 171/171 [00:00<00:00, 1681.05it/s]
2026-05-20:13:52:17 INFO     [api.task:312] Building contexts for hellaswag on rank 0...

  0%|          | 0/10042 [00:00<?, ?it/s]
100%|██████████| 10042/10042 [00:02<00:00, 3980.73it/s]
2026-05-20:13:52:20 INFO     [evaluator:585] Running loglikelihood requests

Tokenizing inputs:   0%|          | 0/100012 [00:00<?, ?it/s]
Tokenizing inputs: 100%|██████████| 100012/100012 [00:31<00:00, 3134.80it/s]

Running loglikelihood requests:   0%|          | 0/100012 [00:00<?, ?it/s]

Process still running.

### [2026-05-20 13:54:16 UTC] ASSISTANT

**Thinking**

The evaluation is running. Let me continue polling to see when it completes.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "salty-gulf",
  "timeout": 600000
}
```

### [2026-05-20 13:56:16 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

Running loglikelihood requests:  33%|███▎      | 33411/100012 [02:00<03:59, 278.33it/s]
Running loglikelihood requests:  36%|███▌      | 35761/100012 [02:10<03:50, 278.33it/s]

Process still running.

### [2026-05-20 13:56:21 UTC] ASSISTANT

**Thinking**

The evaluation is still running. Let me wait more.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "salty-gulf",
  "timeout": 600000
}
```

### [2026-05-20 13:58:21 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

Running loglikelihood requests:  64%|██████▍   | 63770/100012 [04:00<02:17, 263.41it/s]
Running loglikelihood requests:  68%|██████▊   | 68090/100012 [04:10<02:01, 263.41it/s]

Process still running.

### [2026-05-20 13:58:28 UTC] ASSISTANT

**Thinking**

Still running. Let me wait more.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "salty-gulf",
  "timeout": 600000
}
```

### [2026-05-20 13:58:53 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

Running loglikelihood requests: 100%|██████████| 100012/100012 [05:54<00:00, 282.04it/s]
fatal: not a git repository (or any of the parent directories): .git
2026-05-20:13:58:51 INFO     [loggers.evaluation_tracker:247] Saving results aggregated
hf ({'pretrained': '/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128', 'dtype': 'bfloat16', 'device_map': 'auto'}), gen_kwargs: ({'max_gen_toks': 2048}), limit: None, num_fewshot: None, batch_size: 8
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc     |   |0.3394|±  |0.0040|
| - humanities                          |      2|none  |     0|acc     |↑  |0.2884|±  |0.0066|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.3730|±  |0.0433|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.2485|±  |0.0337|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.2647|±  |0.0310|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.3418|±  |0.0309|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.3306|±  |0.0429|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.3333|±  |0.0456|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.3067|±  |0.0362|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.3555|±  |0.0258|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2380|±  |0.0142|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.3441|±  |0.0270|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.3210|±  |0.0260|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.2666|±  |0.0113|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.3041|±  |0.0353|
| - other                               |      2|none  |     0|acc     |↑  |0.3569|±  |0.0085|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.3700|±  |0.0485|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.3358|±  |0.0291|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.2717|±  |0.0339|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.2900|±  |0.0456|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.4036|±  |0.0329|
|  - management                         |      1|none  |     0|acc     |↑  |0.4466|±  |0.0492|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.5342|±  |0.0327|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.3400|±  |0.0476|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.3691|±  |0.0173|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.3791|±  |0.0278|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.3050|±  |0.0275|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.2721|±  |0.0270|
|  - virology                           |      1|none  |     0|acc     |↑  |0.2831|±  |0.0351|
| - social sciences                     |      2|none  |     0|acc     |↑  |0.3773|±  |0.0087|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.2807|±  |0.0423|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.2929|±  |0.0324|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.3575|±  |0.0346|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.3744|±  |0.0245|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.4370|±  |0.0322|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.4385|±  |0.0213|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.4122|±  |0.0432|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.3088|±  |0.0187|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.3818|±  |0.0465|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.3796|±  |0.0311|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.4627|±  |0.0353|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.4200|±  |0.0496|
| - stem                                |      2|none  |     0|acc     |↑  |0.3612|±  |0.0085|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.3000|±  |0.0461|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.2370|±  |0.0367|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.3750|±  |0.0394|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.3889|±  |0.0408|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.3300|±  |0.0473|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.2900|±  |0.0456|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.4400|±  |0.0499|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.3922|±  |0.0486|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.3600|±  |0.0482|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.3915|±  |0.0319|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.3793|±  |0.0404|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.4206|±  |0.0254|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.3903|±  |0.0278|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.3448|±  |0.0334|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.3600|±  |0.0482|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.3296|±  |0.0287|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.3113|±  |0.0378|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.4167|±  |0.0336|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.2054|±  |0.0383|
|hellaswag                              |      1|none  |     0|acc     |↑  |0.3486|±  |0.0048|
|                                       |       |none  |     0|acc_norm|↑  |0.4210|±  |0.0049|
|piqa                                   |      1|none  |     0|acc     |↑  |0.6393|±  |0.0112|
|                                       |       |none  |     0|acc_norm|↑  |0.6344|±  |0.0112|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |   |0.3394|±  |0.0040|
| - humanities     |      2|none  |     0|acc   |↑  |0.2884|±  |0.0066|
| - other          |      2|none  |     0|acc   |↑  |0.3569|±  |0.0085|
| - social sciences|      2|none  |     0|acc   |↑  |0.3773|±  |0.0087|
| - stem           |      2|none  |     0|acc   |↑  |0.3612|±  |0.0085|

Process exited with code 0.

### [2026-05-20 13:58:58 UTC] ASSISTANT

**Thinking**

The evaluation ran successfully. Now I need to parse the raw results and create the accuracy.json file. Let me look at the raw results to get the exact numbers.


**Tool call:** `exec`

```json
{
  "command": "find /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results -name \"results_*.json\" | head -5"
}
```

### [2026-05-20 13:58:58 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/__root__.openclaw__workspace__quantized__Zyphra_ZR1-1.5B-W4A16__ZR1-1.5B-w4g128/results_2026-05-20T13-58-51.367112.json

### [2026-05-20 13:59:03 UTC] ASSISTANT

**Thinking**

Good, the results file exists. Let me now parse the results and write the accuracy.json. I'll need to extract:
- piqa: acc and acc_stderr
- mmlu: acc and acc_stderr
- hellaswag: acc and acc_stderr

From the output:
- piqa: acc = 0.6393, acc_stderr = 0.0112
- mmlu: acc = 0.3394, acc_stderr = 0.0040
- hellaswag: acc = 0.3486, acc_stderr = 0.0048 (acc_norm = 0.4210 but we use acc as primary metric per skill)

I need to read the JSON file to get exact values.


**Tool call:** `exec`

```json
{
  "command": "cat \"/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/__root__.openclaw__workspace__quantized__Zyphra_ZR1-1.5B-W4A16__ZR1-1.5B-w4g128/results_2026-05-20T13-58-51.367112.json\""
}
```

### [2026-05-20 13:59:03 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "results": {
    "piqa": {
      "name": "piqa",
      "alias": "piqa",
      "sample_len": 1838,
      "acc,none": 0.6392818280739935,
      "acc_stderr,none": 0.01120406480908839,
      "acc_norm,none": 0.6343852013057671,
      "acc_norm_stderr,none": 0.01123657167900628
    },
    "mmlu_abstract_algebra": {
      "name": "mmlu_abstract_algebra",
      "alias": "abstract_algebra",
      "sample_len": 100,
      "acc,none": 0.3,
      "acc_stderr,none": 0.04605661864718382
    },
    "mmlu_anatomy": {
      "name": "mmlu_anatomy",
      "alias": "anatomy",
      "sample_len": 135,
      "acc,none": 0.23703703703703705,
      "acc_stderr,none": 0.03673731683969509
    },
    "mmlu_astronomy": {
      "name": "mmlu_astronomy",
      "alias": "astronomy",
      "sample_len": 152,
      "acc,none": 0.375,
      "acc_stderr,none": 0.039397364351956274
    },
    "mmlu_college_biology": {
      "name": "mmlu_college_biology",
      "alias": "college_biology",
      "sample_len": 144,
      "acc,none": 0.3888888888888889,
      "acc_stderr,none": 0.040766632539185714
    },
    "mmlu_college_chemistry": {
      "name": "mmlu_college_chemistry",
      "alias": "college_chemistry",
      "sample_len": 100,
      "acc,none": 0.33,
      "acc_stderr,none": 0.04725815626252609
    },
    "mmlu_college_computer_science": {
      "name": "mmlu_college_computer_science",
      "alias": "college_computer_science",
      "sample_len": 100,
      "acc,none": 0.29,
      "acc_stderr,none": 0.045604802157206865
    },
    "mmlu_college_mathematics": {
      "name": "mmlu_college_mathematics",
      "alias": "college_mathematics",
      "sample_len": 100,
      "acc,none": 0.44,
      "acc_stderr,none": 0.049888765156985884
    },
    "mmlu_college_physics": {
      "name": "mmlu_college_physics",
      "alias": "college_physics",
      "sample_len": 102,
      "acc,none": 0.39215686274509803,
      "acc_stderr,none": 0.04858083574266346
    },
    "mmlu_computer_security": {
      "name": "mmlu_computer_security",
      "alias": "computer_security",
      "sample_len": 100,
      "acc,none": 0.36,
      "acc_stderr,none": 0.048241815132442176
    },
    "mmlu_conceptual_physics": {
      "name": "mmlu_conceptual_physics",
      "alias": "conceptual_physics",
      "sample_len": 235,
      "acc,none": 0.39148936170212767,
      "acc_stderr,none": 0.031907012423268065
    },
    "mmlu_electrical_engineering": {
      "name": "mmlu_electrical_engineering",
      "alias": "electrical_engineering",
      "sample_len": 145,
      "acc,none": 0.3793103448275862,
      "acc_stderr,none": 0.040434618619167494
    },
    "mmlu_elementary_mathematics": {
      "name": "mmlu_elementary_mathematics",
      "alias": "elementary_mathematics",
      "sample_len": 378,
      "acc,none": 0.42063492063492064,
      "acc_stderr,none": 0.02542483508692397
    },
    "mmlu_high_school_biology": {
      "name": "mmlu_high_school_biology",
      "alias": "high_school_biology",
      "sample_len": 310,
      "acc,none": 0.3903225806451613,
      "acc_stderr,none": 0.02775125663696957
    },
    "mmlu_high_school_chemistry": {
      "name": "mmlu_high_school_chemistry",
      "alias": "high_school_chemistry",
      "sample_len": 203,
      "acc,none": 0.3448275862068966,
      "acc_stderr,none": 0.03344283744280457
    },
    "mmlu_high_school_computer_science": {
      "name": "mmlu_high_school_computer_science",
      "alias": "high_school_computer_science",
      "sample_len": 100,
      "acc,none": 0.36,
      "acc_stderr,none": 0.048241815132442176
    },
    "mmlu_high_school_mathematics": {
      "name": "mmlu_high_school_mathematics",
      "alias": "high_school_mathematics",
      "sample_len": 270,
      "acc,none": 0.3296296296296296,
      "acc_stderr,none": 0.028661201116524572
    },
    "mmlu_high_school_physics": {
      "name": "mmlu_high_school_physics",
      "alias": "high_school_physics",
      "sample_len": 151,
      "acc,none": 0.31125827814569534,
      "acc_stderr,none": 0.03780445850526736
    },
    "mmlu_high_school_statistics": {
      "name": "mmlu_high_school_statistics",
      "alias": "high_school_statistics",
      "sample_len": 216,
      "acc,none": 0.4166666666666667,
      "acc_stderr,none": 0.03362277436608037
    },
    "mmlu_machine_learning": {
      "name": "mmlu_machine_learning",
      "alias": "machine_learning",
      "sample_len": 112,
      "acc,none": 0.20535714285714285,
      "acc_stderr,none": 0.038342410214190735
    },
    "mmlu_business_ethics": {
      "name": "mmlu_business_ethics",
      "alias": "business_ethics",
      "sample_len": 100,
      "acc,none": 0.37,
      "acc_stderr,none": 0.048523658709390974
    },
    "mmlu_clinical_knowledge": {
      "name": "mmlu_clinical_knowledge",
      "alias": "clinical_knowledge",
      "sample_len": 265,
      "acc,none": 0.33584905660377357,
      "acc_stderr,none": 0.02906722014664489
    },
    "mmlu_college_medicine": {
      "name": "mmlu_college_medicine",
      "alias": "college_medicine",
      "sample_len": 173,
      "acc,none": 0.27167630057803466,
      "acc_stderr,none": 0.03391750322321658
    },
    "mmlu_global_facts": {
      "name": "mmlu_global_facts",
      "alias": "global_facts",
      "sample_len": 100,
      "acc,none": 0.29,
      "acc_stderr,none": 0.045604802157206865
    },
    "mmlu_human_aging": {
      "name": "mmlu_human_aging",
      "alias": "human_aging",
      "sample_len": 223,
      "acc,none": 0.40358744394618834,
      "acc_stderr,none": 0.03292802819330314
    },
    "mmlu_management": {
      "name": "mmlu_management",
      "alias": "management",
      "sample_len": 103,
      "acc,none": 0.44660194174757284,
      "acc_stderr,none": 0.04922424153458936
    },
    "mmlu_marketing": {
      "name": "mmlu_marketing",
      "alias": "marketing",
      "sample_len": 234,
      "acc,none": 0.5341880341880342,
      "acc_stderr,none": 0.0326794273408122
    },
    "mmlu_medical_genetics": {
      "name": "mmlu_medical_genetics",
      "alias": "medical_genetics",
      "sample_len": 100,
      "acc,none": 0.34,
      "acc_stderr,none": 0.04760952285695233
    },
    "mmlu_miscellaneous": {
      "name": "mmlu_miscellaneous",
      "alias": "miscellaneous",
      "sample_len": 783,
      "acc,none": 0.36909323116219667,
      "acc_stderr,none": 0.017256283109124578
    },
    "mmlu_nutrition": {
      "name": "mmlu_nutrition",
      "alias": "nutrition",
      "sample_len": 306,
      "acc,none": 0.3790849673202614,
      "acc_stderr,none": 0.02778014120702335
    },
    "mmlu_professional_accounting": {
      "name": "mmlu_professional_accounting",
      "alias": "professional_accounting",
      "sample_len": 282,
      "acc,none": 0.3049645390070922,
      "acc_stderr,none": 0.027464708442022118
    },
    "mmlu_professional_medicine": {
      "name": "mmlu_professional_medicine",
      "alias": "professional_medicine",
      "sample_len": 272,
      "acc,none": 0.27205882352941174,
      "acc_stderr,none": 0.027033041151681446
    },
    "mmlu_virology": {
      "name": "mmlu_virology",
      "alias": "virology",
      "sample_len": 166,
      "acc,none": 0.28313253012048195,
      "acc_stderr,none": 0.035072954313705176
    },
    "mmlu_econometrics": {
      "name": "mmlu_econometrics",
      "alias": "econometrics",
      "sample_len": 114,
      "acc,none": 0.2807017543859649,
      "acc_stderr,none": 0.042270544512322025
    },
    "mmlu_high_school_geography": {
      "name": "mmlu_high_school_geography",
      "alias": "high_school_geography",
      "sample_len": 198,
      "acc,none": 0.29292929292929293,
      "acc_stderr,none": 0.032424979581788145
    },
    "mmlu_high_school_government_and_politics": {
      "name": "mmlu_high_school_government_and_politics",
      "alias": "high_school_government_and_politics",
      "sample_len": 193,
      "acc,none": 0.35751295336787564,
      "acc_stderr,none": 0.03458816042181008
    },
    "mmlu_high_school_macroeconomics": {
      "name": "mmlu_high_school_macroeconomics",
      "alias": "high_school_macroeconomics",
      "sample_len": 390,
      "acc,none": 0.37435897435897436,
      "acc_stderr,none": 0.024537591572830485
    },
    "mmlu_high_school_microeconomics": {
      "name": "mmlu_high_school_microeconomics",
      "alias": "high_school_microeconomics",
      "sample_len": 238,
      "acc,none": 0.4369747899159664,
      "acc_stderr,none": 0.032219436365662
    },
    "mmlu_high_school_psychology": {
      "name": "mmlu_high_school_psychology",
      "alias": "high_school_psychology",
      "sample_len": 545,
      "acc,none": 0.43853211009174314,
      "acc_stderr,none": 0.021274713073954548
    },
    "mmlu_human_sexuality": {
      "name": "mmlu_human_sexuality",
      "alias": "human_sexuality",
      "sample_len": 131,
      "acc,none": 0.4122137404580153,
      "acc_stderr,none": 0.043171711948702576
    },
    "mmlu_professional_psychology": {
      "name": "mmlu_professional_psychology",
      "alias": "professional_psychology",
      "sample_len": 612,
      "acc,none": 0.3088235294117647,
      "acc_stderr,none": 0.01869085027359537
    },
    "mmlu_public_relations": {
      "name": "mmlu_public_relations",
      "alias": "public_relations",
      "sample_len": 110,
      "acc,none": 0.38181818181818183,
      "acc_stderr,none": 0.04653429807913508
    },
    "mmlu_security_studies": {
      "name": "mmlu_security_studies",
      "alias": "security_studies",
      "sample_len": 245,
      "acc,none": 0.3795918367346939,
      "acc_stderr,none": 0.03106721126287244
    },
    "mmlu_sociology": {
      "name": "mmlu_sociology",
      "alias": "sociology",
      "sample_len": 201,
      "acc,none": 0.4626865671641791,
      "acc_stderr,none": 0.03525675167467975
    },
    "mmlu_us_foreign_policy": {
      "name": "mmlu_us_foreign_policy",
      "alias": "us_foreign_policy",
      "sample_len": 100,
      "acc,none": 0.42,
      "acc_stderr,none": 0.04960449637488583
    },
    "mmlu_formal_logic": {
      "name": "mmlu_formal_logic",
      "alias": "formal_logic",
      "sample_len": 126,
      "acc,none": 0.373015873015873,
      "acc_stderr,none": 0.043255060420170875
    },
    "mmlu_high_school_european_history": {
      "name": "mmlu_high_school_european_history",
      "alias": "high_school_european_history",
      "sample_len": 165,
      "acc,none": 0.24848484848484848,
      "acc_stderr,none": 0.03374402644139407
    },
    "mmlu_high_school_us_history": {
      "name": "mmlu_high_school_us_history",
      "alias": "high_school_us_history",
      "sample_len": 204,
      "acc,none": 0.2647058823529412,
      "acc_stderr,none": 0.030964517926923372
    },
    "mmlu_high_school_world_history": {
      "name": "mmlu_high_school_world_history",
      "alias": "high_school_world_history",
      "sample_len": 237,
      "acc,none": 0.34177215189873417,
      "acc_stderr,none": 0.03087453753755363
    },
    "mmlu_international_law": {
      "name": "mmlu_international_law",
      "alias": "international_law",
      "sample_len": 121,
      "acc,none": 0.3305785123966942,
      "acc_stderr,none": 0.042943408452120926
    },
    "mmlu_jurisprudence": {
      "name": "mmlu_jurisprudence",
      "alias": "jurisprudence",
      "sample_len": 108,
      "acc,none": 0.3333333333333333,
      "acc_stderr,none": 0.04557239513497755
    },
    "mmlu_logical_fallacies": {
      "name": "mmlu_logical_fallacies",
      "alias": "logical_fallacies",
      "sample_len": 163,
      "acc,none": 0.3067484662576687,
      "acc_stderr,none": 0.036230899157241474
    },
    "mmlu_moral_disputes": {
      "name": "mmlu_moral_disputes",
      "alias": "moral_disputes",
      "sample_len": 346,
      "acc,none": 0.3554913294797688,
      "acc_stderr,none": 0.025770292082977198
    },
    "mmlu_moral_scenarios": {
      "name": "mmlu_moral_scenarios",
      "alias": "moral_scenarios",
      "sample_len": 895,
      "acc,none": 0.23798882681564246,
      "acc_stderr,none": 0.014242630070574904
    },
    "mmlu_philosophy": {
      "name": "mmlu_philosophy",
      "alias": "philosophy",
      "sample_len": 311,
      "acc,none": 0.3440514469453376,
      "acc_stderr,none": 0.02698147804364805
    },
    "mmlu_prehistory": {
      "name": "mmlu_prehistory",
      "alias": "prehistory",
      "sample_len": 324,
      "acc,none": 0.32098765432098764,
      "acc_stderr,none": 0.025976566010862776
    },
    "mmlu_professional_law": {
      "name": "mmlu_professional_law",
      "alias": "professional_law",
      "sample_len": 1534,
      "acc,none": 0.2666232073011734,
      "acc_stderr,none": 0.011293836031612223
    },
    "mmlu_world_religions": {
      "name": "mmlu_world_religions",
      "alias": "world_religions",
      "sample_len": 171,
      "acc,none": 0.30409356725146197,
      "acc_stderr,none": 0.03528211258245229
    },
    "hellaswag": {
      "name": "hellaswag",
      "alias": "hellaswag",
      "sample_len": 10042,
      "acc,none": 0.34863572993427605,
      "acc_stderr,none": 0.004755645016263529,
      "acc_norm,none": 0.42103166699860584,
      "acc_norm_stderr,none": 0.004927155882598323
    },
    "mmlu_stem": {
      "alias": "stem",
      "name": "mmlu_stem",
      "sample_len": 3153,
      "acc,none": 0.3612432603869331,
      "acc_stderr,none": 0.008522239767701185,
      "sample_count": {
        "acc,none": 3153
      }
    },
    "mmlu_other": {
      "alias": "other",
      "name": "mmlu_other",
      "sample_len": 3107,
      "acc,none": 0.35693595107821047,
      "acc_stderr,none": 0.008525593342271057,
      "sample_count": {
        "acc,none": 3107
      }
    },
    "mmlu_social_sciences": {
      "alias": "social sciences",
      "name": "mmlu_social_sciences",
      "sample_len": 3077,
      "acc,none": 0.3773155671108222,
      "acc_stderr,none": 0.008693782497917019,
      "sample_count": {
        "acc,none": 3077
      }
    },
    "mmlu_humanities": {
      "alias": "humanities",
      "name": "mmlu_humanities",
      "sample_len": 4705,
      "acc,none": 0.2884165781083953,
      "acc_stderr,none": 0.006585029955446784,
      "sample_count": {
        "acc,none": 4705
      }
    },
    "mmlu": {
      "alias": "mmlu",
      "name": "mmlu",
      "sample_len": 14042,
      "acc,none": 0.33941034040734935,
      "acc_stderr,none": 0.003964485452533085,
      "sample_count": {
        "acc,none": 14042
      }
    }
  },
  "groups": {
    "mmlu_stem": {
      "alias": "stem",
      "name": "mmlu_stem",
      "sample_len": 3153,
      "acc,none": 0.3612432603869331,
      "acc_stderr,none": 0.008522239767701185,
      "sample_count": {
        "acc,none": 3153
      }
    },
    "mmlu_other": {
      "alias": "other",
      "name": "mmlu_other",
      "sample_len": 3107,
      "acc,none": 0.35693595107821047,
      "acc_stderr,none": 0.008525593342271057,
      "sample_count": {
        "acc,none": 3107
      }
    },
    "mmlu_social_sciences": {
      "alias": "social sciences",
      "name": "mmlu_social_sciences",
      "sample_len": 3077,
      "acc,none": 0.3773155671108222,
      "acc_stderr,none": 0.008693782497917019,
      "sample_count": {
        "acc,none": 3077
      }
    },
    "mmlu_humanities": {
      "alias": "humanities",
      "name": "mmlu_humanities",
      "sample_len": 4705,
      "acc,none": 0.2884165781083953,
      "acc_stderr,none": 0.006585029955446784,
      "sample_count": {
        "acc,none": 4705
      }
    },
    "mmlu": {
      "alias": "mmlu",
      "name": "mmlu",
      "sample_len": 14042,
      "acc,none": 0.33941034040734935,
      "acc_stderr,none": 0.003964485452533085,
      "sample_count": {
        "acc,none": 14042
      }
    }
  },
  "group_subtasks": {
    "mmlu": [
      "mmlu_stem",
      "mmlu_other",
      "mmlu_social_sciences",
      "mmlu_humanities"
    ],
    "mmlu_stem": [
      "mmlu_abstract_algebra",
      "mmlu_anatomy",
      "mmlu_astronomy",
      "mmlu_college_biology",
      "mmlu_college_chemistry",
      "mmlu_college_computer_science",
      "mmlu_college_mathematics",
      "mmlu_college_physics",
      "mmlu_computer_security",
      "mmlu_conceptual_physics",
      "mmlu_electrical_engineering",
      "mmlu_elementary_mathematics",
      "mmlu_high_school_biology",
      "mmlu_high_school_chemistry",
      "mmlu_high_school_computer_science",
      "mmlu_high_school_mathematics",
      "mmlu_high_school_physics",
      "mmlu_high_school_statistics",
      "mmlu_machine_learning"
    ],
    "mmlu_other": [
      "mmlu_business_ethics",
      "mmlu_clinical_knowledge",
      "mmlu_college_medicine",
      "mmlu_global_facts",
      "mmlu_human_aging",
      "mmlu_management",
      "mmlu_marketing",
      "mmlu_medical_genetics",
      "mmlu_miscellaneous",
      "mmlu_nutrition",
      "mmlu_professional_accounting",
      "mmlu_professional_medicine",
      "mmlu_virology"
    ],
    "mmlu_social_sciences": [
      "mmlu_econometrics",
      "mmlu_high_school_geography",
      "mmlu_high_school_government_and_politics",
      "mmlu_high_school_macroeconomics",
      "mmlu_high_school_microeconomics",
      "mmlu_high_school_psychology",
      "mmlu_human_sexuality",
      "mmlu_professional_psychology",
      "mmlu_public_relations",
      "mmlu_security_studies",
      "mmlu_sociology",
      "mmlu_us_foreign_policy"
    ],
    "mmlu_humanities": [
      "mmlu_formal_logic",
      "mmlu_high_school_european_history",
      "mmlu_high_school_us_history",
      "mmlu_high_school_world_history",
      "mmlu_international_law",
      "mmlu_jurisprudence",
      "mmlu_logical_fallacies",
      "mmlu_moral_disputes",
      "mmlu_moral_scenarios",
      "mmlu_philosophy",
      "mmlu_prehistory",
      "mmlu_professional_law",
      "mmlu_world_religions"
    ]
  },
  "configs": {
    "hellaswag": {
      "task": "hellaswag",
      "dataset_path": "Rowan/hellaswag",
      "training_split": "train",
      "validation_split": "validation",
      "process_docs": "def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:\n    def _process_doc(doc):\n        ctx = doc[\"ctx_a\"] + \" \" + doc[\"ctx_b\"].capitalize()\n        out_doc = {\n            \"query\": preprocess(doc[\"activity_label\"] + \": \" + ctx),\n            \"choices\": [preprocess(ending) for ending in doc[\"endings\"]],\n            \"gold\": int(doc[\"label\"]),\n        }\n        return out_doc\n\n    return dataset.map(_process_doc)\n",
      "doc_to_text": "{{query}}",
      "doc_to_target": "{{label}}",
      "unsafe_code": false,
      "doc_to_choice": "choices",
      "description": "",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "default",
        "split": null,
        "process_docs": "<function process_docs at 0x7f0e96fca0c0>",
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{query}}",
        "doc_to_choice": "choices",
        "doc_to_target": "{{label}}",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        },
        {
          "metric": "acc_norm",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/hellaswag/hellaswag.yaml"
      }
    },
    "mmlu_abstract_algebra": {
      "task": "mmlu_abstract_algebra",
      "task_alias": "abstract_algebra",
      "dataset_path": "cais/mmlu",
      "dataset_name": "abstract_algebra",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about abstract algebra.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_abstract_algebra.yaml"
      }
    },
    "mmlu_anatomy": {
      "task": "mmlu_anatomy",
      "task_alias": "anatomy",
      "dataset_path": "cais/mmlu",
      "dataset_name": "anatomy",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about anatomy.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_anatomy.yaml"
      }
    },
    "mmlu_astronomy": {
      "task": "mmlu_astronomy",
      "task_alias": "astronomy",
      "dataset_path": "cais/mmlu",
      "dataset_name": "astronomy",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about astronomy.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_astronomy.yaml"
      }
    },
    "mmlu_business_ethics": {
      "task": "mmlu_business_ethics",
      "task_alias": "business_ethics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "business_ethics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about business ethics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_business_ethics.yaml"
      }
    },
    "mmlu_clinical_knowledge": {
      "task": "mmlu_clinical_knowledge",
      "task_alias": "clinical_knowledge",
      "dataset_path": "cais/mmlu",
      "dataset_name": "clinical_knowledge",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about clinical knowledge.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_clinical_knowledge.yaml"
      }
    },
    "mmlu_college_biology": {
      "task": "mmlu_college_biology",
      "task_alias": "college_biology",
      "dataset_path": "cais/mmlu",
      "dataset_name": "college_biology",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about college biology.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_biology.yaml"
      }
    },
    "mmlu_college_chemistry": {
      "task": "mmlu_college_chemistry",
      "task_alias": "college_chemistry",
      "dataset_path": "cais/mmlu",
      "dataset_name": "college_chemistry",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about college chemistry.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_chemistry.yaml"
      }
    },
    "mmlu_college_computer_science": {
      "task": "mmlu_college_computer_science",
      "task_alias": "college_computer_science",
      "dataset_path": "cais/mmlu",
      "dataset_name": "college_computer_science",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about college computer science.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_computer_science.yaml"
      }
    },
    "mmlu_college_mathematics": {
      "task": "mmlu_college_mathematics",
      "task_alias": "college_mathematics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "college_mathematics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about college mathematics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_mathematics.yaml"
      }
    },
    "mmlu_college_medicine": {
      "task": "mmlu_college_medicine",
      "task_alias": "college_medicine",
      "dataset_path": "cais/mmlu",
      "dataset_name": "college_medicine",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about college medicine.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_medicine.yaml"
      }
    },
    "mmlu_college_physics": {
      "task": "mmlu_college_physics",
      "task_alias": "college_physics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "college_physics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about college physics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_physics.yaml"
      }
    },
    "mmlu_computer_security": {
      "task": "mmlu_computer_security",
      "task_alias": "computer_security",
      "dataset_path": "cais/mmlu",
      "dataset_name": "computer_security",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about computer security.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_computer_security.yaml"
      }
    },
    "mmlu_conceptual_physics": {
      "task": "mmlu_conceptual_physics",
      "task_alias": "conceptual_physics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "conceptual_physics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about conceptual physics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_conceptual_physics.yaml"
      }
    },
    "mmlu_econometrics": {
      "task": "mmlu_econometrics",
      "task_alias": "econometrics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "econometrics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about econometrics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_econometrics.yaml"
      }
    },
    "mmlu_electrical_engineering": {
      "task": "mmlu_electrical_engineering",
      "task_alias": "electrical_engineering",
      "dataset_path": "cais/mmlu",
      "dataset_name": "electrical_engineering",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about electrical engineering.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_electrical_engineering.yaml"
      }
    },
    "mmlu_elementary_mathematics": {
      "task": "mmlu_elementary_mathematics",
      "task_alias": "elementary_mathematics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "elementary_mathematics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about elementary mathematics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_elementary_mathematics.yaml"
      }
    },
    "mmlu_formal_logic": {
      "task": "mmlu_formal_logic",
      "task_alias": "formal_logic",
      "dataset_path": "cais/mmlu",
      "dataset_name": "formal_logic",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about formal logic.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_formal_logic.yaml"
      }
    },
    "mmlu_global_facts": {
      "task": "mmlu_global_facts",
      "task_alias": "global_facts",
      "dataset_path": "cais/mmlu",
      "dataset_name": "global_facts",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about global facts.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_global_facts.yaml"
      }
    },
    "mmlu_high_school_biology": {
      "task": "mmlu_high_school_biology",
      "task_alias": "high_school_biology",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_biology",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school biology.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_biology.yaml"
      }
    },
    "mmlu_high_school_chemistry": {
      "task": "mmlu_high_school_chemistry",
      "task_alias": "high_school_chemistry",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_chemistry",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school chemistry.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_chemistry.yaml"
      }
    },
    "mmlu_high_school_computer_science": {
      "task": "mmlu_high_school_computer_science",
      "task_alias": "high_school_computer_science",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_computer_science",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school computer science.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_computer_science.yaml"
      }
    },
    "mmlu_high_school_european_history": {
      "task": "mmlu_high_school_european_history",
      "task_alias": "high_school_european_history",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_european_history",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school european history.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_european_history.yaml"
      }
    },
    "mmlu_high_school_geography": {
      "task": "mmlu_high_school_geography",
      "task_alias": "high_school_geography",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_geography",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school geography.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_geography.yaml"
      }
    },
    "mmlu_high_school_government_and_politics": {
      "task": "mmlu_high_school_government_and_politics",
      "task_alias": "high_school_government_and_politics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_government_and_politics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school government and politics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_government_and_politics.yaml"
      }
    },
    "mmlu_high_school_macroeconomics": {
      "task": "mmlu_high_school_macroeconomics",
      "task_alias": "high_school_macroeconomics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_macroeconomics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school macroeconomics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_macroeconomics.yaml"
      }
    },
    "mmlu_high_school_mathematics": {
      "task": "mmlu_high_school_mathematics",
      "task_alias": "high_school_mathematics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_mathematics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school mathematics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_mathematics.yaml"
      }
    },
    "mmlu_high_school_microeconomics": {
      "task": "mmlu_high_school_microeconomics",
      "task_alias": "high_school_microeconomics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_microeconomics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school microeconomics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_microeconomics.yaml"
      }
    },
    "mmlu_high_school_physics": {
      "task": "mmlu_high_school_physics",
      "task_alias": "high_school_physics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_physics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school physics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_physics.yaml"
      }
    },
    "mmlu_high_school_psychology": {
      "task": "mmlu_high_school_psychology",
      "task_alias": "high_school_psychology",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_psychology",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school psychology.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_psychology.yaml"
      }
    },
    "mmlu_high_school_statistics": {
      "task": "mmlu_high_school_statistics",
      "task_alias": "high_school_statistics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_statistics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school statistics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_statistics.yaml"
      }
    },
    "mmlu_high_school_us_history": {
      "task": "mmlu_high_school_us_history",
      "task_alias": "high_school_us_history",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_us_history",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school us history.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_us_history.yaml"
      }
    },
    "mmlu_high_school_world_history": {
      "task": "mmlu_high_school_world_history",
      "task_alias": "high_school_world_history",
      "dataset_path": "cais/mmlu",
      "dataset_name": "high_school_world_history",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about high school world history.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_world_history.yaml"
      }
    },
    "mmlu_human_aging": {
      "task": "mmlu_human_aging",
      "task_alias": "human_aging",
      "dataset_path": "cais/mmlu",
      "dataset_name": "human_aging",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about human aging.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_human_aging.yaml"
      }
    },
    "mmlu_human_sexuality": {
      "task": "mmlu_human_sexuality",
      "task_alias": "human_sexuality",
      "dataset_path": "cais/mmlu",
      "dataset_name": "human_sexuality",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about human sexuality.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_human_sexuality.yaml"
      }
    },
    "mmlu_international_law": {
      "task": "mmlu_international_law",
      "task_alias": "international_law",
      "dataset_path": "cais/mmlu",
      "dataset_name": "international_law",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about international law.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_international_law.yaml"
      }
    },
    "mmlu_jurisprudence": {
      "task": "mmlu_jurisprudence",
      "task_alias": "jurisprudence",
      "dataset_path": "cais/mmlu",
      "dataset_name": "jurisprudence",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about jurisprudence.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_jurisprudence.yaml"
      }
    },
    "mmlu_logical_fallacies": {
      "task": "mmlu_logical_fallacies",
      "task_alias": "logical_fallacies",
      "dataset_path": "cais/mmlu",
      "dataset_name": "logical_fallacies",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about logical fallacies.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_logical_fallacies.yaml"
      }
    },
    "mmlu_machine_learning": {
      "task": "mmlu_machine_learning",
      "task_alias": "machine_learning",
      "dataset_path": "cais/mmlu",
      "dataset_name": "machine_learning",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about machine learning.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_machine_learning.yaml"
      }
    },
    "mmlu_management": {
      "task": "mmlu_management",
      "task_alias": "management",
      "dataset_path": "cais/mmlu",
      "dataset_name": "management",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about management.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_management.yaml"
      }
    },
    "mmlu_marketing": {
      "task": "mmlu_marketing",
      "task_alias": "marketing",
      "dataset_path": "cais/mmlu",
      "dataset_name": "marketing",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about marketing.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_marketing.yaml"
      }
    },
    "mmlu_medical_genetics": {
      "task": "mmlu_medical_genetics",
      "task_alias": "medical_genetics",
      "dataset_path": "cais/mmlu",
      "dataset_name": "medical_genetics",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about medical genetics.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_medical_genetics.yaml"
      }
    },
    "mmlu_miscellaneous": {
      "task": "mmlu_miscellaneous",
      "task_alias": "miscellaneous",
      "dataset_path": "cais/mmlu",
      "dataset_name": "miscellaneous",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about miscellaneous.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_miscellaneous.yaml"
      }
    },
    "mmlu_moral_disputes": {
      "task": "mmlu_moral_disputes",
      "task_alias": "moral_disputes",
      "dataset_path": "cais/mmlu",
      "dataset_name": "moral_disputes",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about moral disputes.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_moral_disputes.yaml"
      }
    },
    "mmlu_moral_scenarios": {
      "task": "mmlu_moral_scenarios",
      "task_alias": "moral_scenarios",
      "dataset_path": "cais/mmlu",
      "dataset_name": "moral_scenarios",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about moral scenarios.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_moral_scenarios.yaml"
      }
    },
    "mmlu_nutrition": {
      "task": "mmlu_nutrition",
      "task_alias": "nutrition",
      "dataset_path": "cais/mmlu",
      "dataset_name": "nutrition",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about nutrition.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_nutrition.yaml"
      }
    },
    "mmlu_philosophy": {
      "task": "mmlu_philosophy",
      "task_alias": "philosophy",
      "dataset_path": "cais/mmlu",
      "dataset_name": "philosophy",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about philosophy.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_philosophy.yaml"
      }
    },
    "mmlu_prehistory": {
      "task": "mmlu_prehistory",
      "task_alias": "prehistory",
      "dataset_path": "cais/mmlu",
      "dataset_name": "prehistory",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about prehistory.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_prehistory.yaml"
      }
    },
    "mmlu_professional_accounting": {
      "task": "mmlu_professional_accounting",
      "task_alias": "professional_accounting",
      "dataset_path": "cais/mmlu",
      "dataset_name": "professional_accounting",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about professional accounting.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_accounting.yaml"
      }
    },
    "mmlu_professional_law": {
      "task": "mmlu_professional_law",
      "task_alias": "professional_law",
      "dataset_path": "cais/mmlu",
      "dataset_name": "professional_law",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about professional law.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_law.yaml"
      }
    },
    "mmlu_professional_medicine": {
      "task": "mmlu_professional_medicine",
      "task_alias": "professional_medicine",
      "dataset_path": "cais/mmlu",
      "dataset_name": "professional_medicine",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about professional medicine.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_medicine.yaml"
      }
    },
    "mmlu_professional_psychology": {
      "task": "mmlu_professional_psychology",
      "task_alias": "professional_psychology",
      "dataset_path": "cais/mmlu",
      "dataset_name": "professional_psychology",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about professional psychology.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_psychology.yaml"
      }
    },
    "mmlu_public_relations": {
      "task": "mmlu_public_relations",
      "task_alias": "public_relations",
      "dataset_path": "cais/mmlu",
      "dataset_name": "public_relations",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about public relations.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_public_relations.yaml"
      }
    },
    "mmlu_security_studies": {
      "task": "mmlu_security_studies",
      "task_alias": "security_studies",
      "dataset_path": "cais/mmlu",
      "dataset_name": "security_studies",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about security studies.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_security_studies.yaml"
      }
    },
    "mmlu_sociology": {
      "task": "mmlu_sociology",
      "task_alias": "sociology",
      "dataset_path": "cais/mmlu",
      "dataset_name": "sociology",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about sociology.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_sociology.yaml"
      }
    },
    "mmlu_us_foreign_policy": {
      "task": "mmlu_us_foreign_policy",
      "task_alias": "us_foreign_policy",
      "dataset_path": "cais/mmlu",
      "dataset_name": "us_foreign_policy",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about us foreign policy.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_us_foreign_policy.yaml"
      }
    },
    "mmlu_virology": {
      "task": "mmlu_virology",
      "task_alias": "virology",
      "dataset_path": "cais/mmlu",
      "dataset_name": "virology",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about virology.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_virology.yaml"
      }
    },
    "mmlu_world_religions": {
      "task": "mmlu_world_religions",
      "task_alias": "world_religions",
      "dataset_path": "cais/mmlu",
      "dataset_name": "world_religions",
      "test_split": "test",
      "fewshot_split": "dev",
      "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
      "doc_to_target": "answer",
      "unsafe_code": false,
      "doc_to_choice": [
        "A",
        "B",
        "C",
        "D"
      ],
      "description": "The following are multiple choice questions (with answers) about world religions.\n\n",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "first_n",
        "split": "dev",
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:",
        "doc_to_choice": [
          "A",
          "B",
          "C",
          "D"
        ],
        "doc_to_target": "answer",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": false,
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_world_religions.yaml"
      }
    },
    "piqa": {
      "task": "piqa",
      "dataset_path": "baber/piqa",
      "training_split": "train",
      "validation_split": "validation",
      "doc_to_text": "Question: {{goal}}\nAnswer:",
      "doc_to_target": "label",
      "unsafe_code": false,
      "doc_to_choice": "{{[sol1, sol2]}}",
      "description": "",
      "target_delimiter": " ",
      "fewshot_delimiter": "\n\n",
      "fewshot_config": {
        "sampler": "default",
        "split": null,
        "process_docs": null,
        "fewshot_indices": null,
        "samples": null,
        "doc_to_text": "Question: {{goal}}\nAnswer:",
        "doc_to_choice": "{{[sol1, sol2]}}",
        "doc_to_target": "label",
        "gen_prefix": null,
        "fewshot_delimiter": "\n\n",
        "target_delimiter": " "
      },
      "num_fewshot": 0,
      "metric_list": [
        {
          "metric": "acc",
          "aggregation": "mean",
          "higher_is_better": true
        },
        {
          "metric": "acc_norm",
          "aggregation": "mean",
          "higher_is_better": true
        }
      ],
      "output_type": "multiple_choice",
      "repeats": 1,
      "should_decontaminate": true,
      "doc_to_decontamination_query": "goal",
      "metadata": {
        "version": 1.0,
        "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/lib/python3.12/site-packages/lm_eval/tasks/piqa/piqa.yaml"
      }
    }
  },
  "versions": {
    "hellaswag": 1.0,
    "mmlu": "2",
    "mmlu_abstract_algebra": 1.0,
    "mmlu_anatomy": 1.0,
    "mmlu_astronomy": 1.0,
    "mmlu_business_ethics": 1.0,
    "mmlu_clinical_knowledge": 1.0,
    "mmlu_college_biology": 1.0,
    "mmlu_college_chemistry": 1.0,
    "mmlu_college_computer_science": 1.0,
    "mmlu_college_mathematics": 1.0,
    "mmlu_college_medicine": 1.0,
    "mmlu_college_physics": 1.0,
    "mmlu_computer_security": 1.0,
    "mmlu_conceptual_physics": 1.0,
    "mmlu_econometrics": 1.0,
    "mmlu_electrical_engineering": 1.0,
    "mmlu_elementary_mathematics": 1.0,
    "mmlu_formal_logic": 1.0,
    "mmlu_global_facts": 1.0,
    "mmlu_high_school_biology": 1.0,
    "mmlu_high_school_chemistry": 1.0,
    "mmlu_high_school_computer_science": 1.0,
    "mmlu_high_school_european_history": 1.0,
    "mmlu_high_school_geography": 1.0,
    "mmlu_high_school_government_and_politics": 1.0,
    "mmlu_high_school_macroeconomics": 1.0,
    "mmlu_high_school_mathematics": 1.0,
    "mmlu_high_school_microeconomics": 1.0,
    "mmlu_high_school_physics": 1.0,
    "mmlu_high_school_psychology": 1.0,
    "mmlu_high_school_statistics": 1.0,
    "mmlu_high_school_us_history": 1.0,
    "mmlu_high_school_world_history": 1.0,
    "mmlu_human_aging": 1.0,
    "mmlu_human_sexuality": 1.0,
    "mmlu_humanities": "2",
    "mmlu_international_law": 1.0,
    "mmlu_jurisprudence": 1.0,
    "mmlu_logical_fallacies": 1.0,
    "mmlu_machine_learning": 1.0,
    "mmlu_management": 1.0,
    "mmlu_marketing": 1.0,
    "mmlu_medical_genetics": 1.0,
    "mmlu_miscellaneous": 1.0,
    "mmlu_moral_disputes": 1.0,
    "mmlu_moral_scenarios": 1.0,
    "mmlu_nutrition": 1.0,
    "mmlu_other": "2",
    "mmlu_philosophy": 1.0,
    "mmlu_prehistory": 1.0,
    "mmlu_professional_accounting": 1.0,
    "mmlu_professional_law": 1.0,
    "mmlu_professional_medicine": 1.0,
    "mmlu_professional_psychology": 1.0,
    "mmlu_public_relations": 1.0,
    "mmlu_security_studies": 1.0,
    "mmlu_social_sciences": "2",
    "mmlu_sociology": 1.0,
    "mmlu_stem": "2",
    "mmlu_us_foreign_policy": 1.0,
    "mmlu_virology": 1.0,
    "mmlu_world_religions": 1.0,
    "piqa": 1.0
  },
  "n-shot": {
    "hellaswag": 0,
    "mmlu_abstract_algebra": 0,
    "mmlu_anatomy": 0,
    "mmlu_astronomy": 0,
    "mmlu_business_ethics": 0,
    "mmlu_clinical_knowledge": 0,
    "mmlu_college_biology": 0,
    "mmlu_college_chemistry": 0,
    "mmlu_college_computer_science": 0,
    "mmlu_college_mathematics": 0,
    "mmlu_college_medicine": 0,
    "mmlu_college_physics": 0,
    "mmlu_computer_security": 0,
    "mmlu_conceptual_physics": 0,
    "mmlu_econometrics": 0,
    "mmlu_electrical_engineering": 0,
    "mmlu_elementary_mathematics": 0,
    "mmlu_formal_logic": 0,
    "mmlu_global_facts": 0,
    "mmlu_high_school_biology": 0,
    "mmlu_high_school_chemistry": 0,
    "mmlu_high_school_computer_science": 0,
    "mmlu_high_school_european_history": 0,
    "mmlu_high_school_geography": 0,
    "mmlu_high_school_government_and_politics": 0,
    "mmlu_high_school_macroeconomics": 0,
    "mmlu_high_school_mathematics": 0,
    "mmlu_high_school_microeconomics": 0,
    "mmlu_high_school_physics": 0,
    "mmlu_high_school_psychology": 0,
    "mmlu_high_school_statistics": 0,
    "mmlu_high_school_us_history": 0,
    "mmlu_high_school_world_history": 0,
    "mmlu_human_aging": 0,
    "mmlu_human_sexuality": 0,
    "mmlu_humanities": 0,
    "mmlu_international_law": 0,
    "mmlu_jurisprudence": 0,
    "mmlu_logical_fallacies": 0,
    "mmlu_machine_learning": 0,
    "mmlu_management": 0,
    "mmlu_marketing": 0,
    "mmlu_medical_genetics": 0,
    "mmlu_miscellaneous": 0,
    "mmlu_moral_disputes": 0,
    "mmlu_moral_scenarios": 0,
    "mmlu_nutrition": 0,
    "mmlu_other": 0,
    "mmlu_philosophy": 0,
    "mmlu_prehistory": 0,
    "mmlu_professional_accounting": 0,
    "mmlu_professional_law": 0,
    "mmlu_professional_medicine": 0,
    "mmlu_professional_psychology": 0,
    "mmlu_public_relations": 0,
    "mmlu_security_studies": 0,
    "mmlu_social_sciences": 0,
    "mmlu_sociology": 0,
    "mmlu_stem": 0,
    "mmlu_us_foreign_policy": 0,
    "mmlu_virology": 0,
    "mmlu_world_religions": 0,
    "piqa": 0
  },
  "higher_is_better": {
    "hellaswag": {
      "acc": true,
      "acc_norm": true
    },
    "mmlu_abstract_algebra": {
      "acc": true
    },
    "mmlu_anatomy": {
      "acc": true
    },
    "mmlu_astronomy": {
      "acc": true
    },
    "mmlu_business_ethics": {
      "acc": true
    },
    "mmlu_clinical_knowledge": {
      "acc": true
    },
    "mmlu_college_biology": {
      "acc": true
    },
    "mmlu_college_chemistry": {
      "acc": true
    },
    "mmlu_college_computer_science": {
      "acc": true
    },
    "mmlu_college_mathematics": {
      "acc": true
    },
    "mmlu_college_medicine": {
      "acc": true
    },
    "mmlu_college_physics": {
      "acc": true
    },
    "mmlu_computer_security": {
      "acc": true
    },
    "mmlu_conceptual_physics": {
      "acc": true
    },
    "mmlu_econometrics": {
      "acc": true
    },
    "mmlu_electrical_engineering": {
      "acc": true
    },
    "mmlu_elementary_mathematics": {
      "acc": true
    },
    "mmlu_formal_logic": {
      "acc": true
    },
    "mmlu_global_facts": {
      "acc": true
    },
    "mmlu_high_school_biology": {
      "acc": true
    },
    "mmlu_high_school_chemistry": {
      "acc": true
    },
    "mmlu_high_school_computer_science": {
      "acc": true
    },
    "mmlu_high_school_european_history": {
      "acc": true
    },
    "mmlu_high_school_geography": {
      "acc": true
    },
    "mmlu_high_school_government_and_politics": {
      "acc": true
    },
    "mmlu_high_school_macroeconomics": {
      "acc": true
    },
    "mmlu_high_school_mathematics": {
      "acc": true
    },
    "mmlu_high_school_microeconomics": {
      "acc": true
    },
    "mmlu_high_school_physics": {
      "acc": true
    },
    "mmlu_high_school_psychology": {
      "acc": true
    },
    "mmlu_high_school_statistics": {
      "acc": true
    },
    "mmlu_high_school_us_history": {
      "acc": true
    },
    "mmlu_high_school_world_history": {
      "acc": true
    },
    "mmlu_human_aging": {
      "acc": true
    },
    "mmlu_human_sexuality": {
      "acc": true
    },
    "mmlu_humanities": {
      "acc": true
    },
    "mmlu_international_law": {
      "acc": true
    },
    "mmlu_jurisprudence": {
      "acc": true
    },
    "mmlu_logical_fallacies": {
      "acc": true
    },
    "mmlu_machine_learning": {
      "acc": true
    },
    "mmlu_management": {
      "acc": true
    },
    "mmlu_marketing": {
      "acc": true
    },
    "mmlu_medical_genetics": {
      "acc": true
    },
    "mmlu_miscellaneous": {
      "acc": true
    },
    "mmlu_moral_disputes": {
      "acc": true
    },
    "mmlu_moral_scenarios": {
      "acc": true
    },
    "mmlu_nutrition": {
      "acc": true
    },
    "mmlu_other": {
      "acc": true
    },
    "mmlu_philosophy": {
      "acc": true
    },
    "mmlu_prehistory": {
      "acc": true
    },
    "mmlu_professional_accounting": {
      "acc": true
    },
    "mmlu_professional_law": {
      "acc": true
    },
    "mmlu_professional_medicine": {
      "acc": true
    },
    "mmlu_professional_psychology": {
      "acc": true
    },
    "mmlu_public_relations": {
      "acc": true
    },
    "mmlu_security_studies": {
      "acc": true
    },
    "mmlu_social_sciences": {
      "acc": true
    },
    "mmlu_sociology": {
      "acc": true
    },
    "mmlu_stem": {
      "acc": true
    },
    "mmlu_us_foreign_policy": {
      "acc": true
    },
    "mmlu_virology": {
      "acc": true
    },
    "mmlu_world_religions": {
      "acc": true
    },
    "piqa": {
      "acc": true,
      "acc_norm": true
    }
  },
  "n-samples": {
    "piqa": {
      "original": 1838,
      "effective": 1838
    },
    "mmlu_abstract_algebra": {
      "original": 100,
      "effective": 100
    },
    "mmlu_anatomy": {
      "original": 135,
      "effective": 135
    },
    "mmlu_astronomy": {
      "original": 152,
      "effective": 152
    },
    "mmlu_college_biology": {
      "original": 144,
      "effective": 144
    },
    "mmlu_college_chemistry": {
      "original": 100,
      "effective": 100
    },
    "mmlu_college_computer_science": {
      "original": 100,
      "effective": 100
    },
    "mmlu_college_mathematics": {
      "original": 100,
      "effective": 100
    },
    "mmlu_college_physics": {
      "original": 102,
      "effective": 102
    },
    "mmlu_computer_security": {
      "original": 100,
      "effective": 100
    },
    "mmlu_conceptual_physics": {
      "original": 235,
      "effective": 235
    },
    "mmlu_electrical_engineering": {
      "original": 145,
      "effective": 145
    },
    "mmlu_elementary_mathematics": {
      "original": 378,
      "effective": 378
    },
    "mmlu_high_school_biology": {
      "original": 310,
      "effective": 310
    },
    "mmlu_high_school_chemistry": {
      "original": 203,
      "effective": 203
    },
    "mmlu_high_school_computer_science": {
      "original": 100,
      "effective": 100
    },
    "mmlu_high_school_mathematics": {
      "original": 270,
      "effective": 270
    },
    "mmlu_high_school_physics": {
      "original": 151,
      "effective": 151
    },
    "mmlu_high_school_statistics": {
      "original": 216,
      "effective": 216
    },
    "mmlu_machine_learning": {
      "original": 112,
      "effective": 112
    },
    "mmlu_business_ethics": {
      "original": 100,
      "effective": 100
    },
    "mmlu_clinical_knowledge": {
      "original": 265,
      "effective": 265
    },
    "mmlu_college_medicine": {
      "original": 173,
      "effective": 173
    },
    "mmlu_global_facts": {
      "original": 100,
      "effective": 100
    },
    "mmlu_human_aging": {
      "original": 223,
      "effective": 223
    },
    "mmlu_management": {
      "original": 103,
      "effective": 103
    },
    "mmlu_marketing": {
      "original": 234,
      "effective": 234
    },
    "mmlu_medical_genetics": {
      "original": 100,
      "effective": 100
    },
    "mmlu_miscellaneous": {
      "original": 783,
      "effective": 783
    },
    "mmlu_nutrition": {
      "original": 306,
      "effective": 306
    },
    "mmlu_professional_accounting": {
      "original": 282,
      "effective": 282
    },
    "mmlu_professional_medicine": {
      "original": 272,
      "effective": 272
    },
    "mmlu_virology": {
      "original": 166,
      "effective": 166
    },
    "mmlu_econometrics": {
      "original": 114,
      "effective": 114
    },
    "mmlu_high_school_geography": {
      "original": 198,
      "effective": 198
    },
    "mmlu_high_school_government_and_politics": {
      "original": 193,
      "effective": 193
    },
    "mmlu_high_school_macroeconomics": {
      "original": 390,
      "effective": 390
    },
    "mmlu_high_school_microeconomics": {
      "original": 238,
      "effective": 238
    },
    "mmlu_high_school_psychology": {
      "original": 545,
      "effective": 545
    },
    "mmlu_human_sexuality": {
      "original": 131,
      "effective": 131
    },
    "mmlu_professional_psychology": {
      "original": 612,
      "effective": 612
    },
    "mmlu_public_relations": {
      "original": 110,
      "effective": 110
    },
    "mmlu_security_studies": {
      "original": 245,
      "effective": 245
    },
    "mmlu_sociology": {
      "original": 201,
      "effective": 201
    },
    "mmlu_us_foreign_policy": {
      "original": 100,
      "effective": 100
    },
    "mmlu_formal_logic": {
      "original": 126,
      "effective": 126
    },
    "mmlu_high_school_european_history": {
      "original": 165,
      "effective": 165
    },
    "mmlu_high_school_us_history": {
      "original": 204,
      "effective": 204
    },
    "mmlu_high_school_world_history": {
      "original": 237,
      "effective": 237
    },
    "mmlu_international_law": {
      "original": 121,
      "effective": 121
    },
    "mmlu_jurisprudence": {
      "original": 108,
      "effective": 108
    },
    "mmlu_logical_fallacies": {
      "original": 163,
      "effective": 163
    },
    "mmlu_moral_disputes": {
      "original": 346,
      "effective": 346
    },
    "mmlu_moral_scenarios": {
      "original": 895,
      "effective": 895
    },
    "mmlu_philosophy": {
      "original": 311,
      "effective": 311
    },
    "mmlu_prehistory": {
      "original": 324,
      "effective": 324
    },
    "mmlu_professional_law": {
      "original": 1534,
      "effective": 1534
    },
    "mmlu_world_religions": {
      "original": 171,
      "effective": 171
    },
    "hellaswag": {
      "original": 10042,
      "effective": 10042
    }
  },
  "config": {
    "model": "hf",
    "model_args": {
      "pretrained": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
      "dtype": "bfloat16",
      "device_map": "auto",
      "trust_remote_code": true
    },
    "model_num_parameters": 466834944,
    "model_dtype": "torch.bfloat16",
    "model_revision": "main",
    "model_sha": "",
    "batch_size": "8",
    "batch_sizes": [],
    "device": "cuda",
    "use_cache": null,
    "limit": null,
    "bootstrap_iters": 100000,
    "gen_kwargs": {
      "max_gen_toks": 2048
    },
    "random_seed": 0,
    "numpy_seed": 1234,
    "torch_seed": 1234,
    "fewshot_seed": 1234
  },
  "git_hash": null,
  "date": 1779284983.6194167,
  "pretty_env_info": "PyTorch version: 2.6.0+cu124\nIs debug build: False\nCUDA used to build PyTorch: 12.4\nROCM used to build PyTorch: N/A\n\nOS: Ubuntu 24.04.4 LTS (x86_64)\nGCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\nClang version: Could not collect\nCMake version: Could not collect\nLibc version: glibc-2.39\n\nPython version: 3.12.13 (main, Apr 14 2026, 14:29:00) [Clang 22.1.3 ] (64-bit runtime)\nPython platform: Linux-6.8.0-52-generic-x86_64-with-glibc2.39\nIs CUDA available: True\nCUDA runtime version: 12.0.140\nCUDA_MODULE_LOADING set to: LAZY\nGPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090\nNvidia driver version: 565.57.01\ncuDNN version: Could not collect\nHIP runtime version: N/A\nMIOpen runtime version: N/A\nIs XNNPACK available: True\n\nCPU:\nArchitecture:                         x86_64\nCPU op-mode(s):                       32-bit, 64-bit\nAddress sizes:                        48 bits physical, 48 bits virtual\nByte Order:                           Little Endian\nCPU(s):                               32\nOn-line CPU(s) list:                  0-31\nVendor ID:                            AuthenticAMD\nModel name:                           AMD Ryzen 9 7950X 16-Core Processor\nCPU family:                           25\nModel:                                97\nThread(s) per core:                   2\nCore(s) per socket:                   16\nSocket(s):                            1\nStepping:                             2\nCPU(s) scaling MHz:                   57%\nCPU max MHz:                          5881.0000\nCPU min MHz:                          400.0000\nBogoMIPS:                             8983.04\nFlags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d\nVirtualization:                       AMD-V\nL1d cache:                            512 KiB (16 instances)\nL1i cache:                            512 KiB (16 instances)\nL2 cache:                             16 MiB (16 instances)\nL3 cache:                             64 MiB (2 instances)\nNUMA node(s):                         1\nNUMA node0 CPU(s):                    0-31\nVulnerability Gather data sampling:   Not affected\nVulnerability Itlb multihit:          Not affected\nVulnerability L1tf:                   Not affected\nVulnerability Mds:                    Not affected\nVulnerability Meltdown:               Not affected\nVulnerability Mmio stale data:        Not affected\nVulnerability Reg file data sampling: Not affected\nVulnerability Retbleed:               Not affected\nVulnerability Spec rstack overflow:   Mitigation; Safe RET\nVulnerability Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl\nVulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization\nVulnerability Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected\nVulnerability Srbds:                  Not affected\nVulnerability Tsx async abort:        Not affected\n\nVersions of relevant libraries:\n[pip3] numpy==2.4.6\n[pip3] nvidia-cublas-cu12==12.4.5.8\n[pip3] nvidia-cuda-cupti-cu12==12.4.127\n[pip3] nvidia-cuda-nvrtc-cu12==12.4.127\n[pip3] nvidia-cuda-runtime-cu12==12.4.127\n[pip3] nvidia-cudnn-cu12==9.1.0.70\n[pip3] nvidia-cufft-cu12==11.2.1.3\n[pip3] nvidia-curand-cu12==10.3.5.147\n[pip3] nvidia-cusolver-cu12==11.6.1.9\n[pip3] nvidia-cusparse-cu12==12.3.1.170\n[pip3] nvidia-cusparselt-cu12==0.6.2\n[pip3] nvidia-nccl-cu12==2.21.5\n[pip3] nvidia-nvjitlink-cu12==12.4.127\n[pip3] nvidia-nvtx-cu12==12.4.127\n[pip3] torch==2.6.0+cu124\n[pip3] triton==3.2.0\n[conda] Could not collect",
  "transformers_version": "5.8.1",
  "lm_eval_version": "0.4.12",
  "upper_git_hash": null,
  "tokenizer_pad_token": [
    "<｜end▁of▁sentence｜>",
    "151643"
  ],
  "tokenizer_eos_token": [
    "<｜end▁of▁sentence｜>",
    "151643"
  ],
  "tokenizer_bos_token": [
    "<｜begin▁of▁sentence｜>",
    "151646"
  ],
  "eot_token_id": 151643,
  "max_length": 131072,
  "task_hashes": {},
  "model_source": "hf",
  "model_name": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128",
  "model_name_sanitized": "__root__.openclaw__workspace__quantized__Zyphra_ZR1-1.5B-W4A16__ZR1-1.5B-w4g128",
  "system_instruction": null,
  "system_instruction_sha": null,
  "fewshot_as_multiturn": null,
  "chat_template": null,
  "chat_template_sha": null,
  "total_evaluation_time_seconds": "553.1652030795813"
}

### [2026-05-20 13:59:16 UTC] ASSISTANT

**Thinking**

The raw results are all there. Now I need to parse the results and write the accuracy.json. Let me extract the key values:

- mmlu: acc = 0.33941034040734935, acc_stderr = 0.003964485452533085
- hellaswag: acc = 0.34863572993427605, acc_stderr = 0.004755645016263529
- piqa: acc = 0.6392818280739935, acc_stderr = 0.01120406480908839

Let me compute the duration from the log. The eval started at some point and finished. I'll compute from the timestamps in the results JSON. The total_evaluation_time_seconds = 553.1652030795813

Now let me write the Python script to parse and create accuracy.json.


**Tool call:** `exec`

```json
{
  "command": "cat > /tmp/parse_results.py << 'PYEOF'\nimport json\n\nresults_path = \"/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/__root__.openclaw__workspace__quantized__Zyphra_ZR1-1.5B-W4A16__ZR1-1.5B-w4g128/results_2026-05-20T13-58-51.367112.json\"\n\nwith open(results_path) as f:\n    data = json.load(f)\n\nresults = data[\"results\"]\n\n# Extract results for each task\nmmlu = results.get(\"mmlu\", {})\nhellaswag = results.get(\"hellaswag\", {})\npiqa = results.get(\"piqa\", {})\n\nprint(f\"mmlu acc: {mmlu.get('acc,none')}, stderr: {mmlu.get('acc_stderr,none')}\")\nprint(f\"hellaswag acc: {hellaswag.get('acc,none')}, stderr: {hellaswag.get('acc_stderr,none')}\")\nprint(f\"piqa acc: {piqa.get('acc,none')}, stderr: {piqa.get('acc_stderr,none')}\")\nprint(f\"total_time: {data['config'].get('total_evaluation_time_seconds')}\")\n\nPYEOF\n/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python /tmp/parse_results.py"
}
```

### [2026-05-20 13:59:16 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

mmlu acc: 0.33941034040734935, stderr: 0.003964485452533085
hellaswag acc: 0.34863572993427605, stderr: 0.004755645016263529
piqa acc: 0.6392818280739935, stderr: 0.01120406480908839
total_time: None

### [2026-05-20 13:59:28 UTC] ASSISTANT

**Thinking**

I see total_time is None, but I saw "total_evaluation_time_seconds": "553.1652030795813" in the config section. Let me look at the structure more carefully.

Actually I see it's inside `config` not at top level. Let me extract properly and also get the duration from the execution log.


**Tool call:** `exec`

```json
{
  "command": "cat > /tmp/make_accuracy.py << 'PYEOF'\nimport json\nimport os\n\nresults_path = \"/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/__root__.openclaw__workspace__quantized__Zyphra_ZR1-1.5B-W4A16__ZR1-1.5B-w4g128/results_2026-05-20T13-58-51.367112.json\"\noutput_path = \"/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json\"\n\nwith open(results_path) as f:\n    data = json.load(f)\n\nresults = data[\"results\"]\n\n# Extract accuracy for each task\nmmlu = results.get(\"mmlu\", {})\nhellaswag = results.get(\"hellaswag\", {})\npiqa = results.get(\"piqa\", {})\n\n# Duration\nduration = data.get(\"config\", {}).get(\"total_evaluation_time_seconds\", 553.1652030795813)\nif isinstance(duration, str):\n    duration = float(duration)\n\naccuracy_data = {\n    \"model_id\": \"Zyphra/ZR1-1.5B\",\n    \"model_path\": \"/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16\",\n    \"scheme\": \"W4A16\",\n    \"device\": \"cuda:0\",\n    \"num_gpus\": \"1\",\n    \"tasks\": {\n        \"mmlu\": {\n            \"accuracy\": mmlu.get(\"acc,none\"),\n            \"accuracy_stderr\": mmlu.get(\"acc_stderr,none\")\n        },\n        \"hellaswag\": {\n            \"accuracy\": hellaswag.get(\"acc,none\"),\n            \"accuracy_stderr\": hellaswag.get(\"acc_stderr,none\")\n        },\n        \"piqa\": {\n            \"accuracy\": piqa.get(\"acc,none\"),\n            \"accuracy_stderr\": piqa.get(\"acc_stderr,none\")\n        }\n    },\n    \"status\": \"success\",\n    \"duration_seconds\": duration,\n    \"eval_framework\": \"lm_eval+hf\",\n    \"errors\": []\n}\n\n# Write atomically\ntmp_path = output_path + \".tmp\"\nwith open(tmp_path, \"w\") as f:\n    json.dump(accuracy_data, f, indent=2)\nos.rename(tmp_path, output_path)\n\nprint(f\"Written accuracy.json\")\nprint(json.dumps(accuracy_data, indent=2))\nPYEOF\n/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python /tmp/make_accuracy.py"
}
```

### [2026-05-20 13:59:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Written accuracy.json
{
  "model_id": "Zyphra/ZR1-1.5B",
  "model_path": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16",
  "scheme": "W4A16",
  "device": "cuda:0",
  "num_gpus": "1",
  "tasks": {
    "mmlu": {
      "accuracy": 0.33941034040734935,
      "accuracy_stderr": 0.003964485452533085
    },
    "hellaswag": {
      "accuracy": 0.34863572993427605,
      "accuracy_stderr": 0.004755645016263529
    },
    "piqa": {
      "accuracy": 0.6392818280739935,
      "accuracy_stderr": 0.01120406480908839
    }
  },
  "status": "success",
  "duration_seconds": 553.1652030795813,
  "eval_framework": "lm_eval+hf",
  "errors": []
}

### [2026-05-20 13:59:33 UTC] ASSISTANT

**Thinking**

Done. Let me verify the final artifacts.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json"
}
```

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh"
}
```

### [2026-05-20 13:59:33 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "model_id": "Zyphra/ZR1-1.5B",
  "model_path": "/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16",
  "scheme": "W4A16",
  "device": "cuda:0",
  "num_gpus": "1",
  "tasks": {
    "mmlu": {
      "accuracy": 0.33941034040734935,
      "accuracy_stderr": 0.003964485452533085
    },
    "hellaswag": {
      "accuracy": 0.34863572993427605,
      "accuracy_stderr": 0.004755645016263529
    },
    "piqa": {
      "accuracy": 0.6392818280739935,
      "accuracy_stderr": 0.01120406480908839
    }
  },
  "status": "success",
  "duration_seconds": 553.1652030795813,
  "eval_framework": "lm_eval+hf",
  "errors": []
}

### [2026-05-20 13:59:33 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

#!/bin/bash
set -e

MODEL_PATH="/root/.openclaw/workspace/quantized/Zyphra_ZR1-1.5B-W4A16/ZR1-1.5B-w4g128"
OUTPUT_PATH="/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results"
TASKS="piqa,mmlu,hellaswag"
BATCH_SIZE=8
NUM_GPUS=1

VENV_PY="/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/venv/bin/python"

export CUDA_VISIBLE_DEVICES=0

$VENV_PY -m lm_eval --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,device_map=auto,trust_remote_code=True" \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --gen_kwargs max_gen_toks=2048 \
    --device cuda

### [2026-05-20 13:59:42 UTC] ASSISTANT

**Thinking**

Everything is in order. Let me provide a summary of the completed evaluation.




<final>
## Evaluation Complete ✅

**Model:** Zyphra/ZR1-1.5B (W4A16, auto_round quantization)
**Framework:** lm_eval + HuggingFace backend
**Device:** CUDA (1× RTX 4090)
**Duration:** ~553 seconds (~9 min)

### Results

| Task | Accuracy | Stderr |
|------|----------|--------|
| **piqa** | 0.6393 | ±0.0112 |
| **mmlu** | 0.3394 | ±0.0040 |
| **hellaswag** | 0.3486 | ±0.0048 |

### Artifacts

- **evaluate.sh:** `/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/evaluate.sh`
- **accuracy.json:** `/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/accuracy.json`
- **Raw results:** `/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/lm_eval_results/`
- **Exec log:** `/root/.openclaw/workspace/quantized/runs/Zyphra_ZR1-1.5B-W4A16/logs/eval_exec.log`

### Notes

- Stage A (lm_eval HF backend) completed successfully with `max_gen_toks=2048`
- Stage B parsing extracted aggregated task accuracies from the nested JSON results structure
- The model used bfloat16 with `device_map=auto` for inference
- `trust_remote_code=True` was required for this model architecture
</final>
