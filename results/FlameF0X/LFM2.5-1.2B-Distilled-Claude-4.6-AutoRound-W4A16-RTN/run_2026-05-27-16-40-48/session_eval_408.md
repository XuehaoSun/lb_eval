# Session: autoeval_eval_408

- **Session ID:** `autoeval_eval_408`
- **Timestamp:** 2026-05-27 16:31:22 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 2: Evaluation

### [2026-05-27 16:31:22 UTC] USER

You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_eval/SKILL.md

Quantized model path: /root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN
Raw lm_eval output directory: /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results
Evaluation script path: /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/evaluate.sh
Evaluation execution log path: /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/logs/eval_exec.log
Evaluation tasks: piqa,mmlu,hellaswag
Batch size: 8
Num gpus: 1

The quantized model was produced by auto_quant with scheme=W4A16, export_format=auto_round.
A venv may already exist at /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/venv (created by auto_quant with --system-site-packages).

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- If a venv already exists at /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/venv, reuse it - just install lm_eval and vllm into it.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- Write evaluation outputs, logs, prompts, copied request/session files, and other runtime artifacts to: /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN
- Before starting evaluation, you MUST first generate the evaluation script file:
    /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/evaluate.sh
- The file name must be exactly: evaluate.sh
- The script must be a standalone shell program runnable with:
    bash /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/evaluate.sh
- The generated evaluate.sh must focus on Stage A raw lm_eval execution only.
- Do NOT put venv creation, pip/uv installation, package bootstrap, JSON parsing, accuracy.json writing, or destructive cleanup/recreation into evaluate.sh.
- In this same OpenClaw task, prepare or reuse the evaluation environment separately before executing evaluate.sh.
- Prefer direct lm_eval CLI commands and standard shell environment variables over ad-hoc Python wrappers.
- In this same OpenClaw task, first write evaluate.sh, then execute that generated script yourself.
- When you execute evaluate.sh or a direct lm_eval command, you MUST stream stdout/stderr into this log file while still printing output:
    bash /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/logs/eval_exec.log
- /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json is a final summary artifact, not a progress marker.
- After Stage A raw lm_eval completes, parse the latest raw results and write /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json in a separate finalize step outside evaluate.sh.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json until Stage A raw lm_eval output and Stage B parsing have both completed successfully.
- Write /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json atomically via a temporary file and rename/move it into place only at finalize time.
- If evaluation fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json before exiting non-zero, also atomically.
- The overall evaluation workflow MUST still behave as two idempotent stages:
    Stage A: run lm_eval and persist raw results under /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results
    Stage B: parse the latest raw results into /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json
- Stage A and Stage B MUST be independently rerunnable.
- If /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results already contains a valid raw results file matching , do NOT rerun Stage A; only rerun Stage B parsing.
- If Stage B parsing fails after Stage A already succeeded, exit non-zero but preserve the raw lm_eval outputs, logs, and venv so the next retry only reruns parsing.
- Do NOT delete or recreate /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results when raw results already exist.
- Parsing logic should read the latest results file under /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/**/results_*.json rather than assuming lm_eval must be rerun.
- Do NOT leave the final artifact named eval.sh, eval_script.sh, or evaluate_script.sh.
- When the generated script invokes lm_eval, it MUST pass:
    --output_path /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results
  and MUST set max_gen_toks=2048. The placement depends on the backend:
    - For HF backend (--model hf):   --gen_kwargs max_gen_toks=2048
    - For vLLM backend (--model vllm): append max_gen_toks=2048 inside --model_args
- Do NOT omit --output_path. Keep the raw lm_eval output files under that exact directory for later upload.
- Do NOT omit max_gen_toks=2048 regardless of backend.

IMPORTANT - In this same OpenClaw task, after separate raw-eval and finalize steps, you MUST produce:

/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json - evaluation results:
{
  "model_id": "FlameF0X/LFM2.5-1.2B-Distilled-Claude-4.6",
  "model_path": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
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

/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/ - raw lm_eval output directory created by:
    lm_eval ... --output_path /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, the script may invoke python only for JSON post-processing, but it must still write accuracy.json with status=failed before exiting non-zero.

### [2026-05-27 16:31:28 UTC] ASSISTANT

**Thinking**

Let me start by reading the skill file for auto_eval to understand the evaluation process.

**Tool call:** `read`

```json
{
  "file": "/root/.openclaw/workspace/skills/auto_eval/SKILL.md"
}
```

### [2026-05-27 16:31:29 UTC] TOOL RESULT
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
5. If you ever need to install vllm (e.g. as fallback), **always** use a torch constraint:
   ```bash
   TORCH_VER=$("$VENV_PY" -c "import torch; print(torch.__version__.split('+')[0])")
   echo "torch==${TORCH_VER}" > /tmp/torch_constraint.txt
   uv pip install --python "$VENV_PY" vllm -c /tmp/torch_constraint.txt
   ```
   If that fails, use `--no-deps` and manually install missing dependencies. **Never let vllm replace torch.**

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

### General Debugging Principle

**If standard fixes don't resolve the error, check the model's README (model card) — it may contain useful hints:**

```bash
curl -L https://huggingface.co/{model_id}/resolve/main/README.md | head -200
```

Look for: required library versions, known limitations, or special loading instructions. Not all model cards have useful info — if nothing relevant, move on to other strategies.

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

#### 4.5. Unsupported Model Architecture During Evaluation

**Error:**
```
NotImplementedError: ... is not supported
KeyError: 'xxx' model type not found
ValueError: Unrecognized model architecture
```

**Root cause:** Newer model architectures may not be recognized by the installed `transformers` version.

**Strategy: Always prefer the latest versions of transformers and auto-round.**

```bash
# Upgrade transformers and auto-round to latest
uv pip install --python "$VENV_PY" --upgrade transformers auto-round accelerate

# If still failing, install transformers from source
uv pip install --python "$VENV_PY" --upgrade git+https://github.com/huggingface/transformers.git
```

**Key principle:** When encountering unsupported model type errors, upgrading `transformers` (and `auto-round` if auto_round format is involved) to the latest version is the most likely fix.

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


### [2026-05-27 16:31:33 UTC] ASSISTANT

**Thinking**

Now let me analyze the task:

1. Model path: `/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN`
2. Runtime artifact directory: `/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN`
3. Tasks: piqa, mmlu, hellaswag
4. Batch size: 8
5. Num gpus: 1
6. Scheme: W4A16, export_format=auto_round

The evaluation script path is specified, but I need to generate it. Let me:
1. First check the model directory for quantization config
2. Check if there's an existing venv
3. Generate the evaluate.sh script
4. Set up the environment and run the evaluation
5. Parse results and write accuracy.json

Let me start by examining the model path and quantization config.

**Tool call:** `read`

```json
{
  "file": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN/quantization_config.json"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/ 2>/dev/null || echo \"Directory does not exist yet\""
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/venv/bin/python 2>/dev/null && echo \"VENV_EXISTS\" || echo \"VENV_NOT_EXISTS\""
}
```

### [2026-05-27 16:31:33 UTC] TOOL RESULT
**Tool:** `read`

{
  "bits": 4,
  "data_type": "int",
  "group_size": 128,
  "sym": true,
  "iters": 0,
  "low_gpu_mem_usage": true,
  "seqlen": 512,
  "autoround_version": "0.12.3",
  "quant_method": "auto-round",
  "packing_format": "auto_round:auto_gptq"
}

### [2026-05-27 16:31:33 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 168
drwxr-xr-x 3 root root    138 May 27 16:31 .
drwxr-xr-x 3 root root     73 May 27 16:27 ..
drwxr-xr-x 2 root root    118 May 27 16:31 logs
-rw-r--r-- 1 root root   2112 May 27 16:31 quant_summary.json
-rw-r--r-- 1 root root   1482 May 27 16:29 quantize.py
-rw-r--r-- 1 root root   1100 May 27 16:27 request.json
-rw-r--r-- 1 root root 158637 May 27 16:31 session_quant_408.jsonl

### [2026-05-27 16:31:33 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

VENV_NOT_EXISTS

### [2026-05-27 16:31:36 UTC] ASSISTANT

**Thinking**

Let me check if there's an existing venv at the specified path and also check the raw lm_eval results directory.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/ 2>/dev/null || echo \"lm_eval_results does not exist yet\""
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/logs/ 2>/dev/null"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())\" 2>/dev/null || echo \"torch not available in system python\""
}
```

### [2026-05-27 16:31:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lm_eval_results does not exist yet

### [2026-05-27 16:31:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 152
drwxr-xr-x 2 root root   118 May 27 16:31 .
drwxr-xr-x 3 root root   138 May 27 16:31 ..
-rw-r--r-- 1 root root 81061 May 27 16:31 auto.log
-rw-r--r-- 1 root root  7636 May 27 16:31 eval_script_prompt.txt
-rw-r--r-- 1 root root  4574 May 27 16:30 quant_exec.log
-rw-r--r-- 1 root root  5546 May 27 16:27 quant_prompt.txt

### [2026-05-27 16:31:37 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

CUDA: True Devices: 1

### [2026-05-27 16:31:40 UTC] ASSISTANT

**Thinking**

Good - CUDA is available with 1 GPU. No lm_eval_results exists yet. The venv doesn't exist. Let me:

1. First create the evaluate.sh script
2. Then set up the environment (reuse /root/.venv if available, or create a new one with --system-site-packages)
3. Run the evaluation

Let me check if /root/.venv exists:


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.venv/bin/python 2>/dev/null && echo \"SYSTEM_VENV_EXISTS\" || echo \"SYSTEM_VENV_NOT_EXISTS\""
}
```

### [2026-05-27 16:31:40 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lrwxrwxrwx 1 root root 73 Apr 24 05:53 /root/.venv/bin/python -> /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
SYSTEM_VENV_EXISTS

### [2026-05-27 16:31:44 UTC] ASSISTANT

**Thinking**

Good, `/root/.venv` exists and has torch+cuda. Let me check what's in it and set up the environment.

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import transformers; print('transformers:', transformers.__version__)\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import lm_eval; print('lm_eval:', lm_eval.__version__)\" 2>&1"
}
```

### [2026-05-27 16:31:45 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch: 2.6.0+cu124
CUDA: True Devices: 1

### [2026-05-27 16:31:45 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

transformers: 5.10.0.dev0

### [2026-05-27 16:31:45 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'lm_eval'

(Command exited with code 1)

### [2026-05-27 16:31:48 UTC] ASSISTANT

**Thinking**

lm_eval is not installed in the system venv. I need to install it. Let me also check if auto_round is available for the model loading.

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import auto_round; print('auto_round:', auto_round.__version__)\" 2>&1 || echo \"auto_round not installed\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import flash_attn; print('flash_attn ok')\" 2>&1 || echo \"flash_attn not available\""
}
```

### [2026-05-27 16:31:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

auto_round: 0.12.3

### [2026-05-27 16:31:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'flash_attn'
flash_attn not available

### [2026-05-27 16:31:54 UTC] ASSISTANT

**Thinking**

Good - torch and transformers are available, and auto_round is installed. lm_eval needs to be installed. Let me install lm-eval now.

**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.venv/bin/python \"lm-eval[torch]\" 2>&1 | tail -20",
  "timeout": 120
}
```

### [2026-05-27 16:31:59 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

+ lxml==6.1.1
 + mbstrdecoder==1.1.5
 + more-itertools==11.1.0
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

### [2026-05-27 16:32:03 UTC] ASSISTANT

**Thinking**

Good, lm-eval is installed. Now let me check if it can be invoked via the `lm_eval` command.

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -m lm_eval --version 2>&1 || /root/.venv/bin/lm_eval --version 2>&1 || echo \"lm_eval CLI not found\""
}
```

### [2026-05-27 16:32:04 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

usage: lm-eval [-h] COMMAND ...
lm-eval: error: unrecognized arguments: --version
usage: lm-eval [-h] COMMAND ...
lm-eval: error: unrecognized arguments: --version
lm_eval CLI not found

### [2026-05-27 16:32:07 UTC] ASSISTANT

**Thinking**

The lm_eval package is installed but the CLI isn't directly accessible. Let me check how to run it properly.

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import lm_eval; print(dir(lm_eval))\" 2>&1"
}
```

### [2026-05-27 16:32:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

['__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__getattr__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', 'find_spec', 'importlib', 'logging', 'os']

### [2026-05-27 16:32:10 UTC] ASSISTANT

**Thinking**

lm_eval is installed but doesn't have a CLI accessible via `--version`. Let me check how to invoke it properly.

**Tool call:** `exec`

```json
{
  "command": "which lm_eval 2>/dev/null || echo \"lm_eval not in PATH\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import lm_eval; print(lm_eval.__version__)\""
}
```

### [2026-05-27 16:32:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.venv/bin/lm_eval

### [2026-05-27 16:32:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

0.4.12

### [2026-05-27 16:32:12 UTC] ASSISTANT

**Thinking**

Good, lm_eval 0.4.12 is installed and its CLI is at `/root/.venv/bin/lm_eval`. Let me check the model path structure and then write the evaluate.sh script.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN/"
}
```

### [2026-05-27 16:32:12 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 792700
drwxr-xr-x 2 root root      4096 May 27 16:30 .
drwxr-xr-x 4 root root        90 May 27 16:27 ..
-rw-r--r-- 1 root root       398 May 27 16:30 chat_template.jinja
-rw-r--r-- 1 root root      1666 May 27 16:30 config.json
-rw-r--r-- 1 root root       180 May 27 16:30 generation_config.json
-rw-r--r-- 1 root root 806962496 May 27 16:30 model.safetensors
-rw-r--r-- 1 root root       242 May 27 16:30 quantization_config.json
-rw-r--r-- 1 root root   4733389 May 27 16:30 tokenizer.json
-rw-r--r-- 1 root root       287 May 27 16:30 tokenizer_config.json

### [2026-05-27 16:32:15 UTC] ASSISTANT

**Thinking**

Let me check the config.json to understand the model architecture.

**Tool call:** `read`

```json
{
  "file": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN/config.json"
}
```

### [2026-05-27 16:32:15 UTC] TOOL RESULT
**Tool:** `read`

{
  "architectures": [
    "Lfm2ForCausalLM"
  ],
  "block_auto_adjust_ff_dim": true,
  "block_dim": 2048,
  "block_ffn_dim_multiplier": 1.0,
  "block_mlp_init_scale": 1.0,
  "block_multiple_of": 256,
  "block_norm_eps": 1e-05,
  "block_out_init_scale": 1.0,
  "block_use_swiglu": true,
  "block_use_xavier_init": true,
  "bos_token_id": 1,
  "conv_L_cache": 3,
  "conv_bias": false,
  "conv_dim": 2048,
  "conv_use_xavier_init": true,
  "dtype": "float16",
  "eos_token_id": 7,
  "full_attn_idxs": null,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "layer_types": [
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv"
  ],
  "max_position_embeddings": 128000,
  "model_name": "LiquidAI/LFM2.5-1.2B-Thinking",
  "model_type": "lfm2",
  "norm_eps": 1e-05,
  "num_attention_heads": 32,
  "num_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "quantization_config": {
    "autoround_version": "0.12.3",
    "bits": 4,
    "data_type": "int",
    "group_size": 128,
    "iters": 0,
    "low_gpu_mem_usage": true,
    "packing_format": "auto_round:auto_gptq",
    "quant_method": "auto-round",
    "seqlen": 512,
    "sym": true
  },
  "rope_parameters": {
    "rope_theta": 1000000.0,
    "rope_type": "default"
  },
  "tie_word_embeddings": true,
  "transformers_version": "5.10.0.dev0",
  "unsloth_version": "2026.4.4",
  "use_cache": false,
  "use_pos_enc": true,
  "vocab_size": 65536
}


### [2026-05-27 16:32:19 UTC] ASSISTANT

**Thinking**

The model uses `Lfm2ForCausalLM` architecture which is a custom architecture. I need `trust_remote_code=True`. Also `use_cache=False` is interesting - that might cause issues with generation. Let me now write the evaluate.sh script and then run the evaluation.

The model path: `/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN`
Tasks: piqa,mmlu,hellaswag
Batch size: 8
Output path: `/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results`


Since `use_cache=False` could interfere with generation, I'll rely on `--gen_kwargs max_gen_toks=2048` to handle the generation length. Now I'm ready to create the evaluation script.

**Tool call:** `write`

```json
{
  "content": "#!/bin/bash\nset -e\n\nMODEL_PATH=\"/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN\"\nOUTPUT_PATH=\"/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results\"\nTASKS=\"piqa,mmlu,hellaswag\"\nBATCH_SIZE=8\nGEN_KWARGS=\"max_gen_toks=2048\"\n\nlm_eval \\\n    --model hf \\\n    --model_args \"pretrained=${MODEL_PATH},dtype=bfloat16,device_map=auto,trust_remote_code=True\" \\\n    --tasks \"${TASKS}\" \\\n    --batch_size \"${BATCH_SIZE}\" \\\n    --output_path \"${OUTPUT_PATH}\" \\\n    --gen_kwargs \"${GEN_KWARGS}\" \\\n    --device cuda",
  "file": "/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/evaluate.sh"
}
```

### [2026-05-27 16:32:19 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 594 bytes to /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/evaluate.sh

### [2026-05-27 16:32:22 UTC] ASSISTANT

**Thinking**

Now let me execute the evaluate.sh script and stream stdout/stderr to the log file.


**Tool call:** `exec`

```json
{
  "command": "bash /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/evaluate.sh 2>&1 | tee /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/logs/eval_exec.log",
  "timeout": 600,
  "yieldMs": 120000
}
```

### [2026-05-27 16:34:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session tide-breeze, pid 1837). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-27 16:34:27 UTC] ASSISTANT

**Thinking**

The evaluation is running. Let me check the progress.

**Tool call:** `process`

```json
{
  "action": "log",
  "limit": 100,
  "sessionId": "tide-breeze"
}
```

### [2026-05-27 16:34:27 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`


Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 5887.57 examples/s]

Generating test split:   0%|          | 0/173 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 173/173 [00:00<00:00, 117565.55 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 28737.06 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6516.94 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 74965.22 examples/s]

Generating validation split:   0%|          | 0/10 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 10/10 [00:00<00:00, 12977.43 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6360.79 examples/s]

Generating test split:   0%|          | 0/223 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 223/223 [00:00<00:00, 178600.30 examples/s]

Generating validation split:   0%|          | 0/23 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 23/23 [00:00<00:00, 29866.56 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6891.73 examples/s]

Generating test split:   0%|          | 0/103 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 103/103 [00:00<00:00, 82856.41 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 14261.93 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6905.34 examples/s]

Generating test split:   0%|          | 0/234 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 234/234 [00:00<00:00, 164482.51 examples/s]

Generating validation split:   0%|          | 0/25 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 25/25 [00:00<00:00, 31013.78 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6553.60 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 71210.59 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 14513.16 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7013.89 examples/s]

Generating test split:   0%|          | 0/783 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 783/783 [00:00<00:00, 429355.48 examples/s]

Generating validation split:   0%|          | 0/86 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 86/86 [00:00<00:00, 98878.88 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6155.42 examples/s]

Generating test split:   0%|          | 0/306 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 306/306 [00:00<00:00, 195024.62 examples/s]

Generating validation split:   0%|          | 0/33 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 33/33 [00:00<00:00, 40962.42 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6436.93 examples/s]

Generating test split:   0%|          | 0/282 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 282/282 [00:00<00:00, 180882.97 examples/s]

Generating validation split:   0%|          | 0/31 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 31/31 [00:00<00:00, 37096.55 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6370.45 examples/s]

Generating test split:   0%|          | 0/272 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 272/272 [00:00<00:00, 130011.47 examples/s]

Generating validation split:   0%|          | 0/31 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 31/31 [00:00<00:00, 33528.47 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6395.71 examples/s]

Generating test split:   0%|          | 0/166 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 166/166 [00:00<00:00, 99664.25 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 22336.53 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6376.26 examples/s]

Generating test split:   0%|          | 0/114 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 114/114 [00:00<00:00, 67174.86 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 14618.54 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6173.54 examples/s]

Generating test split:   0%|          | 0/198 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 198/198 [00:00<00:00, 127431.67 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 27235.74 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6828.89 examples/s]

Generating test split:   0%|          | 0/193 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 193/193 [00:00<00:00, 129292.55 examples/s]

Generating validation split:   0%|          | 0/21 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 21/21 [00:00<00:00, 26395.08 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6640.76 examples/s]

Generating test split:   0%|          | 0/390 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 390/390 [00:00<00:00, 234654.79 examples/s]

Generating validation split:   0%|          | 0/43 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 43/43 [00:00<00:00, 52936.62 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6765.01 examples/s]

Generating test split:   0%|          | 0/238 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 238/238 [00:00<00:00, 174548.76 examples/s]

Generating validation split:   0%|          | 0/26 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 26/26 [00:00<00:00, 33513.19 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6670.33 examples/s]

Generating test split:   0%|          | 0/545 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 545/545 [00:00<00:00, 333707.40 examples/s]

Generating validation split:   0%|          | 0/60 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 60/60 [00:00<00:00, 72253.30 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6653.40 examples/s]

Generating test split:   0%|          | 0/131 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 131/131 [00:00<00:00, 95258.98 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 15699.20 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6831.11 examples/s]

Generating test split:   0%|          | 0/612 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 612/612 [00:00<00:00, 403285.79 examples/s]

Generating validation split:   0%|          | 0/69 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 69/69 [00:00<00:00, 84399.82 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6778.13 examples/s]

Generating test split:   0%|          | 0/110 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 110/110 [00:00<00:00, 76411.63 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 15173.85 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6632.36 examples/s]

Generating test split:   0%|          | 0/245 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 245/245 [00:00<00:00, 149927.70 examples/s]

Generating validation split:   0%|          | 0/27 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 27/27 [00:00<00:00, 34337.84 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6601.05 examples/s]

Generating test split:   0%|          | 0/201 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 201/201 [00:00<00:00, 134931.99 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 29386.84 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6584.46 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 79891.50 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13842.59 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6605.20 examples/s]

Generating test split:   0%|          | 0/126 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 126/126 [00:00<00:00, 99432.23 examples/s]

Generating validation split:   0%|          | 0/14 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 14/14 [00:00<00:00, 18665.05 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7070.64 examples/s]

Generating test split:   0%|          | 0/165 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 165/165 [00:00<00:00, 88873.78 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 20989.01 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6594.82 examples/s]

Generating test split:   0%|          | 0/204 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 204/204 [00:00<00:00, 118035.32 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 28141.11 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6983.52 examples/s]

Generating test split:   0%|          | 0/237 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 237/237 [00:00<00:00, 123407.83 examples/s]

Generating validation split:   0%|          | 0/26 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 26/26 [00:00<00:00, 31858.58 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6842.26 examples/s]

Generating test split:   0%|          | 0/121 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 121/121 [00:00<00:00, 92865.65 examples/s]

Generating validation split:   0%|          | 0/13 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 13/13 [00:00<00:00, 17244.13 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7226.57 examples/s]

Generating test split:   0%|          | 0/108 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 108/108 [00:00<00:00, 84638.42 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 14591.19 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7016.23 examples/s]

Generating test split:   0%|          | 0/163 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 163/163 [00:00<00:00, 124871.52 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 23637.28 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7063.50 examples/s]

Generating test split:   0%|          | 0/346 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 346/346 [00:00<00:00, 235283.59 examples/s]

Generating validation split:   0%|          | 0/38 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 38/38 [00:00<00:00, 49822.93 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 4851.15 examples/s]

Generating test split:   0%|          | 0/895 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 895/895 [00:00<00:00, 492573.43 examples/s]

Generating validation split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 100/100 [00:00<00:00, 118717.92 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7229.07 examples/s]

Generating test split:   0%|          | 0/311 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 311/311 [00:00<00:00, 231527.96 examples/s]

Generating validation split:   0%|          | 0/34 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 34/34 [00:00<00:00, 47487.96 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7051.62 examples/s]

### [2026-05-27 16:34:31 UTC] ASSISTANT

**Thinking**

The evaluation is still running. Let me wait for it to complete by polling.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "tide-breeze",
  "timeout": 300000
}
```

### [2026-05-27 16:36:31 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

0:00, 174548.76 examples/s]

Generating validation split:   0%|          | 0/26 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 26/26 [00:00<00:00, 33513.19 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6670.33 examples/s]

Generating test split:   0%|          | 0/545 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 545/545 [00:00<00:00, 333707.40 examples/s]

Generating validation split:   0%|          | 0/60 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 60/60 [00:00<00:00, 72253.30 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6653.40 examples/s]

Generating test split:   0%|          | 0/131 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 131/131 [00:00<00:00, 95258.98 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 15699.20 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6831.11 examples/s]

Generating test split:   0%|          | 0/612 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 612/612 [00:00<00:00, 403285.79 examples/s]

Generating validation split:   0%|          | 0/69 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 69/69 [00:00<00:00, 84399.82 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6778.13 examples/s]

Generating test split:   0%|          | 0/110 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 110/110 [00:00<00:00, 76411.63 examples/s]

Generating validation split:   0%|          | 0/12 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 12/12 [00:00<00:00, 15173.85 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6632.36 examples/s]

Generating test split:   0%|          | 0/245 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 245/245 [00:00<00:00, 149927.70 examples/s]

Generating validation split:   0%|          | 0/27 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 27/27 [00:00<00:00, 34337.84 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6601.05 examples/s]

Generating test split:   0%|          | 0/201 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 201/201 [00:00<00:00, 134931.99 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 29386.84 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6584.46 examples/s]

Generating test split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 100/100 [00:00<00:00, 79891.50 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 13842.59 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6605.20 examples/s]

Generating test split:   0%|          | 0/126 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 126/126 [00:00<00:00, 99432.23 examples/s]

Generating validation split:   0%|          | 0/14 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 14/14 [00:00<00:00, 18665.05 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7070.64 examples/s]

Generating test split:   0%|          | 0/165 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 165/165 [00:00<00:00, 88873.78 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 20989.01 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6594.82 examples/s]

Generating test split:   0%|          | 0/204 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 204/204 [00:00<00:00, 118035.32 examples/s]

Generating validation split:   0%|          | 0/22 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 22/22 [00:00<00:00, 28141.11 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6983.52 examples/s]

Generating test split:   0%|          | 0/237 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 237/237 [00:00<00:00, 123407.83 examples/s]

Generating validation split:   0%|          | 0/26 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 26/26 [00:00<00:00, 31858.58 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 6842.26 examples/s]

Generating test split:   0%|          | 0/121 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 121/121 [00:00<00:00, 92865.65 examples/s]

Generating validation split:   0%|          | 0/13 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 13/13 [00:00<00:00, 17244.13 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7226.57 examples/s]

Generating test split:   0%|          | 0/108 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 108/108 [00:00<00:00, 84638.42 examples/s]

Generating validation split:   0%|          | 0/11 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 11/11 [00:00<00:00, 14591.19 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7016.23 examples/s]

Generating test split:   0%|          | 0/163 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 163/163 [00:00<00:00, 124871.52 examples/s]

Generating validation split:   0%|          | 0/18 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 18/18 [00:00<00:00, 23637.28 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7063.50 examples/s]

Generating test split:   0%|          | 0/346 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 346/346 [00:00<00:00, 235283.59 examples/s]

Generating validation split:   0%|          | 0/38 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 38/38 [00:00<00:00, 49822.93 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 4851.15 examples/s]

Generating test split:   0%|          | 0/895 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 895/895 [00:00<00:00, 492573.43 examples/s]

Generating validation split:   0%|          | 0/100 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 100/100 [00:00<00:00, 118717.92 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7229.07 examples/s]

Generating test split:   0%|          | 0/311 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 311/311 [00:00<00:00, 231527.96 examples/s]

Generating validation split:   0%|          | 0/34 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 34/34 [00:00<00:00, 47487.96 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7051.62 examples/s]

Generating test split:   0%|          | 0/324 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 324/324 [00:00<00:00, 243846.13 examples/s]

Generating validation split:   0%|          | 0/35 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 35/35 [00:00<00:00, 46236.42 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7274.20 examples/s]

Generating test split:   0%|          | 0/1534 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 1534/1534 [00:00<00:00, 311637.23 examples/s]

Generating validation split:   0%|          | 0/170 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 170/170 [00:00<00:00, 162347.83 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7070.64 examples/s]

Generating test split:   0%|          | 0/171 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 171/171 [00:00<00:00, 129580.12 examples/s]

Generating validation split:   0%|          | 0/19 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 19/19 [00:00<00:00, 27161.48 examples/s]

Generating dev split:   0%|          | 0/5 [00:00<?, ? examples/s]
Generating dev split: 100%|██████████| 5/5 [00:00<00:00, 7603.89 examples/s]

Generating train split:   0%|          | 0/39905 [00:00<?, ? examples/s]
Generating train split: 100%|██████████| 39905/39905 [00:00<00:00, 579103.67 examples/s]

Generating test split:   0%|          | 0/10003 [00:00<?, ? examples/s]
Generating test split: 100%|██████████| 10003/10003 [00:00<00:00, 706454.44 examples/s]

Generating validation split:   0%|          | 0/10042 [00:00<?, ? examples/s]
Generating validation split: 100%|██████████| 10042/10042 [00:00<00:00, 666917.91 examples/s]

Map:   0%|          | 0/39905 [00:00<?, ? examples/s]
Map: 100%|██████████| 39905/39905 [00:01<00:00, 21302.03 examples/s]

Map:   0%|          | 0/10042 [00:00<?, ? examples/s]
Map: 100%|██████████| 10042/10042 [00:00<00:00, 16671.83 examples/s]
2026-05-27:16:34:40 INFO     [evaluator_utils:446] Selected tasks:
2026-05-27:16:34:40 INFO     [evaluator_utils:462] Group: mmlu
2026-05-27:16:34:40 INFO     [evaluator_utils:462]   Group: mmlu_stem
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_abstract_algebra (mmlu/default/mmlu_abstract_algebra.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_anatomy (mmlu/default/mmlu_anatomy.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_astronomy (mmlu/default/mmlu_astronomy.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_college_biology (mmlu/default/mmlu_college_biology.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_college_chemistry (mmlu/default/mmlu_college_chemistry.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_college_computer_science (mmlu/default/mmlu_college_computer_science.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_college_mathematics (mmlu/default/mmlu_college_mathematics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_college_physics (mmlu/default/mmlu_college_physics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_computer_security (mmlu/default/mmlu_computer_security.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_conceptual_physics (mmlu/default/mmlu_conceptual_physics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_electrical_engineering (mmlu/default/mmlu_electrical_engineering.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_elementary_mathematics (mmlu/default/mmlu_elementary_mathematics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_biology (mmlu/default/mmlu_high_school_biology.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_chemistry (mmlu/default/mmlu_high_school_chemistry.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_computer_science (mmlu/default/mmlu_high_school_computer_science.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_mathematics (mmlu/default/mmlu_high_school_mathematics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_physics (mmlu/default/mmlu_high_school_physics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_statistics (mmlu/default/mmlu_high_school_statistics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_machine_learning (mmlu/default/mmlu_machine_learning.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:462]   Group: mmlu_other
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_business_ethics (mmlu/default/mmlu_business_ethics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_clinical_knowledge (mmlu/default/mmlu_clinical_knowledge.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_college_medicine (mmlu/default/mmlu_college_medicine.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_global_facts (mmlu/default/mmlu_global_facts.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_human_aging (mmlu/default/mmlu_human_aging.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_management (mmlu/default/mmlu_management.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_marketing (mmlu/default/mmlu_marketing.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_medical_genetics (mmlu/default/mmlu_medical_genetics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_miscellaneous (mmlu/default/mmlu_miscellaneous.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_nutrition (mmlu/default/mmlu_nutrition.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_professional_accounting (mmlu/default/mmlu_professional_accounting.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_professional_medicine (mmlu/default/mmlu_professional_medicine.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_virology (mmlu/default/mmlu_virology.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:462]   Group: mmlu_social_sciences
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_econometrics (mmlu/default/mmlu_econometrics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_geography (mmlu/default/mmlu_high_school_geography.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_government_and_politics (mmlu/default/mmlu_high_school_government_and_politics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_macroeconomics (mmlu/default/mmlu_high_school_macroeconomics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_microeconomics (mmlu/default/mmlu_high_school_microeconomics.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_psychology (mmlu/default/mmlu_high_school_psychology.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_human_sexuality (mmlu/default/mmlu_human_sexuality.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_professional_psychology (mmlu/default/mmlu_professional_psychology.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_public_relations (mmlu/default/mmlu_public_relations.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_security_studies (mmlu/default/mmlu_security_studies.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_sociology (mmlu/default/mmlu_sociology.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_us_foreign_policy (mmlu/default/mmlu_us_foreign_policy.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:462]   Group: mmlu_humanities
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_formal_logic (mmlu/default/mmlu_formal_logic.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_european_history (mmlu/default/mmlu_high_school_european_history.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_us_history (mmlu/default/mmlu_high_school_us_history.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_high_school_world_history (mmlu/default/mmlu_high_school_world_history.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_international_law (mmlu/default/mmlu_international_law.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_jurisprudence (mmlu/default/mmlu_jurisprudence.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_logical_fallacies (mmlu/default/mmlu_logical_fallacies.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_moral_disputes (mmlu/default/mmlu_moral_disputes.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_moral_scenarios (mmlu/default/mmlu_moral_scenarios.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_philosophy (mmlu/default/mmlu_philosophy.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_prehistory (mmlu/default/mmlu_prehistory.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_professional_law (mmlu/default/mmlu_professional_law.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:470]     Task: mmlu_world_religions (mmlu/default/mmlu_world_religions.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:480] Task: hellaswag (hellaswag/hellaswag.yaml)
2026-05-27:16:34:40 INFO     [evaluator_utils:480] Task: piqa (piqa/piqa.yaml)
2026-05-27:16:34:40 INFO     [api.task:312] Building contexts for piqa on rank 0...

  0%|          | 0/1838 [00:00<?, ?it/s]
100%|██████████| 1838/1838 [00:00<00:00, 2291.53it/s]
2026-05-27:16:34:40 INFO     [api.task:312] Building contexts for mmlu_abstract_algebra on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1954.04it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_anatomy on rank 0...

  0%|          | 0/135 [00:00<?, ?it/s]
100%|██████████| 135/135 [00:00<00:00, 1974.78it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_astronomy on rank 0...

  0%|          | 0/152 [00:00<?, ?it/s]
100%|██████████| 152/152 [00:00<00:00, 1969.19it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_college_biology on rank 0...

  0%|          | 0/144 [00:00<?, ?it/s]
100%|██████████| 144/144 [00:00<00:00, 1980.32it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_college_chemistry on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1962.57it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_college_computer_science on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1953.40it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_college_mathematics on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1960.42it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_college_physics on rank 0...

  0%|          | 0/102 [00:00<?, ?it/s]
100%|██████████| 102/102 [00:00<00:00, 1966.38it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_computer_security on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1958.98it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_conceptual_physics on rank 0...

  0%|          | 0/235 [00:00<?, ?it/s]
100%|██████████| 235/235 [00:00<00:00, 1968.11it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_electrical_engineering on rank 0...

  0%|          | 0/145 [00:00<?, ?it/s]
100%|██████████| 145/145 [00:00<00:00, 1965.70it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_elementary_mathematics on rank 0...

  0%|          | 0/378 [00:00<?, ?it/s]
100%|██████████| 378/378 [00:00<00:00, 1952.73it/s]
2026-05-27:16:34:41 INFO     [api.task:312] Building contexts for mmlu_high_school_biology on rank 0...

  0%|          | 0/310 [00:00<?, ?it/s]
100%|██████████| 310/310 [00:00<00:00, 1952.88it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_high_school_chemistry on rank 0...

  0%|          | 0/203 [00:00<?, ?it/s]
100%|██████████| 203/203 [00:00<00:00, 1947.99it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_high_school_computer_science on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1946.22it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_high_school_mathematics on rank 0...

  0%|          | 0/270 [00:00<?, ?it/s]
100%|██████████| 270/270 [00:00<00:00, 1941.56it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_high_school_physics on rank 0...

  0%|          | 0/151 [00:00<?, ?it/s]
100%|██████████| 151/151 [00:00<00:00, 1949.16it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_high_school_statistics on rank 0...

  0%|          | 0/216 [00:00<?, ?it/s]
100%|██████████| 216/216 [00:00<00:00, 1940.96it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_machine_learning on rank 0...

  0%|          | 0/112 [00:00<?, ?it/s]
100%|██████████| 112/112 [00:00<00:00, 1941.61it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_business_ethics on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1939.80it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_clinical_knowledge on rank 0...

  0%|          | 0/265 [00:00<?, ?it/s]
100%|██████████| 265/265 [00:00<00:00, 1946.18it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_college_medicine on rank 0...

  0%|          | 0/173 [00:00<?, ?it/s]
100%|██████████| 173/173 [00:00<00:00, 1932.63it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_global_facts on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1934.18it/s]
2026-05-27:16:34:42 INFO     [api.task:312] Building contexts for mmlu_human_aging on rank 0...

  0%|          | 0/223 [00:00<?, ?it/s]
100%|██████████| 223/223 [00:00<00:00, 1936.54it/s]
2026-05-27:16:34:43 INFO     [api.task:312] Building contexts for mmlu_management on rank 0...

  0%|          | 0/103 [00:00<?, ?it/s]
100%|██████████| 103/103 [00:00<00:00, 1927.46it/s]
2026-05-27:16:34:43 INFO     [api.task:312] Building contexts for mmlu_marketing on rank 0...

  0%|          | 0/234 [00:00<?, ?it/s]
100%|██████████| 234/234 [00:00<00:00, 1950.89it/s]
2026-05-27:16:34:43 INFO     [api.task:312] Building contexts for mmlu_medical_genetics on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1936.29it/s]
2026-05-27:16:34:43 INFO     [api.task:312] Building contexts for mmlu_miscellaneous on rank 0...

  0%|          | 0/783 [00:00<?, ?it/s]
100%|██████████| 783/783 [00:00<00:00, 1944.15it/s]
2026-05-27:16:34:43 INFO     [api.task:312] Building contexts for mmlu_nutrition on rank 0...

  0%|          | 0/306 [00:00<?, ?it/s]
100%|██████████| 306/306 [00:00<00:00, 1953.42it/s]
2026-05-27:16:34:43 INFO     [api.task:312] Building contexts for mmlu_professional_accounting on rank 0...

  0%|          | 0/282 [00:00<?, ?it/s]
100%|██████████| 282/282 [00:00<00:00, 1947.80it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_professional_medicine on rank 0...

  0%|          | 0/272 [00:00<?, ?it/s]
100%|██████████| 272/272 [00:00<00:00, 1942.88it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_virology on rank 0...

  0%|          | 0/166 [00:00<?, ?it/s]
100%|██████████| 166/166 [00:00<00:00, 1949.87it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_econometrics on rank 0...

  0%|          | 0/114 [00:00<?, ?it/s]
100%|██████████| 114/114 [00:00<00:00, 1943.46it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_high_school_geography on rank 0...

  0%|          | 0/198 [00:00<?, ?it/s]
100%|██████████| 198/198 [00:00<00:00, 1950.41it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_high_school_government_and_politics on rank 0...

  0%|          | 0/193 [00:00<?, ?it/s]
100%|██████████| 193/193 [00:00<00:00, 1955.66it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_high_school_macroeconomics on rank 0...

  0%|          | 0/390 [00:00<?, ?it/s]
100%|██████████| 390/390 [00:00<00:00, 1956.79it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_high_school_microeconomics on rank 0...

  0%|          | 0/238 [00:00<?, ?it/s]
100%|██████████| 238/238 [00:00<00:00, 1956.63it/s]
2026-05-27:16:34:44 INFO     [api.task:312] Building contexts for mmlu_high_school_psychology on rank 0...

  0%|          | 0/545 [00:00<?, ?it/s]
100%|██████████| 545/545 [00:00<00:00, 1967.04it/s]
2026-05-27:16:34:45 INFO     [api.task:312] Building contexts for mmlu_human_sexuality on rank 0...

  0%|          | 0/131 [00:00<?, ?it/s]
100%|██████████| 131/131 [00:00<00:00, 1958.79it/s]
2026-05-27:16:34:45 INFO     [api.task:312] Building contexts for mmlu_professional_psychology on rank 0...

  0%|          | 0/612 [00:00<?, ?it/s]
100%|██████████| 612/612 [00:00<00:00, 996.21it/s]
2026-05-27:16:34:45 INFO     [api.task:312] Building contexts for mmlu_public_relations on rank 0...

  0%|          | 0/110 [00:00<?, ?it/s]
100%|██████████| 110/110 [00:00<00:00, 1958.38it/s]
2026-05-27:16:34:45 INFO     [api.task:312] Building contexts for mmlu_security_studies on rank 0...

  0%|          | 0/245 [00:00<?, ?it/s]
100%|██████████| 245/245 [00:00<00:00, 1966.51it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_sociology on rank 0...

  0%|          | 0/201 [00:00<?, ?it/s]
100%|██████████| 201/201 [00:00<00:00, 1966.98it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_us_foreign_policy on rank 0...

  0%|          | 0/100 [00:00<?, ?it/s]
100%|██████████| 100/100 [00:00<00:00, 1948.41it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_formal_logic on rank 0...

  0%|          | 0/126 [00:00<?, ?it/s]
100%|██████████| 126/126 [00:00<00:00, 1968.91it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_high_school_european_history on rank 0...

  0%|          | 0/165 [00:00<?, ?it/s]
100%|██████████| 165/165 [00:00<00:00, 1957.55it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_high_school_us_history on rank 0...

  0%|          | 0/204 [00:00<?, ?it/s]
100%|██████████| 204/204 [00:00<00:00, 1959.79it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_high_school_world_history on rank 0...

  0%|          | 0/237 [00:00<?, ?it/s]
100%|██████████| 237/237 [00:00<00:00, 1965.50it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_international_law on rank 0...

  0%|          | 0/121 [00:00<?, ?it/s]
100%|██████████| 121/121 [00:00<00:00, 1973.86it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_jurisprudence on rank 0...

  0%|          | 0/108 [00:00<?, ?it/s]
100%|██████████| 108/108 [00:00<00:00, 1972.42it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_logical_fallacies on rank 0...

  0%|          | 0/163 [00:00<?, ?it/s]
100%|██████████| 163/163 [00:00<00:00, 1978.66it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_moral_disputes on rank 0...

  0%|          | 0/346 [00:00<?, ?it/s]
100%|██████████| 346/346 [00:00<00:00, 1981.26it/s]
2026-05-27:16:34:46 INFO     [api.task:312] Building contexts for mmlu_moral_scenarios on rank 0...

  0%|          | 0/895 [00:00<?, ?it/s]
100%|██████████| 895/895 [00:00<00:00, 1974.66it/s]
2026-05-27:16:34:47 INFO     [api.task:312] Building contexts for mmlu_philosophy on rank 0...

  0%|          | 0/311 [00:00<?, ?it/s]
100%|██████████| 311/311 [00:00<00:00, 1971.46it/s]
2026-05-27:16:34:47 INFO     [api.task:312] Building contexts for mmlu_prehistory on rank 0...

  0%|          | 0/324 [00:00<?, ?it/s]
100%|██████████| 324/324 [00:00<00:00, 1974.12it/s]
2026-05-27:16:34:47 INFO     [api.task:312] Building contexts for mmlu_professional_law on rank 0...

  0%|          | 0/1534 [00:00<?, ?it/s]
100%|██████████| 1534/1534 [00:00<00:00, 1973.36it/s]
2026-05-27:16:34:48 INFO     [api.task:312] Building contexts for mmlu_world_religions on rank 0...

  0%|          | 0/171 [00:00<?, ?it/s]
100%|██████████| 171/171 [00:00<00:00, 1980.45it/s]
2026-05-27:16:34:48 INFO     [api.task:312] Building contexts for hellaswag on rank 0...

  0%|          | 0/10042 [00:00<?, ?it/s]
100%|██████████| 10042/10042 [00:02<00:00, 4578.94it/s]
2026-05-27:16:34:51 INFO     [evaluator:585] Running loglikelihood requests

Tokenizing inputs:   0%|          | 0/100012 [00:00<?, ?it/s]
Tokenizing inputs: 100%|██████████| 100012/100012 [00:21<00:00, 4719.04it/s]

Running loglikelihood requests:   0%|          | 0/100012 [00:00<?, ?it/s]

Process still running.

### [2026-05-27 16:36:35 UTC] ASSISTANT

**Thinking**

It's still running - let me continue polling.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "tide-breeze",
  "timeout": 600000
}
```

### [2026-05-27 16:38:35 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

Running loglikelihood requests:  40%|████      | 40293/100012 [02:00<02:57, 335.72it/s]
Running loglikelihood requests:  43%|████▎     | 43309/100012 [02:10<02:48, 335.72it/s]

Process still running.

### [2026-05-27 16:38:39 UTC] ASSISTANT

**Thinking**

The evaluation is still running, with about 43% complete. Let me wait longer.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "tide-breeze",
  "timeout": 900000
}
```

### [2026-05-27 16:39:52 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

Running loglikelihood requests:  90%|█████████ | 90343/100012 [04:00<00:25, 383.58it/s]
Running loglikelihood requests:  93%|█████████▎| 93066/100012 [04:10<00:18, 383.58it/s]
Running loglikelihood requests: 100%|██████████| 100012/100012 [04:35<00:00, 363.02it/s]
fatal: not a git repository (or any of the parent directories): .git
2026-05-27:16:39:51 INFO     [loggers.evaluation_tracker:247] Saving results aggregated
hf ({'pretrained': '/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN', 'dtype': 'bfloat16', 'device_map': 'auto'}), gen_kwargs: ({'max_gen_toks': 2048}), limit: None, num_fewshot: None, batch_size: 8
|                 Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
|mmlu                                   |      2|none  |      |acc     |   |0.3199|±  |0.0039|
| - humanities                          |      2|none  |     0|acc     |↑  |0.2982|±  |0.0066|
|  - formal_logic                       |      1|none  |     0|acc     |↑  |0.1905|±  |0.0351|
|  - high_school_european_history       |      1|none  |     0|acc     |↑  |0.3212|±  |0.0365|
|  - high_school_us_history             |      1|none  |     0|acc     |↑  |0.3431|±  |0.0333|
|  - high_school_world_history          |      1|none  |     0|acc     |↑  |0.3291|±  |0.0306|
|  - international_law                  |      1|none  |     0|acc     |↑  |0.5124|±  |0.0456|
|  - jurisprudence                      |      1|none  |     0|acc     |↑  |0.3889|±  |0.0471|
|  - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.2699|±  |0.0349|
|  - moral_disputes                     |      1|none  |     0|acc     |↑  |0.2948|±  |0.0245|
|  - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2637|±  |0.0147|
|  - philosophy                         |      1|none  |     0|acc     |↑  |0.3601|±  |0.0273|
|  - prehistory                         |      1|none  |     0|acc     |↑  |0.3580|±  |0.0267|
|  - professional_law                   |      1|none  |     0|acc     |↑  |0.2705|±  |0.0113|
|  - world_religions                    |      1|none  |     0|acc     |↑  |0.2865|±  |0.0347|
| - other                               |      2|none  |     0|acc     |↑  |0.3370|±  |0.0084|
|  - business_ethics                    |      1|none  |     0|acc     |↑  |0.2300|±  |0.0423|
|  - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.3887|±  |0.0300|
|  - college_medicine                   |      1|none  |     0|acc     |↑  |0.3873|±  |0.0371|
|  - global_facts                       |      1|none  |     0|acc     |↑  |0.3300|±  |0.0473|
|  - human_aging                        |      1|none  |     0|acc     |↑  |0.3184|±  |0.0313|
|  - management                         |      1|none  |     0|acc     |↑  |0.4854|±  |0.0495|
|  - marketing                          |      1|none  |     0|acc     |↑  |0.3376|±  |0.0310|
|  - medical_genetics                   |      1|none  |     0|acc     |↑  |0.3300|±  |0.0473|
|  - miscellaneous                      |      1|none  |     0|acc     |↑  |0.3563|±  |0.0171|
|  - nutrition                          |      1|none  |     0|acc     |↑  |0.3954|±  |0.0280|
|  - professional_accounting            |      1|none  |     0|acc     |↑  |0.2447|±  |0.0256|
|  - professional_medicine              |      1|none  |     0|acc     |↑  |0.2353|±  |0.0258|
|  - virology                           |      1|none  |     0|acc     |↑  |0.3313|±  |0.0366|
| - social sciences                     |      2|none  |     0|acc     |↑  |0.3552|±  |0.0086|
|  - econometrics                       |      1|none  |     0|acc     |↑  |0.2368|±  |0.0400|
|  - high_school_geography              |      1|none  |     0|acc     |↑  |0.3990|±  |0.0349|
|  - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.3575|±  |0.0346|
|  - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.3179|±  |0.0236|
|  - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.3445|±  |0.0309|
|  - high_school_psychology             |      1|none  |     0|acc     |↑  |0.4037|±  |0.0210|
|  - human_sexuality                    |      1|none  |     0|acc     |↑  |0.3817|±  |0.0426|
|  - professional_psychology            |      1|none  |     0|acc     |↑  |0.3301|±  |0.0190|
|  - public_relations                   |      1|none  |     0|acc     |↑  |0.3273|±  |0.0449|
|  - security_studies                   |      1|none  |     0|acc     |↑  |0.3551|±  |0.0306|
|  - sociology                          |      1|none  |     0|acc     |↑  |0.3731|±  |0.0342|
|  - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.4200|±  |0.0496|
| - stem                                |      2|none  |     0|acc     |↑  |0.3010|±  |0.0082|
|  - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2700|±  |0.0446|
|  - anatomy                            |      1|none  |     0|acc     |↑  |0.3481|±  |0.0412|
|  - astronomy                          |      1|none  |     0|acc     |↑  |0.3618|±  |0.0391|
|  - college_biology                    |      1|none  |     0|acc     |↑  |0.2917|±  |0.0380|
|  - college_chemistry                  |      1|none  |     0|acc     |↑  |0.2700|±  |0.0446|
|  - college_computer_science           |      1|none  |     0|acc     |↑  |0.2500|±  |0.0435|
|  - college_mathematics                |      1|none  |     0|acc     |↑  |0.2700|±  |0.0446|
|  - college_physics                    |      1|none  |     0|acc     |↑  |0.1863|±  |0.0387|
|  - computer_security                  |      1|none  |     0|acc     |↑  |0.4000|±  |0.0492|
|  - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.2681|±  |0.0290|
|  - electrical_engineering             |      1|none  |     0|acc     |↑  |0.3517|±  |0.0398|
|  - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.2937|±  |0.0235|
|  - high_school_biology                |      1|none  |     0|acc     |↑  |0.3484|±  |0.0271|
|  - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.3399|±  |0.0333|
|  - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.2500|±  |0.0435|
|  - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.2593|±  |0.0267|
|  - high_school_physics                |      1|none  |     0|acc     |↑  |0.3179|±  |0.0380|
|  - high_school_statistics             |      1|none  |     0|acc     |↑  |0.2593|±  |0.0299|
|  - machine_learning                   |      1|none  |     0|acc     |↑  |0.3482|±  |0.0452|
|hellaswag                              |      1|none  |     0|acc     |↑  |0.3951|±  |0.0049|
|                                       |       |none  |     0|acc_norm|↑  |0.5160|±  |0.0050|
|piqa                                   |      1|none  |     0|acc     |↑  |0.6877|±  |0.0108|
|                                       |       |none  |     0|acc_norm|↑  |0.6844|±  |0.0108|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|-----:|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |   |0.3199|±  |0.0039|
| - humanities     |      2|none  |     0|acc   |↑  |0.2982|±  |0.0066|
| - other          |      2|none  |     0|acc   |↑  |0.3370|±  |0.0084|
| - social sciences|      2|none  |     0|acc   |↑  |0.3552|±  |0.0086|
| - stem           |      2|none  |     0|acc   |↑  |0.3010|±  |0.0082|

Process exited with code 0.

### [2026-05-27 16:39:57 UTC] ASSISTANT

**Thinking**

The evaluation completed successfully. Now I need to parse the results and write the accuracy.json file. Let me first check the raw lm_eval results to find the results JSON file.



Evaluation completed successfully. Now parsing results and writing accuracy.json.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/"
}
```

### [2026-05-27 16:39:57 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 0
drwxr-xr-x 3 root root 115 May 27 16:39 .
drwxr-xr-x 4 root root 188 May 27 16:39 ..
drwxr-xr-x 2 root root  61 May 27 16:39 __root__.openclaw__workspace__quantized__LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN

### [2026-05-27 16:40:00 UTC] ASSISTANT

**Thinking**

It looks like there's a weirdly named directory with `__root__` in it. Let me explore this further.

**Tool call:** `exec`

```json
{
  "command": "find /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/ -name \"results*.json\" 2>/dev/null"
}
```

### [2026-05-27 16:40:00 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/__root__.openclaw__workspace__quantized__LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN/results_2026-05-27T16-39-51.247490.json

### [2026-05-27 16:40:03 UTC] ASSISTANT

**Thinking**

Let me read the results file and parse the accuracies.

**Tool call:** `read`

```json
{
  "file": "/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/__root__.openclaw__workspace__quantized__LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN/results_2026-05-27T16-39-51.247490.json"
}
```

### [2026-05-27 16:40:03 UTC] TOOL RESULT
**Tool:** `read`

{
  "results": {
    "piqa": {
      "name": "piqa",
      "alias": "piqa",
      "sample_len": 1838,
      "acc,none": 0.6877040261153428,
      "acc_stderr,none": 0.010812581599154419,
      "acc_norm,none": 0.6844396082698585,
      "acc_norm_stderr,none": 0.010843119201758775
    },
    "mmlu_abstract_algebra": {
      "name": "mmlu_abstract_algebra",
      "alias": "abstract_algebra",
      "sample_len": 100,
      "acc,none": 0.27,
      "acc_stderr,none": 0.04461960433384737
    },
    "mmlu_anatomy": {
      "name": "mmlu_anatomy",
      "alias": "anatomy",
      "sample_len": 135,
      "acc,none": 0.34814814814814815,
      "acc_stderr,none": 0.04115324610336956
    },
    "mmlu_astronomy": {
      "name": "mmlu_astronomy",
      "alias": "astronomy",
      "sample_len": 152,
      "acc,none": 0.3618421052631579,
      "acc_stderr,none": 0.03910525752849724
    },
    "mmlu_college_biology": {
      "name": "mmlu_college_biology",
      "alias": "college_biology",
      "sample_len": 144,
      "acc,none": 0.2916666666666667,
      "acc_stderr,none": 0.03800968060554863
    },
    "mmlu_college_chemistry": {
      "name": "mmlu_college_chemistry",
      "alias": "college_chemistry",
      "sample_len": 100,
      "acc,none": 0.27,
      "acc_stderr,none": 0.04461960433384737
    },
    "mmlu_college_computer_science": {
      "name": "mmlu_college_computer_science",
      "alias": "college_computer_science",
      "sample_len": 100,
      "acc,none": 0.25,
      "acc_stderr,none": 0.04351941398892446
    },
    "mmlu_college_mathematics": {
      "name": "mmlu_college_mathematics",
      "alias": "college_mathematics",
      "sample_len": 100,
      "acc,none": 0.27,
      "acc_stderr,none": 0.04461960433384737
    },
    "mmlu_college_physics": {
      "name": "mmlu_college_physics",
      "alias": "college_physics",
      "sample_len": 102,
      "acc,none": 0.18627450980392157,
      "acc_stderr,none": 0.03873958714149351
    },
    "mmlu_computer_security": {
      "name": "mmlu_computer_security",
      "alias": "computer_security",
      "sample_len": 100,
      "acc,none": 0.4,
      "acc_stderr,none": 0.0492365963917331
    },
    "mmlu_conceptual_physics": {
      "name": "mmlu_conceptual_physics",
      "alias": "conceptual_physics",
      "sample_len": 235,
      "acc,none": 0.2680851063829787,
      "acc_stderr,none": 0.02895734278834235
    },
    "mmlu_electrical_engineering": {
      "name": "mmlu_electrical_engineering",
      "alias": "electrical_engineering",
      "sample_len": 145,
      "acc,none": 0.35172413793103446,
      "acc_stderr,none": 0.03979236637497416
    },
    "mmlu_elementary_mathematics": {
      "name": "mmlu_elementary_mathematics",
      "alias": "elementary_mathematics",
      "sample_len": 378,
      "acc,none": 0.29365079365079366,
      "acc_stderr,none": 0.02345603738398198
    },
    "mmlu_high_school_biology": {
      "name": "mmlu_high_school_biology",
      "alias": "high_school_biology",
      "sample_len": 310,
      "acc,none": 0.34838709677419355,
      "acc_stderr,none": 0.027104826328100906
    },
    "mmlu_high_school_chemistry": {
      "name": "mmlu_high_school_chemistry",
      "alias": "high_school_chemistry",
      "sample_len": 203,
      "acc,none": 0.3399014778325123,
      "acc_stderr,none": 0.033327690684107965
    },
    "mmlu_high_school_computer_science": {
      "name": "mmlu_high_school_computer_science",
      "alias": "high_school_computer_science",
      "sample_len": 100,
      "acc,none": 0.25,
      "acc_stderr,none": 0.04351941398892446
    },
    "mmlu_high_school_mathematics": {
      "name": "mmlu_high_school_mathematics",
      "alias": "high_school_mathematics",
      "sample_len": 270,
      "acc,none": 0.25925925925925924,
      "acc_stderr,none": 0.02671924078371222
    },
    "mmlu_high_school_physics": {
      "name": "mmlu_high_school_physics",
      "alias": "high_school_physics",
      "sample_len": 151,
      "acc,none": 0.31788079470198677,
      "acc_stderr,none": 0.038020397601078997
    },
    "mmlu_high_school_statistics": {
      "name": "mmlu_high_school_statistics",
      "alias": "high_school_statistics",
      "sample_len": 216,
      "acc,none": 0.25925925925925924,
      "acc_stderr,none": 0.029886910547626995
    },
    "mmlu_machine_learning": {
      "name": "mmlu_machine_learning",
      "alias": "machine_learning",
      "sample_len": 112,
      "acc,none": 0.3482142857142857,
      "acc_stderr,none": 0.04521829902833581
    },
    "mmlu_business_ethics": {
      "name": "mmlu_business_ethics",
      "alias": "business_ethics",
      "sample_len": 100,
      "acc,none": 0.23,
      "acc_stderr,none": 0.04229525846816507
    },
    "mmlu_clinical_knowledge": {
      "name": "mmlu_clinical_knowledge",
      "alias": "clinical_knowledge",
      "sample_len": 265,
      "acc,none": 0.3886792452830189,
      "acc_stderr,none": 0.03000048544867597
    },
    "mmlu_college_medicine": {
      "name": "mmlu_college_medicine",
      "alias": "college_medicine",
      "sample_len": 173,
      "acc,none": 0.3872832369942196,
      "acc_stderr,none": 0.03714325906302067
    },
    "mmlu_global_facts": {
      "name": "mmlu_global_facts",
      "alias": "global_facts",
      "sample_len": 100,
      "acc,none": 0.33,
      "acc_stderr,none": 0.04725815626252609
    },
    "mmlu_human_aging": {
      "name": "mmlu_human_aging",
      "alias": "human_aging",
      "sample_len": 223,
      "acc,none": 0.3183856502242152,
      "acc_stderr,none": 0.03126580522513711
    },
    "mmlu_management": {
      "name": "mmlu_management",
      "alias": "management",
      "sample_len": 103,
      "acc,none": 0.4854368932038835,
      "acc_stderr,none": 0.049486373240266356
    },
    "mmlu_marketing": {
      "name": "mmlu_marketing",
      "alias": "marketing",
      "sample_len": 234,
      "acc,none": 0.33760683760683763,
      "acc_stderr,none": 0.030980296992618544
    },
    "mmlu_medical_genetics": {
      "name": "mmlu_medical_genetics",
      "alias": "medical_genetics",
      "sample_len": 100,
      "acc,none": 0.33,
      "acc_stderr,none": 0.04725815626252609
    },
    "mmlu_miscellaneous": {
      "name": "mmlu_miscellaneous",
      "alias": "miscellaneous",
      "sample_len": 783,
      "acc,none": 0.3563218390804598,
      "acc_stderr,none": 0.017125853762755817
    },
    "mmlu_nutrition": {
      "name": "mmlu_nutrition",
      "alias": "nutrition",
      "sample_len": 306,
      "acc,none": 0.3954248366013072,
      "acc_stderr,none": 0.02799672318063147
    },
    "mmlu_professional_accounting": {
      "name": "mmlu_professional_accounting",
      "alias": "professional_accounting",
      "sample_len": 282,
      "acc,none": 0.24468085106382978,
      "acc_stderr,none": 0.025645553622266795
    },
    "mmlu_professional_medicine": {
      "name": "mmlu_professional_medicine",
      "alias": "professional_medicine",
      "sample_len": 272,
      "acc,none": 0.23529411764705882,
      "acc_stderr,none": 0.025767252010855998
    },
    "mmlu_virology": {
      "name": "mmlu_virology",
      "alias": "virology",
      "sample_len": 166,
      "acc,none": 0.3313253012048193,
      "acc_stderr,none": 0.036643147772880844
    },
    "mmlu_econometrics": {
      "name": "mmlu_econometrics",
      "alias": "econometrics",
      "sample_len": 114,
      "acc,none": 0.23684210526315788,
      "acc_stderr,none": 0.03999423879281335
    },
    "mmlu_high_school_geography": {
      "name": "mmlu_high_school_geography",
      "alias": "high_school_geography",
      "sample_len": 198,
      "acc,none": 0.398989898989899,
      "acc_stderr,none": 0.03488901616852734
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
      "acc,none": 0.31794871794871793,
      "acc_stderr,none": 0.023610884308927833
    },
    "mmlu_high_school_microeconomics": {
      "name": "mmlu_high_school_microeconomics",
      "alias": "high_school_microeconomics",
      "sample_len": 238,
      "acc,none": 0.3445378151260504,
      "acc_stderr,none": 0.030868682604121605
    },
    "mmlu_high_school_psychology": {
      "name": "mmlu_high_school_psychology",
      "alias": "high_school_psychology",
      "sample_len": 545,
      "acc,none": 0.4036697247706422,
      "acc_stderr,none": 0.021035704856575025
    },
    "mmlu_human_sexuality": {
      "name": "mmlu_human_sexuality",
      "alias": "human_sexuality",
      "sample_len": 131,
      "acc,none": 0.3816793893129771,
      "acc_stderr,none": 0.04260735157644566
    },
    "mmlu_professional_psychology": {
      "name": "mmlu_professional_psychology",
      "alias": "professional_psychology",
      "sample_len": 612,
      "acc,none": 0.3300653594771242,
      "acc_stderr,none": 0.019023726160724508
    },
    "mmlu_public_relations": {
      "name": "mmlu_public_relations",
      "alias": "public_relations",
      "sample_len": 110,
      "acc,none": 0.32727272727272727,
      "acc_stderr,none": 0.04494290866252085
    },
    "mmlu_security_studies": {
      "name": "mmlu_security_studies",
      "alias": "security_studies",
      "sample_len": 245,
      "acc,none": 0.3551020408163265,
      "acc_stderr,none": 0.03063565515038758
    },
    "mmlu_sociology": {
      "name": "mmlu_sociology",
      "alias": "sociology",
      "sample_len": 201,
      "acc,none": 0.373134328358209,
      "acc_stderr,none": 0.03419832608176
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
      "acc,none": 0.19047619047619047,
      "acc_stderr,none": 0.03512207412302052
    },
    "mmlu_high_school_european_history": {
      "name": "mmlu_high_school_european_history",
      "alias": "high_school_european_history",
      "sample_len": 165,
      "acc,none": 0.3212121212121212,
      "acc_stderr,none": 0.03646204963253815
    },
    "mmlu_high_school_us_history": {
      "name": "mmlu_high_school_us_history",
      "alias": "high_school_us_history",
      "sample_len": 204,
      "acc,none": 0.3431372549019608,
      "acc_stderr,none": 0.03332139944668079
    },
    "mmlu_high_school_world_history": {
      "name": "mmlu_high_school_world_history",
      "alias": "high_school_world_history",
      "sample_len": 237,
      "acc,none": 0.3291139240506329,
      "acc_stderr,none": 0.030587326294702347
    },
    "mmlu_international_law": {
      "name": "mmlu_international_law",
      "alias": "international_law",
      "sample_len": 121,
      "acc,none": 0.512396694214876,
      "acc_stderr,none": 0.045629515481807645
    },
    "mmlu_jurisprudence": {
      "name": "mmlu_jurisprudence",
      "alias": "jurisprudence",
      "sample_len": 108,
      "acc,none": 0.3888888888888889,
      "acc_stderr,none": 0.04712821257426773
    },
    "mmlu_logical_fallacies": {
      "name": "mmlu_logical_fallacies",
      "alias": "logical_fallacies",
      "sample_len": 163,
      "acc,none": 0.26993865030674846,
      "acc_stderr,none": 0.034878251684978955
    },
    "mmlu_moral_disputes": {
      "name": "mmlu_moral_disputes",
      "alias": "moral_disputes",
      "sample_len": 346,
      "acc,none": 0.2947976878612717,
      "acc_stderr,none": 0.02454761779480379
    },
    "mmlu_moral_scenarios": {
      "name": "mmlu_moral_scenarios",
      "alias": "moral_scenarios",
      "sample_len": 895,
      "acc,none": 0.2636871508379888,
      "acc_stderr,none": 0.014736926383761872
    },
    "mmlu_philosophy": {
      "name": "mmlu_philosophy",
      "alias": "philosophy",
      "sample_len": 311,
      "acc,none": 0.36012861736334406,
      "acc_stderr,none": 0.027264297599803998
    },
    "mmlu_prehistory": {
      "name": "mmlu_prehistory",
      "alias": "prehistory",
      "sample_len": 324,
      "acc,none": 0.35802469135802467,
      "acc_stderr,none": 0.026675611926037082
    },
    "mmlu_professional_law": {
      "name": "mmlu_professional_law",
      "alias": "professional_law",
      "sample_len": 1534,
      "acc,none": 0.27053455019556716,
      "acc_stderr,none": 0.011345996743539222
    },
    "mmlu_world_religions": {
      "name": "mmlu_world_religions",
      "alias": "world_religions",
      "sample_len": 171,
      "acc,none": 0.28654970760233917,
      "acc_stderr,none": 0.03467826685703828
    },
    "hellaswag": {
      "name": "hellaswag",
      "alias": "hellaswag",
      "sample_len": 10042,
      "acc,none": 0.3951404102768373,
      "acc_stderr,none": 0.004878816961012051,
      "acc_norm,none": 0.5160326628161721,
      "acc_norm_stderr,none": 0.004987215542259626
    },
    "mmlu_stem": {
      "alias": "stem",
      "name": "mmlu_stem",
      "sample_len": 3153,
      "acc,none": 0.3009831906121154,
      "acc_stderr,none": 0.008151338437239995,
      "sample_count": {
        "acc,none": 3153
      }
    },
    "mmlu_other": {
      "alias": "other",
      "name": "mmlu_other",
      "sample_len": 3107,
      "acc,none": 0.33698101062117797,
      "acc_stderr,none": 0.00842891400379223,
      "sample_count": {
        "acc,none": 3107
      }
    },
    "mmlu_social_sciences": {
      "alias": "social sciences",
      "name": "mmlu_social_sciences",
      "sample_len": 3077,
      "acc,none": 0.35521611959701005,
      "acc_stderr,none": 0.008614515006033453,
      "sample_count": {
        "acc,none": 3077
      }
    },
    "mmlu_humanities": {
      "alias": "humanities",
      "name": "mmlu_humanities",
      "sample_len": 4705,
      "acc,none": 0.29819341126461213,
      "acc_stderr,none": 0.006632973653300106,
      "sample_count": {
        "acc,none": 4705
      }
    },
    "mmlu": {
      "alias": "mmlu",
      "name": "mmlu",
      "sample_len": 14042,
      "acc,none": 0.31989745050562596,
      "acc_stderr,none": 0.003915491117517266,
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
      "acc,none": 0.3009831906121154,
      "acc_stderr,none": 0.008151338437239995,
      "sample_count": {
        "acc,none": 3153
      }
    },
    "mmlu_other": {
      "alias": "other",
      "name": "mmlu_other",
      "sample_len": 3107,
      "acc,none": 0.33698101062117797,
      "acc_stderr,none": 0.00842891400379223,
      "sample_count": {
        "acc,none": 3107
      }
    },
    "mmlu_social_sciences": {
      "alias": "social sciences",
      "name": "mmlu_social_sciences",
      "sample_len": 3077,
      "acc,none": 0.35521611959701005,
      "acc_stderr,none": 0.008614515006033453,
      "sample_count": {
        "acc,none": 3077
      }
    },
    "mmlu_humanities": {
      "alias": "humanities",
      "name": "mmlu_humanities",
      "sample_len": 4705,
      "acc,none": 0.29819341126461213,
      "acc_stderr,none": 0.006632973653300106,
      "sample_count": {
        "acc,none": 4705
      }
    },
    "mmlu": {
      "alias": "mmlu",
      "name": "mmlu",
      "sample_len": 14042,
      "acc,none": 0.31989745050562596,
      "acc_stderr,none": 0.003915491117517266,
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
        "process_docs": "<function process_docs at 0x7e10d5ca3ce0>",
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/hellaswag/hellaswag.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_abstract_algebra.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_anatomy.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_astronomy.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_business_ethics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_clinical_knowledge.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_biology.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_chemistry.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_computer_science.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_mathematics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_medicine.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_college_physics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_computer_security.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_conceptual_physics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_econometrics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_electrical_engineering.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_elementary_mathematics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_formal_logic.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_global_facts.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_biology.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_chemistry.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_computer_science.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_european_history.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_geography.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_government_and_politics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_macroeconomics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_mathematics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_microeconomics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_physics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_psychology.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_statistics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_us_history.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_high_school_world_history.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_human_aging.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_human_sexuality.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_international_law.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_jurisprudence.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_logical_fallacies.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_machine_learning.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_management.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_marketing.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_medical_genetics.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_miscellaneous.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_moral_disputes.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_moral_scenarios.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_nutrition.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_philosophy.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_prehistory.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_accounting.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_law.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_medicine.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_professional_psychology.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_public_relations.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_security_studies.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_sociology.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_us_foreign_policy.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_virology.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default/mmlu_world_religions.yaml"
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
        "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": true,
        "config_source": "/root/.venv/lib/python3.12/site-packages/lm_eval/tasks/piqa/piqa.yaml"
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
      "pretrained": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
      "dtype": "bfloat16",
      "device_map": "auto",
      "trust_remote_code": true
    },
    "model_num_parameters": 134347520,
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
  "date": 1779899546.8709028,
  "pretty_env_info": "'NoneType' object has no attribute 'splitlines'",
  "transformers_version": "5.10.0.dev0",
  "lm_eval_version": "0.4.12",
  "upper_git_hash": null,
  "tokenizer_pad_token": [
    "<|pad|>",
    "0"
  ],
  "tokenizer_eos_token": [
    "<|im_end|>",
    "7"
  ],
  "tokenizer_bos_token": [
    "<|startoftext|>",
    "1"
  ],
  "eot_token_id": 7,
  "max_length": 128000,
  "task_hashes": {},
  "model_source": "hf",
  "model_name": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
  "model_name_sanitized": "__root__.openclaw__workspace__quantized__LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
  "system_instruction": null,
  "system_instruction_sha": null,
  "fewshot_as_multiturn": null,
  "chat_template": null,
  "chat_template_sha": null,
  "total_evaluation_time_seconds": "448.5044663157314"
}

### [2026-05-27 16:40:09 UTC] ASSISTANT

**Thinking**

Now I have all the data I need. Let me parse this and create the accuracy.json file.

From the results:
- mmlu: acc = 0.3199, stderr = 0.0039
- hellaswag: acc = 0.3951, acc_stderr = 0.0049
- piqa: acc = 0.6877, acc_stderr = 0.0108

The evaluation took ~448 seconds.

Let me create the accuracy.json file atomically.

**Tool call:** `write`

```json
{
  "content": "{\n  \"model_id\": \"FlameF0X/LFM2.5-1.2B-Distilled-Claude-4.6\",\n  \"model_path\": \"/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN\",\n  \"scheme\": \"W4A16\",\n  \"device\": \"cuda:0\",\n  \"num_gpus\": \"1\",\n  \"tasks\": {\n    \"piqa\": {\n      \"accuracy\": 0.6877040261153428,\n      \"accuracy_stderr\": 0.010812581599154419\n    },\n    \"mmlu\": {\n      \"accuracy\": 0.31989745050562596,\n      \"accuracy_stderr\": 0.003915491117517266\n    },\n    \"hellaswag\": {\n      \"accuracy\": 0.3951404102768373,\n      \"accuracy_stderr\": 0.004878816961012051\n    }\n  },\n  \"status\": \"success\",\n  \"duration_seconds\": 448.5044663157314,\n  \"eval_framework\": \"lm_eval+hf\",\n  \"errors\": []\n}",
  "file": "/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json.tmp"
}
```

### [2026-05-27 16:40:09 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 683 bytes to /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json.tmp

### [2026-05-27 16:40:13 UTC] ASSISTANT

**Thinking**

Now atomically rename the tmp file to the final location.

**Tool call:** `exec`

```json
{
  "command": "mv /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json.tmp /root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json"
}
```

### [2026-05-27 16:40:13 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-27 16:40:16 UTC] ASSISTANT

**Thinking**

Let me verify the final files are in place.

**Tool call:** `read`

```json
{
  "file": "/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json"
}
```

### [2026-05-27 16:40:16 UTC] TOOL RESULT
**Tool:** `read`

{
  "model_id": "FlameF0X/LFM2.5-1.2B-Distilled-Claude-4.6",
  "model_path": "/root/.openclaw/workspace/quantized/LFM2.5-1.2B-Distilled-Claude-4.6-AutoRound-W4A16-RTN",
  "scheme": "W4A16",
  "device": "cuda:0",
  "num_gpus": "1",
  "tasks": {
    "piqa": {
      "accuracy": 0.6877040261153428,
      "accuracy_stderr": 0.010812581599154419
    },
    "mmlu": {
      "accuracy": 0.31989745050562596,
      "accuracy_stderr": 0.003915491117517266
    },
    "hellaswag": {
      "accuracy": 0.3951404102768373,
      "accuracy_stderr": 0.004878816961012051
    }
  },
  "status": "success",
  "duration_seconds": 448.5044663157314,
  "eval_framework": "lm_eval+hf",
  "errors": []
}

### [2026-05-27 16:40:20 UTC] ASSISTANT

**Thinking**

All done! Let me provide a summary of what was accomplished.



<final>
Evaluation completed successfully. Here's the summary:

## Results

| Task | Accuracy | Stderr |
|------|----------|--------|
| **piqa** | 0.6877 | ±0.0108 |
| **mmlu** | 0.3199 | ±0.0039 |
| **hellaswag** | 0.3951 | ±0.0049 |

- **Model**: FlameF0X/LFM2.5-1.2B-Distilled-Claude-4.6 (W4A16 AutoRound quantized)
- **Eval framework**: lm_eval + HuggingFace HF backend
- **Duration**: ~448 seconds
- **Raw results**: `/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/lm_eval_results/`
- **Summary**: `/root/.openclaw/workspace/quantized/runs/FlameF0X_LFM2.5-1.2B-Distilled-Claude-4.6-W4A16-RTN/accuracy.json`
</final>
