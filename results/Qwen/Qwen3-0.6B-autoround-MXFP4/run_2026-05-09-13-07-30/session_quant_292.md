# Session: autoeval_quant_292

- **Session ID:** `autoeval_quant_292`
- **Timestamp:** 2026-05-09 12:43:27 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 1: Quantization

### [2026-05-09 12:43:27 UTC] USER

You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_quant/SKILL.md

Model: Qwen/Qwen3-0.6B
Quantization: MXFP4 / RTN
Export format: auto_round
Quantized Model Output directory: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4
Quantization execution log path: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log
Runtime device: cuda
Num gpus: 1

Directory responsibilities:
- Write exported model files to: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4
- Write runtime artifacts such as quant_summary.json, quantize.py, logs, prompts, copied request/session files, and the venv to: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4

CRITICAL SCRIPT REQUIREMENT:
- Before starting quantization, you MUST first generate the quantization script file:
    /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quantize.py
- The file name must be exactly: quantize.py
- The script must be a standalone Python program runnable with:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quantize.py
- The generated quantize.py must focus on the core quantization/export logic only.
- Do NOT put venv creation, pip/uv installation, proxy/bootstrap shell setup, summary generation, or unrelated orchestration into quantize.py.
- In this same OpenClaw task, prepare or reuse the Python environment separately before executing quantize.py.
- In this same OpenClaw task, first write quantize.py, then execute that generated script yourself.
- When you execute quantize.py, you MUST stream stdout/stderr into this log file while still printing output:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quantize.py 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json is a final summary artifact, not a progress marker.
- After quantize.py finishes, inspect the exported artifacts and write /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json in a separate finalize step outside quantize.py.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json until quantization has finished and the exported artifacts are ready.
- Write /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json atomically via a temporary file and rename/move it into place only at finalize time.
- If quantization fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json before exiting non-zero, also atomically.
- Do not use quantize_script.py as the final artifact name

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- This workflow is CUDA-focused. For AutoRound device selection:
    - if Num gpus == 1, prefer device="cuda"
    - if Num gpus > 1, prefer device_map="auto"
  Do NOT default to device_map="0" or device_map="0,1,2,3" unless manual mapping is truly required after auto placement fails.

IMPORTANT - In this same OpenClaw task, after separate quantization and finalize steps, you MUST produce:

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json - structured summary:
{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "MXFP4",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4",
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "original_size_mb": <float or null>,
  "quantized_size_mb": <float or null>,
  "compression_ratio": <float or null>,
  "errors": [<list of error strings>],
  "solutions": [<list of solution strings>],
  "output_files": [<list of file paths in runtime_output_dir>]
}

Write as valid JSON. If quantization fails, still write quant_summary.json with status=failed.

### [2026-05-09 12:43:36 UTC] ASSISTANT

**Thinking**

Let me start by reading the skill instructions in the specified file to understand how to proceed with the quantization task.


**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
}
```

### [2026-05-09 12:43:36 UTC] TOOL RESULT
**Tool:** `read`

---
name: auto_quant
description: Quantize Large Language Models (LLMs) using Intel's Auto-Round library with sign-gradient descent for ultra-low bit widths (2-8 bits). Supports multiple formats (AutoRound, GGUF, GPTQ, AWQ, LLM-Compressor).
metadata:
  openclaw:
    emoji: "⚡"
    homepage: https://github.com/intel/auto-round
    skillKey: auto-quant
    requires:
      bins: []
      env: []
      config: []
---

# Auto-Round Model Quantization Skill

Use this skill when the user wants to quantize Large Language Models (LLMs) using Intel's Auto-Round library. This skill provides comprehensive guidance including error handling, troubleshooting, and model-specific optimizations.

## Overview

AutoRound is an advanced quantization toolkit for LLMs that achieves high accuracy at ultra-low bit widths (2-4 bits) using **sign-gradient descent**. It supports multiple formats (AutoRound, GGUF, GPTQ, AWQ, LLM-Compressor) and inference backends.

**Key capabilities:**
- Quantization schemes: W4A16, W8A16, W2A16, W3A16, MXFP4, MXFP8, NVFP4, GGUF:Q4_K_M, etc.
- Export formats: auto_round, auto_gptq, auto_awq, llm_compressor, gguf
- Inference backends: Transformers, vLLM, SGLang, IPEX, Marlin, ExLLaMAV2

---

## Input Parameters

| Parameter | Description | Required | Default |
|-----------|-------------|----------|---------|
| `model_path` | HuggingFace model ID or local path | Yes | - |
| `output_dir` | Output directory for quantized model | Yes | - |
| `quant_type` / `scheme` | Quantization scheme | No | `W4A16` |
| `iters` | Training iterations (0=RTN) | No | `200` |
| `nsamples` | Calibration samples | No | `128` |
| `format` | Export format | No | `auto_round` |
| `device` / `device_map` | CUDA device selection for quantization | No | Single GPU: `device="cuda"`; Multi-GPU: `device_map="auto"` |

### CUDA Device Rules (CRITICAL)

This workflow is primarily for **CUDA / NVIDIA GPU** quantization.

When generating a quantization script for this repo, follow these rules:

1. **Single GPU CUDA**: use `device="cuda"` in the AutoRound API
2. **Multi-GPU CUDA**: use `device_map="auto"` in the AutoRound API
3. **Do not default to** `device_map="0"` or `device_map="0,1,2,3"` in generated scripts
4. Only use a manual explicit map or comma-separated device list when:
   - `device_map="auto"` fails
   - or you are intentionally debugging manual placement

Examples:

```python
# Single GPU (recommended default)
ar = AutoRound(..., device="cuda")

# Multi-GPU (recommended default)
ar = AutoRound(..., device_map="auto")
```

CLI equivalents:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 auto-round --model Qwen/Qwen3-0.6B --scheme W4A16 --device cuda

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 auto-round --model Qwen/Qwen3-0.6B --scheme W4A16 --device auto
```

### Quantization Schemes

| Scheme | Description | Bits | Group Size | Notes |
|--------|-------------|------|------------|-------|
| `W4A16` / `int4` | INT4 weight, FP16 activation | 4 | 128 | **Recommended** for production |
| `W8A16` | INT8 weight, FP16 activation | 8 | 128 | High accuracy |
| `W3A16` | INT3 weight, FP16 activation | 3 | 128 | Experimental |
| `W2A16` | INT2 weight, FP16 activation | 2 | 128 | Lowest bits, use `auto-round-best` |
| `MXFP4` | MXFP4 format | 4 | 32 | **Research only, no kernel** |
| `MXFP8` | MXFP8 format | 8 | 32 | **Research only, no kernel** |
| `NVFP4` | NVIDIA FP4 | 4 | 16 | Use `llm_compressor` format |
| `GGUF:Q4_K_M` | GGUF Q4 | 4 | - | For llama.cpp |

### Export Formats

| Format | Schemes Supported | Best For |
|--------|-------------------|----------|
| `auto_round` | W4A16, W2A16, W3A16, W8A16, MXFP4, MXFP8, NVFP4 | CPU, NVIDIA GPU, CUDA, HPU |
| `auto_gptq` | W4A16, W2A16, W3A16, W8A16 | CUDA (symmetric) |
| `auto_awq` | W4A16 | CUDA (asymmetric) |
| `llm_compressor` | NVFP4, MXFP4, MXFP8 | vLLM, SGLang |
| `gguf:q4_k_m` | GGUF:Q*_K, Q*_0, Q*_1 | llama.cpp, CPU |

---

## Step 1: Analyze Model from HuggingFace

**CRITICAL: Always fetch model information before quantization.**

### Fetch Model Card and Config

```bash
# README (model card) - contains usage instructions, quantization notes
curl -L https://huggingface.co/{model_id}/resolve/main/README.md -o /tmp/{model_id}_README.md

# config.json - architecture details (model_type, num_layers, hidden_size)
curl -L https://huggingface.co/{model_id}/resolve/main/config.json -o /tmp/{model_id}_config.json

# tokenizer_config.json - tokenizer type and special tokens
curl -L https://huggingface.co/{model_id}/resolve/main/tokenizer_config.json -o /tmp/{model_id}_tokenizer.json
```

Replace `{model_id}` with HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`).

### What to Look For

1. **Architecture**: Check `config.json` → `model_type`
   - Common types: `llama`, `qwen`, `mistral`, `gemma`, `falcon`, `deepseek_v2`, `mixtral`
   
2. **Quantization notes**: Search README for:
   - "quantize", "quantization", "AWQ", "GPTQ", "GGUF"
   - Special requirements or limitations
   
3. **Model size**: Estimate VRAM needed (~1.2-1.5x model size in BF16)

4. **Special requirements**:
   - Token required for gated models (Llama, etc.)
   - Trust remote code requirements
   - Special dependencies

---

## Step 2: Set Up Environment

### Step 2.0: Check for Shared Workspace (model_info.json)

**IMPORTANT: Before creating any venv, check if `auto_run` has already set up the environment for this model.**

The `auto_run` skill writes a `model_info.json` file to the shared workspace directory after environment setup. If this file exists, reuse the venv from it instead of creating a new one.

**Also check for a prebuilt system venv first:**

- If `/root/.venv/bin/python` exists, reuse `/root/.venv`
- Do **not** create a new venv if `/root/.venv` is already suitable
- Install dependencies with `uv pip`, not plain `pip install`
- If `torch` or `flash_attn` already import successfully from the reused venv, keep them; do not reinstall them unless they are missing or incompatible

The shared workspace directory is typically the `auto_run` output directory for this model:
- e.g., `/storage/lkk/inference/Qwen_Qwen3-0.6B/model_info.json`
- The task prompt may explicitly specify it as `workspace_dir`

```python
import json
from pathlib import Path

# Check if model_info.json exists in workspace_dir (passed via task prompt)
workspace_dir = "{workspace_dir}"   # e.g. /storage/lkk/inference/Qwen_Qwen3-0.6B
info_path = Path(workspace_dir) / "model_info.json"

if info_path.exists():
    model_info = json.loads(info_path.read_text())
    venv_path = model_info["venv_path"]          # e.g. /storage/.../venv
    venv_py   = f"{venv_path}/bin/python"
    venv_uv   = f"uv pip --python {venv_py}"
    print(f"✅ Reusing shared venv from auto_run: {venv_path}")
    # → Skip Steps 2.1-2.2, go directly to Step 3
elif Path("/root/.venv/bin/python").exists():
    venv_path = "/root/.venv"
    venv_py   = f"{venv_path}/bin/python"
    venv_uv   = f"uv pip --python {venv_py}"
    print(f"✅ Reusing system venv: {venv_path}")
    # → Skip Steps 2.1-2.2, go directly to Step 3
else:
    print("ℹ️  No model_info.json found, will create standalone venv in output_dir")
    venv_path = "{output_dir}/venv"
    venv_py   = f"{venv_path}/bin/python"
    venv_uv   = f"uv pip --python {venv_py}"
    # → Continue with Steps 2.1-2.2 below
```

### Create Isolated Virtual Environment

**Only run the steps below if model_info.json was NOT found above.**

```bash
# Create output directory
mkdir -p {output_dir}
mkdir -p {output_dir}/logs

# Create virtual environment
python3 -m venv --system-site-packages {output_dir}/venv

# Bootstrap uv in the venv and use uv pip for package installation
{output_dir}/venv/bin/python -m pip install -U uv
uv pip install --python {output_dir}/venv/bin/python -U pip setuptools wheel
```

### Install Auto-Round

**Option A: From local source (editable - allows source modifications)**
```bash
# Copy source if needed
cp -r /storage/lkk/auto-round {output_dir}/auto-round-src

# Install in editable mode
uv pip install --python {output_dir}/venv/bin/python -e {output_dir}/auto-round-src
```

**Option B: From GitHub**
```bash
uv pip install --python {output_dir}/venv/bin/python git+https://github.com/intel/auto-round.git
```

**Option C: From PyPI**
```bash
uv pip install --python {output_dir}/venv/bin/python auto-round
```

### Install Additional Dependencies

```bash
# Verify inherited CUDA packages first; keep them if they already work
{output_dir}/venv/bin/python -c "import torch; print('torch ok:', torch.__version__)"
{output_dir}/venv/bin/python -c "import flash_attn; print('flash_attn ok')" || true

# Install or update non-CUDA packages with uv pip
uv pip install --python {output_dir}/venv/bin/python transformers accelerate datasets

# For specific formats
uv pip install --python {output_dir}/venv/bin/python compressed-tensors  # For better compression
uv pip install --python {output_dir}/venv/bin/python llama-cpp-python   # For GGUF inference
uv pip install --python {output_dir}/venv/bin/python gptqmodel          # For GPTQ inference

# Only if torch is missing or incompatible, install a matching CUDA wheel
# uv pip install --python {output_dir}/venv/bin/python --index-url https://download.pytorch.org/whl/cu124 torch

# Only if flash_attn is required and missing, install it explicitly
# uv pip install --python {output_dir}/venv/bin/python flash-attn --no-build-isolation
```

---

## Step 3: Generate Quantization Script

### Basic Script Template

```python
#!/usr/bin/env python3
"""
Auto-Round Quantization Script
Generated by auto_quant skill

Model: {model_path}
Output: {output_dir}
Scheme: {scheme}
Iterations: {iters}
Samples: {nsamples}
Format: {format}
"""

from auto_round import AutoRound

# Configuration
model_name_or_path = "{model_path}"
output_dir = "{output_dir}"
scheme = "{scheme}"  # e.g., "W4A16", "MXFP4", "GGUF:Q4_K_M"
iters = {iters}      # 0 for RTN mode, 200 for default, 1000 for best
nsamples = {nsamples}
format_str = "{format}"  # "auto_round", "llm_compressor", "gguf:q4_k_m"
num_gpus = 1  # replace with the actual GPU count for this run

# CUDA device selection rule for this repo:
# - single GPU: device="cuda"
# - multi-GPU: device_map="auto"
autoround_device_kwargs = {"device": "cuda"} if num_gpus <= 1 else {"device_map": "auto"}

print(f"Loading model: {{model_name_or_path}}")
print(f"Scheme: {{scheme}}")
print(f"Iters: {{iters}}")
print(f"nsamples: {{nsamples}}")
print(f"Format: {{format_str}}")
print(f"Device args: {{autoround_device_kwargs}}")

# Create AutoRound instance
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    # Optional optimizations
    # enable_torch_compile=True,  # Faster quantization (PyTorch 2.6+)
    # low_gpu_mem_usage=True,    # Lower VRAM, ~30% slower
    # disable_opt_rtn=True,      # For GGUF: use pure RTN
    **autoround_device_kwargs,
)

# Quantize and save
print("Starting quantization...")
ar.quantize_and_save(output_dir=output_dir, format=format_str)

print(f"Quantization complete! Output: {{output_dir}}")
```

### Recipe Recommendations

| Recipe | iters | nsamples | seqlen | Accuracy | Speed |
|--------|-------|----------|--------|----------|-------|
| `default` | 200 | 128 | 2048 | Good | Baseline |
| `best` | 1000 | 512 | 2048 | **Best** | 4-5x slower |
| `light` | 50 | 128 | 2048 | Slight drop | 2-3x faster |

**Recommendation:**
- **W4A16**: Use default recipe (`iters=200`)
- **W2A16**: Use best recipe (`iters=1000`, `enable_alg_ext=True`)
- **GGUF**: Use RTN (`iters=0`)

---

## Step 4: Execute and Handle Errors (CRITICAL!)

When quantization fails, you MUST diagnose and fix. **Do NOT simply report errors without attempting solutions.**

### Error Handling Workflow

```
ERROR → Analyze → Search → Try Solutions → Verify → Document
```

### Common Errors and Solutions

#### 1. ImportError / ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'auto_round'
ImportError: cannot import name 'AutoRound' from 'auto_round'
```

**Solutions:**
```bash
# Reinstall auto-round
uv pip install --python {venv}/bin/python --upgrade auto-round

# Or from source
uv pip install --python {venv}/bin/python -e /path/to/auto-round --force-reinstall

# Check installation
{venv}/bin/pip show auto-round
```

#### 2. CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.OutOfMemoryError: CUDA out of memory: tried to allocate X GiB
```

**Solutions (try in order):**
```python
# Solution A: Reduce memory usage - add to AutoRound initialization
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    enable_torch_compile=True,    # PyTorch 2.6+ recommended
    low_gpu_mem_usage=True,       # Offload to CPU, ~20% more time
    device="cuda",                # Keep single-GPU CUDA explicit
)

# Solution B: Reduce batch size
    batch_size=1,
    gradient_accumulate_steps=8,

# Solution C: Reduce seqlen (may affect accuracy)
    seqlen=512,

# Solution D: Use RTN mode (fastest, no calibration)
    iters=0,
    disable_opt_rtn=True,  # For GGUF format

# Solution E: Use multiple GPUs
    device_map="auto",           # Recommended multi-GPU default
```

**CLI alternatives:**
```bash
# Use light recipe
auto-round-light --model ... --scheme W4A16

# Low memory mode
auto-round --model ... --scheme W4A16 --low_gpu_mem_usage

# Multi-GPU CUDA
CUDA_VISIBLE_DEVICES=0,1,2,3 auto-round --model ... --scheme W4A16 --device auto
```

#### 3. Version Conflicts

**Symptoms:**
```
ImportError: cannot import name 'xxx' from 'transformers'
AttributeError: module 'torch' has no attribute 'xxx'
VersionConflict: transformers x.x.x is incompatible with...
```

**Solutions:**
```bash
# Check current versions
{venv}/bin/pip show torch transformers accelerate

# Upgrade/downgrade transformers
uv pip install --python {venv}/bin/python "transformers>=4.35.0"
uv pip install --python {venv}/bin/python "transformers==4.40.0"

# Upgrade torch only when it is actually missing or incompatible
uv pip install --python {venv}/bin/python "torch>=2.5.0"
uv pip install --python {venv}/bin/python --index-url https://download.pytorch.org/whl/cu124 torch

# Install flash-attn only if required by the model/runtime and currently missing
uv pip install --python {venv}/bin/python flash-attn --no-build-isolation

# Reinstall auto-round dependencies
uv pip install --python {venv}/bin/python -r /path/to/auto-round/requirements.txt
```

#### 4. Model Loading Errors

**Symptoms:**
```
OSError: Can't load tokenizer for ...
FileNotFoundError: tokenizer_config.json not found
ValueError: xxx requires a HuggingFace token
```

**Solutions:**
```bash
# For gated models (Llama, etc.), set token
import os
os.environ["HF_TOKEN"] = "your_token_here"

# Or use CLI
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --token $HF_TOKEN

# Download model first
git lfs clone https://huggingface.co/{model_id} /local/path

# Use trust_remote_code for custom models
ar = AutoRound(
    model_name_or_path,
    trust_remote_code=True,
)
```

#### 5. Quantization Scheme Errors

**Symptoms:**
```
ValueError: Unsupported quantization scheme 'xxx'
KeyError: scheme 'xxx' not found
```

**Solutions:**
```bash
# Check supported schemes
auto-round list scheme

# Use correct scheme name (case-sensitive)
scheme = "W4A16"   # Correct
scheme = "w4a16"   # May not work

# For GGUF format
scheme = "GGUF:Q4_K_M"  # Correct format
```

#### 6. Export Format Errors

**Symptoms:**
```
ValueError: Export format 'xxx' not supported
RuntimeError: Failed to export to gguf format
```

**Solutions:**
```python
# Try different format combinations
format = "auto_round"                    # Most compatible
format = "llm_compressor"                # For NVFP4/MXFP4
format = "gguf:q4_k_m"                   # For GGUF
format = "auto_gptq,auto_awq,auto_round" # Multiple formats

# For GGUF, use iters=0 (RTN)
ar = AutoRound(
    model_name_or_path,
    scheme="W4A16",
    iters=0,  # RTN mode
)
```

#### 7. GPU Not Found / CUDA Errors

**Symptoms:**
```
RuntimeError: CUDA not available
AssertionError: CUDA device not found
```

**Solutions:**
```bash
# Check CUDA availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python script.py
CUDA_VISIBLE_DEVICES=0,1 python script.py

# Use CPU instead
device_map = "cpu"
```

#### 8. Calibration Dataset Errors

**Symptoms:**
```
RuntimeError: Error loading dataset 'xxx'
DatasetNotFoundError: Couldn't find dataset 'xxx'
```

**Solutions:**
```python
# Use default dataset
dataset = "NeelNanda/pile-10k"

# Use alternative dataset
dataset = "swift/pile-val-backup"  # For China region
dataset = "BAAI/CCI3-HQ"           # Chinese
dataset = "mbpp"                   # Code

# Use local dataset
dataset = "/path/to/local_dataset.json"

# Specify dataset split
dataset = "NeelNanda/pile-10k:train"
dataset = "NeelNanda/pile-10k:train+validation"
```

---

## Step 5: Advanced Troubleshooting

### When Standard Solutions Don't Work

#### A. Web Search Strategy

Search for the exact error message:
```
# Search patterns
"auto-round" "CUDA out of memory"
"auto-round" "ImportError" transformers
"intel auto-round" github issues
"auto-round" "ValueError" scheme
```

#### B. Check GitHub Issues

```bash
# Search auto-round issues
curl -s "https://api.github.com/search/issues?q=repo:intel/auto-round+out+of+memory" | jq '.items[:5] | .[] | {title, url}'

# Check recent issues
curl -s "https://api.github.com/repos/intel/auto-round/issues?state=open" | jq '.[:10] | .[] | {title, number}'
```

#### C. Source Code Investigation

If error is in auto-round itself:
```bash
# Look at auto-round source
ls /path/to/auto-round/auto_round/

# Check specific module
cat /path/to/auto-round/auto_round/autoround.py | head -100

# Search for error source
grep -r "error_message" /path/to/auto-round/auto_round/
```

#### D. Try Different Approaches

```python
# Approach 1: Different scheme
scheme = "W4A16"  # Instead of MXFP4

# Approach 2: Different format
format = "auto_round"  # Instead of gguf

# Approach 3: Different recipe
# default → light → best

# Approach 4: Use CLI instead of API
import subprocess
subprocess.run([
    "auto-round",
    "--model", model_path,
    "--scheme", "W4A16",
    "--format", "auto_round",
    "--output_dir", output_dir,
])
```

---

## Step 6: Verify and Save

After successful quantization:

1. **Verify output files:**
```bash
ls -la {output_dir}/
ls -la {output_dir}/quantized_model/  # or output_dir/
```

2. **Save script:**
```python
# Save the quantization script to output directory
with open(f"{output_dir}/quantize_script.py", "w") as f:
    f.write(script_content)
```

3. **Document solutions (if errors occurred):**
```markdown
# {output_dir}/solutions.md

## Error 1: [Error Description]
- **Cause**: [Root cause]
- **Solution**: [What worked]
- **Command**: [Command used]

## Error 2: ...
```

---

## Step 6.5: Generate Summary (RECOMMENDED)

After quantization completes (success or failure), generate a `summary.md` to document the entire process. This helps with debugging, reproducibility, and tracking issues.

### Summary Template

```python
#!/usr/bin/env python3
"""
Generate quantization summary
Run this after quantization completes (success or failure)
"""

import json
import os
from datetime import datetime
from pathlib import Path

def generate_summary(
    output_dir: str,
    model_path: str,
    scheme: str,
    iters: int,
    nsamples: int,
    format_str: str,
    start_time: float,
    errors: list = None,
    solutions: list = None,
    notes: str = None
):
    """Generate a comprehensive summary markdown file."""
    
    import time
    end_time = time.time()
    duration = end_time - start_time
    
    # Collect output files
    output_path = Path(output_dir)
    files_info = []
    if output_path.exists():
        for f in sorted(output_path.rglob("*")):
            if f.is_file() and not f.name.endswith(('.pyc', '.pyo', '__pycache__')):
                size = f.stat().st_size
                size_str = f"{size/1024/1024:.2f} MB" if size > 1024*1024 else f"{size/1024:.2f} KB"
                files_info.append(f"  - {f.relative_to(output_path)} ({size_str})")
    
    # Build summary markdown
    summary = f"""# Quantization Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Model Information

| Field | Value |
|-------|-------|
| Model Path | `{model_path}` |
| Scheme | `{scheme}` |
| Iterations | `{iters}` |
| Calibration Samples | `{nsamples}` |
| Export Format | `{format_str}` |

## Timing

| Phase | Duration |
|-------|----------|
| Total | {duration:.2f} seconds ({duration/60:.2f} minutes) |

## Output Files

```
{chr(10).join(files_info) if files_info else "  (no files found)"}
```

## Errors Encountered

{chr(10).join(f"- {err}" for err in (errors or ["(none)"]))}

## Solutions Applied

{chr(10).join(f"- {sol}" for sol in (solutions or ["(none)"]))}

## Additional Notes

{notes or "(none)"}

## Environment

```bash
# Python version
python3 --version

# Key packages
python -m pip show torch transformers auto-round
```

## Reproduce Command

```bash
# Recreate this quantization
auto-round --model {model_path} --scheme "{scheme}" --format {format_str} --output_dir {output_dir} --iters {iters} --nsamples {nsamples}
```
"""
    
    # Write summary
    summary_path = Path(output_dir) / "summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)
    
    print(f"Summary written to: {summary_path}")
    return summary_path

# Usage example:
if __name__ == "__main__":
    import time
    start_time = time.time()  # Set this at the beginning of quantization
    
    # Your quantization code here...
    
    # Generate summary at the end
    generate_summary(
        output_dir="/storage/quantized/llama-8b-w4a16",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        scheme="W4A16",
        iters=200,
        nsamples=128,
        format_str="auto_round",
        start_time=start_time,
        errors=["CUDA OOM - tried to allocate 12GB", "Fixed by enable_torch_compile=True"],
        solutions=["Added low_gpu_mem_usage=True", "Reduced batch_size to 1"],
        notes="Model quantized successfully with minor memory optimizations"
    )
```

### Integration with Quantization Script

Add summary generation to your quantization script:

```python
#!/usr/bin/env python3
import time
import json
from pathlib import Path

# Track start time
start_time = time.time()

# Track errors and solutions
errors = []
solutions = []

try:
    # Your quantization code here
    ar = AutoRound(...)
    ar.quantize_and_save(...)
    
except Exception as e:
    errors.append(str(e))
    
    # Try to recover
    try:
        # Attempted solution 1
        solutions.append("Attempted solution description")
    except:
        pass
    
    # Try more solutions...
    finally:
        # Always generate summary even if quantization failed
        generate_summary(
            output_dir=output_dir,
            model_path=model_path,
            scheme=scheme,
            iters=iters,
            nsamples=nsamples,
            format_str=format_str,
            start_time=start_time,
            errors=errors,
            solutions=solutions,
            notes="Quantization failed, see errors above"
        )
        raise

# Success path - generate summary
generate_summary(
    output_dir=output_dir,
    model_path=model_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    format_str=format_str,
    start_time=start_time,
    notes="Quantization completed successfully"
)
```

### Summary Output Example

The generated `summary.md` will look like:

```markdown
# Quantization Summary

Generated: 2026-03-20 00:51 UTC

## Model Information

| Field | Value |
|-------|-------|
| Model Path | `meta-llama/Llama-3.1-8B-Instruct` |
| Scheme | `W4A16` |
| Iterations | `200` |
| Calibration Samples | `128` |
| Export Format | `auto_round` |

## Timing

| Phase | Duration |
|-------|----------|
| Total | 845.32 seconds (14.09 minutes) |

## Output Files

```
- quantized_model/adapter_config.json (1.23 KB)
- quantized_model/adapter_model.safetensors (3.87 GB)
- quantize_script.py (2.45 KB)
- summary.md (1.89 KB)
```

## Errors Encountered

- (none)

## Solutions Applied

- (none)

## Additional Notes

- Model quantized successfully with default settings

## Environment

```bash
# Python version
Python 3.10.12

# Key packages
torch: 2.5.0
transformers: 4.40.0
auto-round: 0.2.1
```

## Reproduce Command

```bash
auto-round --model meta-llama/Llama-3.1-8B-Instruct --scheme "W4A16" --format auto_round --output_dir /storage/quantized/llama-8b-w4a16 --iters 200 --nsamples 128
```
```

---

## Complete Example Workflow

### User Request
> Quantize meta-llama/Llama-3.1-8B-Instruct to W4A16 format, output to /storage/quantized/llama-8b-w4a16

### Agent Actions

**1. Query HuggingFace:**
```bash
curl -L https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json
curl -L https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/README.md
```

**2. Set up environment:**
```bash
mkdir -p /storage/quantized/llama-8b-w4a16/logs
if [ -x /root/.venv/bin/python ]; then
  VENV_PY=/root/.venv/bin/python
else
  python3 -m venv --system-site-packages /storage/quantized/llama-8b-w4a16/venv
  VENV_PY=/storage/quantized/llama-8b-w4a16/venv/bin/python
fi
$VENV_PY -m pip install -U uv
uv pip install --python "$VENV_PY" -e /storage/lkk/auto-round
```

**3. Create quantization script:**
```python
# /storage/quantized/llama-8b-w4a16/quantize_script.py
from auto_round import AutoRound

ar = AutoRound(
    "meta-llama/Llama-3.1-8B-Instruct",
    scheme="W4A16",
    iters=200,
    nsamples=128,
    device="cuda",
    enable_torch_compile=True,
)
ar.quantize_and_save(output_dir="/storage/quantized/llama-8b-w4a16", format="auto_round")
```

**Multi-GPU CUDA example:**
```python
from auto_round import AutoRound

ar = AutoRound(
    "meta-llama/Llama-3.1-8B-Instruct",
    scheme="W4A16",
    iters=200,
    nsamples=128,
    device_map="auto",
    low_gpu_mem_usage=True,
    enable_torch_compile=True,
)
ar.quantize_and_save(output_dir="/storage/quantized/llama-8b-w4a16", format="auto_round")
```

**4. Execute:**
```bash
cd /storage/quantified/llama-8b-w4a16
./venv/bin/python quantize_script.py >> logs/quantize.log 2>&1
```

**5. Handle any errors** using the error handling workflow above

**6. Verify:**
```bash
ls /storage/quantized/llama-8b-w4a16/
```

---

## Reference: Auto-Round CLI

```bash
# Basic usage
auto-round --model Qwen/Qwen3-0.6B --scheme "W4A16" --format auto_round

# With options
auto-round \
    --model Qwen/Qwen3-0.6B \
    --scheme "W4A16" \
    --format "auto_gptq,auto_awq,auto_round" \
    --output_dir ./output \
    --iters 200 \
    --nsamples 128

# List available options
auto-round --help
auto-round list scheme
auto-round list format
```

---

## Quick Reference Card

| Need | Solution |
|------|----------|
| Fast quantization | `iters=0` (RTN mode) |
| Best accuracy | `iters=1000`, `enable_alg_ext=True` |
| Low VRAM | `low_gpu_mem_usage=True`, `batch_size=1` |
| GGUF format | `format="gguf:q4_k_m"`, `iters=0` |
| Single GPU CUDA | `device="cuda"` |
| Multiple GPUs | `CUDA_VISIBLE_DEVICES=0,1,2,3` + `device_map="auto"` |
| 2-bit quantization | Use `W2A16` + best recipe |
| MXFP4/MXFP8 | Research only, no kernel |

---

## Notes

- **VRAM Requirements**: ~10-20GB for 8B models, ~20-40GB for 70B models
- **Time**: ~10-20 min for 8B model with default settings
- **MXFP4/MXFP8**: Research features, no real inference kernels
- **Gated models**: May require HuggingFace token
- **Editable install**: Install auto-round in editable mode to modify source if needed


### [2026-05-09 12:43:52 UTC] ASSISTANT

**Thinking**

Let me analyze the task:

1. Model: Qwen/Qwen3-0.6B
2. Quantization: MXFP4 / RTN
3. Export format: auto_round
4. Output directory: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4
5. Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4
6. Log path: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log
7. Runtime device: cuda
8. Num gpus: 1

Key requirements:
- Single GPU CUDA → use device="cuda" (not device_map)
- For RTN mode → iters=0
- Scheme: MXFP4
- Format: auto_round

Let me start by:
1. Creating the directories
2. Writing the quantize.py script
3. Setting up the environment
4. Executing the quantization
5. Writing the quant_summary.json

First, let me check the model info to understand the model architecture and then create the script.


**Tool call:** `exec`

```json
{
  "command": "mkdir -p /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs && mkdir -p /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4"
}
```

**Tool call:** `exec`

```json
{
  "command": "curl -sL https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json | head -50"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())\" 2>&1"
}
```

### [2026-05-09 12:43:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-09 12:43:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}

### [2026-05-09 12:43:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-09 12:43:59 UTC] ASSISTANT

**Thinking**

System Python doesn't have torch. I need to use a venv with torch pre-installed. Let me check if /root/.venv exists.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.venv/bin/python 2>/dev/null || echo \"no .venv\""
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -m venv --system-site-packages /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv 2>&1 | tail -5"
}
```

### [2026-05-09 12:43:59 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lrwxrwxrwx 1 root root 73 Apr 24 05:53 /root/.venv/bin/python -> /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12

### [2026-05-09 12:44:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-09 12:44:07 UTC] ASSISTANT

**Thinking**

Good, /root/.venv exists. Let me reuse it and check what's available, then install auto-round.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -m pip install -U uv 2>&1 | tail -3"
}
```

### [2026-05-09 12:44:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-09 12:44:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.venv/bin/python: No module named pip

### [2026-05-09 12:44:12 UTC] ASSISTANT

**Thinking**

The system Python has torch+cuda but no .venv does. Let me check the system Python.


**Tool call:** `exec`

```json
{
  "command": "which python3 && python3 --version && python3 -c \"import torch; print('torch:', torch.__version__)\" 2>&1"
}
```

### [2026-05-09 12:44:12 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

/root/.venv/bin/python3
Python 3.12.13
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-09 12:44:18 UTC] ASSISTANT

**Thinking**

Even system python doesn't have torch. Let me check if there's any python with torch installed.


**Tool call:** `exec`

```json
{
  "command": "find /usr /opt /root -name \"python*\" -type f 2>/dev/null | head -20 && echo \"---\" && ls /opt/ 2>/dev/null && echo \"---\" && ls /usr/local/lib/ 2>/dev/null | head -20"
}
```

### [2026-05-09 12:44:18 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/usr/local/lib/node_modules/openclaw/dist/python-Cubwb1vn.js
/usr/local/lib/node_modules/openclaw/dist/python-PabNzKr5.js
/usr/local/lib/node_modules/openclaw/node_modules/highlight.js/lib/languages/python-repl.js
/usr/local/lib/node_modules/openclaw/node_modules/highlight.js/lib/languages/python.js
/usr/local/lib/node_modules/openclaw/node_modules/playwright-core/lib/server/codegen/python.js
/root/.venv/lib/python3.12/site-packages/textual/tree-sitter/highlights/python.scm
/root/.venv/lib/python3.12/site-packages/pygments/lexers/python.py
/root/.venv/lib/python3.12/site-packages/google/protobuf/internal/python_edition_defaults.py
/root/.venv/lib/python3.12/site-packages/google/protobuf/internal/python_message.py
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12-config
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12/cpython/pythonrun.h
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12/pythonrun.h
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/pkgconfig/python-3.12-embed.pc
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/pkgconfig/python-3.12.pc
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/config-3.12-x86_64-linux-gnu/python-config.py
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/config-3.12-x86_64-linux-gnu/python.o
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/idlelib/Icons/python.gif
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/site-packages/pip/_vendor/pygments/lexers/python.py
/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/share/man/man1/python3.12.1
---
---
node_modules

### [2026-05-09 12:44:26 UTC] ASSISTANT

**Thinking**

The system Python has no torch. I need to create a new venv and install torch from scratch. Let me create a fresh venv and install torch + auto-round.


**Tool call:** `exec`

```json
{
  "command": "python3 -m venv --system-site-packages /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv 2>&1"
}
```

### [2026-05-09 12:44:27 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-09 12:44:32 UTC] ASSISTANT

**Thinking**

Let me check if the venv was created and then bootstrap uv and install torch.


**Tool call:** `exec`

```json
{
  "command": "ls /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/ | head -20"
}
```

### [2026-05-09 12:44:32 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Activate.ps1
activate
activate.csh
activate.fish
pip
pip3
pip3.12
python
python3
python3.12

### [2026-05-09 12:44:39 UTC] ASSISTANT

**Thinking**

The venv was created but with --system-site-packages, pip should be available. Let me check if it can install packages. I need to install torch first.


**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python -m pip install -U uv 2>&1 | tail -5"
}
```

### [2026-05-09 12:44:40 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Installing collected packages: uv
Successfully installed uv-0.11.12

[notice] A new release of pip is available: 25.0.1 -> 26.1.1
[notice] To update, run: python -m pip install --upgrade pip

### [2026-05-09 12:44:48 UTC] ASSISTANT

**Thinking**

Now let me install torch with CUDA support. Let me check what CUDA version is available first.


**Tool call:** `exec`

```json
{
  "command": "nvidia-smi --query-gpu=driver_version,CUDA_version --format=csv,noheader 2>&1 | head -3"
}
```

**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -10"
}
```

### [2026-05-09 12:44:48 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Field "CUDA_version" is not a valid field to query.

### [2026-05-09 12:44:58 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session young-bloom, pid 569). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-09 12:45:10 UTC] ASSISTANT

**Thinking**

Let me check if torch got installed and verify cuda availability while that runs.


**Tool call:** `exec`

```json
{
  "command": "nvidia-smi 2>&1 | head -20"
}
```

### [2026-05-09 12:45:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Sat May  9 12:45:10 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:03:00.0 Off |                  Off |
|  0%   26C    P8             16W /  450W |       2MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

### [2026-05-09 12:45:18 UTC] ASSISTANT

**Thinking**

Good, we have RTX 4090 with CUDA 12.7. Let me wait for torch install.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "young-bloom",
  "timeout": 120000
}
```

### [2026-05-09 12:45:18 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

+ nvidia-cusparse-cu12==12.3.1.170
 + nvidia-cusparselt-cu12==0.6.2
 + nvidia-nccl-cu12==2.21.5
 + nvidia-nvjitlink-cu12==12.4.127
 + nvidia-nvtx-cu12==12.4.127
 + setuptools==70.2.0
 + sympy==1.13.1
 + torch==2.6.0+cu124
 + triton==3.2.0
 + typing-extensions==4.15.0

Process exited with code 0.

### [2026-05-09 12:45:25 UTC] ASSISTANT

**Thinking**

Torch is installed. Now let me install auto-round and other dependencies.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python transformers accelerate datasets 2>&1 | tail -5"
}
```

**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python auto-round 2>&1 | tail -10"
}
```

### [2026-05-09 12:45:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

+ transformers==5.8.0
 + typer==0.25.1
 + urllib3==2.7.0
 + xxhash==3.7.0
 + yarl==1.23.0

### [2026-05-09 12:45:29 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Downloading pydantic-core (2.0MiB)
 Downloaded pydantic-core
Prepared 6 packages in 171ms
Installed 6 packages in 44ms
 + annotated-types==0.7.0
 + auto-round==0.12.3
 + py-cpuinfo==9.0.0
 + pydantic==2.13.4
 + pydantic-core==2.46.4
 + typing-inspection==0.4.2

### [2026-05-09 12:45:43 UTC] ASSISTANT

**Thinking**

Good. Now let me verify torch can see CUDA and check auto-round version:


**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python -c \"import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python -c \"import auto_round; print('auto_round:', auto_round.__version__)\" 2>&1"
}
```

### [2026-05-09 12:45:45 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch: 2.6.0+cu124
cuda: True
gpu: NVIDIA GeForce RTX 4090

### [2026-05-09 12:45:51 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

auto_round: 0.12.3

### [2026-05-09 12:46:11 UTC] ASSISTANT

**Thinking**

Everything looks good. Now let me write the quantization script.




Now let me write the quantization script:


**Tool call:** `write`

```json
{
  "content": "#!/usr/bin/env python3\n\"\"\"\nAuto-Round Quantization Script\nModel: Qwen/Qwen3-0.6B\nScheme: MXFP4 (RTN mode)\nFormat: auto_round\nDevice: cuda (single GPU)\n\"\"\"\n\nfrom auto_round import AutoRound\n\n# Configuration\nmodel_name_or_path = \"Qwen/Qwen3-0.6B\"\noutput_dir = \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4\"\nscheme = \"MXFP4\"\niters = 0          # 0 = RTN (no training)\nnsamples = 128\nformat_str = \"auto_round\"\n\nprint(f\"Loading model: {model_name_or_path}\")\nprint(f\"Scheme: {scheme}\")\nprint(f\"Iters: {iters} (RTN mode)\")\nprint(f\"nsamples: {nsamples}\")\nprint(f\"Format: {format_str}\")\nprint(f\"Device: cuda\")\n\n# Create AutoRound instance\nar = AutoRound(\n    model_name_or_path,\n    scheme=scheme,\n    iters=iters,\n    nsamples=nsamples,\n    device=\"cuda\",\n)\n\n# Quantize and save\nprint(\"Starting quantization...\")\nar.quantize_and_save(output_dir=output_dir, format=format_str)\n\nprint(f\"Quantization complete! Output: {output_dir}\")",
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quantize.py"
}
```

### [2026-05-09 12:46:11 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 935 bytes to /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quantize.py

### [2026-05-09 12:46:19 UTC] ASSISTANT

**Thinking**

Now let me execute the quantize.py script, streaming stdout/stderr to the log file.


**Tool call:** `exec`

```json
{
  "command": "cd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4 && /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quantize.py 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log",
  "timeout": 1800,
  "yieldMs": 1200000
}
```

### [2026-05-09 12:46:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Loading model: Qwen/Qwen3-0.6B
Scheme: MXFP4
Iters: 0 (RTN mode)
nsamples: 128
Format: auto_round
Device: cuda
404 Client Error. (Request ID: Root=1-69ff2c9f-416c60242c399b162ffe7340;3e97070b-7e65-428b-93a1-3447cb8e9563)

Entry Not Found for url: https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model_index.json.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[33;1m2026-05-09 12:46:35 WARNING base.py L407: `device` is deprecated, please use `device_map` instead[0m
[38;20m2026-05-09 12:46:35 INFO base.py L473: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-05-09 12:46:35 INFO base.py L517: using torch.bfloat16 for quantization tuning[0m
Starting quantization...
[33;1m2026-05-09 12:46:35 WARNING formats.py L166: some layers are skipped quantization (shape not divisible by 32): [0m
[33;1m2026-05-09 12:46:35 WARNING modeling_utils.py L4555: `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.[0m
[33;1m2026-05-09 12:46:35 WARNING base.py L1748: immediate_saving is only supported for int quantization, set to False[0m
[38;20m2026-05-09 12:46:35 INFO base.py L1401: RTN mode detected (iters=0): force blockwise quantization to avoid layer-wise full-model materialization.[0m

  0%|          | 0/28 [00:00<?, ?it/s]
Quantizing model.layers.0:   0%|          | 0/28 [00:00<?, ?it/s][38;20m2026-05-09 12:46:36 INFO offload.py L703: OffloadManager (compressor): tempdir = /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/ar_work_space/offload/compressor_gf1fb5_j[0m
[38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.1:   4%|▎         | 1/28 [00:00<00:14,  1.85it/s][38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.2:   7%|▋         | 2/28 [00:00<00:09,  2.80it/s][38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.3:  11%|█         | 3/28 [00:00<00:07,  3.31it/s][38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.4:  14%|█▍        | 4/28 [00:01<00:06,  3.59it/s][38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.5:  18%|█▊        | 5/28 [00:01<00:05,  3.85it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.6:  21%|██▏       | 6/28 [00:01<00:05,  4.06it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.7:  25%|██▌       | 7/28 [00:01<00:04,  4.22it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.8:  29%|██▊       | 8/28 [00:01<00:04,  4.28it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.9:  32%|███▏      | 9/28 [00:02<00:04,  4.33it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.10:  36%|███▌      | 10/28 [00:02<00:04,  4.44it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.11:  39%|███▉      | 11/28 [00:02<00:03,  4.51it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.12:  43%|████▎     | 12/28 [00:02<00:03,  4.61it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.13:  46%|████▋     | 13/28 [00:02<00:03,  4.70it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.14:  50%|█████     | 14/28 [00:02<00:02,  4.76it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.15:  54%|█████▎    | 15/28 [00:03<00:02,  4.81it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.16:  57%|█████▋    | 16/28 [00:03<00:02,  4.86it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.17:  61%|██████    | 17/28 [00:03<00:02,  4.92it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.18:  64%|██████▍   | 18/28 [00:03<00:02,  4.93it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.19:  68%|██████▊   | 19/28 [00:03<00:01,  4.94it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.20:  71%|███████▏  | 20/28 [00:04<00:01,  4.97it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.21:  75%|███████▌  | 21/28 [00:04<00:01,  5.00it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.22:  79%|███████▊  | 22/28 [00:04<00:01,  5.04it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.23:  82%|████████▏ | 23/28 [00:04<00:00,  5.06it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.24:  86%|████████▌ | 24/28 [00:04<00:00,  5.08it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.25:  89%|████████▉ | 25/28 [00:04<00:00,  5.11it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.26:  93%|█████████▎| 26/28 [00:05<00:00,  5.14it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.27:  96%|█████████▋| 27/28 [00:05<00:00,  5.14it/s][38;20m2026-05-09 12:46:41 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.27: 100%|██████████| 28/28 [00:05<00:00,  4.99it/s]

packing:   0%|          | 0/196 [00:00<?, ?it/s]
packing: 100%|██████████| 196/196 [00:00<00:00, 327523.34it/s]
[33;1m2026-05-09 12:46:41 WARNING export_to_nvfp_mxfp.py L249: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4 already exists, this may cause model conflict[0m
[38;20m2026-05-09 12:46:42 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4

### [2026-05-09 12:46:53 UTC] ASSISTANT

**Thinking**

Quantization completed successfully. Now let me verify the output files and calculate the model sizes to write the quant_summary.json.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/ 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "du -sh /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/ 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"\nimport torch\nfrom transformers import AutoModel\nmodel = AutoModel.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)\nparam_size = sum(p.numel() * p.element_size() for p in model.parameters())\nprint(f'Original model size: {param_size / 1024 / 1024:.2f} MB')\n\" 2>&1 | head -5"
}
```

### [2026-05-09 12:46:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 543724
drwxr-xr-x 2 root root      4096 May  9 12:46 .
drwxr-xr-x 4 root root        59 May  9 12:43 ..
-rw-r--r-- 1 root root      4168 May  9 12:46 chat_template.jinja
-rw-r--r-- 1 root root      1784 May  9 12:46 config.json
-rw-r--r-- 1 root root       213 May  9 12:46 generation_config.json
-rw-r--r-- 1 root root 545318232 May  9 12:46 model.safetensors
-rw-r--r-- 1 root root       313 May  9 12:46 quantization_config.json
-rw-r--r-- 1 root root  11422650 May  9 12:46 tokenizer.json
-rw-r--r-- 1 root root       694 May  9 12:46 tokenizer_config.json

### [2026-05-09 12:46:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

531M	/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/

### [2026-05-09 12:46:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Traceback (most recent call last):
  File "<string>", line 2, in <module>
ModuleNotFoundError: No module named 'torch'

### [2026-05-09 12:47:01 UTC] ASSISTANT

**Thinking**

The quantized model is 531MB. Let me get the original model size using the venv.


**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/bin/python -c \"\nimport torch\nfrom transformers import AutoModel\nmodel = AutoModel.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)\nparam_size = sum(p.numel() * p.element_size() for p in model.parameters())\nprint(f'Original model size: {param_size / 1024 / 1024:.2f} MB')\n\" 2>&1 | tail -3"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/ 2>&1"
}
```

### [2026-05-09 12:47:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Original model size: 1136.88 MB

### [2026-05-09 12:47:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 8
drwxr-xr-x 5 root root  114 May  9 12:46 .
drwxr-xr-x 3 root root   43 May  9 12:43 ..
drwxr-xr-x 2 root root    6 May  9 12:46 ar_work_space
drwxr-xr-x 2 root root   84 May  9 12:46 logs
-rw-r--r-- 1 root root  935 May  9 12:46 quantize.py
-rw-r--r-- 1 root root 1059 May  9 12:43 request.json
drwxr-xr-x 6 root root  132 May  9 12:45 venv

### [2026-05-09 12:47:17 UTC] ASSISTANT

**Thinking**

Good. The quantization completed successfully. The original model was ~1136.88 MB (1.1 GB), and the quantized model is 531M (0.5 GB). That's about 2.1x compression ratio.

Now I need to:
1. Check the exported model files in the output dir
2. Write the quant_summary.json

Let me list what files are in the quantized model output directory and the runtime directory.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/ 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log 2>&1 | tail -30"
}
```

**Tool call:** `exec`

```json
{
  "command": "du -sh /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/ 2>&1"
}
```

### [2026-05-09 12:47:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 52
drwxr-xr-x 2 root root    84 May  9 12:46 .
drwxr-xr-x 5 root root   114 May  9 12:46 ..
-rw-r--r-- 1 root root 33864 May  9 12:47 auto.log
-rw-r--r-- 1 root root  7254 May  9 12:46 quant_exec.log
-rw-r--r-- 1 root root  4873 May  9 12:43 quant_prompt.txt

### [2026-05-09 12:47:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Quantizing model.layers.3:  11%|█         | 3/28 [00:00<00:07,  3.31it/s][38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.4:  14%|█▍        | 4/28 [00:01<00:06,  3.59it/s][38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.5:  18%|█▊        | 5/28 [00:01<00:05,  3.85it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.6:  21%|██▏       | 6/28 [00:01<00:05,  4.06it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.7:  25%|██▌       | 7/28 [00:01<00:04,  4.22it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.8:  29%|██▊       | 8/28 [00:01<00:04,  4.28it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.9:  32%|███▏      | 9/28 [00:02<00:04,  4.33it/s][38;20m2026-05-09 12:46:37 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.10:  36%|███▌      | 10/28 [00:02<00:04,  4.44it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.11:  39%|███▉      | 11/28 [00:02<00:03,  4.51it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.12:  43%|████▎     | 12/28 [00:02<00:03,  4.61it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.13:  46%|████▋     | 13/28 [00:02<00:03,  4.70it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.14:  50%|█████     | 14/28 [00:02<00:02,  4.76it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.15:  54%|█████▎    | 15/28 [00:03<00:02,  4.81it/s][38;20m2026-05-09 12:46:38 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.16:  57%|█████▋    | 16/28 [00:03<00:02,  4.86it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.17:  61%|██████    | 17/28 [00:03<00:02,  4.92it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.18:  64%|██████▍   | 18/28 [00:03<00:02,  4.93it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.19:  68%|██████▊   | 19/28 [00:03<00:01,  4.94it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.20:  71%|███████▏  | 20/28 [00:04<00:01,  4.97it/s][38;20m2026-05-09 12:46:39 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.21:  75%|███████▌  | 21/28 [00:04<00:01,  5.00it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.22:  79%|███████▊  | 22/28 [00:04<00:01,  5.04it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.23:  82%|████████▏ | 23/28 [00:04<00:00,  5.06it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.24:  86%|████████▌ | 24/28 [00:04<00:00,  5.08it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.25:  89%|████████▉ | 25/28 [00:04<00:00,  5.11it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.26:  93%|█████████▎| 26/28 [00:05<00:00,  5.14it/s][38;20m2026-05-09 12:46:40 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.27:  96%|█████████▋| 27/28 [00:05<00:00,  5.14it/s][38;20m2026-05-09 12:46:41 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

Quantizing model.layers.27: 100%|██████████| 28/28 [00:05<00:00,  4.99it/s]

packing:   0%|          | 0/196 [00:00<?, ?it/s]
packing: 100%|██████████| 196/196 [00:00<00:00, 327523.34it/s]
[33;1m2026-05-09 12:46:41 WARNING export_to_nvfp_mxfp.py L249: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4 already exists, this may cause model conflict[0m
[38;20m2026-05-09 12:46:42 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4

### [2026-05-09 12:47:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

5.4G	/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/

### [2026-05-09 12:47:26 UTC] ASSISTANT

**Thinking**

Good, now I have the information needed. Let me calculate the compression ratio:
- Original model: 1136.88 MB
- Quantized model: ~520 MB (531 MB directory)
- Compression ratio: 1136.88 / 520 = 2.19x

Now let me find all output files in the runtime_output_dir and write the quant_summary.json.


**Tool call:** `exec`

```json
{
  "command": "find /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4 -type f ! -path \"*/ar_work_space/*\" | sort 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/request.json 2>&1"
}
```

### [2026-05-09 12:47:26 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

modeling_pp_formulanet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/modular_pp_formulanet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/processing_pp_formulanet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/configuration_pp_lcnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/image_processing_pp_lcnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/modeling_pp_lcnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/modular_pp_lcnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/modular_pp_lcnet_v3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/configuration_pp_ocrv5_mobile_det.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/modeling_pp_ocrv5_mobile_det.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/modular_pp_ocrv5_mobile_det.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/configuration_pp_ocrv5_mobile_rec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/modeling_pp_ocrv5_mobile_rec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/modular_pp_ocrv5_mobile_rec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/configuration_pp_ocrv5_server_det.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/image_processing_pp_ocrv5_server_det.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/modeling_pp_ocrv5_server_det.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/modular_pp_ocrv5_server_det.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/configuration_pp_ocrv5_server_rec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/image_processing_pp_ocrv5_server_rec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/modeling_pp_ocrv5_server_rec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/modular_pp_ocrv5_server_rec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/configuration_prompt_depth_anything.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/image_processing_pil_prompt_depth_anything.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/image_processing_prompt_depth_anything.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/modeling_prompt_depth_anything.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/modular_prompt_depth_anything.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prophetnet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prophetnet/configuration_prophetnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prophetnet/modeling_prophetnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/prophetnet/tokenization_prophetnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt/configuration_pvt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt/image_processing_pil_pvt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt/image_processing_pvt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt/modeling_pvt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt_v2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt_v2/configuration_pvt_v2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/pvt_v2/modeling_pvt_v2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/configuration_qianfan_ocr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/modeling_qianfan_ocr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/modular_qianfan_ocr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/processing_qianfan_ocr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2/__pycache__/tokenization_qwen2.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2/configuration_qwen2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2/modular_qwen2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2/tokenization_qwen2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/configuration_qwen2_audio.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/modeling_qwen2_audio.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/processing_qwen2_audio.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/configuration_qwen2_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/modeling_qwen2_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/modular_qwen2_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/configuration_qwen2_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_pil_qwen2_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/processing_qwen2_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/video_processing_qwen2_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3/__pycache__/configuration_qwen3.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3/__pycache__/modeling_qwen3.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3/configuration_qwen3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3/modular_qwen3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/configuration_qwen3_5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modular_qwen3_5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/tokenization_qwen3_5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/configuration_qwen3_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modular_qwen3_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/configuration_qwen3_next.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modular_qwen3_next.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/configuration_qwen3_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modular_qwen3_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rag/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rag/configuration_rag.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rag/modeling_rag.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rag/retrieval_rag.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rag/tokenization_rag.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/configuration_recurrent_gemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/reformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/reformer/configuration_reformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/reformer/tokenization_reformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/regnet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/regnet/configuration_regnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/regnet/modeling_regnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rembert/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rembert/configuration_rembert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rembert/modeling_rembert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rembert/tokenization_rembert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/resnet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/resnet/configuration_resnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/resnet/modeling_resnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta/configuration_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta/modular_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta_old.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/configuration_roberta_prelayernorm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roc_bert/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roc_bert/configuration_roc_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roc_bert/modeling_roc_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roc_bert/tokenization_roc_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roformer/configuration_roformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roformer/modeling_roformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_roformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr_resnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_pil_rt_detr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_rt_detr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr_resnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr/modular_rt_detr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modular_rt_detr_v2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rwkv/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rwkv/configuration_rwkv.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/rwkv/modeling_rwkv.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam/configuration_sam.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_pil_sam.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_sam.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam/modeling_sam.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam/processing_sam.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2/configuration_sam2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2/image_processing_sam2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2/modeling_sam2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2/modular_sam2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2/processing_sam2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2_video/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2_video/configuration_sam2_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2_video/modeling_sam2_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2_video/modular_sam2_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2_video/processing_sam2_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam2_video/video_processing_sam2_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3/configuration_sam3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3/image_processing_sam3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3/modeling_sam3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3/modular_sam3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3/processing_sam3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/configuration_sam3_lite_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modeling_sam3_lite_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modular_sam3_lite_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/configuration_sam3_tracker.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modeling_sam3_tracker.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modular_sam3_tracker.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/processing_sam3_tracker.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modular_sam3_tracker_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/processing_sam3_tracker_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_video/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_video/configuration_sam3_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_video/modeling_sam3_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam3_video/processing_sam3_video.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam_hq/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam_hq/configuration_sam_hq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam_hq/modeling_sam_hq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam_hq/modular_sam_hq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sam_hq/processing_sam_hq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/configuration_seamless_m4t.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/modeling_seamless_m4t.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/processing_seamless_m4t.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/tokenization_seamless_m4t.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seed_oss/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seed_oss/configuration_seed_oss.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seed_oss/modeling_seed_oss.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seed_oss/modular_seed_oss.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/segformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/segformer/configuration_segformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_pil_segformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_segformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/segformer/modeling_segformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/segformer/modular_segformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seggpt/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seggpt/configuration_seggpt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_pil_seggpt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_seggpt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/seggpt/modeling_seggpt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sew/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sew/configuration_sew.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sew/modeling_sew.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sew/modular_sew.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sew_d/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sew_d/configuration_sew_d.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/sew_d/modeling_sew_d.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/configuration_shieldgemma2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/modeling_shieldgemma2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/processing_shieldgemma2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip/configuration_siglip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_pil_siglip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_siglip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip/processing_siglip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip/tokenization_siglip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/configuration_siglip2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_pil_siglip2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_siglip2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/modeling_siglip2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/modular_siglip2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/processing_siglip2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/siglip2/tokenization_siglip2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanet/configuration_slanet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanet/modeling_slanet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanet/modular_slanet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanext/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanext/configuration_slanext.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanext/image_processing_slanext.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanext/modeling_slanext.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/slanext/modular_slanext.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smollm3/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smollm3/configuration_smollm3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smollm3/modeling_smollm3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smollm3/modular_smollm3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/configuration_smolvlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_pil_smolvlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_smolvlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/modeling_smolvlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/modular_smolvlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/processing_smolvlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/smolvlm/video_processing_smolvlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/solar_open/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/solar_open/configuration_solar_open.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/solar_open/modeling_solar_open.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/solar_open/modular_solar_open.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/configuration_speech_to_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/feature_extraction_speech_to_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/modeling_speech_to_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/processing_speech_to_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/tokenization_speech_to_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speecht5/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speecht5/configuration_speecht5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speecht5/feature_extraction_speecht5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speecht5/modeling_speecht5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speecht5/number_normalizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speecht5/processing_speecht5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/speecht5/tokenization_speecht5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/splinter/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/splinter/configuration_splinter.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/splinter/modeling_splinter.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/splinter/tokenization_splinter.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/squeezebert/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/squeezebert/configuration_squeezebert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/squeezebert/modeling_squeezebert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/squeezebert/tokenization_squeezebert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/stablelm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/stablelm/configuration_stablelm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/stablelm/modeling_stablelm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/starcoder2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/starcoder2/configuration_starcoder2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/starcoder2/modeling_starcoder2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/starcoder2/modular_starcoder2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superglue/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superglue/configuration_superglue.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_pil_superglue.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_superglue.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superglue/modeling_superglue.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superpoint/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superpoint/configuration_superpoint.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_pil_superpoint.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_superpoint.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/superpoint/modeling_superpoint.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swiftformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swiftformer/configuration_swiftformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swiftformer/modeling_swiftformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin/configuration_swin.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin/modeling_swin.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin2sr/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin2sr/configuration_swin2sr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_pil_swin2sr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_swin2sr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swin2sr/modeling_swin2sr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swinv2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swinv2/configuration_swinv2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/swinv2/modeling_swinv2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/configuration_switch_transformers.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modular_switch_transformers.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5/configuration_t5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5/tokenization_t5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma/configuration_t5gemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma/modeling_t5gemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma/modular_t5gemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/configuration_t5gemma2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modular_t5gemma2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/table_transformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/table_transformer/configuration_table_transformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/table_transformer/modeling_table_transformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tapas/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tapas/configuration_tapas.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tapas/modeling_tapas.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tapas/tokenization_tapas.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/textnet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/textnet/configuration_textnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_pil_textnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_textnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/textnet/modeling_textnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/configuration_time_series_transformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/modeling_time_series_transformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm/configuration_timesfm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm/modeling_timesfm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm/modular_timesfm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/configuration_timesfm2_5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modeling_timesfm2_5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modular_timesfm2_5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesformer/configuration_timesformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timesformer/modeling_timesformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timm_backbone/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timm_backbone/configuration_timm_backbone.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timm_backbone/modeling_timm_backbone.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/configuration_timm_wrapper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/image_processing_timm_wrapper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/modeling_timm_wrapper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/trocr/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/trocr/configuration_trocr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/trocr/processing_trocr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tvp/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tvp/configuration_tvp.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_pil_tvp.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_tvp.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tvp/modeling_tvp.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/tvp/processing_tvp.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/udop/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/udop/configuration_udop.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/udop/modeling_udop.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/udop/processing_udop.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/udop/tokenization_udop.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/umt5/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/umt5/configuration_umt5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/umt5/modeling_umt5.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech/configuration_unispeech.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech/modeling_unispeech.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech/modular_unispeech.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/configuration_unispeech_sat.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modeling_unispeech_sat.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modular_unispeech_sat.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/univnet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/univnet/configuration_univnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/univnet/feature_extraction_univnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/univnet/modeling_univnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/upernet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/upernet/configuration_upernet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/upernet/modeling_upernet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/uvdoc/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/uvdoc/configuration_uvdoc.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/uvdoc/image_processing_uvdoc.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/uvdoc/modeling_uvdoc.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/uvdoc/modular_uvdoc.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/configuration_vaultgemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modeling_vaultgemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modular_vaultgemma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/configuration_vibevoice_acoustic_tokenizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/feature_extraction_vibevoice_acoustic_tokenizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modeling_vibevoice_acoustic_tokenizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modular_vibevoice_acoustic_tokenizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/configuration_vibevoice_asr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modeling_vibevoice_asr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modular_vibevoice_asr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/processing_vibevoice_asr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/configuration_video_llama_3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_pil_video_llama_3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_video_llama_3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modeling_video_llama_3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modular_video_llama_3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/processing_video_llama_3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/video_processing_video_llama_3.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llava/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llava/configuration_video_llava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llava/image_processing_video_llava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llava/modeling_video_llava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llava/processing_video_llava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/video_llava/video_processing_video_llava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomae/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomae/configuration_videomae.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_pil_videomae.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_videomae.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomae/modeling_videomae.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomae/video_processing_videomae.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomt/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomt/configuration_videomt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomt/modeling_videomt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomt/modular_videomt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/videomt/video_processing_videomt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vilt/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vilt/configuration_vilt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_pil_vilt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_vilt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vilt/modeling_vilt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vilt/processing_vilt.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vipllava/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vipllava/configuration_vipllava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vipllava/modeling_vipllava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vipllava/modular_vipllava.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/visual_bert/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/visual_bert/configuration_visual_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/visual_bert/modeling_visual_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit/configuration_vit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_pil_vit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_vit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit/modeling_vit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit_mae/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit_mae/configuration_vit_mae.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit_mae/modeling_vit_mae.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit_msn/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit_msn/configuration_vit_msn.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vit_msn/modeling_vit_msn.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitdet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitdet/configuration_vitdet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitdet/modeling_vitdet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitmatte/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitmatte/configuration_vitmatte.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_pil_vitmatte.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_vitmatte.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitmatte/modeling_vitmatte.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose/configuration_vitpose.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_pil_vitpose.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_vitpose.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose/modeling_vitpose.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vits/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vits/configuration_vits.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vits/tokenization_vits.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vivit/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vivit/configuration_vivit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vivit/image_processing_vivit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vivit/modeling_vivit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vjepa2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vjepa2/configuration_vjepa2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vjepa2/modeling_vjepa2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/vjepa2/video_processing_vjepa2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral/configuration_voxtral.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral/modeling_voxtral.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral/modular_voxtral.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral/processing_voxtral.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/configuration_voxtral_realtime.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/feature_extraction_voxtral_realtime.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modular_voxtral_realtime.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/processing_voxtral_realtime.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/configuration_wav2vec2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/processing_wav2vec2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/tokenization_wav2vec2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modular_wav2vec2_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/configuration_wav2vec2_conformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modular_wav2vec2_conformer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wavlm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wavlm/configuration_wavlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wavlm/modeling_wavlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/wavlm/modular_wavlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/configuration_whisper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/english_normalizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/feature_extraction_whisper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/generation_whisper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/processing_whisper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/whisper/tokenization_whisper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/x_clip/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/x_clip/configuration_x_clip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/x_clip/modeling_x_clip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/x_clip/modular_x_clip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/x_clip/processing_x_clip.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xcodec/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xcodec/configuration_xcodec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xcodec/modeling_xcodec.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xglm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xglm/configuration_xglm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xglm/tokenization_xglm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm/configuration_xlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm/modeling_xlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm/tokenization_xlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/configuration_xlm_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modeling_xlm_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modular_xlm_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/tokenization_xlm_roberta.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modular_xlm_roberta_xl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlnet/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlnet/configuration_xlnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlnet/tokenization_xlnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlstm/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlstm/configuration_xlstm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xlstm/modeling_xlstm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xmod/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xmod/configuration_xmod.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/xmod/modeling_xmod.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yolos/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yolos/configuration_yolos.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_pil_yolos.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_yolos.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yolos/modeling_yolos.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yolos/modular_yolos.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yoso/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yoso/configuration_yoso.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/yoso/modeling_yoso.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/youtu/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/youtu/configuration_youtu.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/youtu/modeling_youtu.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/youtu/modular_youtu.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zamba/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zamba/configuration_zamba.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zamba/modeling_zamba.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zamba2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zamba2/configuration_zamba2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zamba2/modeling_zamba2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zamba2/modular_zamba2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zoedepth/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zoedepth/configuration_zoedepth.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_pil_zoedepth.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_zoedepth.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/models/zoedepth/modeling_zoedepth.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/monkey_patching.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/optimization.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/any_to_any.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/audio_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/audio_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/automatic_speech_recognition.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/base.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/depth_estimation.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/document_question_answering.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/feature_extraction.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/fill_mask.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/image_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/image_feature_extraction.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/image_segmentation.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/image_text_to_text.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/keypoint_matching.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/mask_generation.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/object_detection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/pt_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/table_question_answering.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/text_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/text_generation.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/text_to_audio.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/token_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/video_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_audio_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_image_classification.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_object_detection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/processing_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/py.typed
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/pytorch_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/auto.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/base.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_aqlm.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_auto_round.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_awq.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bitnet.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_4bit.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_8bit.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_compressed_tensors.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_eetq.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fbgemm_fp8.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_finegrained_fp8.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fouroversix.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fp_quant.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_gptq.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_higgs.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_hqq.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_metal.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_mxfp4.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quanto.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quark.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_sinq.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_spqr.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_torchao.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_vptq.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizers_utils.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/auto.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/base.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_aqlm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_auto_round.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_awq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bitnet.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_4bit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_8bit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_compressed_tensors.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_eetq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fbgemm_fp8.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_finegrained_fp8.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fouroversix.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fp_quant.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_gptq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_higgs.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_hqq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_metal.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_mxfp4.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quanto.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quark.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_sinq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_spqr.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_torchao.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_vptq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/quantizers/quantizers_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/safetensors_conversion.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/testing_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/time_series_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/tokenization_mistral_common.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/tokenization_python.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/tokenization_utils_sentencepiece.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/tokenization_utils_tokenizers.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/trainer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/trainer_callback.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/trainer_jit_checkpoint.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/trainer_optimizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/trainer_pt_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/trainer_seq2seq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/trainer_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/training_args.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/training_args_seq2seq.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/auto_docstring.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_parsing_utils.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_template_utils.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/constants.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/deprecation.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/doc.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_mistral_common_objects.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_sentencepiece_and_tokenizers_objects.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_torchvision_objects.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_vision_objects.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/generic.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/hub.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/import_utils.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/kernel_config.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/loading_report.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/logging.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/metrics.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/output_capturing.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/peft_utils.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/quantization_config.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/type_validators.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/versions.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/attention_visualizer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/auto_docstring.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/backbone_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/chat_parsing_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/chat_template_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/constants.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/deprecation.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/doc.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_detectron2_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_mistral_common_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_music_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_pt_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_sentencepiece_and_tokenizers_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_speech_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_timm_and_torchvision_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_tokenizers_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_torchaudio_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_torchvision_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/dummy_vision_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/generic.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/hp_naming.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/hub.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/import_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/kernel_config.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/loading_report.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/logging.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/metrics.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/network_logging.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/notebook.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/output_capturing.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/peft_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/pytest_helpers.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/quantization_config.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2_new.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/type_validators.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/utils/versions.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/video_processing_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/transformers/video_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/entry_points.txt
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/top_level.txt
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/_C/libproton.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/_C/libtriton.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/__pycache__/errors.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/__pycache__/testing.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/_internal_testing.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/__pycache__/compiler.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/__pycache__/driver.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/compiler.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/driver.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/driver.c
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/driver.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_channel_descriptor.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_device_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_atomic.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_bf16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_bfloat16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_common.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_complex.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_cooperative_groups.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_fp16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_gl_interop.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_math_constants.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime_pt_api.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_unsafe_atomics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_vector_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_math_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_surface_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_warp_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/concepts.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/device_library_decls.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/functional_grid_launch.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch_GGL.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/helpers.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_api_trace.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_assert.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_cooperative_groups_helper.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_gcc.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_math_fwd.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_ldg.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_prof_str.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_runtime_prof.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/host_defines.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hsa_helpers.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/macro_based_grid_launch.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/math_fwd.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/ockl_image.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/program_state.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_fetch_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_indirect_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/channel_descriptor.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/device_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/driver_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_bf16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_bfloat16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_common.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_complex.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_cooperative_groups.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_deprecated.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_ext.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_fp16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_gl_interop.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_hcc.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_math_constants.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_profile.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime_api.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_texture_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_vector_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_version.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hiprtc.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/library_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/math_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/surface_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/texture_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/Brig.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_common.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_elf.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_kernel_code.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_queue.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_signal.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_amd_tool.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_api_trace.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_amd.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_finalize.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_image.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_aqlprofile.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_loader.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/ext/prof_protocol.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hip_ostream_ops.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hsa_ostream_ops.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hsa_prof_str.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_ext.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hcc.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hip.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hsa.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_plugin.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_roctx.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctx.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/lib/ockl.bc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/amd/lib/ocml.bc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/compiler.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/driver.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/compiler.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/driver.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.c
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openacc/cupti_openacc.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/cupti_openmp.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/omp-tools.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/builtin_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/channel_descriptor.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/common_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/async.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_reduce.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_scan.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/driver_abi.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/functional.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/helpers.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/info.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/invoke.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/memory.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/partitioning.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/reduce.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/scan.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/sync.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/memcpy_async.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/reduce.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/scan.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/common_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/cudacc_ext.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/func_macro.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_config.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_defines.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_runtime.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/nvfunctional
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/storage_class.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuComplex.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGL.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGLTypedefs.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGL.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGLTypedefs.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaProfilerTypedefs.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaTypedefs.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAU.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAUTypedefs.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_helpers.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_primitives.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_device_runtime_api.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_egl_interop.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_gl_interop.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_occupancy.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_helpers.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_primitives.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime_api.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_stdint.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_surface_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_texture_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_vdpau_interop.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudart_platform.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity_deprecated.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_callbacks.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_checkpoint.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_common.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_driver_cbid.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_events.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_metrics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_nvtx_cbid.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling_util.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_profiler_target.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_result.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_runtime_cbid.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_sass_metrics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_target.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_version.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_double_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_launch_parameters.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/fatbinary_section.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaGL_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaVDPAU_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_gl_interop_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_runtime_api_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_vdpau_interop_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudart_removed_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_nvtx_meta.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_config.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_defines.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/library_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_constants.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/mma.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvPTXCompiler.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvfunctional
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_common.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_cuda_host.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_host.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_target.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_atomic_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_intrinsics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_indirect_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_fetch_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_indirect_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.hpp
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_types.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcheckpoint.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.12
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.2024.1.0
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_target.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libpcsamplingutil.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/__pycache__/code_generator.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/__pycache__/compiler.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/__pycache__/errors.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/code_generator.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/compiler.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/errors.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/compiler/make_launcher.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/errors.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/instrumentation/libGPUInstrumentationTestLib.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__pycache__/_utils.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__pycache__/core.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__pycache__/math.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__pycache__/random.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__pycache__/semantic.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/__pycache__/standard.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/core.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/_experimental_tma.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/libdevice.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/utils.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/_experimental_tma.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/libdevice.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/cuda/utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/hip/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/libdevice.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/hip/libdevice.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/extra/libdevice.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/math.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/random.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/semantic.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/language/standard.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/profiler/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/profiler/flags.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/profiler/hook.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/profiler/profile.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/profiler/proton.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/profiler/scope.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/profiler/viewer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/autotuner.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/build.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/cache.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/driver.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/errors.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/jit.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/autotuner.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/build.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/cache.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/driver.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/errors.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/interpreter.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/runtime/jit.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/testing.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/__pycache__/disasm.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/build_extern.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/compile.c
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/compile.h
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/compile.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/disasm.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/experimental_descriptor.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/triton/tools/link.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/entry_points.txt
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/licenses/LICENSE
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/.agents/skills/typer/SKILL.md
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/__main__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/_completion_classes.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/_completion_shared.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/_types.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/_typing.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/cli.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/colors.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/completion.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/core.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/main.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/models.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/params.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/py.typed
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/rich_utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/testing.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typer/utils.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/licenses/LICENSE
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_extensions.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/licenses/LICENSE
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/__pycache__/introspection.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/__pycache__/typing_objects.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/introspection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/py.typed
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/typing_objects.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/typing_inspection/typing_objects.pyi
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/licenses/LICENSE.txt
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/_base_connection.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/_collections.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/_request_methods.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/_version.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/connection.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/connectionpool.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/exceptions.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/fields.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/filepost.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/poolmanager.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/__pycache__/response.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/_base_connection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/_collections.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/_request_methods.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/_version.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/connection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/connectionpool.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/socks.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/connection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/emscripten_fetch_worker.js
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/fetch.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/request.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/response.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/pyopenssl.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/contrib/socks.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/exceptions.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/fields.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/filepost.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/http2/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/probe.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/http2/connection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/http2/probe.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/poolmanager.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/py.typed
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/response.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/connection.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/proxy.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/request.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/response.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/retry.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_match_hostname.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssltransport.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/timeout.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/url.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/util.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/wait.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/connection.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/proxy.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/request.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/response.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/retry.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/ssl_.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/ssl_match_hostname.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/ssltransport.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/timeout.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/url.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/util.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/urllib3/util/wait.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/licenses/LICENSE-APACHE
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/licenses/LICENSE-MIT
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv-0.11.12.dist-info/sboms/uv.cyclonedx.json
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv/__main__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv/__pycache__/__main__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv/__pycache__/_find_uv.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv/_find_uv.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/uv/py.typed
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/licenses/LICENSE
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/top_level.txt
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash/__init__.pyi
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash/__pycache__/version.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash/_xxhash.cpython-312-x86_64-linux-gnu.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash/py.typed
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/xxhash/version.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/composer.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/constructor.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/cyaml.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/dumper.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/emitter.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/error.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/events.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/loader.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/nodes.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/parser.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/reader.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/representer.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/resolver.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/scanner.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/serializer.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/__pycache__/tokens.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/_yaml.cpython-312-x86_64-linux-gnu.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/composer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/constructor.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/cyaml.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/dumper.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/emitter.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/error.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/events.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/loader.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/nodes.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/parser.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/reader.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/representer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/resolver.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/scanner.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/serializer.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yaml/tokens.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/INSTALLER
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/METADATA
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/RECORD
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/REQUESTED
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/WHEEL
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/licenses/LICENSE
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/licenses/NOTICE
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/top_level.txt
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__init__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__pycache__/__init__.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__pycache__/_parse.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__pycache__/_path.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__pycache__/_query.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__pycache__/_quoters.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__pycache__/_quoting.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/__pycache__/_url.cpython-312.pyc
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_parse.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_path.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_query.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_quoters.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_quoting.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_quoting_c.cpython-312-x86_64-linux-gnu.so
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_quoting_c.pyx
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_quoting_py.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/_url.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/lib/python3.12/site-packages/yarl/py.typed
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/pyvenv.cfg
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/share/man/man1/isympy.1

### [2026-05-09 12:47:26 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

-rw-r--r-- 1 root root 1059 May  9 12:43 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/request.json

### [2026-05-09 12:47:50 UTC] ASSISTANT

**Thinking**

OK, so the key output files are:
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quantize.py
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/auto.log
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_prompt.txt
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/request.json
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/venv/


- /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/ (exported model)

Now I'm gathering the details: the model is Qwen/Qwen3-0.6B quantized to MXFP4 using RTN and exported as auto_round. The original size is around 1136.88 MB while the quantized version comes in at approximately 531 MB.

Let me calculate the compression ratio—that's about 0.467. The RTX 4090 was the accelerator, and from the logs I can see the quantization finished at 12:46:42 after starting around 12:46:35, so roughly 7-8 seconds total. I should double-check the timing in the logs to make sure I have the duration right.

Now I'm looking at what files are actually in the output directory to get a complete picture of what was generated.


**Tool call:** `exec`

```json
{
  "command": "stat /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log | grep -E \"Modify|Birth\""
}
```

**Tool call:** `exec`

```json
{
  "command": "head -20 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/logs/quant_exec.log"
}
```

### [2026-05-09 12:47:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Modify: 2026-05-09 12:46:42.397431733 +0000
 Birth: 2026-05-09 12:46:19.990071507 +0000

### [2026-05-09 12:47:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Loading model: Qwen/Qwen3-0.6B
Scheme: MXFP4
Iters: 0 (RTN mode)
nsamples: 128
Format: auto_round
Device: cuda
404 Client Error. (Request ID: Root=1-69ff2c9f-416c60242c399b162ffe7340;3e97070b-7e65-428b-93a1-3447cb8e9563)

Entry Not Found for url: https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model_index.json.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[33;1m2026-05-09 12:46:35 WARNING base.py L407: `device` is deprecated, please use `device_map` instead[0m
[38;20m2026-05-09 12:46:35 INFO base.py L473: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-05-09 12:46:35 INFO base.py L517: using torch.bfloat16 for quantization tuning[0m
Starting quantization...
[33;1m2026-05-09 12:46:35 WARNING formats.py L166: some layers are skipped quantization (shape not divisible by 32): [0m
[33;1m2026-05-09 12:46:35 WARNING modeling_utils.py L4555: `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.[0m
[33;1m2026-05-09 12:46:35 WARNING base.py L1748: immediate_saving is only supported for int quantization, set to False[0m
[38;20m2026-05-09 12:46:35 INFO base.py L1401: RTN mode detected (iters=0): force blockwise quantization to avoid layer-wise full-model materialization.[0m

  0%|          | 0/28 [00:00<?, ?it/s]
Quantizing model.layers.0:   0%|          | 0/28 [00:00<?, ?it/s][38;20m2026-05-09 12:46:36 INFO offload.py L703: OffloadManager (compressor): tempdir = /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/ar_work_space/offload/compressor_gf1fb5_j[0m
[38;20m2026-05-09 12:46:36 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 0.26GB[0m

### [2026-05-09 12:48:10 UTC] ASSISTANT

**Thinking**

I now have a clear picture. Let me compute the duration:
- Log started at 12:46:19 (birth time)
- Log modified at 12:46:42 (end time)
- Duration: 23 seconds

But wait, quantization ran from 12:46:35 to 12:46:42 based on the actual log. Let me compute end-start from the timestamps.

Actually let me re-read the log - quantization started at 12:46:35 and completed at 12:46:42, that's 7 seconds of actual quantization work.

Let me finalize the quant_summary.json with all the correct details.




Now let me generate the final quant_summary.json:


**Tool call:** `exec`

```json
{
  "command": "cat > /tmp/quant_summary_tmp.json << 'EOF'\n{\n  \"model_id\": \"Qwen/Qwen3-0.6B\",\n  \"scheme\": \"MXFP4\",\n  \"method\": \"RTN\",\n  \"export_format\": \"auto_round\",\n  \"device\": \"cuda\",\n  \"quant_num_gpus\": \"1\",\n  \"num_gpus\": \"1\",\n  \"output_dir\": \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4\",\n  \"runtime_output_dir\": \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4\",\n  \"quantized_model_dir\": \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4\",\n  \"status\": \"success\",\n  \"duration_seconds\": 7.0,\n  \"original_size_mb\": 1136.88,\n  \"quantized_size_mb\": 520.0,\n  \"compression_ratio\": 0.4574,\n  \"errors\": [],\n  \"solutions\": [],\n  \"output_files\": [\n    \"logs/quant_exec.log\",\n    \"logs/auto.log\",\n    \"logs/quant_prompt.txt\",\n    \"quantize.py\",\n    \"request.json\",\n    \"venv/\"\n  ]\n}\nEOF\nmv /tmp/quant_summary_tmp.json /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json && echo \"written\""
}
```

### [2026-05-09 12:48:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

written

### [2026-05-09 12:48:22 UTC] ASSISTANT

**Thinking**

Let me verify everything is correct - the quantized model output dir has the model files, the runtime dir has the artifacts, and the quant_summary.json is in the right place.


**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json"
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/ | head -20"
}
```

### [2026-05-09 12:48:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "MXFP4",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4",
  "status": "success",
  "duration_seconds": 7.0,
  "original_size_mb": 1136.88,
  "quantized_size_mb": 520.0,
  "compression_ratio": 0.4574,
  "errors": [],
  "solutions": [],
  "output_files": [
    "logs/quant_exec.log",
    "logs/auto.log",
    "logs/quant_prompt.txt",
    "quantize.py",
    "request.json",
    "venv/"
  ]
}

### [2026-05-09 12:48:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 543724
drwxr-xr-x 2 root root      4096 May  9 12:46 .
drwxr-xr-x 4 root root        59 May  9 12:43 ..
-rw-r--r-- 1 root root      4168 May  9 12:46 chat_template.jinja
-rw-r--r-- 1 root root      1784 May  9 12:46 config.json
-rw-r--r-- 1 root root       213 May  9 12:46 generation_config.json
-rw-r--r-- 1 root root 545318232 May  9 12:46 model.safetensors
-rw-r--r-- 1 root root       313 May  9 12:46 quantization_config.json
-rw-r--r-- 1 root root  11422650 May  9 12:46 tokenizer.json
-rw-r--r-- 1 root root       694 May  9 12:46 tokenizer_config.json

### [2026-05-09 12:48:32 UTC] ASSISTANT

**Thinking**

Let me double check the model size using the safetensors file directly.


**Tool call:** `exec`

```json
{
  "command": "du -sh /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/model.safetensors"
}
```

### [2026-05-09 12:48:32 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

521M	/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/model.safetensors

### [2026-05-09 12:48:52 UTC] ASSISTANT

**Thinking**

Good. The model.safetensors is 521MB and the total model dir is 531MB. Original was 1136.88MB.

The quantization completed successfully. Let me finalize the quant_summary.json with accurate file listing. I should also double-check the actual compressed model size from the safetensors file since that is the actual model artifact.

Original: 1136.88 MB
Quantized model.safetensors: 521 MB
Quantized total: 531 MB (directory)

The directory contains more than just the model file. Let me update the summary with accurate info.

Also, the duration: the quantization log shows it ran for about 7 seconds based on the timestamps (12:46:35 to 12:46:42). Let me be more precise.

Actually looking at the timestamps:
- Start: 12:46:35 (first log line after "Starting quantization...")
- End: 12:46:42 (last log line "Quantization complete!")

So about 7 seconds.

Let me update the summary file to have the proper compressed sizes and file list.


**Tool call:** `exec`

```json
{
  "command": "cat > /tmp/quant_summary_tmp.json << 'EOF'\n{\n  \"model_id\": \"Qwen/Qwen3-0.6B\",\n  \"scheme\": \"MXFP4\",\n  \"method\": \"RTN\",\n  \"export_format\": \"auto_round\",\n  \"device\": \"cuda\",\n  \"quant_num_gpus\": \"1\",\n  \"num_gpus\": \"1\",\n  \"output_dir\": \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4\",\n  \"runtime_output_dir\": \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4\",\n  \"quantized_model_dir\": \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4\",\n  \"status\": \"success\",\n  \"duration_seconds\": 7.0,\n  \"original_size_mb\": 1136.88,\n  \"quantized_size_mb\": 520.0,\n  \"compression_ratio\": 0.4575,\n  \"errors\": [],\n  \"solutions\": [],\n  \"output_files\": [\n    \"logs/quant_exec.log\",\n    \"logs/auto.log\",\n    \"logs/quant_prompt.txt\",\n    \"quantize.py\",\n    \"request.json\",\n    \"venv/\"\n  ]\n}\nEOF\nmv /tmp/quant_summary_tmp.json /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json"
}
```

### [2026-05-09 12:48:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-09 12:48:59 UTC] ASSISTANT

**Thinking**

Let me verify the final state of all artifacts.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/ && echo \"---\" && cat /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json"
}
```

### [2026-05-09 12:48:59 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 12
drwxr-xr-x 5 root root  144 May  9 12:48 .
drwxr-xr-x 3 root root   43 May  9 12:43 ..
drwxr-xr-x 2 root root    6 May  9 12:46 ar_work_space
drwxr-xr-x 2 root root   84 May  9 12:46 logs
-rw-r--r-- 1 root root  761 May  9 12:48 quant_summary.json
-rw-r--r-- 1 root root  935 May  9 12:46 quantize.py
-rw-r--r-- 1 root root 1059 May  9 12:43 request.json
drwxr-xr-x 6 root root  132 May  9 12:45 venv
---
{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "MXFP4",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4",
  "status": "success",
  "duration_seconds": 7.0,
  "original_size_mb": 1136.88,
  "quantized_size_mb": 520.0,
  "compression_ratio": 0.4575,
  "errors": [],
  "solutions": [],
  "output_files": [
    "logs/quant_exec.log",
    "logs/auto.log",
    "logs/quant_prompt.txt",
    "quantize.py",
    "request.json",
    "venv/"
  ]
}

### [2026-05-09 12:49:10 UTC] ASSISTANT

**Thinking**

Everything looks good. Let me summarize what was accomplished.




Quantization complete. Here's the summary:

**Model:** Qwen/Qwen3-0.6B → MXFP4 (RTN mode, auto_round format)

**Results:**
- Original size: **1136.88 MB** (BF16)
- Quantized size: **520 MB** (model.safetensors only)
- Compression ratio: **~45.8%**
- Duration: **~7 seconds**

**Output files:**
- Quantized model: `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-MXFP4/` (model.safetensors, config.json, tokenizer files, etc.)
- Runtime artifacts: `/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/` (quantize.py, logs, venv)
- Summary: `/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-MXFP4/quant_summary.json`

**Notes:**
- RTN (Round-To-Nearest) mode used — no calibration training, fast quantization
- Some layers skipped (shape not divisible by 32 for MXFP4 block size)
- Peak VRAM usage: 0.26 GB (lightweight quantization with block-wise offloading)
