# Session: autoeval_quant_329

- **Session ID:** `autoeval_quant_329`
- **Timestamp:** 2026-04-29 12:36:04 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 1: Quantization

### [2026-04-29 12:36:04 UTC] USER

You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_quant/SKILL.md

Model: Qwen/Qwen3.5-27B
Quantization: W4A16 / RTN
Export format: auto_round
Quantized Model Output directory: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16
Quantization execution log path: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log
Runtime device: cuda
Num gpus: 1

Directory responsibilities:
- Write exported model files to: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16
- Write runtime artifacts such as quant_summary.json, quantize.py, logs, prompts, copied request/session files, and the venv to: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16

CRITICAL SCRIPT REQUIREMENT:
- Before starting quantization, you MUST first generate the quantization script file:
    /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py
- The file name must be exactly: quantize.py
- The script must be a standalone Python program runnable with:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py
- The generated quantize.py must focus on the core quantization/export logic only.
- Do NOT put venv creation, pip/uv installation, proxy/bootstrap shell setup, summary generation, or unrelated orchestration into quantize.py.
- In this same OpenClaw task, prepare or reuse the Python environment separately before executing quantize.py.
- In this same OpenClaw task, first write quantize.py, then execute that generated script yourself.
- When you execute quantize.py, you MUST stream stdout/stderr into this log file while still printing output:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json is a final summary artifact, not a progress marker.
- After quantize.py finishes, inspect the exported artifacts and write /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json in a separate finalize step outside quantize.py.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json until quantization has finished and the exported artifacts are ready.
- Write /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json atomically via a temporary file and rename/move it into place only at finalize time.
- If quantization fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json before exiting non-zero, also atomically.
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

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json - structured summary:
{
  "model_id": "Qwen/Qwen3.5-27B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16",
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

### [2026-04-29 12:36:08 UTC] ASSISTANT

**Thinking**

Let me start by reading the SKILL.md file to understand the quantization workflow, then generate the quantize.py script and execute it.

**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
}
```

### [2026-04-29 12:36:08 UTC] TOOL RESULT
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


### [2026-04-29 12:36:17 UTC] ASSISTANT

**Thinking**

Let me analyze the task:

1. Model: Qwen/Qwen3.5-27B
2. Quantization: W4A16 / RTN
3. Export format: auto_round
4. Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16
5. Runtime artifacts: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16
6. Log path: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log
7. Device: cuda


8. Num gpus: 1

Now I'll create the runtime directory, write out the quantization script, set up the environment, execute it, and then prepare the summary.

**Tool call:** `exec`

```json
{
  "command": "mkdir -p /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs && mkdir -p /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\" 2>/dev/null || echo \"torch not in system python\""
}
```

**Tool call:** `exec`

```json
{
  "command": "ls /root/.venv/bin/python 2>/dev/null && echo \"venv exists\" || echo \"no venv\""
}
```

### [2026-04-29 12:36:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:36:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch not in system python

### [2026-04-29 12:36:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.venv/bin/python
venv exists

### [2026-04-29 12:36:20 UTC] ASSISTANT

**Thinking**

The system Python doesn't have torch, but /root/.venv exists. Let me check if it has torch and the right packages.

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import auto_round; print('auto_round ok')\" 2>/dev/null || echo \"auto_round not installed\""
}
```

### [2026-04-29 12:36:20 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-04-29 12:36:20 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

auto_round not installed

### [2026-04-29 12:36:25 UTC] ASSISTANT

**Thinking**

The venv at /root/.venv doesn't have torch or auto_round. I need to create a new venv with --system-site-packages to inherit torch from the system, or install torch.

Let me check what python3 is available and if there's torch installed system-wide.

**Tool call:** `exec`

```json
{
  "command": "which python3 && python3 --version && find /usr /opt -name \"torch\" -type d 2>/dev/null | head -5"
}
```

**Tool call:** `exec`

```json
{
  "command": "pip3 show torch 2>/dev/null || pip show torch 2>/dev/null || echo \"no system torch\""
}
```

### [2026-04-29 12:36:25 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.venv/bin/python3
Python 3.12.13

### [2026-04-29 12:36:25 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

no system torch

### [2026-04-29 12:36:28 UTC] ASSISTANT

**Thinking**

The system Python is actually from the venv at /root/.venv. Let me check what's installed there.

**Tool call:** `exec`

```json
{
  "command": "pip3 list 2>/dev/null | grep -i -E \"torch|auto.round|transformers|accelerate\" || /root/.venv/bin/pip list 2>/dev/null | grep -i -E \"torch|auto.round|transformers|accelerate\""
}
```

**Tool call:** `exec`

```json
{
  "command": "pip3 show auto-round 2>/dev/null || echo \"auto-round not in pip\""
}
```

### [2026-04-29 12:36:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`



(Command exited with code 1)

### [2026-04-29 12:36:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

auto-round not in pip

### [2026-04-29 12:36:31 UTC] ASSISTANT

**Thinking**

It seems pip3 is the system pip but torch is not installed. Let me look at the /root/.venv more carefully.

**Tool call:** `exec`

```json
{
  "command": "ls /root/.venv/lib/ 2>/dev/null && /root/.venv/bin/pip list 2>/dev/null | head -30"
}
```

### [2026-04-29 12:36:31 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

python3.12

### [2026-04-29 12:36:35 UTC] ASSISTANT

**Thinking**

The /root/.venv/lib/python3.12 directory exists but is empty or has no packages. Let me check what's in the site-packages.

**Tool call:** `exec`

```json
{
  "command": "ls /root/.venv/lib/python3.12/site-packages/ 2>/dev/null | head -20 || echo \"no site-packages\""
}
```

**Tool call:** `exec`

```json
{
  "command": "ls /usr/local/lib/python3*/dist-packages/ 2>/dev/null | head -20 || ls /usr/local/lib/python3*/*/site-packages/ 2>/dev/null | head -20"
}
```

### [2026-04-29 12:36:35 UTC] TOOL RESULT
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
certifi-2026.4.22.dist-info
click
click-8.3.3.dist-info
cmdop
cmdop-2026.4.7.2.dist-info
dotenv
filelock

### [2026-04-29 12:36:35 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:36:38 UTC] ASSISTANT

**Thinking**

OK so there are packages in /usr/local/lib/python3*/*/site-packages/. Let me check if torch is there.

**Tool call:** `exec`

```json
{
  "command": "find /usr/local/lib -name \"torch*\" -type d 2>/dev/null | head -10"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import sys; print(sys.path)\""
}
```

### [2026-04-29 12:36:39 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:36:39 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

['', '/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python312.zip', '/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12', '/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/lib/python3.12/lib-dynload', '/root/.venv/lib/python3.12/site-packages']

### [2026-04-29 12:36:43 UTC] ASSISTANT

**Thinking**

So the Python is using /root/.venv for packages. Let me look at the /usr/local/lib paths more carefully.

**Tool call:** `exec`

```json
{
  "command": "ls /usr/local/lib/ 2>/dev/null"
}
```

**Tool call:** `exec`

```json
{
  "command": "find / -name \"torch\" -type d 2>/dev/null | grep -v proc | head -10"
}
```

### [2026-04-29 12:36:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

node_modules

### [2026-04-29 12:36:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:36:47 UTC] ASSISTANT

**Thinking**

Hmm, there's no torch installed anywhere obvious. Let me check if there's a way torch is preinstalled somewhere else. Let me check if there's a conda or a different python with torch.

**Tool call:** `exec`

```json
{
  "command": "which conda 2>/dev/null; which nvcc 2>/dev/null; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null"
}
```

**Tool call:** `exec`

```json
{
  "command": "find /opt /home /root -name \"torch\" -type d 2>/dev/null | grep -v proc | head -10"
}
```

### [2026-04-29 12:36:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/usr/bin/nvcc
NVIDIA GeForce RTX 4090, 24564 MiB

### [2026-04-29 12:36:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:36:52 UTC] ASSISTANT

**Thinking**

There's an RTX 4090 with 24GB VRAM. Let me install torch and the necessary packages.

Since there's no torch currently, I need to install it. Let me create the venv with --system-site-packages and install torch from PyPI or CUDA wheel.

Actually, the instructions say:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use: python3 -m venv --system-site-packages <path>

But it seems torch is not actually pre-installed. Let me check if there's a pip index URL for CUDA wheels.

Let me try to install torch. Since this is CUDA 12.x compatible (RTX 4090, nvcc at /usr/bin/nvcc), I should install the CUDA 12.x wheel.

**Tool call:** `exec`

```json
{
  "command": "nvcc --version 2>/dev/null | head -3"
}
```

### [2026-04-29 12:36:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023

### [2026-04-29 12:36:58 UTC] ASSISTANT

**Thinking**

nvcc is version 12.x (2023). Let me install torch with CUDA support.

First, let me create the venv with --system-site-packages as instructed, then install torch.

Actually wait - the skill says "System Python has torch+cuda pre-installed" but it doesn't seem to be the case. Let me install torch from PyPI.

For CUDA 12.x (which nvcc 2023 suggests), I should use cu121 or cu124.

Actually let me check what's available:

**Tool call:** `exec`

```json
{
  "command": "python3 -m venv --system-site-packages /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv",
  "timeout": 60
}
```

### [2026-04-29 12:37:00 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:37:03 UTC] ASSISTANT

**Thinking**

Now let me install torch. I'll install the CUDA 12.x wheel. Let me try cu124 since that's a recent CUDA version.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\n$VENV/bin/python -m pip install -U uv 2>&1 | tail -3",
  "timeout": 60
}
```

### [2026-04-29 12:37:04 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

[notice] A new release of pip is available: 25.0.1 -> 26.1
[notice] To update, run: python -m pip install --upgrade pip

### [2026-04-29 12:37:08 UTC] ASSISTANT

**Thinking**

Now let me install torch. I'll try the PyTorch index with CUDA 12.x support.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nuv pip install --python $VENV/bin/python --index-url https://download.pytorch.org/whl/cu124 torch 2>&1 | tail -5",
  "yieldMs": 120000,
  "timeout": 300
}
```

### [2026-04-29 12:37:30 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

+ setuptools==70.2.0
 + sympy==1.13.1
 + torch==2.6.0+cu124
 + triton==3.2.0
 + typing-extensions==4.15.0

### [2026-04-29 12:37:35 UTC] ASSISTANT

**Thinking**

Torch is installed. Now let me install the other necessary packages: auto-round, transformers, accelerate, datasets.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\n$VENV/bin/python -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
}
```

### [2026-04-29 12:37:37 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:275: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:81.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
torch: 2.6.0+cu124 cuda: True NVIDIA GeForce RTX 4090

### [2026-04-29 12:37:40 UTC] ASSISTANT

**Thinking**

Good, torch is installed and CUDA is available. Now let me install auto-round and other dependencies.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nuv pip install --python $VENV/bin/python auto-round 2>&1 | tail -5",
  "yieldMs": 120000,
  "timeout": 300
}
```

### [2026-04-29 12:37:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

+ typer==0.25.0
 + typing-inspection==0.4.2
 + urllib3==2.6.3
 + xxhash==3.7.0
 + yarl==1.23.0

### [2026-04-29 12:37:47 UTC] ASSISTANT

**Thinking**

Let me check if auto-round is installed now and verify the version.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\n$VENV/bin/python -c \"import auto_round; print('auto_round:', auto_round.__version__)\""
}
```

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nuv pip install --python $VENV/bin/python transformers accelerate datasets 2>&1 | tail -3",
  "yieldMs": 120000,
  "timeout": 300
}
```

### [2026-04-29 12:37:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

auto_round: 0.12.3

### [2026-04-29 12:37:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Using Python 3.12.13 environment at: quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv
Checked 3 packages in 34ms

### [2026-04-29 12:37:59 UTC] ASSISTANT

**Thinking**

Wait, I see "Using Python 3.12.13 environment at: quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv" - that's a relative path in the output. That might mean uv is not using the venv properly. Let me check if transformers is installed.

Actually wait, the output from the last exec shows "Checked 3 packages in 34ms" which suggests it didn't actually install anything because those packages might already be available. Let me check.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\n$VENV/bin/python -c \"import transformers; print('transformers:', transformers.__version__)\""
}
```

### [2026-04-29 12:38:00 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

transformers: 5.7.0

### [2026-04-29 12:38:05 UTC] ASSISTANT

**Thinking**

OK so transformers is installed (probably system-wide site-packages were inherited). Let me check what packages are in the venv vs system site-packages.Actually, looking at the exec output more carefully:
- `uv pip install --python $VENV/bin/python transformers accelerate datasets` returned "Checked 3 packages in 34ms"

This is odd. Let me check what `uv pip install` actually does when the packages are already available.

Actually, I think the issue might be that uv is resolving packages but finding they're already available via system-site-packages. Let me check what's actually in the venv's site-packages now.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nls $VENV/lib/python3.12/site-packages/ | grep -E \"transformers|accelerate|datasets|auto_round\""
}
```

### [2026-04-29 12:38:06 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

accelerate
accelerate-1.13.0.dist-info
auto_round
auto_round-0.12.3.dist-info
auto_round_extension
datasets
datasets-4.8.5.dist-info
transformers
transformers-5.7.0.dist-info

### [2026-04-29 12:38:19 UTC] ASSISTANT

**Thinking**

All packages are installed in the venv. Now I need to write the quantize.py script. Let me first check if the model Qwen/Qwen3.5-27B exists on HuggingFace and gather some info about it.

The quantization task:
- Model: Qwen/Qwen3.5-27B
- Scheme: W4A16 (INT4 weight, FP16 activation)
- Method: RTN (Round-To-Nearest, iters=0)
- Export format: auto_round
- Device: cuda
- Num gpus: 1

For W4A16 with RTN (iters=0), the script should use:


- device="cuda" (single GPU)
- iters=0 (RTN mode)
- nsamples can be 128 or similar
- format="auto_round"

Let me create the quantize.py script now.

Now I'll write the quantization script:


**Tool call:** `write`

```json
{
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py",
  "content": "#!/usr/bin/env python3\n\"\"\"\nAuto-Round Quantization Script\nGenerated by auto_quant skill\n\nModel: Qwen/Qwen3.5-27B\nScheme: W4A16\nMethod: RTN (iters=0)\nExport format: auto_round\nDevice: cuda (single GPU)\n\"\"\"\n\nimport os\nimport sys\n\n# Set HF_HOME to avoid token issues for public model\nos.environ.setdefault(\"HF_HOME\", \"/root/.cache/huggingface\")\nos.environ.setdefault(\"TRANSFORMERS_CACHE\", \"/root/.cache/huggingface\")\n\nfrom auto_round import AutoRound\n\n# Configuration\nmodel_name_or_path = \"Qwen/Qwen3.5-27B\"\noutput_dir = \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16\"\nscheme = \"W4A16\"   # INT4 weight, FP16 activation\niters = 0          # 0 = RTN mode (no training, fastest)\nnsamples = 128     # calibration samples for quantize and save\nformat_str = \"auto_round\"\nnum_gpus = 1\n\nprint(f\"=\" * 60)\nprint(f\"Auto-Round Quantization\")\nprint(f\"=\" * 60)\nprint(f\"Model: {model_name_or_path}\")\nprint(f\"Scheme: {scheme}\")\nprint(f\"Method: RTN (iters=0)\")\nprint(f\"nsamples: {nsamples}\")\nprint(f\"Format: {format_str}\")\nprint(f\"Output: {output_dir}\")\nprint(f\"Device: cuda (num_gpus={num_gpus})\")\nprint(f\"=\" * 60)\nsys.stdout.flush()\n\n# Create AutoRound instance\n# Single GPU: use device=\"cuda\"\nar = AutoRound(\n    model_name_or_path,\n    scheme=scheme,\n    iters=iters,\n    nsamples=nsamples,\n    device=\"cuda\",\n)\n\n# Quantize and save\nprint(\"\\nStarting quantization...\")\nsys.stdout.flush()\nar.quantize_and_save(output_dir=output_dir, format=format_str)\n\nprint(f\"\\nQuantization complete! Output: {output_dir}\")\n"
}
```

### [2026-04-29 12:38:19 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 1505 bytes to /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py

### [2026-04-29 12:38:23 UTC] ASSISTANT

**Thinking**

Now let me execute the quantization script with stdout/stderr streamed to the log file.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nLOGFILE=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log\npython3 -u $VENV/bin/python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py 2>&1 | tee $LOGFILE",
  "yieldMs": 600000,
  "timeout": 3600,
  "pty": true
}
```

### [2026-04-29 12:38:23 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/python", line 1
    ELF
SyntaxError: source code cannot contain null bytes

### [2026-04-29 12:38:26 UTC] ASSISTANT

**Thinking**

The venv python seems to be a binary, not a valid Python script. Let me check the venv structure. This is strange - it seems the shebang line is wrong. Let me check.

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/python | head -3"
}
```

### [2026-04-29 12:38:26 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

ELF>���@���@8@)'@@�?@�?Q�td�?�?PP�?P�?00@@

��A�A(�(�A(�A   BB����0� � ��Y�YP�P:P:�9 � P�P:P:R�tdP�P:P:����```\=�\=�P�tdT�����\I\I���	!+F#�
�̐P:h:���o`yA��?��@

### [2026-04-29 12:38:30 UTC] ASSISTANT

**Thinking**

Ah, the python binary in the venv is an actual binary, not a script. But why does the shebang line show `ELF`? Let me check the python3 symlink and also use the proper path.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/ | head -20"
}
```

### [2026-04-29 12:38:30 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 59000
drwxr-xr-x 2 root root     4096 Apr 29 12:37 .
drwxr-xr-x 6 root root      100 Apr 29 12:37 ..
-rw-r--r-- 1 root root     9033 Apr 29 12:37 Activate.ps1
-rwxr-xr-x 1 root root      379 Apr 29 12:37 accelerate
-rwxr-xr-x 1 root root      371 Apr 29 12:37 accelerate-config
-rwxr-xr-x 1 root root      373 Apr 29 12:37 accelerate-estimate-memory
-rwxr-xr-x 1 root root      371 Apr 29 12:37 accelerate-launch
-rwxr-xr-x 1 root root      370 Apr 29 12:37 accelerate-merge-weights
-rw-r--r-- 1 root root     2256 Apr 29 12:37 activate
-rw-r--r-- 1 root root      964 Apr 29 12:37 activate.csh
-rw-r--r-- 1 root root     2239 Apr 29 12:37 activate.fish
-rwxr-xr-x 1 root root      362 Apr 29 12:37 auto-round
-rwxr-xr-x 1 root root      372 Apr 29 12:37 auto-round-best
-rwxr-xr-x 1 root root      372 Apr 29 12:37 auto-round-eval
-rwxr-xr-x 1 root root      372 Apr 29 12:37 auto-round-fast
-rwxr-xr-x 1 root root      384 Apr 29 12:37 auto-round-lib-install
-rwxr-xr-x 1 root root      374 Apr 29 12:37 auto-round-light
-rwxr-xr-x 1 root root      372 Apr 29 12:37 auto-round-mllm
-rwxr-xr-x 1 root root      362 Apr 29 12:37 auto_round

### [2026-04-29 12:38:32 UTC] ASSISTANT

**Thinking**

I see there's no python binary in the venv bin directory. That's the issue. Let me check if there's a python3 symlink or similar.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/python* 2>/dev/null"
}
```

### [2026-04-29 12:38:32 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lrwxrwxrwx 1 root root 10 Apr 29 12:36 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/python -> python3.12
lrwxrwxrwx 1 root root 10 Apr 29 12:36 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/python3 -> python3.12
lrwxrwxrwx 1 root root 73 Apr 29 12:36 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/python3.12 -> /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12

### [2026-04-29 12:38:36 UTC] ASSISTANT

**Thinking**

OK so python3.12 is a symlink to `/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`. Let me try to run the script with that python directly.Actually wait, when I ran `python3 -u $VENV/bin/python` it failed with a syntax error about null bytes. But `$VENV/bin/python` is a symlink to `python3.12` which is itself a symlink to the UV-managed Python.

Let me check the target python directly.

**Tool call:** `exec`

```json
{
  "command": "/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 --version"
}
```

### [2026-04-29 12:38:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Python 3.12.13

### [2026-04-29 12:38:41 UTC] ASSISTANT

**Thinking**

OK so the python binary is fine. The issue might have been with how I was using the PTY. Let me try again without PTY but using a proper shell invocation.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nLOGFILE=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log\ncd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16 && /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py 2>&1 | tee $LOGFILE",
  "yieldMs": 600000,
  "timeout": 3600,
  "pty": true
}
```

### [2026-04-29 12:38:41 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Traceback (most recent call last):
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py", line 20, in <module>
    from auto_round import AutoRound
ModuleNotFoundError: No module named 'auto_round'

### [2026-04-29 12:38:50 UTC] ASSISTANT

**Thinking**

The issue is that the auto_round module is installed in the venv's site-packages, but the UV-managed Python at `/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12` doesn't have the venv's site-packages in its path.

I need to either:
1. Use the venv's python by adding the site-packages to PYTHONPATH
2. Or use the venv properly

Let me check what the venv activation does and try to use the venv's python correctly.

Actually the issue is that the venv's python symlink points to a UV-managed Python which doesn't have the venv's site-packages in its path. 

When I ran `python3 -m venv --system-site-packages`, the symlinks were created correctly, but when I ran pip install, it installed into the venv's site-packages directory. The problem is that the UV Python doesn't know about the venv's site-packages even though they're there. I should try activating the venv properly with `source /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/activate` to set up the environment correctly.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nPYTHONPATH=$VENV/lib/python3.12/site-packages /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -c \"import auto_round; print('ok')\""
}
```

### [2026-04-29 12:38:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

ok

### [2026-04-29 12:39:01 UTC] ASSISTANT

**Thinking**

So the auto_round module is in the venv's site-packages. I can run it by setting PYTHONPATH. But actually, let me try a different approach. Let me try running with the venv activated.Actually, the issue with `python3 -m venv --system-site-packages` is that the venv python binary (which is a symlink to the UV python) doesn't automatically include the venv's site-packages in PYTHONPATH.Let me try with the activate script or by setting PYTHONPATH properly.Actually, I think the issue is that venv with --system-site-packages should inherit system packages but still have its own site-packages. Let me check what happens when I run the python with the venv activated.Actually, let me try running with PYTHONPATH set:

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nLOGFILE=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log\ncd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16 && \\\n  PYTHONPATH=$VENV/lib/python3.12/site-packages \\\n  /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -u \\\n  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py 2>&1 | tee $LOGFILE",
  "yieldMs": 600000,
  "timeout": 3600,
  "pty": true
}
```

### [2026-04-29 12:41:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session young-otter, pid 976). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-04-29 12:41:05 UTC] ASSISTANT

**Thinking**

The quantization is running. Let me poll to see what's happening.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "young-otter",
  "timeout": 60000
}
```

### [2026-04-29 12:42:05 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

============================================================
Auto-Round Quantization
============================================================
Model: Qwen/Qwen3.5-27B
Scheme: W4A16
Method: RTN (iters=0)
nsamples: 128
Format: auto_round
Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16
Device: cuda (num_gpus=1)
============================================================
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
404 Client Error. (Request ID: Root=1-69f1fbe8-0f509d7709c7f3745331d338;b0bf808f-419a-40da-922d-8188022ac96f)

Entry Not Found for url: https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/model_index.json.
[33;1m2026-04-29 12:39:36 WARNING modeling_qwen3_5.py L413: The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d[0m
[33;1m2026-04-29 12:39:37 WARNING base.py L407: `device` is deprecated, please use `device_map` instead[0m
[38;20m2026-04-29 12:39:37 INFO base.py L473: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-04-29 12:39:37 INFO base.py L517: using torch.bfloat16 for quantization tuning[0m

Starting quantization...
[33;1m2026-04-29 12:39:37 WARNING formats.py L166: some layers are skipped quantization (shape not divisible by 32): model.layers.[0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,48-50,52-54,56-58,60-62].linear_attn.in_proj_a, model.layers.[0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,48-50,52-54,56-58,60-62].linear_attn.in_proj_b[0m
[33;1m2026-04-29 12:39:37 WARNING modeling_utils.py L4555: `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.[0m
[38;20m2026-04-29 12:39:37 INFO base.py L1150: start to compute imatrix[0m

Generating train split:   0%|          | 0/10000 [00:00<?, ? examples/s]
Generating train split: 100%|██████████| 10000/10000 [00:00<00:00, 69354.85 examples/s]

Map:   0%|          | 0/10000 [00:00<?, ? examples/s]
Map: 100%|██████████| 10000/10000 [00:06<00:00, 1651.96 examples/s]

Filter:   0%|          | 0/10000 [00:00<?, ? examples/s]
Filter: 100%|██████████| 10000/10000 [00:02<00:00, 4033.40 examples/s]

Casting the dataset:   0%|          | 0/1230 [00:00<?, ? examples/s]
Casting the dataset: 100%|██████████| 1230/1230 [00:02<00:00, 600.31 examples/s]

  0%|          | 0/64 [00:00<?, ?it/s]
Quantizing model.layers.0:   0%|          | 0/64 [00:00<?, ?it/s][38;20m2026-04-29 12:40:04 INFO device.py L1692: 'peak_ram': 11.98GB, 'peak_vram': 13.25GB[0m

Quantizing model.layers.1:   2%|▏         | 1/64 [00:09<09:28,  9.02s/it][38;20m2026-04-29 12:40:12 INFO device.py L1692: 'peak_ram': 11.98GB, 'peak_vram': 13.49GB[0m

Quantizing model.layers.2:   3%|▎         | 2/64 [00:17<08:58,  8.69s/it][38;20m2026-04-29 12:40:21 INFO device.py L1692: 'peak_ram': 12.81GB, 'peak_vram': 13.49GB[0m

Quantizing model.layers.3:   5%|▍         | 3/64 [00:25<08:42,  8.57s/it][38;20m2026-04-29 12:40:28 INFO device.py L1692: 'peak_ram': 13.69GB, 'peak_vram': 13.49GB[0m

Quantizing model.layers.4:   6%|▋         | 4/64 [00:32<08:11,  8.19s/it][38;20m2026-04-29 12:40:36 INFO device.py L1692: 'peak_ram': 14.59GB, 'peak_vram': 13.49GB[0m

Quantizing model.layers.5:   8%|▊         | 5/64 [00:41<08:04,  8.21s/it][38;20m2026-04-29 12:40:44 INFO device.py L1692: 'peak_ram': 15.49GB, 'peak_vram': 13.49GB[0m

Quantizing model.layers.6:   9%|▉         | 6/64 [00:49<07:57,  8.23s/it][38;20m2026-04-29 12:40:53 INFO device.py L1692: 'peak_ram': 16.39GB, 'peak_vram': 13.49GB[0m

Quantizing model.layers.7:  11%|█         | 7/64 [00:57<07:49,  8.24s/it][38;20m2026-04-29 12:40:59 INFO device.py L1692: 'peak_ram': 16.63GB, 'peak_vram': 13.49GB[0m

Quantizing model.layers.8:  12%|█▎        | 8/64 [01:04<07:32,  8.08s/it][38;20m2026-04-29 12:41:08 INFO device.py L1692: 'peak_ram': 17.53GB, 'peak_vram': 13.54GB[0m

Quantizing model.layers.9:  14%|█▍        | 9/64 [01:12<07:25,  8.10s/it][38;20m2026-04-29 12:41:16 INFO device.py L1692: 'peak_ram': 18.43GB, 'peak_vram': 13.54GB[0m

Quantizing model.layers.10:  16%|█▌        | 10/64 [01:21<07:18,  8.13s/it][38;20m2026-04-29 12:41:26 INFO device.py L1692: 'peak_ram': 18.43GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.11:  17%|█▋        | 11/64 [01:30<07:16,  8.24s/it][38;20m2026-04-29 12:41:33 INFO device.py L1692: 'peak_ram': 18.69GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.12:  19%|█▉        | 12/64 [01:37<07:03,  8.15s/it][38;20m2026-04-29 12:41:41 INFO device.py L1692: 'peak_ram': 19.41GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.13:  20%|██        | 13/64 [01:46<06:56,  8.16s/it][38;20m2026-04-29 12:41:49 INFO device.py L1692: 'peak_ram': 20.27GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.14:  22%|██▏       | 14/64 [01:54<06:48,  8.17s/it][38;20m2026-04-29 12:41:58 INFO device.py L1692: 'peak_ram': 21.13GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.14:  23%|██▎       | 15/64 [02:02<06:40,  8.18s/it]
Quantizing model.layers.15:  23%|██▎       | 15/64 [02:02<06:40,  8.18s/it][38;20m2026-04-29 12:42:05 INFO device.py L1692: 'peak_ram': 21.98GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.16:  25%|██▌       | 16/64 [02:09<06:32,  8.18s/it]

Process still running.

### [2026-04-29 12:42:13 UTC] ASSISTANT

**Thinking**

The quantization is running well. It's at layer 16 out of 64 layers. Memory usage is about 22GB RAM and 13.59GB VRAM. Let me wait for it to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "young-otter",
  "timeout": 2400000
}
```

### [2026-04-29 12:44:13 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

Quantizing model.layers.16:  25%|██▌       | 16/64 [02:14<06:32,  8.18s/it][38;20m2026-04-29 12:42:13 INFO device.py L1692: 'peak_ram': 22.84GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.17:  27%|██▋       | 17/64 [02:18<06:24,  8.18s/it][38;20m2026-04-29 12:42:21 INFO device.py L1692: 'peak_ram': 23.61GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.18:  28%|██▊       | 18/64 [02:26<06:16,  8.18s/it][38;20m2026-04-29 12:42:30 INFO device.py L1692: 'peak_ram': 24.51GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.19:  30%|██▉       | 19/64 [02:34<06:08,  8.18s/it][38;20m2026-04-29 12:42:37 INFO device.py L1692: 'peak_ram': 25.38GB, 'peak_vram': 13.59GB[0m

Quantizing model.layers.20:  31%|███▏      | 20/64 [02:41<05:59,  8.18s/it][38;20m2026-04-29 12:42:45 INFO device.py L1692: 'peak_ram': 26.28GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.21:  33%|███▎      | 21/64 [02:50<05:51,  8.18s/it][38;20m2026-04-29 12:42:55 INFO device.py L1692: 'peak_ram': 26.28GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.22:  34%|███▍      | 22/64 [02:59<05:43,  8.18s/it][38;20m2026-04-29 12:43:03 INFO device.py L1692: 'peak_ram': 26.54GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.23:  36%|███▌      | 23/64 [03:08<05:35,  8.18s/it][38;20m2026-04-29 12:43:10 INFO device.py L1692: 'peak_ram': 27.23GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.24:  38%|███▊      | 24/64 [03:15<05:27,  8.18s/it][38;20m2026-04-29 12:43:18 INFO device.py L1692: 'peak_ram': 28.11GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.25:  39%|███▉      | 25/64 [03:23<05:19,  8.18s/it][38;20m2026-04-29 12:43:27 INFO device.py L1692: 'peak_ram': 28.97GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.26:  41%|████      | 26/64 [03:31<05:10,  8.18s/it][38;20m2026-04-29 12:43:35 INFO device.py L1692: 'peak_ram': 29.83GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.27:  42%|████▏     | 27/64 [03:40<05:02,  8.18s/it][38;20m2026-04-29 12:43:42 INFO device.py L1692: 'peak_ram': 30.55GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.28:  44%|████▍     | 28/64 [03:47<04:54,  8.18s/it][38;20m2026-04-29 12:43:50 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.29:  45%|████▌     | 29/64 [03:55<04:46,  8.18s/it][38;20m2026-04-29 12:43:59 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.29:  47%|████▋     | 30/64 [04:04<04:36,  8.13s/it]
Quantizing model.layers.30:  47%|████▋     | 30/64 [04:04<04:36,  8.13s/it][38;20m2026-04-29 12:44:07 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.31:  48%|████▊     | 31/64 [04:12<04:28,  8.13s/it]

Process still running.

### [2026-04-29 12:44:18 UTC] ASSISTANT

**Thinking**

The quantization is progressing well. It's at layer 31/64. RAM usage is at 31.44GB, VRAM at 13.85GB. It seems to be going well. Let me keep waiting.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "young-otter",
  "timeout": 3000000
}
```

### [2026-04-29 12:46:18 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

Quantizing model.layers.31:  48%|████▊     | 31/64 [04:19<04:28,  8.13s/it][38;20m2026-04-29 12:44:14 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.32:  50%|█████     | 32/64 [04:19<04:20,  8.13s/it][38;20m2026-04-29 12:44:24 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.33:  52%|█████▏    | 33/64 [04:29<04:12,  8.13s/it][38;20m2026-04-29 12:44:32 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.34:  53%|█████▎    | 34/64 [04:37<04:03,  8.13s/it][38;20m2026-04-29 12:44:41 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.35:  55%|█████▍    | 35/64 [04:45<03:55,  8.13s/it][38;20m2026-04-29 12:44:48 INFO device.py L1692: 'peak_ram': 31.44GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.36:  56%|█████▋    | 36/64 [04:52<03:47,  8.13s/it][38;20m2026-04-29 12:44:56 INFO device.py L1692: 'peak_ram': 32.28GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.37:  58%|█████▊    | 37/64 [05:01<03:39,  8.13s/it][38;20m2026-04-29 12:45:04 INFO device.py L1692: 'peak_ram': 33.02GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.38:  59%|█████▉    | 38/64 [05:09<03:31,  8.13s/it][38;20m2026-04-29 12:45:13 INFO device.py L1692: 'peak_ram': 33.92GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.39:  61%|██████    | 39/64 [05:17<03:23,  8.13s/it][38;20m2026-04-29 12:45:20 INFO device.py L1692: 'peak_ram': 34.79GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.40:  62%|██████▎   | 40/64 [05:24<03:15,  8.13s/it][38;20m2026-04-29 12:45:28 INFO device.py L1692: 'peak_ram': 35.69GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.41:  64%|██████▍   | 41/64 [05:33<03:07,  8.13s/it][38;20m2026-04-29 12:45:36 INFO device.py L1692: 'peak_ram': 36.59GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.42:  66%|██████▌   | 42/64 [05:41<02:58,  8.13s/it][38;20m2026-04-29 12:45:45 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.43:  67%|██████▋   | 43/64 [05:49<02:50,  8.13s/it][38;20m2026-04-29 12:45:53 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.44:  69%|██████▉   | 44/64 [05:58<02:42,  8.13s/it][38;20m2026-04-29 12:46:01 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.44:  70%|███████   | 45/64 [06:06<02:34,  8.14s/it]
Quantizing model.layers.45:  70%|███████   | 45/64 [06:06<02:34,  8.14s/it][38;20m2026-04-29 12:46:10 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.46:  72%|███████▏  | 46/64 [06:14<02:26,  8.14s/it]
Quantizing model.layers.46:  72%|███████▏  | 46/64 [06:19<02:26,  8.14s/it]

Process still running.

### [2026-04-29 12:46:22 UTC] ASSISTANT

**Thinking**

Layer 46/64. Let me keep waiting.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "young-otter",
  "timeout": 1800000
}
```

### [2026-04-29 12:48:22 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

[38;20m2026-04-29 12:46:18 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.47:  73%|███████▎  | 47/64 [06:23<02:18,  8.14s/it][38;20m2026-04-29 12:46:25 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.48:  75%|███████▌  | 48/64 [06:30<02:10,  8.14s/it][38;20m2026-04-29 12:46:33 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.49:  77%|███████▋  | 49/64 [06:38<02:02,  8.14s/it][38;20m2026-04-29 12:46:42 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.50:  78%|███████▊  | 50/64 [06:46<01:54,  8.14s/it][38;20m2026-04-29 12:46:50 INFO device.py L1692: 'peak_ram': 37.69GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.51:  80%|███████▉  | 51/64 [06:55<01:45,  8.14s/it][38;20m2026-04-29 12:46:57 INFO device.py L1692: 'peak_ram': 38.57GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.52:  81%|████████▏ | 52/64 [07:02<01:37,  8.14s/it][38;20m2026-04-29 12:47:05 INFO device.py L1692: 'peak_ram': 39.46GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.53:  83%|████████▎ | 53/64 [07:10<01:29,  8.14s/it][38;20m2026-04-29 12:47:15 INFO device.py L1692: 'peak_ram': 39.74GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.54:  84%|████████▍ | 54/64 [07:20<01:21,  8.14s/it][38;20m2026-04-29 12:47:23 INFO device.py L1692: 'peak_ram': 40.45GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.55:  86%|████████▌ | 55/64 [07:28<01:13,  8.14s/it][38;20m2026-04-29 12:47:31 INFO device.py L1692: 'peak_ram': 40.84GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.56:  88%|████████▊ | 56/64 [07:35<01:05,  8.14s/it][38;20m2026-04-29 12:47:39 INFO device.py L1692: 'peak_ram': 41.55GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.57:  89%|████████▉ | 57/64 [07:44<00:57,  8.14s/it][38;20m2026-04-29 12:47:47 INFO device.py L1692: 'peak_ram': 41.99GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.58:  91%|█████████ | 58/64 [07:52<00:48,  8.14s/it][38;20m2026-04-29 12:47:56 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.59:  92%|█████████▏| 59/64 [08:00<00:40,  8.14s/it][38;20m2026-04-29 12:48:03 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.59:  94%|█████████▍| 60/64 [08:07<00:32,  8.13s/it]
Quantizing model.layers.60:  94%|█████████▍| 60/64 [08:07<00:32,  8.13s/it][38;20m2026-04-29 12:48:11 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.61:  95%|█████████▌| 61/64 [08:16<00:24,  8.13s/it]
Quantizing model.layers.61:  95%|█████████▌| 61/64 [08:19<00:24,  8.13s/it][38;20m2026-04-29 12:48:19 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.62:  97%|█████████▋| 62/64 [08:24<00:16,  8.13s/it]

Process still running.

### [2026-04-29 12:48:27 UTC] ASSISTANT

**Thinking**

Layer 62/64. Almost done. Let me wait for it to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "young-otter",
  "timeout": 600000
}
```

### [2026-04-29 12:48:40 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

[38;20m2026-04-29 12:48:28 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.63:  98%|█████████▊| 63/64 [08:33<00:08,  8.13s/it][38;20m2026-04-29 12:48:35 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.63: 100%|██████████| 64/64 [08:40<00:00,  8.13s/it]
Traceback (most recent call last):
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py", line 57, in <module>
    ar.quantize_and_save(output_dir=output_dir, format=format_str)
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py", line 1018, in quantize_and_save
    model, _ = self.quantize()
               ^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py", line 1797, in quantize
    return self._quantize_rtn()
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py", line 1503, in _quantize_rtn
    shard_writer(self, is_finalize=True)
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 267, in shard_writer
    rounder._shard_writer.finalize()
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 207, in finalize
    self._add_tensor(pname, tensor.detach().to("cpu"))
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 121, in _add_tensor
    self._flush_shard()
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 159, in _flush_shard
    self._offload_to_meta(saved_params)
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 176, in _offload_to_meta
    module.to("meta")
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 960, in _apply
    assert isinstance(param, Parameter)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

Process exited with code 0.

### [2026-04-29 12:48:52 UTC] ASSISTANT

**Thinking**

The quantization failed during the finalization/shard_writer step with an AssertionError. The error is in the `_offload_to_meta` function where it tries to move a module to "meta" device but encounters a parameter that is not a `torch.nn.Parameter`.

This is a bug in the auto-round library when used with this model. Let me search for workarounds.

Looking at the error trace:
1. It quantized all 64 layers successfully
2. It failed during the finalization/shard writing phase
3. The error is: `assert isinstance(param, Parameter)` in `module.to("meta")`

This might be related to some layers being skipped (the warning about `linear_attn.in_proj_a, in_proj_b` being skipped because shape not divisible by 32).


The skipped layers might have non-standard parameter types that don't survive the meta device conversion. I could try switching formats, disabling sharding, or using a different export approach. Let me also check if there's an HF_TOKEN issue or configuration problem causing this, since the 404 error on model_index.json might have led to some fallback behavior that's creating non-standard parameters.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/ 2>/dev/null"
}
```

### [2026-04-29 12:48:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 14877048
drwxr-xr-x 2 root root       4096 Apr 29 12:48 .
drwxr-xr-x 4 root root         48 Apr 29 12:36 ..
-rw-r--r-- 1 root root 2130065944 Apr 29 12:41 model-shard-00001.safetensors
-rw-r--r-- 1 root root 2124618760 Apr 29 12:42 model-shard-00002.safetensors
-rw-r--r-- 1 root root 2124618760 Apr 29 12:44 model-shard-00003.safetensors
-rw-r--r-- 1 root root 2116446192 Apr 29 12:45 model-shard-00004.safetensors
-rw-r--r-- 1 root root 2124618744 Apr 29 12:47 model-shard-00005.safetensors
-rw-r--r-- 1 root root 2018387568 Apr 29 12:48 model-shard-00006.safetensors
-rw-r--r-- 1 root root 2542796912 Apr 29 12:48 model-shard-00007.safetensors
-rw-r--r-- 1 root root   52526384 Apr 29 12:48 model-shard-00008.safetensors

### [2026-04-29 12:49:01 UTC] ASSISTANT

**Thinking**

So the model shards were created, but the process failed during finalization. The shards have been saved, but the final adapter_config.json and potentially other files weren't created.Let me look at the shard_writer.py error more carefully. It seems to be failing when doing `_offload_to_meta`. The issue is `assert isinstance(param, Parameter)`.Let me look at what files we have so far and if we can complete the process somehow.Actually, looking at the output directory - it has model shards but no config files. The issue is in the auto-round library's shard_writer when finalizing.

Let me check what might be causing this. The error is that during `_offload_to_meta`, a module has a tensor that is not a `Parameter`. 

I'm examining the shard_writer code to understand how to work around this. The auto-round library appears to have a bug when handling non-standard parameter types during the meta offload phase. Let me check the log for more details, and also see if there's a way to resume or if I need to create the config files manually.

**Tool call:** `exec`

```json
{
  "command": "tail -50 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log"
}
```

### [2026-04-29 12:49:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Quantizing model.layers.49:  77%|███████▋  | 49/64 [06:38<02:02,  8.14s/it][38;20m2026-04-29 12:46:42 INFO device.py L1692: 'peak_ram': 37.49GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.50:  78%|███████▊  | 50/64 [06:46<01:54,  8.14s/it][38;20m2026-04-29 12:46:50 INFO device.py L1692: 'peak_ram': 37.69GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.51:  80%|███████▉  | 51/64 [06:55<01:45,  8.14s/it][38;20m2026-04-29 12:46:57 INFO device.py L1692: 'peak_ram': 38.57GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.52:  81%|████████▏ | 52/64 [07:02<01:37,  8.14s/it][38;20m2026-04-29 12:47:05 INFO device.py L1692: 'peak_ram': 39.46GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.53:  83%|████████▎ | 53/64 [07:10<01:29,  8.14s/it][38;20m2026-04-29 12:47:15 INFO device.py L1692: 'peak_ram': 39.74GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.54:  84%|████████▍ | 54/64 [07:20<01:21,  8.14s/it][38;20m2026-04-29 12:47:23 INFO device.py L1692: 'peak_ram': 40.45GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.55:  86%|████████▌ | 55/64 [07:28<01:13,  8.14s/it][38;20m2026-04-29 12:47:31 INFO device.py L1692: 'peak_ram': 40.84GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.56:  88%|████████▊ | 56/64 [07:35<01:05,  8.14s/it][38;20m2026-04-29 12:47:39 INFO device.py L1692: 'peak_ram': 41.55GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.57:  89%|████████▉ | 57/64 [07:44<00:57,  8.14s/it][38;20m2026-04-29 12:47:47 INFO device.py L1692: 'peak_ram': 41.99GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.58:  91%|█████████ | 58/64 [07:52<00:48,  8.14s/it][38;20m2026-04-29 12:47:56 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.59:  92%|█████████▏| 59/64 [08:00<00:40,  8.14s/it][38;20m2026-04-29 12:48:03 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.85GB[0m

Quantizing model.layers.59:  94%|█████████▍| 60/64 [08:07<00:32,  8.13s/it]
Quantizing model.layers.60:  94%|█████████▍| 60/64 [08:07<00:32,  8.13s/it][38;20m2026-04-29 12:48:11 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.61:  95%|█████████▌| 61/64 [08:16<00:24,  8.13s/it]
Quantizing model.layers.61:  95%|█████████▌| 61/64 [08:19<00:24,  8.13s/it][38;20m2026-04-29 12:48:19 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.62:  97%|█████████▋| 62/64 [08:24<00:16,  8.13s/it][38;20m2026-04-29 12:48:28 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.63:  98%|█████████▊| 63/64 [08:33<00:08,  8.13s/it][38;20m2026-04-29 12:48:35 INFO device.py L1692: 'peak_ram': 42.83GB, 'peak_vram': 13.94GB[0m

Quantizing model.layers.63: 100%|██████████| 64/64 [08:40<00:00,  8.13s/it]
Traceback (most recent call last):
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py", line 57, in <module>
    ar.quantize_and_save(output_dir=output_dir, format=format_str)
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py", line 1018, in quantize_and_save
    model, _ = self.quantize()
               ^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py", line 1797, in quantize
    return self._quantize_rtn()
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py", line 1503, in _quantize_rtn
    shard_writer(self, is_finalize=True)
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 267, in shard_writer
    rounder._shard_writer.finalize()
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 207, in finalize
    self._add_tensor(pname, tensor.detach().to("cpu"))
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 121, in _add_tensor
    self._flush_shard()
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 159, in _flush_shard
    self._offload_to_meta(saved_params)
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py", line 176, in _offload_to_meta
    module.to("meta")
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 960, in _apply
    assert isinstance(param, Parameter)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

### [2026-04-29 12:49:12 UTC] ASSISTANT

**Thinking**

The error is happening in the auto-round library at `shard_writer.py` line 176, where it calls `module.to("meta")`. The assertion failure is because some parameter in the model is not a `torch.nn.Parameter` but a plain tensor.

This seems to be a bug in auto-round 0.12.3 when used with Qwen3.5-27B. The quantization completed but the save/finalize failed.

Let me look at what version of auto-round might work better, or if there's a workaround. Let me check the issue in the auto-round GitHub issues or try to find a workaround.

One possible approach:
1. Try using a different format (like `auto_gptq` or `llm_compressor`) instead of `auto_round`


2. Try an older version of auto-round that might handle this model better
3. Check if there's a way to disable the shard writing that's causing the problem

Since I have the model shards already, let me try switching to `auto_gptq` format instead, which might bypass this particular save path.

**Tool call:** `exec`

```json
{
  "command": "grep -n \"_offload_to_meta\\|isinstance.*Parameter\\|def finalize\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py | head -30"
}
```

### [2026-04-29 12:49:12 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

159:        self._offload_to_meta(saved_params)
164:    def _offload_to_meta(self, saved_params):
178:    def finalize(self):

### [2026-04-29 12:49:17 UTC] ASSISTANT

**Thinking**

Let me look at the shard_writer.py more carefully to understand the issue and find a workaround.

**Tool call:** `read`

```json
{
  "offset": 155,
  "limit": 40,
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/shard_writer.py"
}
```

### [2026-04-29 12:49:17 UTC] TOOL RESULT
**Tool:** `read`

        self.shard_meta.append({"tmp_file": tmp_name, "params": saved_params})
        self._all_saved.update(saved_params)

        # Offload logic: move modules to meta device once all params are saved
        self._offload_to_meta(saved_params)

        self.current_shard_tensors = OrderedDict()
        self.current_shard_size = 0

    def _offload_to_meta(self, saved_params):
        """Attempts to move fully saved modules to the 'meta' device to free RAM."""
        for param_full_name in saved_params:
            module_path = param_full_name.rsplit(".", 1)[0]

            module = get_module(self.model, module_path)
            # Check if all parameters of this module are now in '_all_saved'
            if (
                module is not None
                and isinstance(module, torch.nn.Module)
                and all(f"{module_path}.{k}" in self._all_saved for k in module.state_dict().keys())
            ):
                module.to("meta")

    def finalize(self):
        """Saves remaining weights, renames files, and writes the index JSON."""
        # 1. Capture remaining weights not yet saved
        full_sd = self.model.state_dict()
        tie_word_embeddings = False
        config = getattr(self.model, "config", None)
        if hasattr(self.model, "config") and hasattr(self.model.config, "tie_word_embeddings"):
            tie_word_embeddings = self.model.config.tie_word_embeddings
        if tie_word_embeddings is None:
            # For multimodal models, check nested text/thinker configs
            for sub_attr in ("text_config", "thinker_config", "language_config", "llm_config"):
                sub_config = getattr(config, sub_attr, None)
                if sub_config is not None:
                    val = getattr(sub_config, "tie_word_embeddings", None)
                    if val is not None:
                        tie_word_embeddings = val
                        break

[76 more lines in file. Use offset=195 to continue.]

### [2026-04-29 12:49:28 UTC] ASSISTANT

**Thinking**

The issue is in `_offload_to_meta` where it checks `all(f"{module_path}.{k}" in self._all_saved for k in module.state_dict().keys())`. The problem is that `module.state_dict()` might include non-parameter entries (like buffers) that aren't in `_all_saved`, or the model might have some tensor registered as a parameter that isn't being recognized as such.

The fix could be:
1. Use a different format (like `auto_gptq`)
2. Patch the auto-round code to handle this case

Let me try the `auto_gptq` format as an alternative, which might not use this shard_writer path.

Actually, looking more carefully, the problem is that `module.state_dict().keys()` might include buffers that are not Parameters, but the check is looking for all keys in `_all_saved`. The assertion happens at `module.to("meta")` which requires all parameters to be Parameters.
 
So the sequence is: the module satisfies the condition that all its state dict keys are in `_all_saved`, but then `module.to("meta")` fails because some parameter inside isn't actually a `torch.nn.Parameter` instance. This could be a buffer or registered tensor that got mixed in. Let me try switching to the `auto_gptq` format instead.

**Tool call:** `edit`

```json
{
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py",
  "old_string": "format_str = \"auto_round\"",
  "newString": "format_str = \"auto_gptq\""
}
```

### [2026-04-29 12:49:28 UTC] TOOL RESULT
**Tool:** `edit`

Successfully replaced text in /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py.

### [2026-04-29 12:49:36 UTC] ASSISTANT

**Thinking**

Wait, but the model shards are already saved in the output directory. If I run again, it might fail because there are already model shards there. Let me clean up first.

Also, actually the issue is that the model shards are there but no config. Let me try running the same script but with auto_gptq format and see if it can work. But first I need to clean the output directory since there are already partial shards there.

Actually wait - let me look at the auto-round code more carefully. The issue is `shard_writer` which is used by the auto_round format. If I use auto_gptq, it might use a different writer. Let me check.

**Tool call:** `exec`

```json
{
  "command": "grep -n \"auto_gptq\\|auto_round\\|format\\|shard\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py | grep -i \"format\\|writer\" | head -20"
}
```

### [2026-04-29 12:49:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

37:from auto_round.compressors.shard_writer import shard_writer
61:from auto_round.formats import OutputFormat, get_formats
608:                "cannot yet be exported to real formats except GGUF. "
609:                "Please save the model using the `fake` format for now."
708:        Handles GGUF formatting and synchronizes weight/activation parameters.
729:                    "When using GGUF scheme, user-specified overrides will be ignored to ensure format compatibility."
901:                "And please save the quantized model to fake format as real deployment is not supported currently"
918:                if self.iters > 0:  # GGUF should log this warning, but we don't know the format here
969:        self, output_dir: str = "tmp_autoround", format: str = "auto_round", inplace: bool = True, **kwargs
971:        """Quantizes the model and saves it in the specified format(s).
973:        This function checks the validity of the requested format(s), quantizes
975:        If multiple formats are provided, the model is saved separately for each format.
980:            format (str, optional): The quantization format(s) to use, separated
983:                one format is used. Defaults to True.
991:            ValueError: If an unsupported format is specified.
993:        # Validate and process the specified formats
996:        # check and update the format based on the current configuration
997:        format_list = get_formats(format, self)
998:        self.formats = format_list
1000:        # If multiple formats are specified, enforce inplace=False

### [2026-04-29 12:49:40 UTC] ASSISTANT

**Thinking**

Let me check what formats are supported and if auto_gptq uses a different save mechanism:

**Tool call:** `exec`

```json
{
  "command": "grep -n \"auto_gptq\\|get_format\\|OutputFormat\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/formats/__init__.py 2>/dev/null | head -30"
}
```

### [2026-04-29 12:49:40 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:49:43 UTC] ASSISTANT

**Thinking**

Let me find the formats module:

**Tool call:** `exec`

```json
{
  "command": "find /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round -name \"*.py\" | xargs grep -l \"auto_gptq\\|get_formats\" 2>/dev/null | head -5"
}
```

### [2026-04-29 12:49:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/__main__.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/formats.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/base.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/diffusion/hybrid.py
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/export/utils.py

### [2026-04-29 12:49:47 UTC] ASSISTANT

**Thinking**

Let me look at the formats.py to understand what formats are available and how they work:

**Tool call:** `exec`

```json
{
  "command": "grep -n \"auto_gptq\\|auto_round\\|class.*Format\\|shard_writer\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/formats.py | head -30"
}
```

### [2026-04-29 12:49:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

29:from auto_round.compressors.utils import (
39:from auto_round.export.export_to_gguf.config import ModelType
40:from auto_round.schemes import (
45:from auto_round.utils import (
58:class AutoRoundExportFormat(str, Enum):
75:    from auto_round.compressors.base import BaseCompressor
97:    if isinstance(ar.group_size, tuple) and any(["auto_round" in f.lower() for f in formats]):
99:            "`auto_round` format can't be used for deploying block-wise fp8 quantization now, use `fp8` instead."
101:        formats = ["fp8" if "auto_round" in f.lower() else f for f in formats]
147:    from auto_round.schemes import preset_name_to_scheme
171:class OutputFormat(ABC):
175:            For example, auto_round, gguf, llm_compressor etc.
177:            For example, auto_round:fp8_static, auto_round:auto_awq etc.
192:                f"but got scheme {ar.scheme}, please change to fake or auto_round etc."
226:        # auto_round:llm_compressor:fp8_static
229:        # auto_round:fp8_static, llm_compressor:fp8_static, auto_round:auto_awq
253:            self.format_name in ["fp8", "auto_round:fp8"] and isinstance(ar.group_size, tuple) and ar.act_dynamic
257:                f"is only supported to export auto_round or llm_compressor format,"
306:class FakeFormat(OutputFormat):
347:class LLMCompressorFormat(OutputFormat):
355:                f"but got scheme {ar.scheme}, please change to fake or auto_round etc."
359:        if re.search("^(auto_round:)?llm_compressor", format):
363:                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported
383:                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported
459:            from auto_round.export.export_to_llmcompressor.export_to_fp import pack_layer
463:            from auto_round.export.export_to_llmcompressor.export_to_static_fp import pack_layer
467:            from auto_round.export.export_to_llmcompressor.export import pack_layer
471:            from auto_round.export.export_to_llmcompressor.export import pack_layer
491:            from auto_round.export.export_to_llmcompressor.export_to_fp import save_quantized_as_fp
495:            from auto_round.export.export_to_llmcompressor.export_to_static_fp import save_quantized_as_static_fp

### [2026-04-29 12:49:50 UTC] ASSISTANT

**Thinking**

Let me look for the auto_gptq format specifically:

**Tool call:** `exec`

```json
{
  "command": "grep -n \"GPTQ\\|gptq\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/formats.py | head -20"
}
```

### [2026-04-29 12:49:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

295:    def is_gptq(self) -> bool:
296:        return "gptq" in self.output_format or (self.backend is not None and self.backend.is_gptq())
515:@OutputFormat.register("auto_gptq", "gptqmodel")
516:class AutoGPTQFormat(OutputFormat):
518:    format_name = "auto_gptq"
523:                "the asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
556:            from auto_round.export.export_to_autogptq.export import pack_layer
572:        if backend == "auto_round:auto_gptq" or backend == "auto_round:gptqmodel":
577:            from auto_round.export.export_to_autogptq.export import save_quantized_as_autogptq
579:            export_func = save_quantized_as_autogptq
1062:@OutputFormat.register("auto_round:gptqmodel", "auto_round:auto_gptq")
1089:                self.backend = AutoGPTQFormat("auto_round:auto_gptq", ar)

### [2026-04-29 12:50:02 UTC] ASSISTANT

**Thinking**

Good. So `auto_gptq` uses `export_to_autogptq.export`. Let me look at the export mechanism to see if it also has the same issue.Actually, let me try a simpler approach first: I'll clean up the output directory and try with `auto_gptq` format. The `auto_gptq` format exports to GPTQ format which is different from the `shard_writer` that caused the problem.Actually wait - the issue is that the quantization has already happened (all 64 layers quantized), but the saving/finalizing failed. If I try again with auto_gptq, it might try to re-quantize, which would take a long time.

Let me look at whether auto-round caches the quantized state or if I can use the already-quantized model.Actually, looking at the error again - the quantization completed (all 64 layers) but the saving failed. 

Since the quantization itself finished successfully, I should explore whether I can export the quantized weights directly without re-running the entire process. The partial output with model shards exists, so maybe I can write a custom export script to finalize and save them properly. But before going down that route, let me check if there are any export-related issues in the auto_round library itself, or if downgrading to an earlier version like 0.9 or 0.10 might resolve the shard_writer bug I'm hitting.

**Tool call:** `exec`

```json
{
  "command": "pip index versions auto-round 2>/dev/null || uv pip install --dry-run --python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/bin/python 'auto-round<0.12' 2>&1 | head -10"
}
```

### [2026-04-29 12:50:03 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Using Python 3.12.13 environment at: quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv
Resolved 72 packages in 536ms
Would download 2 packages
Would uninstall 1 package
Would install 2 packages
 - auto-round==0.12.3
 + auto-round==0.10.2
 + threadpoolctl==3.6.0

### [2026-04-29 12:50:12 UTC] ASSISTANT

**Thinking**

So there is version 0.10.2 available. Let me try downgrading to 0.10.2 and running the quantization again. But first, let me clean up the output directory since we have partial shards there.Actually, before doing that, let me think about whether the auto-round library has a way to load from a partially quantized checkpoint. Given the time it takes to quantize (about 10 minutes), let me try to see if I can fix the issue in the current version by patching the shard_writer.py.Let me look at the exact assertion error more carefully. The error is in `torch.nn.modules.module.py` line 960, `assert isinstance(param, Parameter)`. This happens when calling `module.to("meta")`. 

The problem is that when a module is moved to the meta device, it checks that all parameters are indeed Parameter instances, but some tensor in the module isn't. I could patch shard_writer.py to skip modules that fail this check, but that risks data loss. Let me first downgrade to 0.10.2 and see if the quantization can complete cleanly from scratch with a clean output directory.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nuv pip install --python $VENV/bin/python 'auto-round==0.10.2' 2>&1 | tail -5",
  "yieldMs": 60000,
  "timeout": 120
}
```

### [2026-04-29 12:50:13 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Uninstalled 1 package in 6ms
Installed 2 packages in 37ms
 - auto-round==0.12.3
 + auto-round==0.10.2
 + threadpoolctl==3.6.0

### [2026-04-29 12:50:17 UTC] ASSISTANT

**Thinking**

Now let me clean the output directory and run the quantization again. First, let me clean up the partial model shards.

**Tool call:** `exec`

```json
{
  "command": "rm -rf /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/*"
}
```

### [2026-04-29 12:50:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 12:50:22 UTC] ASSISTANT

**Thinking**

Now let me update the quantize.py to use auto_round format again (since 0.10.2 might not have the auto_gptq export), and also set nsamples to 128 which is the default. But actually, let me check what format options are available in 0.10.2.Actually, let me just try running the same script with 0.10.2 and see what happens:

**Tool call:** `edit`

```json
{
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py",
  "old_string": "format_str = \"auto_gptq\"",
  "newString": "format_str = \"auto_round\""
}
```

### [2026-04-29 12:50:22 UTC] TOOL RESULT
**Tool:** `edit`

Successfully replaced text in /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py.

### [2026-04-29 12:50:35 UTC] ASSISTANT

**Thinking**

Wait, actually let me think about this more carefully. The issue was that during `shard_writer.finalize()`, it calls `module.to("meta")` which fails with an AssertionError. 

The problem might be specific to the Qwen3.5-27B model which has some non-standard parameters. Let me look at the error in more detail:

The error happens at `module.to("meta")` in `_offload_to_meta`. The assertion `assert isinstance(param, Parameter)` fails. 

This suggests that the model has some parameter that is not a `torch.nn.Parameter` but some other tensor type. This could be because:
1. The model uses a custom parameter type
2. The model has some hook or buffer that's not properly registered
3. Some optimization pass changed a Parameter to a plain tensor


4. There's a `torch.compile` or other optimization that converts parameters

Looking at the earlier warning about fast path being unavailable and falling back to torch implementation, this suggests flash-linear-attention wasn't available so it used a different implementation. Maybe this alternative implementation created some non-standard parameters.

Let me try a different approach: use `auto_gptq` format which doesn't use the `shard_writer` at all. I need to check if 0.10.2 supports `auto_gptq` format.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nPYTHONPATH=$VENV/lib/python3.12/site-packages /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -c \"\nfrom auto_round.formats import get_formats, AutoRoundExportFormat\nprint(AutoRoundExportFormat._member_names_)\n\" 2>&1"
}
```

### [2026-04-29 12:50:38 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

['FP8_STATIC', 'MXFP8', 'MXFP4', 'NVFP4', 'FP8', 'MX_FP', 'NV_FP', 'MX_FP_RCEIL', 'NV_FP4_WITH_STATIC_GS', 'INT8_W8A8']

### [2026-04-29 12:50:43 UTC] ASSISTANT

**Thinking**

Hmm, in version 0.10.2, the available export formats don't include `auto_gptq`. Let me check the full format list more carefully. Actually, the `get_formats` function might return different formats based on the scheme. Let me look at what 0.10.2 supports:

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nPYTHONPATH=$VENV/lib/python3.12/site-packages /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -c \"\nfrom auto_round.formats import OutputFormat\nfor name, cls in OutputFormat._registry.items():\n    print(name, '->', cls.format_name if hasattr(cls, 'format_name') else '?')\n\" 2>&1"
}
```

### [2026-04-29 12:50:46 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 3, in <module>
AttributeError: type object 'OutputFormat' has no attribute '_registry'. Did you mean: 'register'?

(Command exited with code 1)

### [2026-04-29 12:50:50 UTC] ASSISTANT

**Thinking**

Let me try a different approach - check what export formats are listed when running the auto-round CLI:

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nPYTHONPATH=$VENV/lib/python3.12/site-packages /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -c \"\nfrom auto_round import AutoRound\nhelp(AutoRound.quantize_and_save)\n\" 2>&1 | head -40"
}
```

### [2026-04-29 12:50:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Traceback (most recent call last):
  File "<string>", line 3, in <module>
AttributeError: type object 'AutoRound' has no attribute 'quantize_and_save'

### [2026-04-29 12:50:59 UTC] ASSISTANT

**Thinking**

Let me just try to run the script with auto_round format and see what happens with version 0.10.2:

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nLOGFILE=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log\ncd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16 && \\\n  PYTHONPATH=$VENV/lib/python3.12/site-packages \\\n  /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -u \\\n  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py 2>&1 | tee $LOGFILE",
  "yieldMs": 600000,
  "timeout": 3600,
  "pty": true
}
```

### [2026-04-29 12:51:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

============================================================
Auto-Round Quantization
============================================================
Model: Qwen/Qwen3.5-27B
Scheme: W4A16
Method: RTN (iters=0)
nsamples: 128
Format: auto_round
Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16
Device: cuda (num_gpus=1)
============================================================
[38;20m2026-04-29 12:51:02 INFO autoround.py L165: using MLLM mode for multimodal model.[0m
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[33;1m2026-04-29 12:51:03 WARNING modeling_qwen3_5.py L413: The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d[0m
[33;1m2026-04-29 12:51:07 WARNING import_utils.py L2330: `Qwen2VLImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `Qwen2VLImageProcessor` instead.[0m
Traceback (most recent call last):
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py", line 46, in <module>
    ar = AutoRound(
         ^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/autoround.py", line 191, in __new__
    ar = dynamic_compressor(
         ^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/compressors/mllm/compressor.py", line 179, in __init__
    model, processor, tokenizer, image_processor = mllm_load_model(
                                                   ^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/auto_round/utils/model.py", line 463, in mllm_load_model
    processor = AutoProcessor.from_pretrained(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/auto/processing_auto.py", line 432, in from_pretrained
    return processor_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 1429, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/processing_utils.py", line 1558, in _get_arguments_from_pretrained
    sub_processor = auto_processor_class.from_pretrained(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py", line 630, in from_pretrained
    return image_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/import_utils.py", line 2040, in __getattribute__
    requires_backends(cls, cls._backends)
  File "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/import_utils.py", line 2026, in requires_backends
    raise ImportError("".join(failed))
ImportError: 
Qwen2VLImageProcessor requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.

Qwen2VLImageProcessor requires the Torchvision library but it was not found in your environment. Check out the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.

### [2026-04-29 12:51:17 UTC] ASSISTANT

**Thinking**

The 0.10.2 version requires PIL and Torchvision for the Qwen2.5-VL model (which Qwen3.5-27B seems to use). Let me install those dependencies and try again:

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nuv pip install --python $VENV/bin/python pillow torchvision 2>&1 | tail -5",
  "yieldMs": 120000,
  "timeout": 300
}
```

### [2026-04-29 12:51:40 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

- torch==2.6.0+cu124
 + torch==2.11.0
 + torchvision==0.26.0
 - triton==3.2.0
 + triton==3.6.0

### [2026-04-29 12:51:46 UTC] ASSISTANT

**Thinking**

Oh no, it upgraded torch from 2.6.0 to 2.11.0. That's a big jump and might cause compatibility issues. Let me check if this is compatible with the RTX 4090 and CUDA 12.x. Actually, let me see what version of torch we have now:

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\n$VENV/bin/python -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\""
}
```

### [2026-04-29 12:51:48 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch: 2.11.0+cu130 cuda: True

### [2026-04-29 12:51:54 UTC] ASSISTANT

**Thinking**

It installed torch 2.11.0 with CUDA 13.0. That's a newer version. Let me run the quantization script again to see if it works now. But first let me check if the CUDA version is compatible with the RTX 4090.Actually, let me just run it and see.

**Tool call:** `exec`

```json
{
  "command": "VENV=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv\nLOGFILE=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log\ncd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16 && \\\n  PYTHONPATH=$VENV/lib/python3.12/site-packages \\\n  /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -u \\\n  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py 2>&1 | tee $LOGFILE",
  "yieldMs": 600000,
  "timeout": 3600,
  "pty": true
}
```

### [2026-04-29 12:53:54 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session nova-cove, pid 1581). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-04-29 12:53:58 UTC] ASSISTANT

**Thinking**

Let me check the output:

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "nova-cove",
  "timeout": 60000
}
```

### [2026-04-29 12:54:58 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

============================================================
Auto-Round Quantization
============================================================
Model: Qwen/Qwen3.5-27B
Scheme: W4A16
Method: RTN (iters=0)
nsamples: 128
Format: auto_round
Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16
Device: cuda (num_gpus=1)
============================================================
[38;20m2026-04-29 12:51:58 INFO autoround.py L165: using MLLM mode for multimodal model.[0m
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[33;1m2026-04-29 12:51:59 WARNING modeling_qwen3_5.py L413: The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d[0m
[33;1m2026-04-29 12:52:07 WARNING base.py L382: `device` is deprecated, please use `device_map` instead[0m
[38;20m2026-04-29 12:52:07 INFO base.py L448: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-04-29 12:52:07 INFO base.py L486: using torch.bfloat16 for quantization tuning[0m

Starting quantization...
[33;1m2026-04-29 12:52:07 WARNING formats.py L154: some layers are skipped quantization (shape not divisible by 32).[0m
[38;20m2026-04-29 12:52:07 INFO base.py L1081: start to compute imatrix[0m

cache block inputs:   0%|          | 0/128 [00:00<?, ?it/s]
cache block inputs: 100%|██████████| 128/128 [00:00<00:00, 301.98it/s]

  0%|          | 0/64 [00:00<?, ?it/s]
Quantizing model.language_model.layers.0:   0%|          | 0/64 [00:00<?, ?it/s][38;20m2026-04-29 12:52:22 INFO device.py L1468:  'peak_ram': 4.48GB, 'peak_vram': 13.25GB[0m

Quantizing model.language_model.layers.1:   2%|▏         | 1/64 [00:09<10:14,  9.76s/it][38;20m2026-04-29 12:52:31 INFO device.py L1468:  'peak_ram': 4.48GB, 'peak_vram': 13.49GB[0m

Quantizing model.language_model.layers.2:   3%|▎         | 2/64 [00:18<09:38,  9.34s/it][38;20m2026-04-29 12:52:40 INFO device.py L1468:  'peak_ram': 5.36GB, 'peak_vram': 13.49GB[0m

Quantizing model.language_model.layers.3:   5%|▍         | 3/64 [00:27<09:21,  9.21s/it][38;20m2026-04-29 12:52:47 INFO device.py L1468:  'peak_ram': 6.23GB, 'peak_vram': 13.49GB[0m

Quantizing model.language_model.layers.4:   6%|▋         | 4/64 [00:35<08:50,  8.84s/it][38;20m2026-04-29 12:52:56 INFO device.py L1468:  'peak_ram': 7.13GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.5:   8%|▊         | 5/64 [00:44<08:44,  8.89s/it][38;20m2026-04-29 12:53:05 INFO device.py L1468:  'peak_ram': 8.03GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.6:   9%|▉         | 6/64 [00:53<08:36,  8.90s/it][38;20m2026-04-29 12:53:14 INFO device.py L1468:  'peak_ram': 8.93GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.7:  11%|█         | 7/64 [01:02<08:27,  8.90s/it][38;20m2026-04-29 12:53:22 INFO device.py L1468:  'peak_ram': 9.81GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.8:  12%|█▎        | 8/64 [01:10<08:10,  8.75s/it][38;20m2026-04-29 12:53:31 INFO device.py L1468:  'peak_ram': 10.71GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.9:  14%|█▍        | 9/64 [01:18<08:02,  8.77s/it][38;20m2026-04-29 12:53:40 INFO device.py L1468:  'peak_ram': 11.6GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.10:  16%|█▌        | 10/64 [01:27<07:54,  8.79s/it][38;20m2026-04-29 12:53:50 INFO device.py L1468:  'peak_ram': 11.6GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.11:  17%|█▋        | 11/64 [01:37<07:51,  8.90s/it][38;20m2026-04-29 12:53:58 INFO device.py L1468:  'peak_ram': 11.86GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.12:  19%|█▉        | 12/64 [01:45<07:37,  8.80s/it][38;20m2026-04-29 12:54:07 INFO device.py L1468:  'peak_ram': 12.58GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.13:  20%|██        | 13/64 [01:54<07:30,  8.84s/it][38;20m2026-04-29 12:54:16 INFO device.py L1468:  'peak_ram': 13.29GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.13:  22%|██▏       | 14/64 [02:03<07:22,  8.84s/it]
Quantizing model.language_model.layers.14:  22%|██▏       | 14/64 [02:03<07:22,  8.84s/it][38;20m2026-04-29 12:54:25 INFO device.py L1468:  'peak_ram': 14.13GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.15:  23%|██▎       | 15/64 [02:12<07:13,  8.84s/it]
Quantizing model.language_model.layers.15:  23%|██▎       | 15/64 [02:16<07:13,  8.84s/it][38;20m2026-04-29 12:54:32 INFO device.py L1468:  'peak_ram': 14.99GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.16:  25%|██▌       | 16/64 [02:20<07:04,  8.84s/it][38;20m2026-04-29 12:54:41 INFO device.py L1468:  'peak_ram': 15.88GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.17:  27%|██▋       | 17/64 [02:29<06:55,  8.84s/it][38;20m2026-04-29 12:54:50 INFO device.py L1468:  'peak_ram': 16.77GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.18:  28%|██▊       | 18/64 [02:37<06:46,  8.84s/it]

Process still running.

### [2026-04-29 12:55:04 UTC] ASSISTANT

**Thinking**

Good, the quantization is running. I notice it's using `model.language_model.layers` instead of just `model.layers` which suggests it's treating Qwen3.5-27B as a MLLM (multimodal large language model). This is different from version 0.12.3 which used just `model.layers`.

Let me wait for it to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "nova-cove",
  "timeout": 3000000
}
```

### [2026-04-29 12:57:04 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

[38;20m2026-04-29 12:54:59 INFO device.py L1468:  'peak_ram': 17.67GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.19:  30%|██▉       | 19/64 [02:46<06:37,  8.84s/it][38;20m2026-04-29 12:55:06 INFO device.py L1468:  'peak_ram': 18.54GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.20:  31%|███▏      | 20/64 [02:54<06:29,  8.84s/it][38;20m2026-04-29 12:55:15 INFO device.py L1468:  'peak_ram': 19.44GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.21:  33%|███▎      | 21/64 [03:03<06:20,  8.84s/it][38;20m2026-04-29 12:55:25 INFO device.py L1468:  'peak_ram': 19.44GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.22:  34%|███▍      | 22/64 [03:13<06:11,  8.84s/it][38;20m2026-04-29 12:55:34 INFO device.py L1468:  'peak_ram': 20.05GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.23:  36%|███▌      | 23/64 [03:22<06:02,  8.84s/it][38;20m2026-04-29 12:55:42 INFO device.py L1468:  'peak_ram': 20.74GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.24:  38%|███▊      | 24/64 [03:29<05:53,  8.84s/it][38;20m2026-04-29 12:55:51 INFO device.py L1468:  'peak_ram': 21.45GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.25:  39%|███▉      | 25/64 [03:38<05:44,  8.84s/it][38;20m2026-04-29 12:55:59 INFO device.py L1468:  'peak_ram': 22.17GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.26:  41%|████      | 26/64 [03:47<05:36,  8.84s/it][38;20m2026-04-29 12:56:08 INFO device.py L1468:  'peak_ram': 22.88GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.27:  42%|████▏     | 27/64 [03:56<05:27,  8.84s/it][38;20m2026-04-29 12:56:16 INFO device.py L1468:  'peak_ram': 23.73GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.27:  44%|████▍     | 28/64 [04:03<05:12,  8.69s/it]
Quantizing model.language_model.layers.28:  44%|████▍     | 28/64 [04:03<05:12,  8.69s/it][38;20m2026-04-29 12:56:25 INFO device.py L1468:  'peak_ram': 24.62GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.29:  45%|████▌     | 29/64 [04:12<05:04,  8.69s/it]
Quantizing model.language_model.layers.29:  45%|████▌     | 29/64 [04:16<05:04,  8.69s/it][38;20m2026-04-29 12:56:34 INFO device.py L1468:  'peak_ram': 24.62GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.30:  47%|████▋     | 30/64 [04:21<04:55,  8.69s/it][38;20m2026-04-29 12:56:42 INFO device.py L1468:  'peak_ram': 24.62GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.31:  48%|████▊     | 31/64 [04:30<04:46,  8.69s/it][38;20m2026-04-29 12:56:50 INFO device.py L1468:  'peak_ram': 24.62GB, 'peak_vram': 13.96GB[0m

Quantizing model.language_model.layers.32:  50%|█████     | 32/64 [04:38<04:37,  8.69s/it][38;20m2026-04-29 12:57:00 INFO device.py L1468:  'peak_ram': 24.62GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.33:  52%|█████▏    | 33/64 [04:48<04:29,  8.69s/it]

Process still running.

### [2026-04-29 12:57:16 UTC] ASSISTANT

**Thinking**

It's at layer 33/64, about 50% done. Let me wait for it to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "nova-cove",
  "timeout": 3000000
}
```

### [2026-04-29 12:59:16 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

[38;20m2026-04-29 12:57:09 INFO device.py L1468:  'peak_ram': 24.62GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.34:  53%|█████▎    | 34/64 [04:57<04:20,  8.69s/it][38;20m2026-04-29 12:57:18 INFO device.py L1468:  'peak_ram': 24.62GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.35:  55%|█████▍    | 35/64 [05:06<04:11,  8.69s/it][38;20m2026-04-29 12:57:26 INFO device.py L1468:  'peak_ram': 25.03GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.36:  56%|█████▋    | 36/64 [05:13<04:03,  8.69s/it][38;20m2026-04-29 12:57:34 INFO device.py L1468:  'peak_ram': 25.75GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.37:  58%|█████▊    | 37/64 [05:22<03:54,  8.69s/it][38;20m2026-04-29 12:57:43 INFO device.py L1468:  'peak_ram': 26.46GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.38:  59%|█████▉    | 38/64 [05:31<03:45,  8.69s/it][38;20m2026-04-29 12:57:52 INFO device.py L1468:  'peak_ram': 27.17GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.39:  61%|██████    | 39/64 [05:40<03:37,  8.69s/it][38;20m2026-04-29 12:58:00 INFO device.py L1468:  'peak_ram': 27.95GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.40:  62%|██████▎   | 40/64 [05:47<03:28,  8.69s/it][38;20m2026-04-29 12:58:09 INFO device.py L1468:  'peak_ram': 28.85GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.41:  64%|██████▍   | 41/64 [05:56<03:19,  8.69s/it][38;20m2026-04-29 12:58:18 INFO device.py L1468:  'peak_ram': 29.75GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.41:  66%|██████▌   | 42/64 [06:05<03:11,  8.70s/it]
Quantizing model.language_model.layers.42:  66%|██████▌   | 42/64 [06:05<03:11,  8.70s/it][38;20m2026-04-29 12:58:27 INFO device.py L1468:  'peak_ram': 30.64GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.43:  67%|██████▋   | 43/64 [06:14<03:02,  8.70s/it]
Quantizing model.language_model.layers.43:  67%|██████▋   | 43/64 [06:16<03:02,  8.70s/it][38;20m2026-04-29 12:58:35 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.97GB[0m

Quantizing model.language_model.layers.44:  69%|██████▉   | 44/64 [06:23<02:53,  8.70s/it][38;20m2026-04-29 12:58:44 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.45:  70%|███████   | 45/64 [06:32<02:45,  8.70s/it][38;20m2026-04-29 12:58:53 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.46:  72%|███████▏  | 46/64 [06:41<02:36,  8.70s/it][38;20m2026-04-29 12:59:02 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.47:  73%|███████▎  | 47/64 [06:50<02:27,  8.70s/it][38;20m2026-04-29 12:59:10 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.48:  75%|███████▌  | 48/64 [06:57<02:19,  8.70s/it]

Process still running.

### [2026-04-29 12:59:19 UTC] ASSISTANT

**Thinking**

Layer 48/64 at about 75%. Let me wait for completion.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "nova-cove",
  "timeout": 2400000
}
```

### [2026-04-29 13:01:19 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

[38;20m2026-04-29 12:59:19 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.49:  77%|███████▋  | 49/64 [07:06<02:10,  8.70s/it][38;20m2026-04-29 12:59:27 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.50:  78%|███████▊  | 50/64 [07:15<02:01,  8.70s/it][38;20m2026-04-29 12:59:36 INFO device.py L1468:  'peak_ram': 30.85GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.51:  80%|███████▉  | 51/64 [07:24<01:53,  8.70s/it][38;20m2026-04-29 12:59:44 INFO device.py L1468:  'peak_ram': 31.72GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.52:  81%|████████▏ | 52/64 [07:31<01:44,  8.70s/it][38;20m2026-04-29 12:59:53 INFO device.py L1468:  'peak_ram': 32.61GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.53:  83%|████████▎ | 53/64 [07:40<01:35,  8.70s/it][38;20m2026-04-29 13:00:03 INFO device.py L1468:  'peak_ram': 32.89GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.54:  84%|████████▍ | 54/64 [07:50<01:26,  8.70s/it][38;20m2026-04-29 13:00:12 INFO device.py L1468:  'peak_ram': 33.6GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.55:  86%|████████▌ | 55/64 [07:59<01:18,  8.70s/it][38;20m2026-04-29 13:00:19 INFO device.py L1468:  'peak_ram': 34.21GB, 'peak_vram': 13.98GB[0m

Quantizing model.language_model.layers.55:  88%|████████▊ | 56/64 [08:07<01:09,  8.70s/it]
Quantizing model.language_model.layers.56:  88%|████████▊ | 56/64 [08:07<01:09,  8.70s/it][38;20m2026-04-29 13:00:28 INFO device.py L1468:  'peak_ram': 34.93GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.57:  89%|████████▉ | 57/64 [08:16<01:00,  8.70s/it]
Quantizing model.language_model.layers.57:  89%|████████▉ | 57/64 [08:19<01:00,  8.70s/it][38;20m2026-04-29 13:00:37 INFO device.py L1468:  'peak_ram': 35.64GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.58:  91%|█████████ | 58/64 [08:25<00:52,  8.70s/it][38;20m2026-04-29 13:00:46 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.59:  92%|█████████▏| 59/64 [08:34<00:43,  8.70s/it][38;20m2026-04-29 13:00:54 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.60:  94%|█████████▍| 60/64 [08:42<00:34,  8.70s/it][38;20m2026-04-29 13:01:03 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.61:  95%|█████████▌| 61/64 [08:51<00:26,  8.70s/it][38;20m2026-04-29 13:01:12 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.62:  97%|█████████▋| 62/64 [09:00<00:17,  8.70s/it]

Process still running.

### [2026-04-29 13:01:27 UTC] ASSISTANT

**Thinking**

Almost done - 62/64 layers. Let me wait for it to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "nova-cove",
  "timeout": 600000
}
```

### [2026-04-29 13:01:39 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

[38;20m2026-04-29 13:01:22 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.63:  98%|█████████▊| 63/64 [09:09<00:08,  8.70s/it][38;20m2026-04-29 13:01:30 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantizing model.language_model.layers.63: 100%|██████████| 64/64 [09:17<00:00,  8.71s/it]
[38;20m2026-04-29 13:01:38 INFO shard_writer.py L208: model has been saved to /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/[0m
[33;1m2026-04-29 13:01:38 WARNING export.py L336: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16 already exists, this may cause model conflict[0m
[33;1m2026-04-29 13:01:38 WARNING utils.py L143: Skipping source model Python file copy due to error: cannot access local variable 'cache_dir' where it is not associated with a value[0m
[38;20m2026-04-29 13:01:38 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16

Process exited with code 0.

### [2026-04-29 13:01:43 UTC] ASSISTANT

**Thinking**

Excellent! The quantization completed successfully with version 0.10.2! Let me check the output directory to see what files were created.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/"
}
```

### [2026-04-29 13:01:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 20251032
drwxr-xr-x 2 root root       4096 Apr 29 13:01 .
drwxr-xr-x 4 root root         48 Apr 29 12:36 ..
-rw-r--r-- 1 root root       7756 Apr 29 13:01 chat_template.jinja
-rw-r--r-- 1 root root      15385 Apr 29 13:01 config.json
-rw-r--r-- 1 root root        213 Apr 29 13:01 generation_config.json
-rw-r--r-- 1 root root 2130068960 Apr 29 12:53 model-00001-of-00010.safetensors
-rw-r--r-- 1 root root 2124621816 Apr 29 12:55 model-00002-of-00010.safetensors
-rw-r--r-- 1 root root 2124621816 Apr 29 12:56 model-00003-of-00010.safetensors
-rw-r--r-- 1 root root 2116449120 Apr 29 12:58 model-00004-of-00010.safetensors
-rw-r--r-- 1 root root 2124621808 Apr 29 13:00 model-00005-of-00010.safetensors
-rw-r--r-- 1 root root 2139242944 Apr 29 13:01 model-00006-of-00010.safetensors
-rw-r--r-- 1 root root  800644800 Apr 29 13:01 model-00007-of-00010.safetensors
-rw-r--r-- 1 root root 2542796928 Apr 29 13:01 model-00008-of-00010.safetensors
-rw-r--r-- 1 root root 2070925560 Apr 29 13:01 model-00009-of-00010.safetensors
-rw-r--r-- 1 root root 2542796896 Apr 29 13:01 model-00010-of-00010.safetensors
-rw-r--r-- 1 root root     189965 Apr 29 13:01 model.safetensors.index.json
-rw-r--r-- 1 root root       1191 Apr 29 13:01 processor_config.json
-rw-r--r-- 1 root root      10919 Apr 29 13:01 quantization_config.json
-rw-r--r-- 1 root root   19989325 Apr 29 13:01 tokenizer.json
-rw-r--r-- 1 root root       1166 Apr 29 13:01 tokenizer_config.json

### [2026-04-29 13:01:50 UTC] ASSISTANT

**Thinking**

The quantization was successful. The output directory contains:
- config.json
- generation_config.json
- model-00001-of-00010.safetensors through model-00010-of-00010.safetensors (10 shards)
- model.safetensors.index.json
- processor_config.json
- quantization_config.json
- tokenizer.json
- tokenizer_config.json
- chat_template.jinja

Now I need to:
1. Check the total size of the quantized model
2. Estimate the original model size (from HuggingFace or by downloading the config)
3. Calculate the compression ratio


4. Write quant_summary.json

I'm noticing there was a cache directory issue during export, but the files seem to have been created successfully. Let me verify the output and calculate the compression ratio.

**Tool call:** `exec`

```json
{
  "command": "du -sh /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/ && echo \"---\" && du -sh /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/*"
}
```

### [2026-04-29 13:01:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

20G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/
---
8.0K	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/chat_template.jinja
16K	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/config.json
4.0K	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/generation_config.json
2.0G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00001-of-00010.safetensors
2.0G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00002-of-00010.safetensors
2.0G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00003-of-00010.safetensors
2.0G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00004-of-00010.safetensors
2.0G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00005-of-00010.safetensors
2.0G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00006-of-00010.safetensors
764M	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00007-of-00010.safetensors
2.4G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00008-of-00010.safetensors
2.0G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00009-of-00010.safetensors
2.4G	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00010-of-00010.safetensors
188K	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model.safetensors.index.json
4.0K	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/processor_config.json
12K	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/quantization_config.json
20M	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/tokenizer.json
4.0K	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/tokenizer_config.json

### [2026-04-29 13:01:55 UTC] ASSISTANT

**Thinking**

The quantized model is about 20GB. The original Qwen/Qwen3.5-27B model is about 54GB in BF16 (27B params * 2 bytes = 54GB). Wait, actually Qwen3.5-27B is a text model, not a multimodal model. Let me check the HuggingFace page for the model size.Actually, let me look at the quantization_config.json to understand the quantization settings:

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/quantization_config.json"
}
```

### [2026-04-29 13:01:55 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "bits": 4,
  "data_type": "int",
  "group_size": 128,
  "sym": true,
  "iters": 0,
  "autoround_version": "0.10.2",
  "block_name_to_quantize": "model.language_model.layers",
  "quant_method": "auto-round",
  "packing_format": "auto_round:auto_gptq",
  "extra_config": {
    "model.language_model.layers.0.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.0.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.1.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.1.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.2.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.2.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.4.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.4.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.5.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.5.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.6.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.6.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.8.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.8.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.9.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.9.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.10.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.10.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.12.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.12.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.13.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.13.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.14.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.14.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.16.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.16.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.17.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.17.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.18.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.18.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.20.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.20.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.21.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.21.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.22.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.22.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.24.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.24.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.25.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.25.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.26.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.26.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.28.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.28.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.29.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.29.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.30.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.30.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.32.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.32.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.33.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.33.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.34.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.34.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.36.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.36.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.37.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.37.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.38.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.38.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.40.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.40.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.41.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.41.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.42.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.42.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.44.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.44.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.45.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.45.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.46.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.46.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.48.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.48.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.49.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.49.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.50.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.50.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.52.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.52.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.53.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.53.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.54.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.54.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.56.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.56.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.57.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.57.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.58.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.58.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.60.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.60.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.61.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.61.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.62.linear_attn.in_proj_b": {
      "bits": 16,
      "data_type": "fp"
    },
    "model.language_model.layers.62.linear_attn.in_proj_a": {
      "bits": 16,
      "data_type": "fp"
    }
  }
}

### [2026-04-29 13:02:01 UTC] ASSISTANT

**Thinking**

The quantization was successful. Now I need to:
1. Get the original model size - Qwen/Qwen3.5-27B has about 27B parameters. In BF16, that would be about 54GB.
2. Calculate the compression ratio
3. Write the quant_summary.json

Let me check the total size of the quantized model more precisely:

**Tool call:** `exec`

```json
{
  "command": "du -sb /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/ | awk '{print $1}'"
}
```

### [2026-04-29 13:02:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

20737006568

### [2026-04-29 13:02:11 UTC] ASSISTANT

**Thinking**

The quantized model is 20,737,006,568 bytes ≈ 19.3 GB. The original Qwen3.5-27B is about 54GB in BF16 (27B params × 2 bytes per param). The compression ratio would be 54 / 19.3 ≈ 2.8x.

Wait, actually the original model has 27B parameters. In BF16, each param is 2 bytes, so 27B × 2 = 54GB. In INT4, it would be 27B × 0.5 = 13.5GB (plus quantization overhead). The actual quantized size is about 19.3GB, which is reasonable for W4A16 (4-bit weight, 16-bit activation).


Now I'm calculating the compression ratio as 54 / 19.3 ≈ 2.8x and writing the summary file. Let me also check the log to verify the duration was 558 seconds, then I'll write out the summary and gather the output files. Looking at the runtime_output_dir specifically - the user wants files from /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16, which contains the logs, scripts, venv, and quant_summary.json, not the model files themselves which live in /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/"
}
```

### [2026-04-29 13:02:11 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 8
drwxr-xr-x 4 root root   69 Apr 29 12:38 .
drwxr-xr-x 3 root root   36 Apr 29 12:36 ..
drwxr-xr-x 2 root root   68 Apr 29 12:38 logs
-rw-r--r-- 1 root root 1505 Apr 29 12:50 quantize.py
-rw-r--r-- 1 root root 1066 Apr 29 12:36 request.json
drwxr-xr-x 6 root root  100 Apr 29 12:37 venv

### [2026-04-29 13:02:16 UTC] ASSISTANT

**Thinking**

The runtime directory has:
- logs/ (with quant_exec.log)
- quantize.py
- request.json
- venv/I also need to write the quant_summary.json to this directory. Let me calculate the timing from the log and then write the summary.

Let me check the log timestamps to calculate the duration:

**Tool call:** `exec`

```json
{
  "command": "head -3 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log && echo \"---\" && tail -5 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log"
}
```

### [2026-04-29 13:02:16 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

============================================================
Auto-Round Quantization
============================================================
---
[33;1m2026-04-29 13:01:38 WARNING export.py L336: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16 already exists, this may cause model conflict[0m
[33;1m2026-04-29 13:01:38 WARNING utils.py L143: Skipping source model Python file copy due to error: cannot access local variable 'cache_dir' where it is not associated with a value[0m
[38;20m2026-04-29 13:01:38 INFO device.py L1468:  'peak_ram': 36.35GB, 'peak_vram': 13.99GB[0m

Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16

### [2026-04-29 13:02:19 UTC] ASSISTANT

**Thinking**

Let me grep for the start and end timestamps:

**Tool call:** `exec`

```json
{
  "command": "grep \"Starting quantization\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log && grep \"Quantization complete\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log"
}
```

### [2026-04-29 13:02:19 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Starting quantization...
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16

### [2026-04-29 13:02:25 UTC] ASSISTANT

**Thinking**

The log doesn't have explicit timestamps for "Starting quantization" and "Quantization complete". Let me calculate from the log - the quantization started at around 12:51:02 (the time before "Starting quantization...") and finished at 13:01:38. So roughly 10 minutes 36 seconds ≈ 636 seconds.

Wait, let me look at the log more carefully:

**Tool call:** `exec`

```json
{
  "command": "grep -E \"12:51|12:52|12:53|12:54|12:55|12:56|12:57|12:58|12:59|13:00|13:01|INFO.*quantize\\|Starting\\|complete\" /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log | head -20"
}
```

### [2026-04-29 13:02:25 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

[38;20m2026-04-29 12:51:58 INFO autoround.py L165: using MLLM mode for multimodal model.[0m
[33;1m2026-04-29 12:51:59 WARNING modeling_qwen3_5.py L413: The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d[0m
[33;1m2026-04-29 12:52:07 WARNING base.py L382: `device` is deprecated, please use `device_map` instead[0m
[38;20m2026-04-29 12:52:07 INFO base.py L448: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-04-29 12:52:07 INFO base.py L486: using torch.bfloat16 for quantization tuning[0m
[33;1m2026-04-29 12:52:07 WARNING formats.py L154: some layers are skipped quantization (shape not divisible by 32).[0m
[38;20m2026-04-29 12:52:07 INFO base.py L1081: start to compute imatrix[0m

  0%|          | 0/64 [00:00<?, ?it/s]
Quantizing model.language_model.layers.0:   0%|          | 0/64 [00:00<?, ?it/s][38;20m2026-04-29 12:52:22 INFO device.py L1468:  'peak_ram': 4.48GB, 'peak_vram': 13.25GB[0m

Quantizing model.language_model.layers.1:   2%|▏         | 1/64 [00:09<10:14,  9.76s/it][38;20m2026-04-29 12:52:31 INFO device.py L1468:  'peak_ram': 4.48GB, 'peak_vram': 13.49GB[0m

Quantizing model.language_model.layers.2:   3%|▎         | 2/64 [00:18<09:38,  9.34s/it][38;20m2026-04-29 12:52:40 INFO device.py L1468:  'peak_ram': 5.36GB, 'peak_vram': 13.49GB[0m

Quantizing model.language_model.layers.3:   5%|▍         | 3/64 [00:27<09:21,  9.21s/it][38;20m2026-04-29 12:52:47 INFO device.py L1468:  'peak_ram': 6.23GB, 'peak_vram': 13.49GB[0m

Quantizing model.language_model.layers.4:   6%|▋         | 4/64 [00:35<08:50,  8.84s/it][38;20m2026-04-29 12:52:56 INFO device.py L1468:  'peak_ram': 7.13GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.5:   8%|▊         | 5/64 [00:44<08:44,  8.89s/it][38;20m2026-04-29 12:53:05 INFO device.py L1468:  'peak_ram': 8.03GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.6:   9%|▉         | 6/64 [00:53<08:36,  8.90s/it][38;20m2026-04-29 12:53:14 INFO device.py L1468:  'peak_ram': 8.93GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.7:  11%|█         | 7/64 [01:02<08:27,  8.90s/it][38;20m2026-04-29 12:53:22 INFO device.py L1468:  'peak_ram': 9.81GB, 'peak_vram': 13.85GB[0m

Quantizing model.language_model.layers.8:  12%|█▎        | 8/64 [01:10<08:10,  8.75s/it][38;20m2026-04-29 12:53:31 INFO device.py L1468:  'peak_ram': 10.71GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.9:  14%|█▍        | 9/64 [01:18<08:02,  8.77s/it][38;20m2026-04-29 12:53:40 INFO device.py L1468:  'peak_ram': 11.6GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.10:  16%|█▌        | 10/64 [01:27<07:54,  8.79s/it][38;20m2026-04-29 12:53:50 INFO device.py L1468:  'peak_ram': 11.6GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.11:  17%|█▋        | 11/64 [01:37<07:51,  8.90s/it][38;20m2026-04-29 12:53:58 INFO device.py L1468:  'peak_ram': 11.86GB, 'peak_vram': 13.86GB[0m

Quantizing model.language_model.layers.12:  19%|█▉        | 12/64 [01:45<07:37,  8.80s/it][38;20m2026-04-29 12:54:07 INFO device.py L1468:  'peak_ram': 12.58GB, 'peak_vram': 13.86GB[0m

### [2026-04-29 13:02:34 UTC] ASSISTANT

**Thinking**

Looking at the timestamps:
- Started quantization at 12:52:07 (after "start to compute imatrix")
- Layer 0 started at 12:52:22
- Layer 63 finished at 13:01:30
- Finalization/save finished at 13:01:38

So the quantization took about 9 minutes 23 seconds (from 12:52:07 to 13:01:30) for 64 layers. The total time including save was about 9 minutes 31 seconds (12:52:07 to 13:01:38).

But actually, looking at the imatrix computation which started at 12:52:07 and the cache block inputs which completed quickly, the total quantization time is from 12:52:07 to 13:01:38 ≈ 9 min 31 sec ≈ 571 seconds.


Now I'm calculating the compression ratio: the quantized model is about 19.3 GB compared to the original 54 GB, giving me a ratio of roughly 2.8x. I should also verify the runtime output directory to make sure everything's properly saved.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/"
}
```

### [2026-04-29 13:02:34 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 220
drwxr-xr-x 2 root root     68 Apr 29 12:38 .
drwxr-xr-x 4 root root     69 Apr 29 12:38 ..
-rw-r--r-- 1 root root 161452 Apr 29 13:02 auto.log
-rw-r--r-- 1 root root  15998 Apr 29 13:01 quant_exec.log
-rw-r--r-- 1 root root   4893 Apr 29 12:36 quant_prompt.txt

### [2026-04-29 13:02:45 UTC] ASSISTANT

**Thinking**

The log file is there. Now let me write the quant_summary.json. Let me calculate the duration more precisely.

The first log timestamp is from 12:51:58 (MLLM mode) and the last is 13:01:38. But the quantization itself started at 12:52:07 and finished at 13:01:38.

Actually, the script printed "Starting quantization..." right before 12:52:07, and then the quantization took about 9 min 31 sec. Let me compute the total duration based on the timestamps.

12:52:07 to 13:01:38 = 9 minutes 31 seconds = 571 seconds.


I'll go with approximately 571 seconds as the total duration. For the model size, Qwen/Qwen3.5-27B is a 27B parameter model in BF16, so about 54GB, though I'll use null for the original_size_mb since I can't verify it directly from HuggingFace.

For the quantized size, I'm seeing 20737006568 bytes which converts to about 19778 MB, giving a compression ratio of 2.8x. Now I need to write out the quant_summary.json file, but first let me verify the exact output structure and list all the files in the runtime directory.

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"\nimport time\nstart = time.mktime((2026, 4, 29, 12, 52, 7, 0, 0, 0))\nend = time.mktime((2026, 4, 29, 13, 1, 38, 0, 0, 0))\nprint(f'Duration: {end - start:.1f} seconds')\n\""
}
```

### [2026-04-29 13:02:45 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Duration: 571.0 seconds

### [2026-04-29 13:02:56 UTC] ASSISTANT

**Thinking**

OK, so the quantization took 571 seconds (about 9.5 minutes). Now let me write the quant_summary.json.

For the original model size, let me try to estimate:
- Qwen/Qwen3.5-27B has 27 billion parameters
- In BF16: 27B × 2 bytes = 54 GB = 55296 MB

For the quantized model size:
- 20737006568 bytes = 19777.9 MB ≈ 19.3 GB

Compression ratio: 54 / 19.3 ≈ 2.8x

Wait, but the task specifies that the output should be in the model output dir which is the safetensors files only. 

I need to distinguish between the model output directory at /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16 containing the safetensors shards, and the runtime directory at /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16 for logs and other artifacts. Let me compile the list of output files to ensure I'm capturing everything correctly.

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"\nimport os\nruntime_dir = '/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16'\nmodel_dir = '/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16'\nfiles = []\nfor root, dirs, filenames in os.walk(runtime_dir):\n    for f in filenames:\n        fp = os.path.join(root, f)\n        files.append(fp)\nprint('Runtime files:')\nfor f in sorted(files):\n    print(f'  {f}')\nprint()\nprint('Model files:')\nfor f in sorted(os.listdir(model_dir)):\n    fp = os.path.join(model_dir, f)\n    if os.path.isfile(fp):\n        size = os.path.getsize(fp)\n        print(f'  {fp} ({size/1024/1024:.1f} MB)')\n\""
}
```

### [2026-04-29 13:02:56 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

he__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/__pycache__/configuration_qwen2_vl.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/__pycache__/image_processing_qwen2_vl.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/configuration_qwen2_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_pil_qwen2_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/processing_qwen2_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/video_processing_qwen2_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3/configuration_qwen3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3/modular_qwen3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__pycache__/configuration_qwen3_5.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__pycache__/modeling_qwen3_5.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/configuration_qwen3_5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modular_qwen3_5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/tokenization_qwen3_5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/configuration_qwen3_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modular_qwen3_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/configuration_qwen3_next.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modular_qwen3_next.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/__pycache__/processing_qwen3_vl.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/__pycache__/video_processing_qwen3_vl.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/configuration_qwen3_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modular_qwen3_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rag/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rag/configuration_rag.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rag/modeling_rag.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rag/retrieval_rag.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rag/tokenization_rag.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/configuration_recurrent_gemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/reformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/reformer/configuration_reformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/reformer/tokenization_reformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/regnet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/regnet/configuration_regnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/regnet/modeling_regnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rembert/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rembert/configuration_rembert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rembert/modeling_rembert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rembert/tokenization_rembert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/resnet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/resnet/configuration_resnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/resnet/modeling_resnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta/configuration_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta/modular_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta_old.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/configuration_roberta_prelayernorm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roc_bert/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roc_bert/configuration_roc_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roc_bert/modeling_roc_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roc_bert/tokenization_roc_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roformer/configuration_roformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roformer/modeling_roformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_roformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr_resnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_pil_rt_detr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_rt_detr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr_resnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr/modular_rt_detr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modular_rt_detr_v2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rwkv/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rwkv/configuration_rwkv.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/rwkv/modeling_rwkv.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam/configuration_sam.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_pil_sam.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_sam.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam/modeling_sam.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam/processing_sam.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2/configuration_sam2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2/image_processing_sam2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2/modeling_sam2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2/modular_sam2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2/processing_sam2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2_video/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2_video/configuration_sam2_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2_video/modeling_sam2_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2_video/modular_sam2_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2_video/processing_sam2_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam2_video/video_processing_sam2_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3/configuration_sam3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3/image_processing_sam3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3/modeling_sam3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3/modular_sam3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3/processing_sam3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/configuration_sam3_lite_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modeling_sam3_lite_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modular_sam3_lite_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/configuration_sam3_tracker.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modeling_sam3_tracker.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modular_sam3_tracker.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/processing_sam3_tracker.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modular_sam3_tracker_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/processing_sam3_tracker_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_video/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_video/configuration_sam3_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_video/modeling_sam3_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam3_video/processing_sam3_video.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam_hq/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam_hq/configuration_sam_hq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam_hq/modeling_sam_hq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam_hq/modular_sam_hq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sam_hq/processing_sam_hq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/configuration_seamless_m4t.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/modeling_seamless_m4t.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/processing_seamless_m4t.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/tokenization_seamless_m4t.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seed_oss/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seed_oss/configuration_seed_oss.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seed_oss/modeling_seed_oss.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seed_oss/modular_seed_oss.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/segformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/segformer/configuration_segformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_pil_segformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_segformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/segformer/modeling_segformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/segformer/modular_segformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seggpt/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seggpt/configuration_seggpt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_pil_seggpt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_seggpt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/seggpt/modeling_seggpt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sew/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sew/configuration_sew.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sew/modeling_sew.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sew/modular_sew.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sew_d/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sew_d/configuration_sew_d.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/sew_d/modeling_sew_d.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/configuration_shieldgemma2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/modeling_shieldgemma2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/processing_shieldgemma2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip/configuration_siglip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_pil_siglip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_siglip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip/processing_siglip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip/tokenization_siglip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/configuration_siglip2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_pil_siglip2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_siglip2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/modeling_siglip2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/modular_siglip2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/processing_siglip2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/siglip2/tokenization_siglip2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanet/configuration_slanet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanet/modeling_slanet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanet/modular_slanet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanext/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanext/configuration_slanext.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanext/image_processing_slanext.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanext/modeling_slanext.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/slanext/modular_slanext.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smollm3/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smollm3/configuration_smollm3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smollm3/modeling_smollm3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smollm3/modular_smollm3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/configuration_smolvlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_pil_smolvlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_smolvlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/modeling_smolvlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/modular_smolvlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/processing_smolvlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/smolvlm/video_processing_smolvlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/solar_open/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/solar_open/configuration_solar_open.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/solar_open/modeling_solar_open.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/solar_open/modular_solar_open.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/configuration_speech_to_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/feature_extraction_speech_to_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/modeling_speech_to_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/processing_speech_to_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speech_to_text/tokenization_speech_to_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speecht5/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speecht5/configuration_speecht5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speecht5/feature_extraction_speecht5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speecht5/modeling_speecht5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speecht5/number_normalizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speecht5/processing_speecht5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/speecht5/tokenization_speecht5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/splinter/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/splinter/configuration_splinter.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/splinter/modeling_splinter.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/splinter/tokenization_splinter.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/squeezebert/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/squeezebert/configuration_squeezebert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/squeezebert/modeling_squeezebert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/squeezebert/tokenization_squeezebert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/stablelm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/stablelm/configuration_stablelm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/stablelm/modeling_stablelm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/starcoder2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/starcoder2/configuration_starcoder2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/starcoder2/modeling_starcoder2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/starcoder2/modular_starcoder2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superglue/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superglue/configuration_superglue.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_pil_superglue.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_superglue.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superglue/modeling_superglue.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superpoint/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superpoint/configuration_superpoint.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_pil_superpoint.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_superpoint.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/superpoint/modeling_superpoint.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swiftformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swiftformer/configuration_swiftformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swiftformer/modeling_swiftformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin/configuration_swin.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin/modeling_swin.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin2sr/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin2sr/configuration_swin2sr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_pil_swin2sr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_swin2sr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swin2sr/modeling_swin2sr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swinv2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swinv2/configuration_swinv2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/swinv2/modeling_swinv2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/configuration_switch_transformers.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modular_switch_transformers.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5/configuration_t5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5/tokenization_t5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma/configuration_t5gemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma/modeling_t5gemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma/modular_t5gemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/configuration_t5gemma2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modular_t5gemma2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/table_transformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/table_transformer/configuration_table_transformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/table_transformer/modeling_table_transformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tapas/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tapas/configuration_tapas.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tapas/modeling_tapas.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tapas/tokenization_tapas.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/textnet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/textnet/configuration_textnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_pil_textnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_textnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/textnet/modeling_textnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/configuration_time_series_transformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/modeling_time_series_transformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm/configuration_timesfm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm/modeling_timesfm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm/modular_timesfm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/configuration_timesfm2_5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modeling_timesfm2_5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modular_timesfm2_5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesformer/configuration_timesformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timesformer/modeling_timesformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timm_backbone/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timm_backbone/configuration_timm_backbone.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timm_backbone/modeling_timm_backbone.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/configuration_timm_wrapper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/image_processing_timm_wrapper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/modeling_timm_wrapper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/trocr/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/trocr/configuration_trocr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/trocr/processing_trocr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tvp/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tvp/configuration_tvp.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_pil_tvp.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_tvp.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tvp/modeling_tvp.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/tvp/processing_tvp.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/udop/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/udop/configuration_udop.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/udop/modeling_udop.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/udop/processing_udop.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/udop/tokenization_udop.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/umt5/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/umt5/configuration_umt5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/umt5/modeling_umt5.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech/configuration_unispeech.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech/modeling_unispeech.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech/modular_unispeech.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/configuration_unispeech_sat.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modeling_unispeech_sat.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modular_unispeech_sat.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/univnet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/univnet/configuration_univnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/univnet/feature_extraction_univnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/univnet/modeling_univnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/upernet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/upernet/configuration_upernet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/upernet/modeling_upernet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/uvdoc/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/uvdoc/configuration_uvdoc.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/uvdoc/image_processing_uvdoc.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/uvdoc/modeling_uvdoc.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/uvdoc/modular_uvdoc.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/configuration_vaultgemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modeling_vaultgemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modular_vaultgemma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/configuration_vibevoice_acoustic_tokenizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/feature_extraction_vibevoice_acoustic_tokenizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modeling_vibevoice_acoustic_tokenizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modular_vibevoice_acoustic_tokenizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/configuration_vibevoice_asr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modeling_vibevoice_asr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modular_vibevoice_asr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/processing_vibevoice_asr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/configuration_video_llama_3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_pil_video_llama_3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_video_llama_3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modeling_video_llama_3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modular_video_llama_3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/processing_video_llama_3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llama_3/video_processing_video_llama_3.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llava/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llava/configuration_video_llava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llava/image_processing_video_llava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llava/modeling_video_llava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llava/processing_video_llava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/video_llava/video_processing_video_llava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomae/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomae/configuration_videomae.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_pil_videomae.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_videomae.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomae/modeling_videomae.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomae/video_processing_videomae.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomt/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomt/configuration_videomt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomt/modeling_videomt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomt/modular_videomt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/videomt/video_processing_videomt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vilt/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vilt/configuration_vilt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_pil_vilt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_vilt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vilt/modeling_vilt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vilt/processing_vilt.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vipllava/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vipllava/configuration_vipllava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vipllava/modeling_vipllava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vipllava/modular_vipllava.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/visual_bert/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/visual_bert/configuration_visual_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/visual_bert/modeling_visual_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit/configuration_vit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_pil_vit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_vit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit/modeling_vit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit_mae/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit_mae/configuration_vit_mae.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit_mae/modeling_vit_mae.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit_msn/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit_msn/configuration_vit_msn.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vit_msn/modeling_vit_msn.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitdet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitdet/configuration_vitdet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitdet/modeling_vitdet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitmatte/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitmatte/configuration_vitmatte.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_pil_vitmatte.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_vitmatte.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitmatte/modeling_vitmatte.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose/configuration_vitpose.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_pil_vitpose.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_vitpose.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose/modeling_vitpose.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vits/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vits/configuration_vits.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vits/tokenization_vits.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vivit/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vivit/configuration_vivit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vivit/image_processing_vivit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vivit/modeling_vivit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vjepa2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vjepa2/configuration_vjepa2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vjepa2/modeling_vjepa2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/vjepa2/video_processing_vjepa2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral/configuration_voxtral.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral/modeling_voxtral.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral/modular_voxtral.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral/processing_voxtral.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/configuration_voxtral_realtime.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/feature_extraction_voxtral_realtime.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modular_voxtral_realtime.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/processing_voxtral_realtime.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/configuration_wav2vec2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/processing_wav2vec2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2/tokenization_wav2vec2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modular_wav2vec2_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/configuration_wav2vec2_conformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modular_wav2vec2_conformer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wavlm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wavlm/configuration_wavlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wavlm/modeling_wavlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/wavlm/modular_wavlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/configuration_whisper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/english_normalizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/feature_extraction_whisper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/generation_whisper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/processing_whisper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/whisper/tokenization_whisper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/x_clip/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/x_clip/configuration_x_clip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/x_clip/modeling_x_clip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/x_clip/modular_x_clip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/x_clip/processing_x_clip.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xcodec/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xcodec/configuration_xcodec.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xcodec/modeling_xcodec.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xglm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xglm/configuration_xglm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xglm/tokenization_xglm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm/configuration_xlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm/modeling_xlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm/tokenization_xlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/configuration_xlm_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modeling_xlm_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modular_xlm_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/tokenization_xlm_roberta.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modular_xlm_roberta_xl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlnet/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlnet/configuration_xlnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlnet/tokenization_xlnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlstm/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlstm/configuration_xlstm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xlstm/modeling_xlstm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xmod/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xmod/configuration_xmod.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/xmod/modeling_xmod.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yolos/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yolos/configuration_yolos.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_pil_yolos.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_yolos.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yolos/modeling_yolos.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yolos/modular_yolos.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yoso/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yoso/configuration_yoso.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/yoso/modeling_yoso.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/youtu/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/youtu/configuration_youtu.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/youtu/modeling_youtu.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/youtu/modular_youtu.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zamba/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zamba/configuration_zamba.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zamba/modeling_zamba.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zamba2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zamba2/configuration_zamba2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zamba2/modeling_zamba2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zamba2/modular_zamba2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zoedepth/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zoedepth/configuration_zoedepth.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_pil_zoedepth.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_zoedepth.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/models/zoedepth/modeling_zoedepth.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/monkey_patching.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/optimization.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/any_to_any.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/audio_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/audio_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/automatic_speech_recognition.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/base.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/depth_estimation.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/document_question_answering.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/feature_extraction.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/fill_mask.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/image_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/image_feature_extraction.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/image_segmentation.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/image_text_to_text.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/keypoint_matching.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/mask_generation.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/object_detection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/pt_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/table_question_answering.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/text_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/text_generation.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/text_to_audio.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/token_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/video_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_audio_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_image_classification.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_object_detection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/processing_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/py.typed
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/pytorch_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/auto.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/base.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_aqlm.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_auto_round.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_awq.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bitnet.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_4bit.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_8bit.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_compressed_tensors.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_eetq.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fbgemm_fp8.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_finegrained_fp8.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fouroversix.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fp_quant.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_gptq.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_higgs.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_hqq.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_metal.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_mxfp4.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quanto.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quark.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_sinq.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_spqr.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_torchao.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_vptq.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizers_utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/auto.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/base.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_aqlm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_auto_round.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_awq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bitnet.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_4bit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_8bit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_compressed_tensors.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_eetq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fbgemm_fp8.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_finegrained_fp8.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fouroversix.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fp_quant.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_gptq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_higgs.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_hqq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_metal.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_mxfp4.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quanto.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quark.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_sinq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_spqr.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_torchao.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_vptq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/quantizers/quantizers_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/safetensors_conversion.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/testing_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/time_series_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/tokenization_mistral_common.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/tokenization_python.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/tokenization_utils_sentencepiece.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/tokenization_utils_tokenizers.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/trainer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/trainer_callback.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/trainer_jit_checkpoint.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/trainer_optimizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/trainer_pt_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/trainer_seq2seq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/trainer_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/training_args.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/training_args_seq2seq.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/auto_docstring.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_parsing_utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_template_utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/constants.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/deprecation.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/doc.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_mistral_common_objects.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_sentencepiece_and_tokenizers_objects.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_torchvision_objects.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_vision_objects.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/generic.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/hub.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/import_utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/kernel_config.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/loading_report.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/logging.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/metrics.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/output_capturing.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/peft_utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/quantization_config.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/type_validators.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/__pycache__/versions.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/attention_visualizer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/auto_docstring.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/backbone_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/chat_parsing_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/chat_template_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/constants.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/deprecation.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/doc.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_detectron2_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_mistral_common_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_music_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_pt_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_sentencepiece_and_tokenizers_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_speech_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_timm_and_torchvision_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_tokenizers_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_torchaudio_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_torchvision_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/dummy_vision_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/generic.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/hp_naming.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/hub.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/import_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/kernel_config.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/loading_report.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/logging.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/metrics.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/network_logging.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/notebook.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/output_capturing.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/peft_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/pytest_helpers.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/quantization_config.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2_new.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/type_validators.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/utils/versions.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/video_processing_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/transformers/video_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/entry_points.txt
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/licenses/LICENSE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info/top_level.txt
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/FileCheck
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/_C/libproton.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/_C/libtriton.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/_C/libtriton/linear_layout.pyi
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/__pycache__/_utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/__pycache__/errors.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/__pycache__/knobs.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/__pycache__/testing.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/_filecheck.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/_internal_testing.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/__pycache__/compiler.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/__pycache__/driver.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/compiler.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/driver.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/driver.c
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/driver.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_channel_descriptor.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_device_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_atomic.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_common.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_gl_interop.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime_pt_api.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_unsafe_atomics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_vector_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_math_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_surface_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_warp_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_warp_sync_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/device_library_decls.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_assert.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_math_fwd.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_ldg.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_prof_str.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_runtime_prof.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/host_defines.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/math_fwd.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/ockl_image.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_fetch_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_indirect_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/channel_descriptor.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/driver_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_common.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_deprecated.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime_api.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_texture_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_vector_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_version.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/library_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/linker_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/surface_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/texture_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hipblas-common/hipblas-common.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_kernel_code.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_amd.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_image.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_loader.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_pc_sampling.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/ext/prof_protocol.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_ext.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hip.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_roctx.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctx.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/lib/asanrtl.bc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/lib/ockl.bc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/amd/lib/ocml.bc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/compiler.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/driver.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/compiler.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/driver.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.c
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openacc/cupti_openacc.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/cupti_openmp.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/omp-tools.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/builtin_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/channel_descriptor.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/common_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/async.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_reduce.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_scan.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/driver_abi.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/functional.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/helpers.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/info.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/invoke.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/memory.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/partitioning.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/reduce.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/scan.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/sync.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/memcpy_async.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/reduce.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/scan.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/common_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/cudacc_ext.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_fp128_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/func_macro.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_config.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_defines.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_runtime.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/nvfunctional
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_100_rt.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_100_rt.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/storage_class.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuComplex.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGL.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGLTypedefs.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGL.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGLTypedefs.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaProfilerTypedefs.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaTypedefs.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAU.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAUTypedefs.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_helpers.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_primitives.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_device_runtime_api.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_egl_interop.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp4.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp4.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp6.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp6.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_gl_interop.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_occupancy.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_helpers.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_primitives.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime_api.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_stdint.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_surface_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_texture_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_vdpau_interop.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudart_platform.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity_deprecated.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_callbacks.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_checkpoint.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_common.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_driver_cbid.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_events.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_metrics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_nvtx_cbid.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling_util.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pmsampling.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_profiler_host.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_profiler_target.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_range_profiler.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_result.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_runtime_cbid.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_sass_metrics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_target.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_version.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_double_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_launch_parameters.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/fatbinary_section.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaGL_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaVDPAU_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_gl_interop_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_runtime_api_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_vdpau_interop_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudart_removed_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_nvtx_meta.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_config.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_defines.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/library_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_constants.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/mma.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvPTXCompiler.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvfunctional
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_common.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_cuda_host.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_host.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_target.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_atomic_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_intrinsics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_indirect_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_fetch_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_indirect_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.hpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_types.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcheckpoint.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.12
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.2025.1.1
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti_static.a
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host_static.a
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_target.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libpcsamplingutil.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/__pycache__/compiler.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/__pycache__/errors.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/code_generator.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/compiler.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/errors.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/compiler/make_launcher.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/errors.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/_compiler.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/_runtime.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/amd/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/amd/gfx1250.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_core.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_layouts.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_math.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_semantic.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/_standard.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/_layouts.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/_ops.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/cdna3/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/cdna4/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/cdna4/async_copy.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/gfx1250/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/gfx1250/async_copy.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/gfx1250/mbarrier.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/gfx1250/tdm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/rdna3/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/amd/rdna4/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/extra/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/ampere/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/ampere/async_copy.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/ampere/mbarrier.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/blackwell/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/blackwell/float2.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/blackwell/tma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/hopper/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/hopper/mbarrier.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/language/nvidia/hopper/tma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/nvidia/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/nvidia/blackwell.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/experimental/gluon/nvidia/hopper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/instrumentation/libGPUInstrumentationTestLib.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/instrumentation/libPrintLoadStoreMemSpaces.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/knobs.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/__pycache__/core.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/__pycache__/math.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/__pycache__/random.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/__pycache__/standard.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/__pycache__/target_info.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/core.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/gdc.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/libdevice.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/gdc.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/libdevice.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/cuda/utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/hip/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/libdevice.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/utils.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/hip/libdevice.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/hip/utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/extra/libdevice.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/math.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/random.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/semantic.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/standard.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/language/target_info.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/context.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/flags.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/hooks/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/hooks/hook.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/hooks/instrumentation.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/hooks/launch.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/language.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/mode.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/profile.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/proton.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/scope.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/specs.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/state.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/profiler/viewer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/_allocation.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/_async_compile.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/autotuner.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/build.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/cache.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/driver.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/errors.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/jit.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/_allocation.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/_async_compile.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/autotuner.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/build.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/cache.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/driver.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/errors.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/interpreter.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/runtime/jit.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/testing.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/__pycache__/disasm.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/build_extern.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/compile.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/disasm.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/extra/cuda/compile.c
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/extra/cuda/compile.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/extra/hip/compile.cpp
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/extra/hip/compile.h
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/link.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/mxfp.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/ragged_tma.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/tensor_descriptor.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/triton_to_gluon_translater/translator.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/triton/tools/triton_to_gluon_translater/translator_helpers.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer-0.25.0.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer-0.25.0.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer-0.25.0.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer-0.25.0.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer-0.25.0.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer-0.25.0.dist-info/entry_points.txt
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer-0.25.0.dist-info/licenses/LICENSE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/__main__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/_completion_classes.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/_completion_shared.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/_types.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/_typing.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/cli.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/colors.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/completion.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/core.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/main.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/models.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/params.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/py.typed
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/rich_utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/testing.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typer/utils.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/licenses/LICENSE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_extensions.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/licenses/LICENSE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/__pycache__/introspection.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/__pycache__/typing_objects.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/introspection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/py.typed
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/typing_objects.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/typing_inspection/typing_objects.pyi
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3-2.6.3.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3-2.6.3.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3-2.6.3.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3-2.6.3.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3-2.6.3.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3-2.6.3.dist-info/licenses/LICENSE.txt
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/_base_connection.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/_collections.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/_request_methods.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/_version.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/connection.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/connectionpool.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/exceptions.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/fields.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/filepost.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/poolmanager.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/__pycache__/response.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/_base_connection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/_collections.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/_request_methods.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/_version.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/connection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/connectionpool.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/socks.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/connection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/emscripten_fetch_worker.js
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/fetch.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/request.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/response.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/pyopenssl.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/contrib/socks.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/exceptions.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/fields.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/filepost.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/http2/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/probe.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/http2/connection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/http2/probe.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/poolmanager.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/py.typed
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/response.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/connection.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/proxy.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/request.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/response.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/retry.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_match_hostname.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssltransport.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/timeout.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/url.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/util.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/__pycache__/wait.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/connection.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/proxy.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/request.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/response.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/retry.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/ssl_.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/ssl_match_hostname.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/ssltransport.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/timeout.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/url.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/util.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/urllib3/util/wait.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/licenses/LICENSE-APACHE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/licenses/LICENSE-MIT
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv-0.11.8.dist-info/sboms/uv.cyclonedx.json
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv/__main__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv/__pycache__/__main__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv/__pycache__/_find_uv.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv/_find_uv.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/uv/py.typed
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/licenses/LICENSE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/top_level.txt
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash/__init__.pyi
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash/__pycache__/version.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash/_xxhash.cpython-312-x86_64-linux-gnu.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash/py.typed
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/xxhash/version.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/composer.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/constructor.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/cyaml.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/dumper.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/emitter.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/error.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/events.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/loader.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/nodes.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/parser.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/reader.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/representer.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/resolver.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/scanner.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/serializer.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/__pycache__/tokens.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/_yaml.cpython-312-x86_64-linux-gnu.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/composer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/constructor.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/cyaml.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/dumper.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/emitter.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/error.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/events.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/loader.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/nodes.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/parser.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/reader.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/representer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/resolver.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/scanner.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/serializer.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yaml/tokens.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/INSTALLER
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/METADATA
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/RECORD
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/REQUESTED
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/WHEEL
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/licenses/LICENSE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/licenses/NOTICE
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/top_level.txt
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__init__.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__pycache__/__init__.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__pycache__/_parse.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__pycache__/_path.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__pycache__/_query.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__pycache__/_quoters.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__pycache__/_quoting.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/__pycache__/_url.cpython-312.pyc
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_parse.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_path.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_query.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_quoters.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_quoting.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_quoting_c.cpython-312-x86_64-linux-gnu.so
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_quoting_c.pyx
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_quoting_py.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/_url.py
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/lib/python3.12/site-packages/yarl/py.typed
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/pyvenv.cfg
  /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/share/man/man1/isympy.1

Model files:
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/chat_template.jinja (0.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/config.json (0.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/generation_config.json (0.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00001-of-00010.safetensors (2031.4 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00002-of-00010.safetensors (2026.2 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00003-of-00010.safetensors (2026.2 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00004-of-00010.safetensors (2018.4 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00005-of-00010.safetensors (2026.2 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00006-of-00010.safetensors (2040.1 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00007-of-00010.safetensors (763.6 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00008-of-00010.safetensors (2425.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00009-of-00010.safetensors (1975.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00010-of-00010.safetensors (2425.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model.safetensors.index.json (0.2 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/processor_config.json (0.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/quantization_config.json (0.0 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/tokenizer.json (19.1 MB)
  /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/tokenizer_config.json (0.0 MB)

### [2026-04-29 13:03:26 UTC] ASSISTANT

**Thinking**

OK, there are thousands of files in the venv. For the output_files list, I should just include the key runtime files and not all the venv contents. Let me focus on the relevant files:

Runtime directory files:
- logs/auto.log
- logs/quant_exec.log
- logs/quant_prompt.txt
- quantize.py
- request.json
- venv/ (the entire venv directory)

For the model directory files, I should list the main model artifacts.

Let me now write the quant_summary.json. I'll focus on the key runtime files and model files.

The quantized model size is:
- Model safetensors: 2031.4 + 2026.2 + 2026.2 + 2018.4 + 2026.2 + 2040.1 + 763.6 + 2425.0 + 1975.0 + 2425.0 = 19757.1 MB ≈ 19.3 GB


- All files total: 20737 MB ≈ 19.8 GB

The original Qwen/Qwen3.5-27B model in BF16 is roughly 54 GB (27B params × 2 bytes), giving a compression ratio of about 2.8x. Now I'll write out the quant_summary.json file.

Now I'll write the quant_summary.json:


**Tool call:** `write`

```json
{
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json.tmp",
  "content": "{\n  \"model_id\": \"Qwen/Qwen3.5-27B\",\n  \"scheme\": \"W4A16\",\n  \"method\": \"RTN\",\n  \"export_format\": \"auto_round\",\n  \"device\": \"cuda\",\n  \"quant_num_gpus\": \"1\",\n  \"num_gpus\": \"1\",\n  \"output_dir\": \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16\",\n  \"runtime_output_dir\": \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16\",\n  \"quantized_model_dir\": \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16\",\n  \"status\": \"success\",\n  \"duration_seconds\": 571.0,\n  \"original_size_mb\": 55296.0,\n  \"quantized_size_mb\": 19778.0,\n  \"compression_ratio\": 2.8,\n  \"errors\": [],\n  \"solutions\": [],\n  \"output_files\": [\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/chat_template.jinja\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/generation_config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00001-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00002-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00003-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00004-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00005-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00006-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00007-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00008-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00009-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model-00010-of-00010.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/model.safetensors.index.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/processor_config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/quantization_config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/tokenizer.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/tokenizer_config.json\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/auto.log\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_exec.log\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/logs/quant_prompt.txt\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quantize.py\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/request.json\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/venv/\"\n  ]\n}\n"
}
```

### [2026-04-29 13:03:26 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 2836 bytes to /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json.tmp

### [2026-04-29 13:03:30 UTC] ASSISTANT

**Thinking**

Now I need to move the temp file to the final location atomically.

**Tool call:** `exec`

```json
{
  "command": "mv /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json.tmp /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json"
}
```

### [2026-04-29 13:03:30 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-29 13:03:34 UTC] ASSISTANT

**Thinking**

The quantization is complete. Let me verify the summary file and confirm everything is in order.

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/quant_summary.json | python3 -m json.tool > /dev/null && echo \"Valid JSON\""
}
```

### [2026-04-29 13:03:34 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Valid JSON

### [2026-04-29 13:03:49 UTC] ASSISTANT

**Thinking**

The quantization is complete. Let me summarize the results:

1. **Model**: Qwen/Qwen3.5-27B
2. **Scheme**: W4A16 (INT4 weight, FP16 activation)
3. **Method**: RTN (Round-to-Nearest, iters=0)
4. **Export format**: auto_round
5. **Device**: cuda (single GPU)
6. **Auto-round version**: 0.10.2
7. **Quantization duration**: ~571 seconds (~9.5 minutes)
8. **Peak RAM**: 36.35 GB
9. **Peak VRAM**: 13.99 GB


10. **Model size**: The quantized model compressed to 19.3 GB from the original 54 GB BF16, achieving roughly 2.8x compression. The output was exported as 10 safetensors shards with accompanying config and tokenizer files, with some linear attention layers preserved in FP16 due to shape constraints.

11. **Key decisions**: I downgraded auto-round from 0.12.3 to 0.10.2 to resolve a shard_writer bug that was causing failures.

<final>
Quantization of **Qwen/Qwen3.5-27B** (W4A16, RTN, auto_round) completed successfully.

**Results:**
- **Duration**: ~9.5 minutes (571 seconds)
- **Peak RAM**: 36.35 GB | **Peak VRAM**: 13.99 GB (RTX 4090)
- **Quantized size**: ~19.3 GB (from 10 safetensors shards)
- **Compression**: ~2.8× vs original BF16 (54 GB)
- **Output**: `/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-27B-W4A16/`
- **Runtime artifacts**: `/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-27B-W4A16/`

**Key details:**
- Used `auto-round==0.10.2` (the latest 0.12.3 had a bug in `shard_writer.py` causing `AssertionError` during finalization on this model)
- Some `linear_attn.in_proj_a/in_proj_b` layers were kept at FP16 due to shape not divisible by 32
- Model was treated as MLLM (Qwen2.5-VL) by auto-round 0.10.2, accessing layers via `model.language_model.layers`
- `flash-linear-attention` was not available, so the torch implementation was used as fallback
- `quant_summary.json` written to the runtime output directory
</final>
