# Session: autoeval_quant_352

- **Session ID:** `autoeval_quant_352`
- **Timestamp:** 2026-05-19 02:48:06 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 1: Quantization

### [2026-05-19 02:48:06 UTC] USER

You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_quant/SKILL.md

Model: Qwen/WebWorld-8B
Quantization: W4A16 / RTN
Export format: auto_round
Quantized Model Output directory: /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16
Quantization execution log path: /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs/quant_exec.log
Runtime device: cuda
Num gpus: 1

Directory responsibilities:
- Write exported model files to: /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16
- Write runtime artifacts such as quant_summary.json, quantize.py, logs, prompts, copied request/session files, and the venv to: /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16

CRITICAL SCRIPT REQUIREMENT:
- Before starting quantization, you MUST first generate the quantization script file:
    /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quantize.py
- The file name must be exactly: quantize.py
- The script must be a standalone Python program runnable with:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quantize.py
- The generated quantize.py must focus on the core quantization/export logic only.
- Do NOT put venv creation, pip/uv installation, proxy/bootstrap shell setup, summary generation, or unrelated orchestration into quantize.py.
- In this same OpenClaw task, prepare or reuse the Python environment separately before executing quantize.py.
- In this same OpenClaw task, first write quantize.py, then execute that generated script yourself.
- When you execute quantize.py, you MUST stream stdout/stderr into this log file while still printing output:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quantize.py 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs/quant_exec.log
- /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quant_summary.json is a final summary artifact, not a progress marker.
- After quantize.py finishes, inspect the exported artifacts and write /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quant_summary.json in a separate finalize step outside quantize.py.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quant_summary.json until quantization has finished and the exported artifacts are ready.
- Write /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quant_summary.json atomically via a temporary file and rename/move it into place only at finalize time.
- If quantization fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quant_summary.json before exiting non-zero, also atomically.
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

/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quant_summary.json - structured summary:
{
  "model_id": "Qwen/WebWorld-8B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16",
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

### [2026-05-19 02:48:11 UTC] ASSISTANT

**Thinking**

Let me start by reading the SKILL.md file to understand the Intel Auto-Round quantization workflow, then proceed with the task.


**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
}
```

### [2026-05-19 02:48:11 UTC] TOOL RESULT
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


### [2026-05-19 02:48:35 UTC] ASSISTANT

**Thinking**

Let me analyze the task:

1. Model: Qwen/WebWorld-8B
2. Quantization: W4A16 / RTN (Round-Trip Network - iters=0)
3. Export format: auto_round
4. Output directory: /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16
5. Runtime artifacts directory: /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16
6. Log path: /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs/quant_exec.log
7. Runtime device: cuda
8. Num gpus: 1

Key requirements:
- Write quantize.py to /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quantize.py
- Execute it with output streaming to the log file
- Use RTN method (iters=0)
- Use device="cuda" for single GPU
- Finalize with quant_summary.json after quantization completes

Let me start by:
1. Checking if there's a pre-existing venv or model_info.json
2. Creating the output directories
3. Creating the quantize.py script
4. Running it

For RTN mode with W4A16:
- scheme = "W4A16"
- iters = 0 (RTN mode)
- nsamples = 128 (default)
- format = "auto_round"
- device = "cuda" (single GPU)

Let me first check the environment and existing venv.




Let me start by checking the environment and creating the necessary directories.


**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())\" 2>/dev/null || echo \"torch not available\""
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.venv/bin/python 2>/dev/null || echo \"No /root/.venv\""
}
```

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/ 2>/dev/null || echo \"Directory does not exist yet\""
}
```

### [2026-05-19 02:48:35 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch not available

### [2026-05-19 02:48:35 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

lrwxrwxrwx 1 root root 73 Apr 24 05:53 /root/.venv/bin/python -> /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12

### [2026-05-19 02:48:35 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 0
drwxr-xr-x 3 root root  18 May 19 02:48 .
drwxr-xr-x 5 root root 176 May 19 02:48 ..
drwxr-xr-x 3 root root  36 May 19 02:48 runs

### [2026-05-19 02:48:44 UTC] ASSISTANT

**Thinking**

Let me check the venv more carefully and see what's installed, then set up the environment properly.


**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())\" 2>&1"
}
```

**Tool call:** `exec`

```json
{
  "command": "uv pip --python /root/.venv/bin/python list 2>/dev/null | grep -iE \"torch|transformers|auto.round|accelerate\" || echo \"Packages not found\""
}
```

**Tool call:** `exec`

```json
{
  "command": "which auto-round 2>/dev/null || echo \"auto-round not in PATH\""
}
```

### [2026-05-19 02:48:44 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-19 02:48:44 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Packages not found

### [2026-05-19 02:48:44 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

auto-round not in PATH

### [2026-05-19 02:48:53 UTC] ASSISTANT

**Thinking**

The system venv at /root/.venv doesn't have torch installed. I need to set up the environment properly. Let me:

1. Create the output directories
2. Create a new venv with --system-site-packages to inherit torch+cuda
3. Install auto-round and necessary dependencies

First, let me check if torch+cuda is installed system-wide.


**Tool call:** `exec`

```json
{
  "command": "mkdir -p /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import sys; print(sys.version)\""
}
```

**Tool call:** `exec`

```json
{
  "command": "find /usr /opt -name \"torch\" -type d 2>/dev/null | head -5"
}
```

### [2026-05-19 02:48:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-19 02:48:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

3.12.13 (main, Apr 14 2026, 14:29:00) [Clang 22.1.3 ]

### [2026-05-19 02:48:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-19 02:49:06 UTC] ASSISTANT

**Thinking**

System Python 3.12 has no torch. I need to create a venv with --system-site-packages to inherit CUDA packages. But if there's no torch anywhere, I need to install it.

Let me check what's available on the system and create the venv properly. Since torch isn't found, I'll need to install it. Let me first create the venv, then install torch and auto-round.

Actually, the SKILL says: "System Python has torch+cuda pre-installed" - but that's not true in this environment. I need to install torch.

Let me create the venv and install the necessary packages.


**Tool call:** `exec`

```json
{
  "command": "python3 -m venv --system-site-packages /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv 2>&1"
}
```

### [2026-05-19 02:49:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-19 02:49:12 UTC] ASSISTANT

**Thinking**

Now let me install torch with CUDA support. I need to find the right index URL for PyTorch with CUDA.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -20"
}
```

### [2026-05-19 02:49:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session crisp-reef, pid 571). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-19 02:49:29 UTC] ASSISTANT

**Thinking**

Torch is installing. Let me poll for the result.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "crisp-reef",
  "timeout": 120000
}
```

### [2026-05-19 02:49:37 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

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
 + typing-extensions==4.15.0

Process exited with code 0.

### [2026-05-19 02:49:45 UTC] ASSISTANT

**Thinking**

Torch is installed. Now let me install auto-round and other dependencies.


**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv/bin/python transformers accelerate datasets 2>&1 | tail -10"
}
```

**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv/bin/python auto-round 2>&1 | tail -10"
}
```

### [2026-05-19 02:49:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

+ safetensors==0.7.0
 + shellingham==1.5.4
 + six==1.17.0
 + tokenizers==0.22.2
 + tqdm==4.67.3
 + transformers==5.8.1
 + typer==0.25.1
 + urllib3==2.7.0
 + xxhash==3.7.0
 + yarl==1.23.0

### [2026-05-19 02:49:48 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Downloading pydantic-core (2.0MiB)
 Downloaded pydantic-core
Prepared 6 packages in 159ms
Installed 6 packages in 40ms
 + annotated-types==0.7.0
 + auto-round==0.12.3
 + py-cpuinfo==9.0.0
 + pydantic==2.13.4
 + pydantic-core==2.46.4
 + typing-inspection==0.4.2

### [2026-05-19 02:49:56 UTC] ASSISTANT

**Thinking**

Now let me verify torch can import with CUDA support:


**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv/bin/python -c \"import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())\""
}
```

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv/bin/python -c \"import auto_round; print('auto-round ok:', auto_round.__version__)\""
}
```

### [2026-05-19 02:49:58 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch: 2.6.0+cu124
cuda: True

### [2026-05-19 02:50:03 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

auto-round ok: 0.12.3

### [2026-05-19 02:50:17 UTC] ASSISTANT

**Thinking**

Environment is ready. Now let me create the quantize.py script. Based on the task parameters:
- Model: Qwen/WebWorld-8B
- Quantization: W4A16 / RTN
- Export format: auto_round
- Output: /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16
- Runtime artifacts: /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16
- device: cuda
- num_gpus: 1

For RTN mode, iters=0.

Let me write the quantize.py script.




Environment is ready. Now I'll write the quantization script:


**Tool call:** `write`

```json
{
  "content": "#!/usr/bin/env python3\n\"\"\"\nAuto-Round Quantization Script\nModel: Qwen/WebWorld-8B\nScheme: W4A16 (INT4 weight, FP16 activation)\nMethod: RTN (iters=0)\nFormat: auto_round\nDevice: cuda (single GPU)\n\"\"\"\n\nfrom auto_round import AutoRound\n\n# Configuration\nmodel_name_or_path = \"Qwen/WebWorld-8B\"\noutput_dir = \"/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16\"\nscheme = \"W4A16\"\niters = 0  # RTN mode - no calibration training\nnsamples = 128\nformat_str = \"auto_round\"\nnum_gpus = 1\n\nprint(f\"Loading model: {model_name_or_path}\")\nprint(f\"Scheme: {scheme}\")\nprint(f\"Method: RTN (iters={iters})\")\nprint(f\"nsamples: {nsamples}\")\nprint(f\"Format: {format_str}\")\nprint(f\"Device: cuda (num_gpus={num_gpus})\")\n\n# Create AutoRound instance\n# Single GPU: use device=\"cuda\"\nar = AutoRound(\n    model_name_or_path,\n    scheme=scheme,\n    iters=iters,\n    nsamples=nsamples,\n    device=\"cuda\",\n)\n\n# Quantize and save\nprint(\"Starting quantization...\")\nar.quantize_and_save(output_dir=output_dir, format=format_str)\n\nprint(f\"Quantization complete! Output: {output_dir}\")",
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quantize.py"
}
```

### [2026-05-19 02:50:17 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 1054 bytes to /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/quantize.py

### [2026-05-19 02:50:24 UTC] ASSISTANT

**Thinking**

Now let me execute the quantization script and stream output to the log file.


**Tool call:** `exec`

```json
{
  "command": "cd /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16 && /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/venv/bin/python -u quantize.py 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs/quant_exec.log",
  "timeout": 3600,
  "yieldMs": 30000
}
```

### [2026-05-19 02:50:54 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session warm-valley, pid 897). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-19 02:50:59 UTC] ASSISTANT

**Thinking**

The quantization is running. Let me check on its progress periodically.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "warm-valley",
  "timeout": 300000
}
```

### [2026-05-19 02:52:59 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

Loading model: Qwen/WebWorld-8B
Scheme: W4A16
Method: RTN (iters=0)
nsamples: 128
Format: auto_round
Device: cuda (num_gpus=1)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
404 Client Error. (Request ID: Root=1-6a0bcff2-06e7d5af46b180fd0800a279;0450a521-71cf-472e-8fc0-ecec40b8ea16)

Entry Not Found for url: https://huggingface.co/Qwen/WebWorld-8B/resolve/main/model_index.json.
[33;1m2026-05-19 02:51:12 WARNING base.py L407: `device` is deprecated, please use `device_map` instead[0m
[38;20m2026-05-19 02:51:12 INFO base.py L473: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-05-19 02:51:12 INFO base.py L517: using torch.bfloat16 for quantization tuning[0m
Starting quantization...
[33;1m2026-05-19 02:51:12 WARNING formats.py L166: some layers are skipped quantization (shape not divisible by 32): [0m
[33;1m2026-05-19 02:51:12 WARNING modeling_utils.py L4555: `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.[0m
[38;20m2026-05-19 02:51:13 INFO base.py L1150: start to compute imatrix[0m

Generating train split:   0%|          | 0/10000 [00:00<?, ? examples/s]
Generating train split: 100%|██████████| 10000/10000 [00:00<00:00, 78827.56 examples/s]

Map:   0%|          | 0/10000 [00:00<?, ? examples/s]
Map: 100%|██████████| 10000/10000 [00:05<00:00, 1706.25 examples/s]

Filter:   0%|          | 0/10000 [00:00<?, ? examples/s]
Filter: 100%|██████████| 10000/10000 [00:02<00:00, 4130.66 examples/s]

Casting the dataset:   0%|          | 0/1216 [00:00<?, ? examples/s]
Casting the dataset: 100%|██████████| 1216/1216 [00:01<00:00, 636.42 examples/s]

  0%|          | 0/36 [00:00<?, ?it/s]
Quantizing model.layers.0:   0%|          | 0/36 [00:00<?, ?it/s][38;20m2026-05-19 02:51:33 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.1:   3%|▎         | 1/36 [00:04<02:37,  4.49s/it][38;20m2026-05-19 02:51:37 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.2:   6%|▌         | 2/36 [00:08<02:20,  4.15s/it][38;20m2026-05-19 02:51:41 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.3:   8%|▊         | 3/36 [00:12<02:13,  4.04s/it][38;20m2026-05-19 02:51:45 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.4:  11%|█         | 4/36 [00:15<02:07,  3.98s/it][38;20m2026-05-19 02:51:49 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.5:  14%|█▍        | 5/36 [00:19<02:02,  3.95s/it][38;20m2026-05-19 02:51:52 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.6:  17%|█▋        | 6/36 [00:23<01:57,  3.93s/it][38;20m2026-05-19 02:51:56 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.7:  19%|█▉        | 7/36 [00:27<01:53,  3.92s/it][38;20m2026-05-19 02:52:00 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.8:  22%|██▏       | 8/36 [00:31<01:49,  3.91s/it][38;20m2026-05-19 02:52:04 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.9:  25%|██▌       | 9/36 [00:35<01:45,  3.90s/it][38;20m2026-05-19 02:52:08 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.10:  28%|██▊       | 10/36 [00:38<01:41,  3.89s/it][38;20m2026-05-19 02:52:12 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.11:  31%|███       | 11/36 [00:43<01:38,  3.93s/it][38;20m2026-05-19 02:52:16 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.12:  33%|███▎      | 12/36 [00:47<01:34,  3.92s/it][38;20m2026-05-19 02:52:20 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.13:  36%|███▌      | 13/36 [00:50<01:30,  3.92s/it][38;20m2026-05-19 02:52:24 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.14:  39%|███▉      | 14/36 [00:54<01:26,  3.91s/it][38;20m2026-05-19 02:52:28 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.15:  42%|████▏     | 15/36 [00:58<01:22,  3.91s/it][38;20m2026-05-19 02:52:31 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.16:  44%|████▍     | 16/36 [01:02<01:18,  3.90s/it][38;20m2026-05-19 02:52:35 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.17:  47%|████▋     | 17/36 [01:06<01:14,  3.90s/it][38;20m2026-05-19 02:52:39 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.18:  50%|█████     | 18/36 [01:10<01:10,  3.90s/it][38;20m2026-05-19 02:52:43 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.19:  53%|█████▎    | 19/36 [01:14<01:06,  3.90s/it][38;20m2026-05-19 02:52:47 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.20:  56%|█████▌    | 20/36 [01:17<01:02,  3.89s/it][38;20m2026-05-19 02:52:51 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.21:  58%|█████▊    | 21/36 [01:21<00:58,  3.89s/it][38;20m2026-05-19 02:52:55 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.22:  61%|██████    | 22/36 [01:26<00:54,  3.93s/it]

Process still running.

### [2026-05-19 02:53:04 UTC] ASSISTANT

**Thinking**

The quantization is still running. Let me wait for it to complete.


**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "warm-valley",
  "timeout": 600000
}
```

### [2026-05-19 02:53:55 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

[38;20m2026-05-19 02:52:59 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.23:  64%|██████▍   | 23/36 [01:30<00:51,  3.93s/it][38;20m2026-05-19 02:53:03 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.24:  67%|██████▋   | 24/36 [01:34<00:47,  3.92s/it][38;20m2026-05-19 02:53:07 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.25:  69%|██████▉   | 25/36 [01:38<00:43,  3.92s/it][38;20m2026-05-19 02:53:11 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.26:  72%|███████▏  | 26/36 [01:41<00:39,  3.92s/it][38;20m2026-05-19 02:53:15 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.27:  75%|███████▌  | 27/36 [01:45<00:35,  3.92s/it][38;20m2026-05-19 02:53:19 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.28:  78%|███████▊  | 28/36 [01:49<00:31,  3.91s/it][38;20m2026-05-19 02:53:22 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.29:  81%|████████  | 29/36 [01:53<00:27,  3.91s/it][38;20m2026-05-19 02:53:26 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.30:  83%|████████▎ | 30/36 [01:57<00:23,  3.91s/it][38;20m2026-05-19 02:53:30 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.30:  86%|████████▌ | 31/36 [02:01<00:19,  3.91s/it]
Quantizing model.layers.31:  86%|████████▌ | 31/36 [02:01<00:19,  3.91s/it][38;20m2026-05-19 02:53:35 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.32:  89%|████████▉ | 32/36 [02:05<00:15,  3.91s/it][38;20m2026-05-19 02:53:39 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.33:  92%|█████████▏| 33/36 [02:09<00:11,  3.91s/it][38;20m2026-05-19 02:53:43 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.34:  94%|█████████▍| 34/36 [02:13<00:07,  3.91s/it]
Quantizing model.layers.34:  94%|█████████▍| 34/36 [02:15<00:07,  3.91s/it][38;20m2026-05-19 02:53:47 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.35:  97%|█████████▋| 35/36 [02:17<00:03,  3.91s/it][38;20m2026-05-19 02:53:51 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.35: 100%|██████████| 36/36 [02:21<00:00,  3.93s/it]
[38;20m2026-05-19 02:53:53 INFO shard_writer.py L250: model has been saved to /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/[0m
[38;20m2026-05-19 02:53:53 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16

Process exited with code 0.

### [2026-05-19 02:54:02 UTC] ASSISTANT

**Thinking**

The quantization completed successfully. Now I need to verify the output files and create the quant_summary.json. Let me first check what was exported.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/"
}
```

**Tool call:** `exec`

```json
{
  "command": "du -sh /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/ 2>/dev/null; du -sh /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/* 2>/dev/null | head -20"
}
```

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs/quant_exec.log | tail -30"
}
```

### [2026-05-19 02:54:02 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 5966928
drwxr-xr-x 2 root root       4096 May 19 02:53 .
drwxr-xr-x 4 root root         48 May 19 02:51 ..
-rw-r--r-- 1 root root       4116 May 19 02:53 chat_template.jinja
-rw-r--r-- 1 root root       1836 May 19 02:53 config.json
-rw-r--r-- 1 root root        213 May 19 02:53 generation_config.json
-rw-r--r-- 1 root root 1050337864 May 19 02:52 model-00001-of-00007.safetensors
-rw-r--r-- 1 root root 1067770672 May 19 02:52 model-00002-of-00007.safetensors
-rw-r--r-- 1 root root 1063411832 May 19 02:53 model-00003-of-00007.safetensors
-rw-r--r-- 1 root root  427107856 May 19 02:53 model-00004-of-00007.safetensors
-rw-r--r-- 1 root root 1244659824 May 19 02:53 model-00005-of-00007.safetensors
-rw-r--r-- 1 root root     631752 May 19 02:53 model-00006-of-00007.safetensors
-rw-r--r-- 1 root root 1244659808 May 19 02:53 model-00007-of-00007.safetensors
-rw-r--r-- 1 root root      74116 May 19 02:53 model.safetensors.index.json
-rw-r--r-- 1 root root        196 May 19 02:53 quantization_config.json
-rw-r--r-- 1 root root   11422749 May 19 02:53 tokenizer.json
-rw-r--r-- 1 root root        405 May 19 02:53 tokenizer_config.json

### [2026-05-19 02:54:02 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

5.7G	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/
8.0K	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/chat_template.jinja
4.0K	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/config.json
4.0K	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/generation_config.json
1002M	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model-00001-of-00007.safetensors
1019M	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model-00002-of-00007.safetensors
1015M	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model-00003-of-00007.safetensors
408M	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model-00004-of-00007.safetensors
1.2G	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model-00005-of-00007.safetensors
620K	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model-00006-of-00007.safetensors
1.2G	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model-00007-of-00007.safetensors
76K	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/model.safetensors.index.json
4.0K	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/quantization_config.json
11M	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/tokenizer.json
4.0K	/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/tokenizer_config.json

### [2026-05-19 02:54:02 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Quantizing model.layers.10:  28%|██▊       | 10/36 [00:38<01:41,  3.89s/it][38;20m2026-05-19 02:52:12 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.11:  31%|███       | 11/36 [00:43<01:38,  3.93s/it][38;20m2026-05-19 02:52:16 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.12:  33%|███▎      | 12/36 [00:47<01:34,  3.92s/it][38;20m2026-05-19 02:52:20 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.13:  36%|███▌      | 13/36 [00:50<01:30,  3.92s/it][38;20m2026-05-19 02:52:24 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.14:  39%|███▉      | 14/36 [00:54<01:26,  3.91s/it][38;20m2026-05-19 02:52:28 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.15:  42%|████▏     | 15/36 [00:58<01:22,  3.91s/it][38;20m2026-05-19 02:52:31 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.16:  44%|████▍     | 16/36 [01:02<01:18,  3.90s/it][38;20m2026-05-19 02:52:35 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.17:  47%|████▋     | 17/36 [01:06<01:14,  3.90s/it][38;20m2026-05-19 02:52:39 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.18:  50%|█████     | 18/36 [01:10<01:10,  3.90s/it][38;20m2026-05-19 02:52:43 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.19:  53%|█████▎    | 19/36 [01:14<01:06,  3.90s/it][38;20m2026-05-19 02:52:47 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.20:  56%|█████▌    | 20/36 [01:17<01:02,  3.89s/it][38;20m2026-05-19 02:52:51 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.21:  58%|█████▊    | 21/36 [01:21<00:58,  3.89s/it][38;20m2026-05-19 02:52:55 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.22:  61%|██████    | 22/36 [01:26<00:54,  3.93s/it][38;20m2026-05-19 02:52:59 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.23:  64%|██████▍   | 23/36 [01:30<00:51,  3.93s/it][38;20m2026-05-19 02:53:03 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.24:  67%|██████▋   | 24/36 [01:34<00:47,  3.92s/it][38;20m2026-05-19 02:53:07 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.25:  69%|██████▉   | 25/36 [01:38<00:43,  3.92s/it][38;20m2026-05-19 02:53:11 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.26:  72%|███████▏  | 26/36 [01:41<00:39,  3.92s/it][38;20m2026-05-19 02:53:15 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.27:  75%|███████▌  | 27/36 [01:45<00:35,  3.92s/it][38;20m2026-05-19 02:53:19 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.28:  78%|███████▊  | 28/36 [01:49<00:31,  3.91s/it][38;20m2026-05-19 02:53:22 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.29:  81%|████████  | 29/36 [01:53<00:27,  3.91s/it][38;20m2026-05-19 02:53:26 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.30:  83%|████████▎ | 30/36 [01:57<00:23,  3.91s/it][38;20m2026-05-19 02:53:30 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.30:  86%|████████▌ | 31/36 [02:01<00:19,  3.91s/it]
Quantizing model.layers.31:  86%|████████▌ | 31/36 [02:01<00:19,  3.91s/it][38;20m2026-05-19 02:53:35 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.32:  89%|████████▉ | 32/36 [02:05<00:15,  3.91s/it][38;20m2026-05-19 02:53:39 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.33:  92%|█████████▏| 33/36 [02:09<00:11,  3.91s/it][38;20m2026-05-19 02:53:43 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.34:  94%|█████████▍| 34/36 [02:13<00:07,  3.91s/it]
Quantizing model.layers.34:  94%|█████████▍| 34/36 [02:15<00:07,  3.91s/it][38;20m2026-05-19 02:53:47 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.35:  97%|█████████▋| 35/36 [02:17<00:03,  3.91s/it][38;20m2026-05-19 02:53:51 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m

Quantizing model.layers.35: 100%|██████████| 36/36 [02:21<00:00,  3.93s/it]
[38;20m2026-05-19 02:53:53 INFO shard_writer.py L250: model has been saved to /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/[0m
[38;20m2026-05-19 02:53:53 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16

### [2026-05-19 02:54:15 UTC] ASSISTANT

**Thinking**

The quantization completed successfully. Now I need to compute the original model size and create the quant_summary.json.

Let me figure out the original model size. The model is Qwen/WebWorld-8B. I need to check what size the original FP16 model would be. The quantized model is 5.7GB. Since this is W4A16 (INT4 weight only, FP16 activation), the weight compression is roughly 4x (from FP16 16 bits to INT4 4 bits). So original FP16 model would be roughly 5.7GB * 4 = ~22.8GB.

Wait, let me be more precise. The model has 7 shard files plus index. The total is 5.7GB. For W4A16:


- Weights are quantized to 4-bit, while activations remain in FP16
- Compression ratio = 16/4 = 4x
- Original model size ≈ 5.7GB * 4 = ~22.8GB

I should verify this by downloading the config and calculating directly from the model architecture, or try pulling the original model size from HuggingFace to confirm.


**Tool call:** `exec`

```json
{
  "command": "curl -s https://huggingface.co/api/models/Qwen/WebWorld-8B 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print('siblings:', len(d.get('siblings',[]))); print('files:', [f['rfilename'] for f in d.get('siblings',[])[:5]])\" 2>/dev/null || echo \"Could not fetch\""
}
```

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/quantization_config.json"
}
```

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/config.json"
}
```

### [2026-05-19 02:54:16 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

siblings: 16
files: ['.gitattributes', 'LICENSE', 'README.md', 'added_tokens.json', 'config.json']

### [2026-05-19 02:54:16 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "bits": 4,
  "data_type": "int",
  "group_size": 128,
  "sym": true,
  "iters": 0,
  "autoround_version": "0.12.3",
  "quant_method": "auto-round",
  "packing_format": "auto_round:auto_gptq"
}

### [2026-05-19 02:54:16 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "pad_token_id": null,
  "quantization_config": {
    "autoround_version": "0.12.3",
    "bits": 4,
    "data_type": "int",
    "group_size": 128,
    "iters": 0,
    "packing_format": "auto_round:auto_gptq",
    "quant_method": "auto-round",
    "sym": true
  },
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 1000000,
    "rope_type": "default"
  },
  "sliding_window": null,
  "tie_word_embeddings": false,
  "transformers_version": "5.8.1",
  "use_cache": false,
  "use_sliding_window": false,
  "vocab_size": 151936
}

### [2026-05-19 02:54:39 UTC] ASSISTANT

**Thinking**

The quantization completed successfully. Let me estimate the original model size. Based on the model config:
- hidden_size: 4096
- num_hidden_layers: 36
- intermediate_size: 12288
- vocab_size: 151936
- head_dim: 128
- num_attention_heads: 32
- num_key_value_heads: 8

For an 8B model, approximate parameters:
- Embeddings: vocab_size * hidden_size = 151936 * 4096 ≈ 622M
- Each layer: 2 * (hidden_size * intermediate_size + hidden_size * hidden_size) ≈ 2 * (4096 * 12288 + 4096 * 4096) ≈ 2 * (50M + 17M) ≈ 134M per layer


- Total: ~36 * 134M + 622M ≈ 5.4B

The model size is around 8B. Let me calculate the compressed size more directly by summing up the shard files, which gives approximately 5.6GB compared to the original 16GB BF16 size, yielding a compression ratio of about 2.85x with W4A16 quantization. The export directory contains the quantized weights and model metadata.

The output appears to be an auto-round format with safetensors shards, adapter_config.json, tokenizer files, and quantization_config.json all present, though I should verify this is correct rather than making assumptions.

Let me check the runtime output directory to see what artifacts are there.


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/"
}
```

**Tool call:** `exec`

```json
{
  "command": "# Calculate original vs quantized size\nquantized_size=$(du -sb /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/ | cut -f1)\necho \"Quantized size bytes: $quantized_size\"\necho \"Quantized size MB: $(echo \"scale=2; $quantized_size / 1024 / 1024\" | bc)\"\n# Original FP16 8B model roughly 16GB (8B params * 2 bytes)\noriginal_size_mb=16000\nquantized_size_mb=$(echo \"scale=2; $quantized_size / 1024 / 1024\" | bc)\ncompression=$(echo \"scale=2; $original_size_mb / $quantized_size_mb\" | bc)\necho \"Original estimate: ${original_size_mb} MB\"\necho \"Quantized: ${quantized_size_mb} MB\"\necho \"Compression ratio: ${compression}x\""
}
```

**Tool call:** `exec`

```json
{
  "command": "# Get start and end time from log to calculate duration\nhead -1 /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs/quant_exec.log | head -1\ntail -3 /root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16/logs/quant_exec.log"
}
```

### [2026-05-19 02:54:39 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 8
drwxr-xr-x 4 root root   69 May 19 02:50 .
drwxr-xr-x 3 root root   36 May 19 02:48 ..
drwxr-xr-x 2 root root   68 May 19 02:50 logs
-rw-r--r-- 1 root root 1054 May 19 02:50 quantize.py
-rw-r--r-- 1 root root 1078 May 19 02:48 request.json
drwxr-xr-x 6 root root  100 May 19 02:49 venv

### [2026-05-19 02:54:39 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Quantized size bytes: 6110083239
Quantized size MB: 
sh: 4: bc: not found
sh: 7: bc: not found
sh: 8: bc: not found
Original estimate: 16000 MB
Quantized:  MB
Compression ratio: x

### [2026-05-19 02:54:39 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Loading model: Qwen/WebWorld-8B
[38;20m2026-05-19 02:53:53 INFO shard_writer.py L250: model has been saved to /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16/[0m
[38;20m2026-05-19 02:53:53 INFO device.py L1692: 'peak_ram': 24.91GB, 'peak_vram': 7.37GB[0m
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16

### [2026-05-19 02:54:48 UTC] ASSISTANT

**Thinking**

Let me calculate the sizes properly with Python since bc isn't available, and get exact model size from HuggingFace API.


**Tool call:** `exec`

```json
{
  "command": "python3 -c \"\nimport urllib.request, json\n# Get original model size\nurl = 'https://huggingface.co/api/models/Qwen/WebWorld-8B'\nreq = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})\nwith urllib.request.urlopen(req, timeout=10) as r:\n    data = json.loads(r.read())\n# Get the files\nsiblings = data.get('siblings', [])\ntotal_size = sum(f.get('size', 0) for f in siblings)\nprint(f'Original model files count: {len(siblings)}')\nprint(f'Original model total size (MB): {total_size / 1024**2:.2f}')\n\""
}
```

### [2026-05-19 02:54:48 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Original model files count: 16
Original model total size (MB): 0.00

### [2026-05-19 02:55:03 UTC] ASSISTANT

**Thinking**

The API didn't return file sizes. Let me just use the quantization info. The model was loaded as bfloat16, so we can estimate original size. Let me calculate everything properly in Python.


**Tool call:** `exec`

```json
{
  "command": "python3 << 'EOF'\nimport os\nimport json\nfrom pathlib import Path\n\nquantized_dir = Path(\"/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16\")\nruntime_dir = Path(\"/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16\")\n\n# Calculate quantized size\ntotal_bytes = sum(f.stat().st_size for f in quantized_dir.rglob(\"*\") if f.is_file())\nquantized_size_mb = total_bytes / (1024 * 1024)\n\n# Original BF16 size estimate: hidden_size=4096, num_hidden_layers=36, vocab_size=151936\n# Approximate: 8B params * 2 bytes = ~16GB for BF16\noriginal_size_mb = 16000.0\n\ncompression_ratio = original_size_mb / quantized_size_mb\n\nprint(f\"Quantized size: {quantized_size_mb:.2f} MB ({total_bytes} bytes)\")\nprint(f\"Estimated original size: {original_size_mb:.2f} MB\")\nprint(f\"Compression ratio: {compression_ratio:.2f}x\")\n\n# Duration\nstart_time_str = \"2026-05-19 02:51:12\"\nend_time_str = \"2026-05-19 02:53:53\"\nfrom datetime import datetime\nstart = datetime.fromisoformat(start_time_str)\nend = datetime.fromisoformat(end_time_str)\nduration = (end - start).total_seconds()\nprint(f\"Duration: {duration:.2f} seconds\")\n\n# List output files in runtime dir\noutput_files = sorted([str(f.relative_to(runtime_dir)) for f in runtime_dir.rglob(\"*\") if f.is_file()])\nprint(f\"Runtime output files: {output_files}\")\nEOF"
}
```

### [2026-05-19 02:55:04 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

v/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/chunk_utils.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/data_transforms.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/feats.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/loss.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/protein.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/residue_constants.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/rigid_utils.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/openfold_utils/tensor_utils.py', 'venv/lib/python3.12/site-packages/transformers/models/esm/tokenization_esm.py', 'venv/lib/python3.12/site-packages/transformers/models/eurobert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/eurobert/configuration_eurobert.py', 'venv/lib/python3.12/site-packages/transformers/models/eurobert/modeling_eurobert.py', 'venv/lib/python3.12/site-packages/transformers/models/eurobert/modular_eurobert.py', 'venv/lib/python3.12/site-packages/transformers/models/evolla/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/evolla/configuration_evolla.py', 'venv/lib/python3.12/site-packages/transformers/models/evolla/modeling_evolla.py', 'venv/lib/python3.12/site-packages/transformers/models/evolla/modular_evolla.py', 'venv/lib/python3.12/site-packages/transformers/models/evolla/processing_evolla.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4/configuration_exaone4.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4/modeling_exaone4.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4/modular_exaone4.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4_5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4_5/configuration_exaone4_5.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4_5/modeling_exaone4_5.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4_5/modular_exaone4_5.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone4_5/processing_exaone4_5.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone_moe/configuration_exaone_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone_moe/modeling_exaone_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/exaone_moe/modular_exaone_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon/configuration_falcon.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon/modeling_falcon.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_h1/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_h1/configuration_falcon_h1.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_h1/modular_falcon_h1.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_mamba/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_mamba/configuration_falcon_mamba.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_mamba/modeling_falcon_mamba.py', 'venv/lib/python3.12/site-packages/transformers/models/falcon_mamba/modular_falcon_mamba.py', 'venv/lib/python3.12/site-packages/transformers/models/fast_vlm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/fast_vlm/configuration_fast_vlm.py', 'venv/lib/python3.12/site-packages/transformers/models/fast_vlm/modeling_fast_vlm.py', 'venv/lib/python3.12/site-packages/transformers/models/fast_vlm/modular_fast_vlm.py', 'venv/lib/python3.12/site-packages/transformers/models/fastspeech2_conformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/fastspeech2_conformer/configuration_fastspeech2_conformer.py', 'venv/lib/python3.12/site-packages/transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer.py', 'venv/lib/python3.12/site-packages/transformers/models/fastspeech2_conformer/tokenization_fastspeech2_conformer.py', 'venv/lib/python3.12/site-packages/transformers/models/flaubert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/flaubert/configuration_flaubert.py', 'venv/lib/python3.12/site-packages/transformers/models/flaubert/modeling_flaubert.py', 'venv/lib/python3.12/site-packages/transformers/models/flaubert/tokenization_flaubert.py', 'venv/lib/python3.12/site-packages/transformers/models/flava/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/flava/configuration_flava.py', 'venv/lib/python3.12/site-packages/transformers/models/flava/image_processing_flava.py', 'venv/lib/python3.12/site-packages/transformers/models/flava/image_processing_pil_flava.py', 'venv/lib/python3.12/site-packages/transformers/models/flava/modeling_flava.py', 'venv/lib/python3.12/site-packages/transformers/models/flava/processing_flava.py', 'venv/lib/python3.12/site-packages/transformers/models/flex_olmo/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/flex_olmo/configuration_flex_olmo.py', 'venv/lib/python3.12/site-packages/transformers/models/flex_olmo/modeling_flex_olmo.py', 'venv/lib/python3.12/site-packages/transformers/models/flex_olmo/modular_flex_olmo.py', 'venv/lib/python3.12/site-packages/transformers/models/florence2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/florence2/configuration_florence2.py', 'venv/lib/python3.12/site-packages/transformers/models/florence2/modeling_florence2.py', 'venv/lib/python3.12/site-packages/transformers/models/florence2/modular_florence2.py', 'venv/lib/python3.12/site-packages/transformers/models/florence2/processing_florence2.py', 'venv/lib/python3.12/site-packages/transformers/models/fnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/fnet/configuration_fnet.py', 'venv/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py', 'venv/lib/python3.12/site-packages/transformers/models/fnet/tokenization_fnet.py', 'venv/lib/python3.12/site-packages/transformers/models/focalnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/focalnet/configuration_focalnet.py', 'venv/lib/python3.12/site-packages/transformers/models/focalnet/modeling_focalnet.py', 'venv/lib/python3.12/site-packages/transformers/models/fsmt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/fsmt/configuration_fsmt.py', 'venv/lib/python3.12/site-packages/transformers/models/fsmt/modeling_fsmt.py', 'venv/lib/python3.12/site-packages/transformers/models/fsmt/tokenization_fsmt.py', 'venv/lib/python3.12/site-packages/transformers/models/funnel/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/funnel/configuration_funnel.py', 'venv/lib/python3.12/site-packages/transformers/models/funnel/modeling_funnel.py', 'venv/lib/python3.12/site-packages/transformers/models/funnel/tokenization_funnel.py', 'venv/lib/python3.12/site-packages/transformers/models/fuyu/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/fuyu/configuration_fuyu.py', 'venv/lib/python3.12/site-packages/transformers/models/fuyu/image_processing_fuyu.py', 'venv/lib/python3.12/site-packages/transformers/models/fuyu/image_processing_pil_fuyu.py', 'venv/lib/python3.12/site-packages/transformers/models/fuyu/modeling_fuyu.py', 'venv/lib/python3.12/site-packages/transformers/models/fuyu/processing_fuyu.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma/configuration_gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma/modeling_gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma/modular_gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma/tokenization_gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma2/configuration_gemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma2/modular_gemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3/configuration_gemma3.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3/image_processing_gemma3.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3/image_processing_pil_gemma3.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3/modeling_gemma3.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3/modular_gemma3.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3/processing_gemma3.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3n/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3n/configuration_gemma3n.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3n/feature_extraction_gemma3n.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3n/modeling_gemma3n.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3n/modular_gemma3n.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma3n/processing_gemma3n.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/configuration_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/feature_extraction_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/image_processing_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/image_processing_pil_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/modular_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/processing_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4/video_processing_gemma4.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4_assistant/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4_assistant/configuration_gemma4_assistant.py', 'venv/lib/python3.12/site-packages/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py', 'venv/lib/python3.12/site-packages/transformers/models/git/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/git/configuration_git.py', 'venv/lib/python3.12/site-packages/transformers/models/git/modeling_git.py', 'venv/lib/python3.12/site-packages/transformers/models/git/processing_git.py', 'venv/lib/python3.12/site-packages/transformers/models/glm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm/configuration_glm.py', 'venv/lib/python3.12/site-packages/transformers/models/glm/modeling_glm.py', 'venv/lib/python3.12/site-packages/transformers/models/glm/modular_glm.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4/configuration_glm4.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4/modeling_glm4.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4/modular_glm4.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/configuration_glm46v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/image_processing_glm46v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/image_processing_pil_glm46v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/modeling_glm46v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/modular_glm46v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/processing_glm46v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm46v/video_processing_glm46v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe/configuration_glm4_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe/modeling_glm4_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe/modular_glm4_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/configuration_glm4_moe_lite.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/modular_glm4_moe_lite.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/configuration_glm4v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/image_processing_glm4v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/image_processing_pil_glm4v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/modeling_glm4v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/modular_glm4v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/processing_glm4v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v/video_processing_glm4v.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/configuration_glm4v_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/modeling_glm4v_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/modular_glm4v_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_image/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_image/configuration_glm_image.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_image/image_processing_glm_image.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_image/image_processing_pil_glm_image.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_image/modeling_glm_image.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_image/modular_glm_image.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_image/processing_glm_image.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_ocr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_ocr/configuration_glm_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_ocr/modeling_glm_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/glm_ocr/modular_glm_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/glmasr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glmasr/configuration_glmasr.py', 'venv/lib/python3.12/site-packages/transformers/models/glmasr/modeling_glmasr.py', 'venv/lib/python3.12/site-packages/transformers/models/glmasr/modular_glmasr.py', 'venv/lib/python3.12/site-packages/transformers/models/glmasr/processing_glmasr.py', 'venv/lib/python3.12/site-packages/transformers/models/glpn/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/glpn/configuration_glpn.py', 'venv/lib/python3.12/site-packages/transformers/models/glpn/image_processing_glpn.py', 'venv/lib/python3.12/site-packages/transformers/models/glpn/image_processing_pil_glpn.py', 'venv/lib/python3.12/site-packages/transformers/models/glpn/modeling_glpn.py', 'venv/lib/python3.12/site-packages/transformers/models/got_ocr2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/got_ocr2/configuration_got_ocr2.py', 'venv/lib/python3.12/site-packages/transformers/models/got_ocr2/image_processing_got_ocr2.py', 'venv/lib/python3.12/site-packages/transformers/models/got_ocr2/image_processing_pil_got_ocr2.py', 'venv/lib/python3.12/site-packages/transformers/models/got_ocr2/modeling_got_ocr2.py', 'venv/lib/python3.12/site-packages/transformers/models/got_ocr2/modular_got_ocr2.py', 'venv/lib/python3.12/site-packages/transformers/models/got_ocr2/processing_got_ocr2.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt2/configuration_gpt2.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt2/tokenization_gpt2.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_bigcode/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neo/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neo/configuration_gpt_neo.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox/configuration_gpt_neox.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox/modular_gpt_neox.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox/tokenization_gpt_neox.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/configuration_gpt_neox_japanese.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/tokenization_gpt_neox_japanese.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_oss/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_oss/configuration_gpt_oss.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_oss/modular_gpt_oss.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_sw3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gpt_sw3/tokenization_gpt_sw3.py', 'venv/lib/python3.12/site-packages/transformers/models/gptj/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/gptj/configuration_gptj.py', 'venv/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py', 'venv/lib/python3.12/site-packages/transformers/models/granite/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/granite/configuration_granite.py', 'venv/lib/python3.12/site-packages/transformers/models/granite/modeling_granite.py', 'venv/lib/python3.12/site-packages/transformers/models/granite/modular_granite.py', 'venv/lib/python3.12/site-packages/transformers/models/granite4_vision/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/granite4_vision/configuration_granite4_vision.py', 'venv/lib/python3.12/site-packages/transformers/models/granite4_vision/modeling_granite4_vision.py', 'venv/lib/python3.12/site-packages/transformers/models/granite4_vision/modular_granite4_vision.py', 'venv/lib/python3.12/site-packages/transformers/models/granite4_vision/processing_granite4_vision.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech/configuration_granite_speech.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech/feature_extraction_granite_speech.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech/modeling_granite_speech.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech/processing_granite_speech.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/configuration_granite_speech_plus.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/modeling_granite_speech_plus.py', 'venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/modular_granite_speech_plus.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoe/configuration_granitemoe.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoe/modeling_granitemoe.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoe/modular_granitemoe.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/configuration_granitemoehybrid.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/modular_granitemoehybrid.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/configuration_granitemoeshared.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/modeling_granitemoeshared.py', 'venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/modular_granitemoeshared.py', 'venv/lib/python3.12/site-packages/transformers/models/grounding_dino/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/grounding_dino/configuration_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/grounding_dino/image_processing_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/grounding_dino/image_processing_pil_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/grounding_dino/modular_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/grounding_dino/processing_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/groupvit/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/groupvit/configuration_groupvit.py', 'venv/lib/python3.12/site-packages/transformers/models/groupvit/modeling_groupvit.py', 'venv/lib/python3.12/site-packages/transformers/models/helium/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/helium/configuration_helium.py', 'venv/lib/python3.12/site-packages/transformers/models/helium/modeling_helium.py', 'venv/lib/python3.12/site-packages/transformers/models/helium/modular_helium.py', 'venv/lib/python3.12/site-packages/transformers/models/herbert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/herbert/tokenization_herbert.py', 'venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/configuration_hgnet_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/modeling_hgnet_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/modular_hgnet_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/hiera/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/hiera/configuration_hiera.py', 'venv/lib/python3.12/site-packages/transformers/models/hiera/modeling_hiera.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/configuration_higgs_audio_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/generation_higgs_audio_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/modeling_higgs_audio_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/modular_higgs_audio_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/processing_higgs_audio_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/configuration_higgs_audio_v2_tokenizer.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py', 'venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/modular_higgs_audio_v2_tokenizer.py', 'venv/lib/python3.12/site-packages/transformers/models/hubert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/hubert/configuration_hubert.py', 'venv/lib/python3.12/site-packages/transformers/models/hubert/modeling_hubert.py', 'venv/lib/python3.12/site-packages/transformers/models/hubert/modular_hubert.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/configuration_hunyuan_v1_dense.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/modular_hunyuan_v1_dense.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/configuration_hunyuan_v1_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/modular_hunyuan_v1_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/hy_v3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/hy_v3/configuration_hy_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/hy_v3/modeling_hy_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/hy_v3/modular_hy_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/ibert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/ibert/configuration_ibert.py', 'venv/lib/python3.12/site-packages/transformers/models/ibert/modeling_ibert.py', 'venv/lib/python3.12/site-packages/transformers/models/ibert/quant_modules.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/configuration_idefics.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/image_processing_idefics.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/image_processing_pil_idefics.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/modeling_idefics.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/perceiver.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/processing_idefics.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics/vision.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics2/configuration_idefics2.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics2/image_processing_idefics2.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics2/image_processing_pil_idefics2.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics2/modeling_idefics2.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics2/processing_idefics2.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics3/configuration_idefics3.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics3/image_processing_idefics3.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics3/image_processing_pil_idefics3.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py', 'venv/lib/python3.12/site-packages/transformers/models/idefics3/processing_idefics3.py', 'venv/lib/python3.12/site-packages/transformers/models/ijepa/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/ijepa/configuration_ijepa.py', 'venv/lib/python3.12/site-packages/transformers/models/ijepa/modeling_ijepa.py', 'venv/lib/python3.12/site-packages/transformers/models/ijepa/modular_ijepa.py', 'venv/lib/python3.12/site-packages/transformers/models/imagegpt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/imagegpt/configuration_imagegpt.py', 'venv/lib/python3.12/site-packages/transformers/models/imagegpt/image_processing_imagegpt.py', 'venv/lib/python3.12/site-packages/transformers/models/imagegpt/image_processing_pil_imagegpt.py', 'venv/lib/python3.12/site-packages/transformers/models/imagegpt/modeling_imagegpt.py', 'venv/lib/python3.12/site-packages/transformers/models/informer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/informer/configuration_informer.py', 'venv/lib/python3.12/site-packages/transformers/models/informer/modeling_informer.py', 'venv/lib/python3.12/site-packages/transformers/models/informer/modular_informer.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblip/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblip/configuration_instructblip.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblip/modeling_instructblip.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblip/processing_instructblip.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/configuration_instructblipvideo.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/modeling_instructblipvideo.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/modular_instructblipvideo.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/processing_instructblipvideo.py', 'venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/video_processing_instructblipvideo.py', 'venv/lib/python3.12/site-packages/transformers/models/internvl/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/internvl/configuration_internvl.py', 'venv/lib/python3.12/site-packages/transformers/models/internvl/modeling_internvl.py', 'venv/lib/python3.12/site-packages/transformers/models/internvl/modular_internvl.py', 'venv/lib/python3.12/site-packages/transformers/models/internvl/processing_internvl.py', 'venv/lib/python3.12/site-packages/transformers/models/internvl/video_processing_internvl.py', 'venv/lib/python3.12/site-packages/transformers/models/jais2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/jais2/configuration_jais2.py', 'venv/lib/python3.12/site-packages/transformers/models/jais2/modeling_jais2.py', 'venv/lib/python3.12/site-packages/transformers/models/jais2/modular_jais2.py', 'venv/lib/python3.12/site-packages/transformers/models/jamba/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/jamba/configuration_jamba.py', 'venv/lib/python3.12/site-packages/transformers/models/jamba/modeling_jamba.py', 'venv/lib/python3.12/site-packages/transformers/models/jamba/modular_jamba.py', 'venv/lib/python3.12/site-packages/transformers/models/janus/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/janus/configuration_janus.py', 'venv/lib/python3.12/site-packages/transformers/models/janus/image_processing_janus.py', 'venv/lib/python3.12/site-packages/transformers/models/janus/image_processing_pil_janus.py', 'venv/lib/python3.12/site-packages/transformers/models/janus/modeling_janus.py', 'venv/lib/python3.12/site-packages/transformers/models/janus/modular_janus.py', 'venv/lib/python3.12/site-packages/transformers/models/janus/processing_janus.py', 'venv/lib/python3.12/site-packages/transformers/models/jetmoe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/jetmoe/configuration_jetmoe.py', 'venv/lib/python3.12/site-packages/transformers/models/jetmoe/modeling_jetmoe.py', 'venv/lib/python3.12/site-packages/transformers/models/jetmoe/modular_jetmoe.py', 'venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/configuration_jina_embeddings_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/modeling_jina_embeddings_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/modular_jina_embeddings_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2/configuration_kosmos2.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2/modeling_kosmos2.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2/processing_kosmos2.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/configuration_kosmos2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/image_processing_kosmos2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/image_processing_pil_kosmos2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/modeling_kosmos2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/processing_kosmos2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/configuration_kyutai_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/feature_extraction_kyutai_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/modular_kyutai_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/processing_kyutai_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/laguna/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/laguna/configuration_laguna.py', 'venv/lib/python3.12/site-packages/transformers/models/laguna/modeling_laguna.py', 'venv/lib/python3.12/site-packages/transformers/models/laguna/modular_laguna.py', 'venv/lib/python3.12/site-packages/transformers/models/lasr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lasr/configuration_lasr.py', 'venv/lib/python3.12/site-packages/transformers/models/lasr/feature_extraction_lasr.py', 'venv/lib/python3.12/site-packages/transformers/models/lasr/modeling_lasr.py', 'venv/lib/python3.12/site-packages/transformers/models/lasr/modular_lasr.py', 'venv/lib/python3.12/site-packages/transformers/models/lasr/processing_lasr.py', 'venv/lib/python3.12/site-packages/transformers/models/lasr/tokenization_lasr.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlm/configuration_layoutlm.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/configuration_layoutlmv2.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/image_processing_layoutlmv2.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/image_processing_pil_layoutlmv2.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/modeling_layoutlmv2.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/processing_layoutlmv2.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/tokenization_layoutlmv2.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/configuration_layoutlmv3.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/image_processing_layoutlmv3.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/image_processing_pil_layoutlmv3.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/modeling_layoutlmv3.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/processing_layoutlmv3.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/tokenization_layoutlmv3.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutxlm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutxlm/configuration_layoutxlm.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutxlm/modular_layoutxlm.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutxlm/processing_layoutxlm.py', 'venv/lib/python3.12/site-packages/transformers/models/layoutxlm/tokenization_layoutxlm.py', 'venv/lib/python3.12/site-packages/transformers/models/led/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/led/configuration_led.py', 'venv/lib/python3.12/site-packages/transformers/models/led/modeling_led.py', 'venv/lib/python3.12/site-packages/transformers/models/levit/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/levit/configuration_levit.py', 'venv/lib/python3.12/site-packages/transformers/models/levit/image_processing_levit.py', 'venv/lib/python3.12/site-packages/transformers/models/levit/image_processing_pil_levit.py', 'venv/lib/python3.12/site-packages/transformers/models/levit/modeling_levit.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2/configuration_lfm2.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2/modeling_lfm2.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2/modular_lfm2.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/configuration_lfm2_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/modeling_lfm2_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/modular_lfm2_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/configuration_lfm2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/image_processing_lfm2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/modeling_lfm2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/modular_lfm2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/processing_lfm2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/lightglue/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lightglue/configuration_lightglue.py', 'venv/lib/python3.12/site-packages/transformers/models/lightglue/image_processing_lightglue.py', 'venv/lib/python3.12/site-packages/transformers/models/lightglue/image_processing_pil_lightglue.py', 'venv/lib/python3.12/site-packages/transformers/models/lightglue/modeling_lightglue.py', 'venv/lib/python3.12/site-packages/transformers/models/lightglue/modular_lightglue.py', 'venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/configuration_lighton_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/modeling_lighton_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/modular_lighton_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/processing_lighton_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/lilt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lilt/configuration_lilt.py', 'venv/lib/python3.12/site-packages/transformers/models/lilt/modeling_lilt.py', 'venv/lib/python3.12/site-packages/transformers/models/llama/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py', 'venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py', 'venv/lib/python3.12/site-packages/transformers/models/llama/tokenization_llama.py', 'venv/lib/python3.12/site-packages/transformers/models/llama4/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/llama4/configuration_llama4.py', 'venv/lib/python3.12/site-packages/transformers/models/llama4/image_processing_llama4.py', 'venv/lib/python3.12/site-packages/transformers/models/llama4/modeling_llama4.py', 'venv/lib/python3.12/site-packages/transformers/models/llama4/processing_llama4.py', 'venv/lib/python3.12/site-packages/transformers/models/llava/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/llava/configuration_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/llava/image_processing_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/llava/image_processing_pil_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/llava/modeling_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/llava/processing_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next/configuration_llava_next.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_pil_llava_next.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next/modeling_llava_next.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next/processing_llava_next.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next_video/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next_video/configuration_llava_next_video.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next_video/modeling_llava_next_video.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next_video/modular_llava_next_video.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next_video/processing_llava_next_video.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_next_video/video_processing_llava_next_video.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/configuration_llava_onevision.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/image_processing_llava_onevision.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/image_processing_pil_llava_onevision.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/modeling_llava_onevision.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/modular_llava_onevision.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/processing_llava_onevision.py', 'venv/lib/python3.12/site-packages/transformers/models/llava_onevision/video_processing_llava_onevision.py', 'venv/lib/python3.12/site-packages/transformers/models/longcat_flash/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/longcat_flash/configuration_longcat_flash.py', 'venv/lib/python3.12/site-packages/transformers/models/longcat_flash/modeling_longcat_flash.py', 'venv/lib/python3.12/site-packages/transformers/models/longcat_flash/modular_longcat_flash.py', 'venv/lib/python3.12/site-packages/transformers/models/longformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/longformer/configuration_longformer.py', 'venv/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py', 'venv/lib/python3.12/site-packages/transformers/models/longt5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/longt5/configuration_longt5.py', 'venv/lib/python3.12/site-packages/transformers/models/longt5/modeling_longt5.py', 'venv/lib/python3.12/site-packages/transformers/models/luke/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/luke/configuration_luke.py', 'venv/lib/python3.12/site-packages/transformers/models/luke/modeling_luke.py', 'venv/lib/python3.12/site-packages/transformers/models/luke/tokenization_luke.py', 'venv/lib/python3.12/site-packages/transformers/models/lw_detr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lw_detr/configuration_lw_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/lw_detr/modeling_lw_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/lw_detr/modular_lw_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/lxmert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/lxmert/configuration_lxmert.py', 'venv/lib/python3.12/site-packages/transformers/models/lxmert/modeling_lxmert.py', 'venv/lib/python3.12/site-packages/transformers/models/m2m_100/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/m2m_100/configuration_m2m_100.py', 'venv/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py', 'venv/lib/python3.12/site-packages/transformers/models/m2m_100/tokenization_m2m_100.py', 'venv/lib/python3.12/site-packages/transformers/models/mamba/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mamba/configuration_mamba.py', 'venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py', 'venv/lib/python3.12/site-packages/transformers/models/mamba2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mamba2/configuration_mamba2.py', 'venv/lib/python3.12/site-packages/transformers/models/mamba2/modeling_mamba2.py', 'venv/lib/python3.12/site-packages/transformers/models/marian/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/marian/configuration_marian.py', 'venv/lib/python3.12/site-packages/transformers/models/marian/modeling_marian.py', 'venv/lib/python3.12/site-packages/transformers/models/marian/tokenization_marian.py', 'venv/lib/python3.12/site-packages/transformers/models/markuplm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/markuplm/configuration_markuplm.py', 'venv/lib/python3.12/site-packages/transformers/models/markuplm/feature_extraction_markuplm.py', 'venv/lib/python3.12/site-packages/transformers/models/markuplm/modeling_markuplm.py', 'venv/lib/python3.12/site-packages/transformers/models/markuplm/processing_markuplm.py', 'venv/lib/python3.12/site-packages/transformers/models/markuplm/tokenization_markuplm.py', 'venv/lib/python3.12/site-packages/transformers/models/mask2former/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mask2former/configuration_mask2former.py', 'venv/lib/python3.12/site-packages/transformers/models/mask2former/image_processing_mask2former.py', 'venv/lib/python3.12/site-packages/transformers/models/mask2former/image_processing_pil_mask2former.py', 'venv/lib/python3.12/site-packages/transformers/models/mask2former/modeling_mask2former.py', 'venv/lib/python3.12/site-packages/transformers/models/mask2former/modular_mask2former.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/configuration_maskformer.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/configuration_maskformer_swin.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/image_processing_maskformer.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/image_processing_pil_maskformer.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/modeling_maskformer.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/modeling_maskformer_swin.py', 'venv/lib/python3.12/site-packages/transformers/models/maskformer/modular_maskformer.py', 'venv/lib/python3.12/site-packages/transformers/models/mbart/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mbart/configuration_mbart.py', 'venv/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py', 'venv/lib/python3.12/site-packages/transformers/models/mbart/tokenization_mbart.py', 'venv/lib/python3.12/site-packages/transformers/models/mbart50/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mbart50/tokenization_mbart50.py', 'venv/lib/python3.12/site-packages/transformers/models/megatron_bert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/megatron_bert/configuration_megatron_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/megatron_gpt2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py', 'venv/lib/python3.12/site-packages/transformers/models/metaclip_2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/metaclip_2/configuration_metaclip_2.py', 'venv/lib/python3.12/site-packages/transformers/models/metaclip_2/modeling_metaclip_2.py', 'venv/lib/python3.12/site-packages/transformers/models/metaclip_2/modular_metaclip_2.py', 'venv/lib/python3.12/site-packages/transformers/models/mgp_str/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mgp_str/configuration_mgp_str.py', 'venv/lib/python3.12/site-packages/transformers/models/mgp_str/modeling_mgp_str.py', 'venv/lib/python3.12/site-packages/transformers/models/mgp_str/processing_mgp_str.py', 'venv/lib/python3.12/site-packages/transformers/models/mgp_str/tokenization_mgp_str.py', 'venv/lib/python3.12/site-packages/transformers/models/mimi/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mimi/configuration_mimi.py', 'venv/lib/python3.12/site-packages/transformers/models/mimi/modeling_mimi.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/configuration_minicpmv4_6.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/image_processing_minicpmv4_6.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/image_processing_pil_minicpmv4_6.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/modeling_minicpmv4_6.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/modular_minicpmv4_6.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/processing_minicpmv4_6.py', 'venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax/configuration_minimax.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax/modeling_minimax.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax/modular_minimax.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax_m2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax_m2/configuration_minimax_m2.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax_m2/modeling_minimax_m2.py', 'venv/lib/python3.12/site-packages/transformers/models/minimax_m2/modular_minimax_m2.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral/configuration_ministral.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral/modeling_ministral.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral/modular_ministral.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral3/configuration_ministral3.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral3/modeling_ministral3.py', 'venv/lib/python3.12/site-packages/transformers/models/ministral3/modular_ministral3.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral/configuration_mistral.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral/modular_mistral.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral3/configuration_mistral3.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral3/modeling_mistral3.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral3/modular_mistral3.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral4/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral4/configuration_mistral4.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral4/modeling_mistral4.py', 'venv/lib/python3.12/site-packages/transformers/models/mistral4/modular_mistral4.py', 'venv/lib/python3.12/site-packages/transformers/models/mixtral/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mixtral/configuration_mixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/mixtral/modeling_mixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/mixtral/modular_mixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/mlcd/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mlcd/configuration_mlcd.py', 'venv/lib/python3.12/site-packages/transformers/models/mlcd/modeling_mlcd.py', 'venv/lib/python3.12/site-packages/transformers/models/mlcd/modular_mlcd.py', 'venv/lib/python3.12/site-packages/transformers/models/mllama/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mllama/configuration_mllama.py', 'venv/lib/python3.12/site-packages/transformers/models/mllama/image_processing_mllama.py', 'venv/lib/python3.12/site-packages/transformers/models/mllama/image_processing_pil_mllama.py', 'venv/lib/python3.12/site-packages/transformers/models/mllama/modeling_mllama.py', 'venv/lib/python3.12/site-packages/transformers/models/mllama/processing_mllama.py', 'venv/lib/python3.12/site-packages/transformers/models/mluke/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mluke/tokenization_mluke.py', 'venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/configuration_mm_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/modular_mm_grounding_dino.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilebert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilebert/configuration_mobilebert.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilebert/tokenization_mobilebert.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/configuration_mobilenet_v1.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/image_processing_mobilenet_v1.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/image_processing_pil_mobilenet_v1.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/configuration_mobilenet_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/image_processing_mobilenet_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/image_processing_pil_mobilenet_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/modeling_mobilenet_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevit/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevit/configuration_mobilevit.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevit/image_processing_mobilevit.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevit/image_processing_pil_mobilevit.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevit/modeling_mobilevit.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevitv2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevitv2/configuration_mobilevitv2.py', 'venv/lib/python3.12/site-packages/transformers/models/mobilevitv2/modeling_mobilevitv2.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert/configuration_modernbert.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert/modeling_modernbert.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert/modular_modernbert.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/configuration_modernbert_decoder.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/modeling_modernbert_decoder.py', 'venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/modular_modernbert_decoder.py', 'venv/lib/python3.12/site-packages/transformers/models/modernvbert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/modernvbert/configuration_modernvbert.py', 'venv/lib/python3.12/site-packages/transformers/models/modernvbert/modeling_modernvbert.py', 'venv/lib/python3.12/site-packages/transformers/models/modernvbert/modular_modernvbert.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine/configuration_moonshine.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine/modeling_moonshine.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine/modular_moonshine.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/configuration_moonshine_streaming.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/modeling_moonshine_streaming.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/modular_moonshine_streaming.py', 'venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/processing_moonshine_streaming.py', 'venv/lib/python3.12/site-packages/transformers/models/moshi/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/moshi/configuration_moshi.py', 'venv/lib/python3.12/site-packages/transformers/models/moshi/modeling_moshi.py', 'venv/lib/python3.12/site-packages/transformers/models/mpnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mpnet/configuration_mpnet.py', 'venv/lib/python3.12/site-packages/transformers/models/mpnet/modeling_mpnet.py', 'venv/lib/python3.12/site-packages/transformers/models/mpnet/tokenization_mpnet.py', 'venv/lib/python3.12/site-packages/transformers/models/mpt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mpt/configuration_mpt.py', 'venv/lib/python3.12/site-packages/transformers/models/mpt/modeling_mpt.py', 'venv/lib/python3.12/site-packages/transformers/models/mra/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mra/configuration_mra.py', 'venv/lib/python3.12/site-packages/transformers/models/mra/modeling_mra.py', 'venv/lib/python3.12/site-packages/transformers/models/mt5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mt5/configuration_mt5.py', 'venv/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py', 'venv/lib/python3.12/site-packages/transformers/models/musicflamingo/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/musicflamingo/configuration_musicflamingo.py', 'venv/lib/python3.12/site-packages/transformers/models/musicflamingo/modeling_musicflamingo.py', 'venv/lib/python3.12/site-packages/transformers/models/musicflamingo/modular_musicflamingo.py', 'venv/lib/python3.12/site-packages/transformers/models/musicflamingo/processing_musicflamingo.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen/configuration_musicgen.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen/modeling_musicgen.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen/processing_musicgen.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/configuration_musicgen_melody.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/feature_extraction_musicgen_melody.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/modeling_musicgen_melody.py', 'venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/processing_musicgen_melody.py', 'venv/lib/python3.12/site-packages/transformers/models/mvp/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/mvp/configuration_mvp.py', 'venv/lib/python3.12/site-packages/transformers/models/mvp/modeling_mvp.py', 'venv/lib/python3.12/site-packages/transformers/models/myt5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/myt5/tokenization_myt5.py', 'venv/lib/python3.12/site-packages/transformers/models/nanochat/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nanochat/configuration_nanochat.py', 'venv/lib/python3.12/site-packages/transformers/models/nanochat/modeling_nanochat.py', 'venv/lib/python3.12/site-packages/transformers/models/nanochat/modular_nanochat.py', 'venv/lib/python3.12/site-packages/transformers/models/nemotron/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nemotron/configuration_nemotron.py', 'venv/lib/python3.12/site-packages/transformers/models/nemotron/modeling_nemotron.py', 'venv/lib/python3.12/site-packages/transformers/models/nemotron_h/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nemotron_h/configuration_nemotron_h.py', 'venv/lib/python3.12/site-packages/transformers/models/nemotron_h/modeling_nemotron_h.py', 'venv/lib/python3.12/site-packages/transformers/models/nemotron_h/modular_nemotron_h.py', 'venv/lib/python3.12/site-packages/transformers/models/nllb/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nllb/tokenization_nllb.py', 'venv/lib/python3.12/site-packages/transformers/models/nllb_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nllb_moe/configuration_nllb_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/nllb_moe/modeling_nllb_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/nomic_bert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nomic_bert/configuration_nomic_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/nomic_bert/modeling_nomic_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/nomic_bert/modular_nomic_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/nougat/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nougat/configuration_nougat.py', 'venv/lib/python3.12/site-packages/transformers/models/nougat/image_processing_nougat.py', 'venv/lib/python3.12/site-packages/transformers/models/nougat/image_processing_pil_nougat.py', 'venv/lib/python3.12/site-packages/transformers/models/nougat/processing_nougat.py', 'venv/lib/python3.12/site-packages/transformers/models/nougat/tokenization_nougat.py', 'venv/lib/python3.12/site-packages/transformers/models/nystromformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/nystromformer/configuration_nystromformer.py', 'venv/lib/python3.12/site-packages/transformers/models/nystromformer/modeling_nystromformer.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo/configuration_olmo.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo/modeling_olmo.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo/modular_olmo.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo2/configuration_olmo2.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo2/modeling_olmo2.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo2/modular_olmo2.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo3/configuration_olmo3.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo3/modeling_olmo3.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo3/modular_olmo3.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/configuration_olmo_hybrid.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/modeling_olmo_hybrid.py', 'venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/modular_olmo_hybrid.py', 'venv/lib/python3.12/site-packages/transformers/models/olmoe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/olmoe/configuration_olmoe.py', 'venv/lib/python3.12/site-packages/transformers/models/olmoe/modeling_olmoe.py', 'venv/lib/python3.12/site-packages/transformers/models/olmoe/modular_olmoe.py', 'venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/configuration_omdet_turbo.py', 'venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/modeling_omdet_turbo.py', 'venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/processing_omdet_turbo.py', 'venv/lib/python3.12/site-packages/transformers/models/oneformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/oneformer/configuration_oneformer.py', 'venv/lib/python3.12/site-packages/transformers/models/oneformer/image_processing_oneformer.py', 'venv/lib/python3.12/site-packages/transformers/models/oneformer/image_processing_pil_oneformer.py', 'venv/lib/python3.12/site-packages/transformers/models/oneformer/modeling_oneformer.py', 'venv/lib/python3.12/site-packages/transformers/models/oneformer/processing_oneformer.py', 'venv/lib/python3.12/site-packages/transformers/models/openai/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/openai/configuration_openai.py', 'venv/lib/python3.12/site-packages/transformers/models/openai/modeling_openai.py', 'venv/lib/python3.12/site-packages/transformers/models/openai/tokenization_openai.py', 'venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/configuration_openai_privacy_filter.py', 'venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/modeling_openai_privacy_filter.py', 'venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/modular_openai_privacy_filter.py', 'venv/lib/python3.12/site-packages/transformers/models/opt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/opt/configuration_opt.py', 'venv/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py', 'venv/lib/python3.12/site-packages/transformers/models/ovis2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/ovis2/configuration_ovis2.py', 'venv/lib/python3.12/site-packages/transformers/models/ovis2/image_processing_ovis2.py', 'venv/lib/python3.12/site-packages/transformers/models/ovis2/image_processing_pil_ovis2.py', 'venv/lib/python3.12/site-packages/transformers/models/ovis2/modeling_ovis2.py', 'venv/lib/python3.12/site-packages/transformers/models/ovis2/modular_ovis2.py', 'venv/lib/python3.12/site-packages/transformers/models/ovis2/processing_ovis2.py', 'venv/lib/python3.12/site-packages/transformers/models/owlv2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/owlv2/configuration_owlv2.py', 'venv/lib/python3.12/site-packages/transformers/models/owlv2/image_processing_owlv2.py', 'venv/lib/python3.12/site-packages/transformers/models/owlv2/image_processing_pil_owlv2.py', 'venv/lib/python3.12/site-packages/transformers/models/owlv2/modeling_owlv2.py', 'venv/lib/python3.12/site-packages/transformers/models/owlv2/modular_owlv2.py', 'venv/lib/python3.12/site-packages/transformers/models/owlv2/processing_owlv2.py', 'venv/lib/python3.12/site-packages/transformers/models/owlvit/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/owlvit/configuration_owlvit.py', 'venv/lib/python3.12/site-packages/transformers/models/owlvit/image_processing_owlvit.py', 'venv/lib/python3.12/site-packages/transformers/models/owlvit/image_processing_pil_owlvit.py', 'venv/lib/python3.12/site-packages/transformers/models/owlvit/modeling_owlvit.py', 'venv/lib/python3.12/site-packages/transformers/models/owlvit/processing_owlvit.py', 'venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/configuration_paddleocr_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/image_processing_paddleocr_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/image_processing_pil_paddleocr_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/modeling_paddleocr_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/modular_paddleocr_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/processing_paddleocr_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/paligemma/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/paligemma/configuration_paligemma.py', 'venv/lib/python3.12/site-packages/transformers/models/paligemma/modeling_paligemma.py', 'venv/lib/python3.12/site-packages/transformers/models/paligemma/processing_paligemma.py', 'venv/lib/python3.12/site-packages/transformers/models/parakeet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/parakeet/configuration_parakeet.py', 'venv/lib/python3.12/site-packages/transformers/models/parakeet/feature_extraction_parakeet.py', 'venv/lib/python3.12/site-packages/transformers/models/parakeet/modeling_parakeet.py', 'venv/lib/python3.12/site-packages/transformers/models/parakeet/modular_parakeet.py', 'venv/lib/python3.12/site-packages/transformers/models/parakeet/processing_parakeet.py', 'venv/lib/python3.12/site-packages/transformers/models/parakeet/tokenization_parakeet.py', 'venv/lib/python3.12/site-packages/transformers/models/patchtsmixer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/patchtsmixer/configuration_patchtsmixer.py', 'venv/lib/python3.12/site-packages/transformers/models/patchtsmixer/modeling_patchtsmixer.py', 'venv/lib/python3.12/site-packages/transformers/models/patchtst/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/patchtst/configuration_patchtst.py', 'venv/lib/python3.12/site-packages/transformers/models/patchtst/modeling_patchtst.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio/configuration_pe_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio/feature_extraction_pe_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio/modeling_pe_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio/modular_pe_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio/processing_pe_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/configuration_pe_audio_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/modeling_pe_audio_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/modular_pe_audio_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/processing_pe_audio_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_video/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_video/configuration_pe_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_video/modeling_pe_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_video/modular_pe_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_video/processing_pe_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pe_video/video_processing_pe_video.py', 'venv/lib/python3.12/site-packages/transformers/models/pegasus/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pegasus/configuration_pegasus.py', 'venv/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py', 'venv/lib/python3.12/site-packages/transformers/models/pegasus/tokenization_pegasus.py', 'venv/lib/python3.12/site-packages/transformers/models/pegasus_x/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pegasus_x/configuration_pegasus_x.py', 'venv/lib/python3.12/site-packages/transformers/models/pegasus_x/modeling_pegasus_x.py', 'venv/lib/python3.12/site-packages/transformers/models/perceiver/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/perceiver/configuration_perceiver.py', 'venv/lib/python3.12/site-packages/transformers/models/perceiver/image_processing_perceiver.py', 'venv/lib/python3.12/site-packages/transformers/models/perceiver/image_processing_pil_perceiver.py', 'venv/lib/python3.12/site-packages/transformers/models/perceiver/modeling_perceiver.py', 'venv/lib/python3.12/site-packages/transformers/models/perceiver/tokenization_perceiver.py', 'venv/lib/python3.12/site-packages/transformers/models/perception_lm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/perception_lm/configuration_perception_lm.py', 'venv/lib/python3.12/site-packages/transformers/models/perception_lm/image_processing_perception_lm.py', 'venv/lib/python3.12/site-packages/transformers/models/perception_lm/modeling_perception_lm.py', 'venv/lib/python3.12/site-packages/transformers/models/perception_lm/modular_perception_lm.py', 'venv/lib/python3.12/site-packages/transformers/models/perception_lm/processing_perception_lm.py', 'venv/lib/python3.12/site-packages/transformers/models/perception_lm/video_processing_perception_lm.py', 'venv/lib/python3.12/site-packages/transformers/models/persimmon/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/persimmon/configuration_persimmon.py', 'venv/lib/python3.12/site-packages/transformers/models/persimmon/modeling_persimmon.py', 'venv/lib/python3.12/site-packages/transformers/models/phi/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/phi/configuration_phi.py', 'venv/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py', 'venv/lib/python3.12/site-packages/transformers/models/phi/modular_phi.py', 'venv/lib/python3.12/site-packages/transformers/models/phi3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/phi3/configuration_phi3.py', 'venv/lib/python3.12/site-packages/transformers/models/phi3/modeling_phi3.py', 'venv/lib/python3.12/site-packages/transformers/models/phi3/modular_phi3.py', 'venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py', 'venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/feature_extraction_phi4_multimodal.py', 'venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/image_processing_phi4_multimodal.py', 'venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py', 'venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/modular_phi4_multimodal.py', 'venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/processing_phi4_multimodal.py', 'venv/lib/python3.12/site-packages/transformers/models/phimoe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/phimoe/configuration_phimoe.py', 'venv/lib/python3.12/site-packages/transformers/models/phimoe/modeling_phimoe.py', 'venv/lib/python3.12/site-packages/transformers/models/phimoe/modular_phimoe.py', 'venv/lib/python3.12/site-packages/transformers/models/phobert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/phobert/tokenization_phobert.py', 'venv/lib/python3.12/site-packages/transformers/models/pi0/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pi0/configuration_pi0.py', 'venv/lib/python3.12/site-packages/transformers/models/pi0/image_processing_pi0.py', 'venv/lib/python3.12/site-packages/transformers/models/pi0/modeling_pi0.py', 'venv/lib/python3.12/site-packages/transformers/models/pi0/modular_pi0.py', 'venv/lib/python3.12/site-packages/transformers/models/pi0/processing_pi0.py', 'venv/lib/python3.12/site-packages/transformers/models/pix2struct/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pix2struct/configuration_pix2struct.py', 'venv/lib/python3.12/site-packages/transformers/models/pix2struct/image_processing_pil_pix2struct.py', 'venv/lib/python3.12/site-packages/transformers/models/pix2struct/image_processing_pix2struct.py', 'venv/lib/python3.12/site-packages/transformers/models/pix2struct/modeling_pix2struct.py', 'venv/lib/python3.12/site-packages/transformers/models/pix2struct/processing_pix2struct.py', 'venv/lib/python3.12/site-packages/transformers/models/pixio/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pixio/configuration_pixio.py', 'venv/lib/python3.12/site-packages/transformers/models/pixio/modeling_pixio.py', 'venv/lib/python3.12/site-packages/transformers/models/pixio/modular_pixio.py', 'venv/lib/python3.12/site-packages/transformers/models/pixtral/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pixtral/configuration_pixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/pixtral/image_processing_pil_pixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/pixtral/image_processing_pixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/pixtral/modeling_pixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/pixtral/processing_pixtral.py', 'venv/lib/python3.12/site-packages/transformers/models/plbart/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/plbart/configuration_plbart.py', 'venv/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py', 'venv/lib/python3.12/site-packages/transformers/models/plbart/modular_plbart.py', 'venv/lib/python3.12/site-packages/transformers/models/plbart/tokenization_plbart.py', 'venv/lib/python3.12/site-packages/transformers/models/poolformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/poolformer/configuration_poolformer.py', 'venv/lib/python3.12/site-packages/transformers/models/poolformer/image_processing_pil_poolformer.py', 'venv/lib/python3.12/site-packages/transformers/models/poolformer/image_processing_poolformer.py', 'venv/lib/python3.12/site-packages/transformers/models/poolformer/modeling_poolformer.py', 'venv/lib/python3.12/site-packages/transformers/models/pop2piano/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pop2piano/configuration_pop2piano.py', 'venv/lib/python3.12/site-packages/transformers/models/pop2piano/feature_extraction_pop2piano.py', 'venv/lib/python3.12/site-packages/transformers/models/pop2piano/modeling_pop2piano.py', 'venv/lib/python3.12/site-packages/transformers/models/pop2piano/processing_pop2piano.py', 'venv/lib/python3.12/site-packages/transformers/models/pop2piano/tokenization_pop2piano.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/configuration_pp_chart2table.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/image_processing_pil_pp_chart2table.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/image_processing_pp_chart2table.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/modular_pp_chart2table.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/processing_pp_chart2table.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/configuration_pp_doclayout_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/image_processing_pp_doclayout_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/modeling_pp_doclayout_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/modular_pp_doclayout_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/configuration_pp_doclayout_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/image_processing_pp_doclayout_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/modular_pp_doclayout_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/configuration_pp_formulanet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/image_processing_pp_formulanet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/modeling_pp_formulanet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/modular_pp_formulanet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/processing_pp_formulanet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/configuration_pp_lcnet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/image_processing_pp_lcnet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/modeling_pp_lcnet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/modular_pp_lcnet.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/modular_pp_lcnet_v3.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/configuration_pp_ocrv5_mobile_det.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/modeling_pp_ocrv5_mobile_det.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/modular_pp_ocrv5_mobile_det.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/configuration_pp_ocrv5_mobile_rec.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/modeling_pp_ocrv5_mobile_rec.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/modular_pp_ocrv5_mobile_rec.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/configuration_pp_ocrv5_server_det.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/image_processing_pp_ocrv5_server_det.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/modeling_pp_ocrv5_server_det.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/modular_pp_ocrv5_server_det.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/configuration_pp_ocrv5_server_rec.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/image_processing_pp_ocrv5_server_rec.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/modeling_pp_ocrv5_server_rec.py', 'venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/modular_pp_ocrv5_server_rec.py', 'venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/configuration_prompt_depth_anything.py', 'venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/image_processing_pil_prompt_depth_anything.py', 'venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/image_processing_prompt_depth_anything.py', 'venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/modeling_prompt_depth_anything.py', 'venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/modular_prompt_depth_anything.py', 'venv/lib/python3.12/site-packages/transformers/models/prophetnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/prophetnet/configuration_prophetnet.py', 'venv/lib/python3.12/site-packages/transformers/models/prophetnet/modeling_prophetnet.py', 'venv/lib/python3.12/site-packages/transformers/models/prophetnet/tokenization_prophetnet.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt/configuration_pvt.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt/image_processing_pil_pvt.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt/image_processing_pvt.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt/modeling_pvt.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt_v2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt_v2/configuration_pvt_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/pvt_v2/modeling_pvt_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/configuration_qianfan_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/modeling_qianfan_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/modular_qianfan_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/processing_qianfan_ocr.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/models/qwen2/__pycache__/tokenization_qwen2.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/models/qwen2/configuration_qwen2.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2/modular_qwen2.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2/tokenization_qwen2.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/configuration_qwen2_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/modeling_qwen2_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/processing_qwen2_audio.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/configuration_qwen2_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/modeling_qwen2_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/modular_qwen2_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/configuration_qwen2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_pil_qwen2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/processing_qwen2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/video_processing_qwen2_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/models/qwen3/__pycache__/configuration_qwen3.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/models/qwen3/__pycache__/modeling_qwen3.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/models/qwen3/configuration_qwen3.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3/modular_qwen3.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5/configuration_qwen3_5.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modular_qwen3_5.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5/tokenization_qwen3_5.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/configuration_qwen3_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modular_qwen3_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_next/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_next/configuration_qwen3_next.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modular_qwen3_next.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/configuration_qwen3_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modular_qwen3_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py', 'venv/lib/python3.12/site-packages/transformers/models/rag/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/rag/configuration_rag.py', 'venv/lib/python3.12/site-packages/transformers/models/rag/modeling_rag.py', 'venv/lib/python3.12/site-packages/transformers/models/rag/retrieval_rag.py', 'venv/lib/python3.12/site-packages/transformers/models/rag/tokenization_rag.py', 'venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/configuration_recurrent_gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/reformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/reformer/configuration_reformer.py', 'venv/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py', 'venv/lib/python3.12/site-packages/transformers/models/reformer/tokenization_reformer.py', 'venv/lib/python3.12/site-packages/transformers/models/regnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/regnet/configuration_regnet.py', 'venv/lib/python3.12/site-packages/transformers/models/regnet/modeling_regnet.py', 'venv/lib/python3.12/site-packages/transformers/models/rembert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/rembert/configuration_rembert.py', 'venv/lib/python3.12/site-packages/transformers/models/rembert/modeling_rembert.py', 'venv/lib/python3.12/site-packages/transformers/models/rembert/tokenization_rembert.py', 'venv/lib/python3.12/site-packages/transformers/models/resnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/resnet/configuration_resnet.py', 'venv/lib/python3.12/site-packages/transformers/models/resnet/modeling_resnet.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta/configuration_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta/modular_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta_old.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/configuration_roberta_prelayernorm.py', 'venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py', 'venv/lib/python3.12/site-packages/transformers/models/roc_bert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/roc_bert/configuration_roc_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/roc_bert/modeling_roc_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/roc_bert/tokenization_roc_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/roformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/roformer/configuration_roformer.py', 'venv/lib/python3.12/site-packages/transformers/models/roformer/modeling_roformer.py', 'venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_roformer.py', 'venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_utils.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr_resnet.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_pil_rt_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_rt_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr_resnet.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr/modular_rt_detr.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modular_rt_detr_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/rwkv/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/rwkv/configuration_rwkv.py', 'venv/lib/python3.12/site-packages/transformers/models/rwkv/modeling_rwkv.py', 'venv/lib/python3.12/site-packages/transformers/models/sam/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam/configuration_sam.py', 'venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_pil_sam.py', 'venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_sam.py', 'venv/lib/python3.12/site-packages/transformers/models/sam/modeling_sam.py', 'venv/lib/python3.12/site-packages/transformers/models/sam/processing_sam.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2/configuration_sam2.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2/image_processing_sam2.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2/modeling_sam2.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2/modular_sam2.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2/processing_sam2.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2_video/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2_video/configuration_sam2_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2_video/modeling_sam2_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2_video/modular_sam2_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2_video/processing_sam2_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam2_video/video_processing_sam2_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3/configuration_sam3.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3/image_processing_sam3.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3/modeling_sam3.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3/modular_sam3.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3/processing_sam3.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/configuration_sam3_lite_text.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modeling_sam3_lite_text.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modular_sam3_lite_text.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/configuration_sam3_tracker.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modeling_sam3_tracker.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modular_sam3_tracker.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/processing_sam3_tracker.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modular_sam3_tracker_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/processing_sam3_tracker_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_video/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_video/configuration_sam3_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_video/modeling_sam3_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam3_video/processing_sam3_video.py', 'venv/lib/python3.12/site-packages/transformers/models/sam_hq/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sam_hq/configuration_sam_hq.py', 'venv/lib/python3.12/site-packages/transformers/models/sam_hq/modeling_sam_hq.py', 'venv/lib/python3.12/site-packages/transformers/models/sam_hq/modular_sam_hq.py', 'venv/lib/python3.12/site-packages/transformers/models/sam_hq/processing_sam_hq.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/configuration_seamless_m4t.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/modeling_seamless_m4t.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/processing_seamless_m4t.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/tokenization_seamless_m4t.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py', 'venv/lib/python3.12/site-packages/transformers/models/seed_oss/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/seed_oss/configuration_seed_oss.py', 'venv/lib/python3.12/site-packages/transformers/models/seed_oss/modeling_seed_oss.py', 'venv/lib/python3.12/site-packages/transformers/models/seed_oss/modular_seed_oss.py', 'venv/lib/python3.12/site-packages/transformers/models/segformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/segformer/configuration_segformer.py', 'venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_pil_segformer.py', 'venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_segformer.py', 'venv/lib/python3.12/site-packages/transformers/models/segformer/modeling_segformer.py', 'venv/lib/python3.12/site-packages/transformers/models/segformer/modular_segformer.py', 'venv/lib/python3.12/site-packages/transformers/models/seggpt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/seggpt/configuration_seggpt.py', 'venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_pil_seggpt.py', 'venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_seggpt.py', 'venv/lib/python3.12/site-packages/transformers/models/seggpt/modeling_seggpt.py', 'venv/lib/python3.12/site-packages/transformers/models/sew/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sew/configuration_sew.py', 'venv/lib/python3.12/site-packages/transformers/models/sew/modeling_sew.py', 'venv/lib/python3.12/site-packages/transformers/models/sew/modular_sew.py', 'venv/lib/python3.12/site-packages/transformers/models/sew_d/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/sew_d/configuration_sew_d.py', 'venv/lib/python3.12/site-packages/transformers/models/sew_d/modeling_sew_d.py', 'venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/configuration_shieldgemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/modeling_shieldgemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/processing_shieldgemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip/configuration_siglip.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_pil_siglip.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_siglip.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip/processing_siglip.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip/tokenization_siglip.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/configuration_siglip2.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_pil_siglip2.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_siglip2.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/modeling_siglip2.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/modular_siglip2.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/processing_siglip2.py', 'venv/lib/python3.12/site-packages/transformers/models/siglip2/tokenization_siglip2.py', 'venv/lib/python3.12/site-packages/transformers/models/slanet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/slanet/configuration_slanet.py', 'venv/lib/python3.12/site-packages/transformers/models/slanet/modeling_slanet.py', 'venv/lib/python3.12/site-packages/transformers/models/slanet/modular_slanet.py', 'venv/lib/python3.12/site-packages/transformers/models/slanext/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/slanext/configuration_slanext.py', 'venv/lib/python3.12/site-packages/transformers/models/slanext/image_processing_slanext.py', 'venv/lib/python3.12/site-packages/transformers/models/slanext/modeling_slanext.py', 'venv/lib/python3.12/site-packages/transformers/models/slanext/modular_slanext.py', 'venv/lib/python3.12/site-packages/transformers/models/smollm3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/smollm3/configuration_smollm3.py', 'venv/lib/python3.12/site-packages/transformers/models/smollm3/modeling_smollm3.py', 'venv/lib/python3.12/site-packages/transformers/models/smollm3/modular_smollm3.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/configuration_smolvlm.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_pil_smolvlm.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_smolvlm.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/modeling_smolvlm.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/modular_smolvlm.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/processing_smolvlm.py', 'venv/lib/python3.12/site-packages/transformers/models/smolvlm/video_processing_smolvlm.py', 'venv/lib/python3.12/site-packages/transformers/models/solar_open/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/solar_open/configuration_solar_open.py', 'venv/lib/python3.12/site-packages/transformers/models/solar_open/modeling_solar_open.py', 'venv/lib/python3.12/site-packages/transformers/models/solar_open/modular_solar_open.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_to_text/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_to_text/configuration_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_to_text/feature_extraction_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_to_text/modeling_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_to_text/processing_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/speech_to_text/tokenization_speech_to_text.py', 'venv/lib/python3.12/site-packages/transformers/models/speecht5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/speecht5/configuration_speecht5.py', 'venv/lib/python3.12/site-packages/transformers/models/speecht5/feature_extraction_speecht5.py', 'venv/lib/python3.12/site-packages/transformers/models/speecht5/modeling_speecht5.py', 'venv/lib/python3.12/site-packages/transformers/models/speecht5/number_normalizer.py', 'venv/lib/python3.12/site-packages/transformers/models/speecht5/processing_speecht5.py', 'venv/lib/python3.12/site-packages/transformers/models/speecht5/tokenization_speecht5.py', 'venv/lib/python3.12/site-packages/transformers/models/splinter/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/splinter/configuration_splinter.py', 'venv/lib/python3.12/site-packages/transformers/models/splinter/modeling_splinter.py', 'venv/lib/python3.12/site-packages/transformers/models/splinter/tokenization_splinter.py', 'venv/lib/python3.12/site-packages/transformers/models/squeezebert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/squeezebert/configuration_squeezebert.py', 'venv/lib/python3.12/site-packages/transformers/models/squeezebert/modeling_squeezebert.py', 'venv/lib/python3.12/site-packages/transformers/models/squeezebert/tokenization_squeezebert.py', 'venv/lib/python3.12/site-packages/transformers/models/stablelm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/stablelm/configuration_stablelm.py', 'venv/lib/python3.12/site-packages/transformers/models/stablelm/modeling_stablelm.py', 'venv/lib/python3.12/site-packages/transformers/models/starcoder2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/starcoder2/configuration_starcoder2.py', 'venv/lib/python3.12/site-packages/transformers/models/starcoder2/modeling_starcoder2.py', 'venv/lib/python3.12/site-packages/transformers/models/starcoder2/modular_starcoder2.py', 'venv/lib/python3.12/site-packages/transformers/models/superglue/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/superglue/configuration_superglue.py', 'venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_pil_superglue.py', 'venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_superglue.py', 'venv/lib/python3.12/site-packages/transformers/models/superglue/modeling_superglue.py', 'venv/lib/python3.12/site-packages/transformers/models/superpoint/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/superpoint/configuration_superpoint.py', 'venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_pil_superpoint.py', 'venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_superpoint.py', 'venv/lib/python3.12/site-packages/transformers/models/superpoint/modeling_superpoint.py', 'venv/lib/python3.12/site-packages/transformers/models/swiftformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/swiftformer/configuration_swiftformer.py', 'venv/lib/python3.12/site-packages/transformers/models/swiftformer/modeling_swiftformer.py', 'venv/lib/python3.12/site-packages/transformers/models/swin/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/swin/configuration_swin.py', 'venv/lib/python3.12/site-packages/transformers/models/swin/modeling_swin.py', 'venv/lib/python3.12/site-packages/transformers/models/swin2sr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/swin2sr/configuration_swin2sr.py', 'venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_pil_swin2sr.py', 'venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_swin2sr.py', 'venv/lib/python3.12/site-packages/transformers/models/swin2sr/modeling_swin2sr.py', 'venv/lib/python3.12/site-packages/transformers/models/swinv2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/swinv2/configuration_swinv2.py', 'venv/lib/python3.12/site-packages/transformers/models/swinv2/modeling_swinv2.py', 'venv/lib/python3.12/site-packages/transformers/models/switch_transformers/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/switch_transformers/configuration_switch_transformers.py', 'venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py', 'venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modular_switch_transformers.py', 'venv/lib/python3.12/site-packages/transformers/models/t5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/t5/configuration_t5.py', 'venv/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py', 'venv/lib/python3.12/site-packages/transformers/models/t5/tokenization_t5.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma/configuration_t5gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma/modeling_t5gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma/modular_t5gemma.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma2/configuration_t5gemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modular_t5gemma2.py', 'venv/lib/python3.12/site-packages/transformers/models/table_transformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/table_transformer/configuration_table_transformer.py', 'venv/lib/python3.12/site-packages/transformers/models/table_transformer/modeling_table_transformer.py', 'venv/lib/python3.12/site-packages/transformers/models/tapas/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/tapas/configuration_tapas.py', 'venv/lib/python3.12/site-packages/transformers/models/tapas/modeling_tapas.py', 'venv/lib/python3.12/site-packages/transformers/models/tapas/tokenization_tapas.py', 'venv/lib/python3.12/site-packages/transformers/models/textnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/textnet/configuration_textnet.py', 'venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_pil_textnet.py', 'venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_textnet.py', 'venv/lib/python3.12/site-packages/transformers/models/textnet/modeling_textnet.py', 'venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/configuration_time_series_transformer.py', 'venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/modeling_time_series_transformer.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm/configuration_timesfm.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm/modeling_timesfm.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm/modular_timesfm.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/configuration_timesfm2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modeling_timesfm2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modular_timesfm2_5.py', 'venv/lib/python3.12/site-packages/transformers/models/timesformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/timesformer/configuration_timesformer.py', 'venv/lib/python3.12/site-packages/transformers/models/timesformer/modeling_timesformer.py', 'venv/lib/python3.12/site-packages/transformers/models/timm_backbone/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/timm_backbone/configuration_timm_backbone.py', 'venv/lib/python3.12/site-packages/transformers/models/timm_backbone/modeling_timm_backbone.py', 'venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/configuration_timm_wrapper.py', 'venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/image_processing_timm_wrapper.py', 'venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/modeling_timm_wrapper.py', 'venv/lib/python3.12/site-packages/transformers/models/trocr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/trocr/configuration_trocr.py', 'venv/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py', 'venv/lib/python3.12/site-packages/transformers/models/trocr/processing_trocr.py', 'venv/lib/python3.12/site-packages/transformers/models/tvp/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/tvp/configuration_tvp.py', 'venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_pil_tvp.py', 'venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_tvp.py', 'venv/lib/python3.12/site-packages/transformers/models/tvp/modeling_tvp.py', 'venv/lib/python3.12/site-packages/transformers/models/tvp/processing_tvp.py', 'venv/lib/python3.12/site-packages/transformers/models/udop/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/udop/configuration_udop.py', 'venv/lib/python3.12/site-packages/transformers/models/udop/modeling_udop.py', 'venv/lib/python3.12/site-packages/transformers/models/udop/processing_udop.py', 'venv/lib/python3.12/site-packages/transformers/models/udop/tokenization_udop.py', 'venv/lib/python3.12/site-packages/transformers/models/umt5/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/umt5/configuration_umt5.py', 'venv/lib/python3.12/site-packages/transformers/models/umt5/modeling_umt5.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech/configuration_unispeech.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech/modeling_unispeech.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech/modular_unispeech.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/configuration_unispeech_sat.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modeling_unispeech_sat.py', 'venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modular_unispeech_sat.py', 'venv/lib/python3.12/site-packages/transformers/models/univnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/univnet/configuration_univnet.py', 'venv/lib/python3.12/site-packages/transformers/models/univnet/feature_extraction_univnet.py', 'venv/lib/python3.12/site-packages/transformers/models/univnet/modeling_univnet.py', 'venv/lib/python3.12/site-packages/transformers/models/upernet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/upernet/configuration_upernet.py', 'venv/lib/python3.12/site-packages/transformers/models/upernet/modeling_upernet.py', 'venv/lib/python3.12/site-packages/transformers/models/uvdoc/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/uvdoc/configuration_uvdoc.py', 'venv/lib/python3.12/site-packages/transformers/models/uvdoc/image_processing_uvdoc.py', 'venv/lib/python3.12/site-packages/transformers/models/uvdoc/modeling_uvdoc.py', 'venv/lib/python3.12/site-packages/transformers/models/uvdoc/modular_uvdoc.py', 'venv/lib/python3.12/site-packages/transformers/models/vaultgemma/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vaultgemma/configuration_vaultgemma.py', 'venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modeling_vaultgemma.py', 'venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modular_vaultgemma.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/configuration_vibevoice_acoustic_tokenizer.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/feature_extraction_vibevoice_acoustic_tokenizer.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modeling_vibevoice_acoustic_tokenizer.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modular_vibevoice_acoustic_tokenizer.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/configuration_vibevoice_asr.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modeling_vibevoice_asr.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modular_vibevoice_asr.py', 'venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/processing_vibevoice_asr.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/configuration_video_llama_3.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_pil_video_llama_3.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_video_llama_3.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modeling_video_llama_3.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modular_video_llama_3.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/processing_video_llama_3.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llama_3/video_processing_video_llama_3.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llava/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llava/configuration_video_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llava/image_processing_video_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llava/modeling_video_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llava/processing_video_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/video_llava/video_processing_video_llava.py', 'venv/lib/python3.12/site-packages/transformers/models/videomae/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/videomae/configuration_videomae.py', 'venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_pil_videomae.py', 'venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_videomae.py', 'venv/lib/python3.12/site-packages/transformers/models/videomae/modeling_videomae.py', 'venv/lib/python3.12/site-packages/transformers/models/videomae/video_processing_videomae.py', 'venv/lib/python3.12/site-packages/transformers/models/videomt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/videomt/configuration_videomt.py', 'venv/lib/python3.12/site-packages/transformers/models/videomt/modeling_videomt.py', 'venv/lib/python3.12/site-packages/transformers/models/videomt/modular_videomt.py', 'venv/lib/python3.12/site-packages/transformers/models/videomt/video_processing_videomt.py', 'venv/lib/python3.12/site-packages/transformers/models/vilt/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vilt/configuration_vilt.py', 'venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_pil_vilt.py', 'venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_vilt.py', 'venv/lib/python3.12/site-packages/transformers/models/vilt/modeling_vilt.py', 'venv/lib/python3.12/site-packages/transformers/models/vilt/processing_vilt.py', 'venv/lib/python3.12/site-packages/transformers/models/vipllava/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vipllava/configuration_vipllava.py', 'venv/lib/python3.12/site-packages/transformers/models/vipllava/modeling_vipllava.py', 'venv/lib/python3.12/site-packages/transformers/models/vipllava/modular_vipllava.py', 'venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py', 'venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py', 'venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py', 'venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py', 'venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py', 'venv/lib/python3.12/site-packages/transformers/models/visual_bert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/visual_bert/configuration_visual_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/visual_bert/modeling_visual_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/vit/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vit/configuration_vit.py', 'venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_pil_vit.py', 'venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_vit.py', 'venv/lib/python3.12/site-packages/transformers/models/vit/modeling_vit.py', 'venv/lib/python3.12/site-packages/transformers/models/vit_mae/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vit_mae/configuration_vit_mae.py', 'venv/lib/python3.12/site-packages/transformers/models/vit_mae/modeling_vit_mae.py', 'venv/lib/python3.12/site-packages/transformers/models/vit_msn/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vit_msn/configuration_vit_msn.py', 'venv/lib/python3.12/site-packages/transformers/models/vit_msn/modeling_vit_msn.py', 'venv/lib/python3.12/site-packages/transformers/models/vitdet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vitdet/configuration_vitdet.py', 'venv/lib/python3.12/site-packages/transformers/models/vitdet/modeling_vitdet.py', 'venv/lib/python3.12/site-packages/transformers/models/vitmatte/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vitmatte/configuration_vitmatte.py', 'venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_pil_vitmatte.py', 'venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_vitmatte.py', 'venv/lib/python3.12/site-packages/transformers/models/vitmatte/modeling_vitmatte.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose/configuration_vitpose.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_pil_vitpose.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_vitpose.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose/modeling_vitpose.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py', 'venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py', 'venv/lib/python3.12/site-packages/transformers/models/vits/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vits/configuration_vits.py', 'venv/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py', 'venv/lib/python3.12/site-packages/transformers/models/vits/tokenization_vits.py', 'venv/lib/python3.12/site-packages/transformers/models/vivit/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vivit/configuration_vivit.py', 'venv/lib/python3.12/site-packages/transformers/models/vivit/image_processing_vivit.py', 'venv/lib/python3.12/site-packages/transformers/models/vivit/modeling_vivit.py', 'venv/lib/python3.12/site-packages/transformers/models/vjepa2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/vjepa2/configuration_vjepa2.py', 'venv/lib/python3.12/site-packages/transformers/models/vjepa2/modeling_vjepa2.py', 'venv/lib/python3.12/site-packages/transformers/models/vjepa2/video_processing_vjepa2.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral/configuration_voxtral.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral/modeling_voxtral.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral/modular_voxtral.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral/processing_voxtral.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/configuration_voxtral_realtime.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/feature_extraction_voxtral_realtime.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modular_voxtral_realtime.py', 'venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/processing_voxtral_realtime.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2/configuration_wav2vec2.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2/feature_extraction_wav2vec2.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2/processing_wav2vec2.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2/tokenization_wav2vec2.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modular_wav2vec2_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/configuration_wav2vec2_conformer.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modular_wav2vec2_conformer.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py', 'venv/lib/python3.12/site-packages/transformers/models/wavlm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/wavlm/configuration_wavlm.py', 'venv/lib/python3.12/site-packages/transformers/models/wavlm/modeling_wavlm.py', 'venv/lib/python3.12/site-packages/transformers/models/wavlm/modular_wavlm.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/configuration_whisper.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/english_normalizer.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/feature_extraction_whisper.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/generation_whisper.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/processing_whisper.py', 'venv/lib/python3.12/site-packages/transformers/models/whisper/tokenization_whisper.py', 'venv/lib/python3.12/site-packages/transformers/models/x_clip/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/x_clip/configuration_x_clip.py', 'venv/lib/python3.12/site-packages/transformers/models/x_clip/modeling_x_clip.py', 'venv/lib/python3.12/site-packages/transformers/models/x_clip/modular_x_clip.py', 'venv/lib/python3.12/site-packages/transformers/models/x_clip/processing_x_clip.py', 'venv/lib/python3.12/site-packages/transformers/models/xcodec/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xcodec/configuration_xcodec.py', 'venv/lib/python3.12/site-packages/transformers/models/xcodec/modeling_xcodec.py', 'venv/lib/python3.12/site-packages/transformers/models/xglm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xglm/configuration_xglm.py', 'venv/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py', 'venv/lib/python3.12/site-packages/transformers/models/xglm/tokenization_xglm.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm/configuration_xlm.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm/modeling_xlm.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm/tokenization_xlm.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/configuration_xlm_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modeling_xlm_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modular_xlm_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/tokenization_xlm_roberta.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py', 'venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modular_xlm_roberta_xl.py', 'venv/lib/python3.12/site-packages/transformers/models/xlnet/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xlnet/configuration_xlnet.py', 'venv/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py', 'venv/lib/python3.12/site-packages/transformers/models/xlnet/tokenization_xlnet.py', 'venv/lib/python3.12/site-packages/transformers/models/xlstm/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xlstm/configuration_xlstm.py', 'venv/lib/python3.12/site-packages/transformers/models/xlstm/modeling_xlstm.py', 'venv/lib/python3.12/site-packages/transformers/models/xmod/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/xmod/configuration_xmod.py', 'venv/lib/python3.12/site-packages/transformers/models/xmod/modeling_xmod.py', 'venv/lib/python3.12/site-packages/transformers/models/yolos/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/yolos/configuration_yolos.py', 'venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_pil_yolos.py', 'venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_yolos.py', 'venv/lib/python3.12/site-packages/transformers/models/yolos/modeling_yolos.py', 'venv/lib/python3.12/site-packages/transformers/models/yolos/modular_yolos.py', 'venv/lib/python3.12/site-packages/transformers/models/yoso/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/yoso/configuration_yoso.py', 'venv/lib/python3.12/site-packages/transformers/models/yoso/modeling_yoso.py', 'venv/lib/python3.12/site-packages/transformers/models/youtu/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/youtu/configuration_youtu.py', 'venv/lib/python3.12/site-packages/transformers/models/youtu/modeling_youtu.py', 'venv/lib/python3.12/site-packages/transformers/models/youtu/modular_youtu.py', 'venv/lib/python3.12/site-packages/transformers/models/zamba/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/zamba/configuration_zamba.py', 'venv/lib/python3.12/site-packages/transformers/models/zamba/modeling_zamba.py', 'venv/lib/python3.12/site-packages/transformers/models/zamba2/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/zamba2/configuration_zamba2.py', 'venv/lib/python3.12/site-packages/transformers/models/zamba2/modeling_zamba2.py', 'venv/lib/python3.12/site-packages/transformers/models/zamba2/modular_zamba2.py', 'venv/lib/python3.12/site-packages/transformers/models/zoedepth/__init__.py', 'venv/lib/python3.12/site-packages/transformers/models/zoedepth/configuration_zoedepth.py', 'venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_pil_zoedepth.py', 'venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_zoedepth.py', 'venv/lib/python3.12/site-packages/transformers/models/zoedepth/modeling_zoedepth.py', 'venv/lib/python3.12/site-packages/transformers/monkey_patching.py', 'venv/lib/python3.12/site-packages/transformers/optimization.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/__init__.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/any_to_any.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/audio_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/audio_utils.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/automatic_speech_recognition.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/base.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/depth_estimation.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/document_question_answering.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/feature_extraction.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/fill_mask.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/image_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/image_feature_extraction.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/image_segmentation.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/image_text_to_text.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/keypoint_matching.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/mask_generation.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/object_detection.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/pt_utils.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/table_question_answering.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/text_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/text_generation.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/text_to_audio.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/token_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/video_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_audio_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_image_classification.py', 'venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_object_detection.py', 'venv/lib/python3.12/site-packages/transformers/processing_utils.py', 'venv/lib/python3.12/site-packages/transformers/py.typed', 'venv/lib/python3.12/site-packages/transformers/pytorch_utils.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/__init__.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/auto.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/base.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_aqlm.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_auto_round.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_awq.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bitnet.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_4bit.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_8bit.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_compressed_tensors.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_eetq.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fbgemm_fp8.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_finegrained_fp8.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fouroversix.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fp_quant.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_gptq.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_higgs.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_hqq.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_metal.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_mxfp4.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quanto.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quark.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_sinq.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_spqr.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_torchao.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_vptq.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizers_utils.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/quantizers/auto.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/base.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_aqlm.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_auto_round.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_awq.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bitnet.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_4bit.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_8bit.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_compressed_tensors.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_eetq.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fbgemm_fp8.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_finegrained_fp8.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fouroversix.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fp_quant.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_gptq.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_higgs.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_hqq.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_metal.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_mxfp4.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quanto.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quark.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_sinq.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_spqr.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_torchao.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_vptq.py', 'venv/lib/python3.12/site-packages/transformers/quantizers/quantizers_utils.py', 'venv/lib/python3.12/site-packages/transformers/safetensors_conversion.py', 'venv/lib/python3.12/site-packages/transformers/testing_utils.py', 'venv/lib/python3.12/site-packages/transformers/time_series_utils.py', 'venv/lib/python3.12/site-packages/transformers/tokenization_mistral_common.py', 'venv/lib/python3.12/site-packages/transformers/tokenization_python.py', 'venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py', 'venv/lib/python3.12/site-packages/transformers/tokenization_utils_sentencepiece.py', 'venv/lib/python3.12/site-packages/transformers/tokenization_utils_tokenizers.py', 'venv/lib/python3.12/site-packages/transformers/trainer.py', 'venv/lib/python3.12/site-packages/transformers/trainer_callback.py', 'venv/lib/python3.12/site-packages/transformers/trainer_jit_checkpoint.py', 'venv/lib/python3.12/site-packages/transformers/trainer_optimizer.py', 'venv/lib/python3.12/site-packages/transformers/trainer_pt_utils.py', 'venv/lib/python3.12/site-packages/transformers/trainer_seq2seq.py', 'venv/lib/python3.12/site-packages/transformers/trainer_utils.py', 'venv/lib/python3.12/site-packages/transformers/training_args.py', 'venv/lib/python3.12/site-packages/transformers/training_args_seq2seq.py', 'venv/lib/python3.12/site-packages/transformers/utils/__init__.py', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/auto_docstring.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_parsing_utils.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_template_utils.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/constants.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/deprecation.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/doc.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_mistral_common_objects.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_sentencepiece_and_tokenizers_objects.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_torchvision_objects.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_vision_objects.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/generic.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/hub.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/import_utils.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/kernel_config.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/loading_report.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/logging.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/metrics.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/output_capturing.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/peft_utils.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/quantization_config.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/type_validators.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/__pycache__/versions.cpython-312.pyc', 'venv/lib/python3.12/site-packages/transformers/utils/attention_visualizer.py', 'venv/lib/python3.12/site-packages/transformers/utils/auto_docstring.py', 'venv/lib/python3.12/site-packages/transformers/utils/backbone_utils.py', 'venv/lib/python3.12/site-packages/transformers/utils/chat_parsing_utils.py', 'venv/lib/python3.12/site-packages/transformers/utils/chat_template_utils.py', 'venv/lib/python3.12/site-packages/transformers/utils/constants.py', 'venv/lib/python3.12/site-packages/transformers/utils/deprecation.py', 'venv/lib/python3.12/site-packages/transformers/utils/doc.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_detectron2_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_mistral_common_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_music_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_pt_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_sentencepiece_and_tokenizers_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_speech_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_timm_and_torchvision_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_tokenizers_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_torchaudio_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_torchvision_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/dummy_vision_objects.py', 'venv/lib/python3.12/site-packages/transformers/utils/generic.py', 'venv/lib/python3.12/site-packages/transformers/utils/hp_naming.py', 'venv/lib/python3.12/site-packages/transformers/utils/hub.py', 'venv/lib/python3.12/site-packages/transformers/utils/import_utils.py', 'venv/lib/python3.12/site-packages/transformers/utils/kernel_config.py', 'venv/lib/python3.12/site-packages/transformers/utils/loading_report.py', 'venv/lib/python3.12/site-packages/transformers/utils/logging.py', 'venv/lib/python3.12/site-packages/transformers/utils/metrics.py', 'venv/lib/python3.12/site-packages/transformers/utils/network_logging.py', 'venv/lib/python3.12/site-packages/transformers/utils/notebook.py', 'venv/lib/python3.12/site-packages/transformers/utils/output_capturing.py', 'venv/lib/python3.12/site-packages/transformers/utils/peft_utils.py', 'venv/lib/python3.12/site-packages/transformers/utils/pytest_helpers.py', 'venv/lib/python3.12/site-packages/transformers/utils/quantization_config.py', 'venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2.py', 'venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2_new.py', 'venv/lib/python3.12/site-packages/transformers/utils/type_validators.py', 'venv/lib/python3.12/site-packages/transformers/utils/versions.py', 'venv/lib/python3.12/site-packages/transformers/video_processing_utils.py', 'venv/lib/python3.12/site-packages/transformers/video_utils.py', 'venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/INSTALLER', 'venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/METADATA', 'venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/RECORD', 'venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/REQUESTED', 'venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/WHEEL', 'venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/entry_points.txt', 'venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/top_level.txt', 'venv/lib/python3.12/site-packages/triton/_C/libproton.so', 'venv/lib/python3.12/site-packages/triton/_C/libtriton.so', 'venv/lib/python3.12/site-packages/triton/__init__.py', 'venv/lib/python3.12/site-packages/triton/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/__pycache__/errors.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/__pycache__/testing.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/_internal_testing.py', 'venv/lib/python3.12/site-packages/triton/backends/__init__.py', 'venv/lib/python3.12/site-packages/triton/backends/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/backends/__pycache__/compiler.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/backends/__pycache__/driver.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/compiler.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/driver.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py', 'venv/lib/python3.12/site-packages/triton/backends/amd/driver.c', 'venv/lib/python3.12/site-packages/triton/backends/amd/driver.py', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_channel_descriptor.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_device_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_atomic.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_bf16.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_bfloat16.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_common.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_complex.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_cooperative_groups.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_fp16.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_gl_interop.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_math_constants.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime_pt_api.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_unsafe_atomics.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_vector_types.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_math_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_surface_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_warp_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/concepts.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/device_library_decls.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/functional_grid_launch.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch_GGL.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/helpers.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_api_trace.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_assert.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_cooperative_groups_helper.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_gcc.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_math_fwd.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_ldg.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_prof_str.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_runtime_prof.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/host_defines.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hsa_helpers.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/macro_based_grid_launch.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/math_fwd.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/ockl_image.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/program_state.hpp', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_fetch_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_indirect_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/channel_descriptor.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/device_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/driver_types.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_bf16.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_bfloat16.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_common.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_complex.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_cooperative_groups.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_deprecated.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_ext.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_fp16.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_gl_interop.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_hcc.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_math_constants.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_profile.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime_api.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_texture_types.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_vector_types.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_version.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hiprtc.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/library_types.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/math_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/surface_types.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/texture_types.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/Brig.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_common.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_elf.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_kernel_code.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_queue.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_signal.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_amd_tool.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_api_trace.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_amd.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_finalize.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_image.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_aqlprofile.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_loader.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/ext/prof_protocol.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hip_ostream_ops.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hsa_ostream_ops.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hsa_prof_str.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_ext.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hcc.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hip.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hsa.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_plugin.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_roctx.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctx.h', 'venv/lib/python3.12/site-packages/triton/backends/amd/lib/ockl.bc', 'venv/lib/python3.12/site-packages/triton/backends/amd/lib/ocml.bc', 'venv/lib/python3.12/site-packages/triton/backends/compiler.py', 'venv/lib/python3.12/site-packages/triton/backends/driver.py', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/__init__.py', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/compiler.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/driver.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.c', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openacc/cupti_openacc.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/cupti_openmp.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/omp-tools.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/builtin_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/channel_descriptor.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/common_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/async.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_reduce.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_scan.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/driver_abi.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/functional.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/helpers.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/info.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/invoke.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/memory.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/partitioning.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/reduce.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/scan.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/sync.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/memcpy_async.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/reduce.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/scan.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/common_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/cudacc_ext.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/func_macro.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_config.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_defines.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_runtime.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/nvfunctional', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/storage_class.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuComplex.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGL.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGLTypedefs.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGL.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGLTypedefs.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaProfilerTypedefs.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaTypedefs.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAU.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAUTypedefs.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_helpers.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_primitives.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_device_runtime_api.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_egl_interop.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_gl_interop.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_occupancy.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_helpers.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_primitives.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime_api.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_stdint.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_surface_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_texture_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_vdpau_interop.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudart_platform.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity_deprecated.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_callbacks.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_checkpoint.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_common.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_driver_cbid.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_events.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_metrics.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_nvtx_cbid.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling_util.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_profiler_target.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_result.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_runtime_cbid.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_sass_metrics.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_target.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_version.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_double_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_launch_parameters.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/fatbinary_section.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaGL_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaVDPAU_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_gl_interop_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_runtime_api_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_vdpau_interop_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudart_removed_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_nvtx_meta.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_config.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_defines.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/library_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_constants.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/mma.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvPTXCompiler.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvfunctional', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_common.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_cuda_host.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_host.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_target.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_atomic_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_intrinsics.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_indirect_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_fetch_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_indirect_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.hpp', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_types.h', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcheckpoint.so', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.12', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.2024.1.0', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host.so', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_target.so', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libpcsamplingutil.so', 'venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc', 'venv/lib/python3.12/site-packages/triton/compiler/__init__.py', 'venv/lib/python3.12/site-packages/triton/compiler/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/compiler/__pycache__/code_generator.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/compiler/__pycache__/compiler.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/compiler/__pycache__/errors.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/compiler/code_generator.py', 'venv/lib/python3.12/site-packages/triton/compiler/compiler.py', 'venv/lib/python3.12/site-packages/triton/compiler/errors.py', 'venv/lib/python3.12/site-packages/triton/compiler/make_launcher.py', 'venv/lib/python3.12/site-packages/triton/errors.py', 'venv/lib/python3.12/site-packages/triton/instrumentation/libGPUInstrumentationTestLib.so', 'venv/lib/python3.12/site-packages/triton/language/__init__.py', 'venv/lib/python3.12/site-packages/triton/language/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/__pycache__/_utils.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/__pycache__/core.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/__pycache__/math.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/__pycache__/random.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/__pycache__/semantic.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/__pycache__/standard.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/_utils.py', 'venv/lib/python3.12/site-packages/triton/language/core.py', 'venv/lib/python3.12/site-packages/triton/language/extra/__init__.py', 'venv/lib/python3.12/site-packages/triton/language/extra/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/__init__.py', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/_experimental_tma.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/libdevice.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/utils.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/_experimental_tma.py', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/libdevice.py', 'venv/lib/python3.12/site-packages/triton/language/extra/cuda/utils.py', 'venv/lib/python3.12/site-packages/triton/language/extra/hip/__init__.py', 'venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/libdevice.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/language/extra/hip/libdevice.py', 'venv/lib/python3.12/site-packages/triton/language/extra/libdevice.py', 'venv/lib/python3.12/site-packages/triton/language/math.py', 'venv/lib/python3.12/site-packages/triton/language/random.py', 'venv/lib/python3.12/site-packages/triton/language/semantic.py', 'venv/lib/python3.12/site-packages/triton/language/standard.py', 'venv/lib/python3.12/site-packages/triton/profiler/__init__.py', 'venv/lib/python3.12/site-packages/triton/profiler/flags.py', 'venv/lib/python3.12/site-packages/triton/profiler/hook.py', 'venv/lib/python3.12/site-packages/triton/profiler/profile.py', 'venv/lib/python3.12/site-packages/triton/profiler/proton.py', 'venv/lib/python3.12/site-packages/triton/profiler/scope.py', 'venv/lib/python3.12/site-packages/triton/profiler/viewer.py', 'venv/lib/python3.12/site-packages/triton/runtime/__init__.py', 'venv/lib/python3.12/site-packages/triton/runtime/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/runtime/__pycache__/autotuner.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/runtime/__pycache__/build.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/runtime/__pycache__/cache.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/runtime/__pycache__/driver.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/runtime/__pycache__/errors.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/runtime/__pycache__/jit.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/runtime/autotuner.py', 'venv/lib/python3.12/site-packages/triton/runtime/build.py', 'venv/lib/python3.12/site-packages/triton/runtime/cache.py', 'venv/lib/python3.12/site-packages/triton/runtime/driver.py', 'venv/lib/python3.12/site-packages/triton/runtime/errors.py', 'venv/lib/python3.12/site-packages/triton/runtime/interpreter.py', 'venv/lib/python3.12/site-packages/triton/runtime/jit.py', 'venv/lib/python3.12/site-packages/triton/testing.py', 'venv/lib/python3.12/site-packages/triton/tools/__init__.py', 'venv/lib/python3.12/site-packages/triton/tools/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/tools/__pycache__/disasm.cpython-312.pyc', 'venv/lib/python3.12/site-packages/triton/tools/build_extern.py', 'venv/lib/python3.12/site-packages/triton/tools/compile.c', 'venv/lib/python3.12/site-packages/triton/tools/compile.h', 'venv/lib/python3.12/site-packages/triton/tools/compile.py', 'venv/lib/python3.12/site-packages/triton/tools/disasm.py', 'venv/lib/python3.12/site-packages/triton/tools/experimental_descriptor.py', 'venv/lib/python3.12/site-packages/triton/tools/link.py', 'venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/INSTALLER', 'venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/METADATA', 'venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/RECORD', 'venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/REQUESTED', 'venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/WHEEL', 'venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/entry_points.txt', 'venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/licenses/LICENSE', 'venv/lib/python3.12/site-packages/typer/.agents/skills/typer/SKILL.md', 'venv/lib/python3.12/site-packages/typer/__init__.py', 'venv/lib/python3.12/site-packages/typer/__main__.py', 'venv/lib/python3.12/site-packages/typer/_completion_classes.py', 'venv/lib/python3.12/site-packages/typer/_completion_shared.py', 'venv/lib/python3.12/site-packages/typer/_types.py', 'venv/lib/python3.12/site-packages/typer/_typing.py', 'venv/lib/python3.12/site-packages/typer/cli.py', 'venv/lib/python3.12/site-packages/typer/colors.py', 'venv/lib/python3.12/site-packages/typer/completion.py', 'venv/lib/python3.12/site-packages/typer/core.py', 'venv/lib/python3.12/site-packages/typer/main.py', 'venv/lib/python3.12/site-packages/typer/models.py', 'venv/lib/python3.12/site-packages/typer/params.py', 'venv/lib/python3.12/site-packages/typer/py.typed', 'venv/lib/python3.12/site-packages/typer/rich_utils.py', 'venv/lib/python3.12/site-packages/typer/testing.py', 'venv/lib/python3.12/site-packages/typer/utils.py', 'venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/INSTALLER', 'venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/METADATA', 'venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/RECORD', 'venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/REQUESTED', 'venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/WHEEL', 'venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/licenses/LICENSE', 'venv/lib/python3.12/site-packages/typing_extensions.py', 'venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/INSTALLER', 'venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/METADATA', 'venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/RECORD', 'venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/REQUESTED', 'venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/WHEEL', 'venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/licenses/LICENSE', 'venv/lib/python3.12/site-packages/typing_inspection/__init__.py', 'venv/lib/python3.12/site-packages/typing_inspection/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/typing_inspection/__pycache__/introspection.cpython-312.pyc', 'venv/lib/python3.12/site-packages/typing_inspection/__pycache__/typing_objects.cpython-312.pyc', 'venv/lib/python3.12/site-packages/typing_inspection/introspection.py', 'venv/lib/python3.12/site-packages/typing_inspection/py.typed', 'venv/lib/python3.12/site-packages/typing_inspection/typing_objects.py', 'venv/lib/python3.12/site-packages/typing_inspection/typing_objects.pyi', 'venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/INSTALLER', 'venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/METADATA', 'venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/RECORD', 'venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/REQUESTED', 'venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/WHEEL', 'venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/licenses/LICENSE.txt', 'venv/lib/python3.12/site-packages/urllib3/__init__.py', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/_base_connection.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/_collections.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/_request_methods.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/_version.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/connection.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/connectionpool.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/exceptions.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/fields.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/filepost.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/poolmanager.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/__pycache__/response.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/_base_connection.py', 'venv/lib/python3.12/site-packages/urllib3/_collections.py', 'venv/lib/python3.12/site-packages/urllib3/_request_methods.py', 'venv/lib/python3.12/site-packages/urllib3/_version.py', 'venv/lib/python3.12/site-packages/urllib3/connection.py', 'venv/lib/python3.12/site-packages/urllib3/connectionpool.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/__init__.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/socks.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/__init__.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/connection.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/emscripten_fetch_worker.js', 'venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/fetch.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/request.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/response.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/pyopenssl.py', 'venv/lib/python3.12/site-packages/urllib3/contrib/socks.py', 'venv/lib/python3.12/site-packages/urllib3/exceptions.py', 'venv/lib/python3.12/site-packages/urllib3/fields.py', 'venv/lib/python3.12/site-packages/urllib3/filepost.py', 'venv/lib/python3.12/site-packages/urllib3/http2/__init__.py', 'venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/probe.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/http2/connection.py', 'venv/lib/python3.12/site-packages/urllib3/http2/probe.py', 'venv/lib/python3.12/site-packages/urllib3/poolmanager.py', 'venv/lib/python3.12/site-packages/urllib3/py.typed', 'venv/lib/python3.12/site-packages/urllib3/response.py', 'venv/lib/python3.12/site-packages/urllib3/util/__init__.py', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/connection.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/proxy.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/request.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/response.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/retry.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_match_hostname.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssltransport.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/timeout.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/url.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/util.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/__pycache__/wait.cpython-312.pyc', 'venv/lib/python3.12/site-packages/urllib3/util/connection.py', 'venv/lib/python3.12/site-packages/urllib3/util/proxy.py', 'venv/lib/python3.12/site-packages/urllib3/util/request.py', 'venv/lib/python3.12/site-packages/urllib3/util/response.py', 'venv/lib/python3.12/site-packages/urllib3/util/retry.py', 'venv/lib/python3.12/site-packages/urllib3/util/ssl_.py', 'venv/lib/python3.12/site-packages/urllib3/util/ssl_match_hostname.py', 'venv/lib/python3.12/site-packages/urllib3/util/ssltransport.py', 'venv/lib/python3.12/site-packages/urllib3/util/timeout.py', 'venv/lib/python3.12/site-packages/urllib3/util/url.py', 'venv/lib/python3.12/site-packages/urllib3/util/util.py', 'venv/lib/python3.12/site-packages/urllib3/util/wait.py', 'venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/INSTALLER', 'venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/METADATA', 'venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/RECORD', 'venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/REQUESTED', 'venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/WHEEL', 'venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/licenses/LICENSE', 'venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/top_level.txt', 'venv/lib/python3.12/site-packages/xxhash/__init__.py', 'venv/lib/python3.12/site-packages/xxhash/__init__.pyi', 'venv/lib/python3.12/site-packages/xxhash/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/xxhash/__pycache__/version.cpython-312.pyc', 'venv/lib/python3.12/site-packages/xxhash/_xxhash.cpython-312-x86_64-linux-gnu.so', 'venv/lib/python3.12/site-packages/xxhash/py.typed', 'venv/lib/python3.12/site-packages/xxhash/version.py', 'venv/lib/python3.12/site-packages/yaml/__init__.py', 'venv/lib/python3.12/site-packages/yaml/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/composer.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/constructor.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/cyaml.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/dumper.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/emitter.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/error.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/events.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/loader.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/nodes.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/parser.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/reader.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/representer.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/resolver.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/scanner.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/serializer.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/__pycache__/tokens.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yaml/_yaml.cpython-312-x86_64-linux-gnu.so', 'venv/lib/python3.12/site-packages/yaml/composer.py', 'venv/lib/python3.12/site-packages/yaml/constructor.py', 'venv/lib/python3.12/site-packages/yaml/cyaml.py', 'venv/lib/python3.12/site-packages/yaml/dumper.py', 'venv/lib/python3.12/site-packages/yaml/emitter.py', 'venv/lib/python3.12/site-packages/yaml/error.py', 'venv/lib/python3.12/site-packages/yaml/events.py', 'venv/lib/python3.12/site-packages/yaml/loader.py', 'venv/lib/python3.12/site-packages/yaml/nodes.py', 'venv/lib/python3.12/site-packages/yaml/parser.py', 'venv/lib/python3.12/site-packages/yaml/reader.py', 'venv/lib/python3.12/site-packages/yaml/representer.py', 'venv/lib/python3.12/site-packages/yaml/resolver.py', 'venv/lib/python3.12/site-packages/yaml/scanner.py', 'venv/lib/python3.12/site-packages/yaml/serializer.py', 'venv/lib/python3.12/site-packages/yaml/tokens.py', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/INSTALLER', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/METADATA', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/RECORD', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/REQUESTED', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/WHEEL', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/licenses/LICENSE', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/licenses/NOTICE', 'venv/lib/python3.12/site-packages/yarl-1.23.0.dist-info/top_level.txt', 'venv/lib/python3.12/site-packages/yarl/__init__.py', 'venv/lib/python3.12/site-packages/yarl/__pycache__/__init__.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yarl/__pycache__/_parse.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yarl/__pycache__/_path.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yarl/__pycache__/_query.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yarl/__pycache__/_quoters.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yarl/__pycache__/_quoting.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yarl/__pycache__/_url.cpython-312.pyc', 'venv/lib/python3.12/site-packages/yarl/_parse.py', 'venv/lib/python3.12/site-packages/yarl/_path.py', 'venv/lib/python3.12/site-packages/yarl/_query.py', 'venv/lib/python3.12/site-packages/yarl/_quoters.py', 'venv/lib/python3.12/site-packages/yarl/_quoting.py', 'venv/lib/python3.12/site-packages/yarl/_quoting_c.cpython-312-x86_64-linux-gnu.so', 'venv/lib/python3.12/site-packages/yarl/_quoting_c.pyx', 'venv/lib/python3.12/site-packages/yarl/_quoting_py.py', 'venv/lib/python3.12/site-packages/yarl/_url.py', 'venv/lib/python3.12/site-packages/yarl/py.typed', 'venv/pyvenv.cfg', 'venv/share/man/man1/isympy.1']

### [2026-05-19 02:55:26 UTC] ASSISTANT

**Thinking**

OK, the heredoc worked but the Python output got cut off. Let me run a simpler command to get the key metrics.


**Tool call:** `exec`

```json
{
  "command": "python3 -c \"\nimport os\nfrom pathlib import Path\n\nquantized_dir = Path('/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16')\nruntime_dir = Path('/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16')\n\ntotal_bytes = sum(f.stat().st_size for f in quantized_dir.rglob('*') if f.is_file())\nquantized_size_mb = total_bytes / (1024 * 1024)\n\noriginal_size_mb = 16000.0\ncompression_ratio = original_size_mb / quantized_size_mb\n\nprint(f'Quantized size MB: {quantized_size_mb:.2f}')\nprint(f'Original estimate MB: {original_size_mb:.2f}')\nprint(f'Compression ratio: {compression_ratio:.2f}x')\nprint(f'Duration: 161 seconds')\n\noutput_files = sorted([str(f.relative_to(runtime_dir)) for f in runtime_dir.rglob('*') if f.is_file() and 'venv' not in str(f) and '__pycache__' not in str(f)])\nfor f in output_files:\n    print(f'  {f}')\n\""
}
```

### [2026-05-19 02:55:26 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Quantized size MB: 5827.03
Original estimate MB: 16000.00
Compression ratio: 2.75x
Duration: 161 seconds
  logs/auto.log
  logs/quant_exec.log
  logs/quant_prompt.txt
  quantize.py
  request.json

### [2026-05-19 02:55:43 UTC] ASSISTANT

**Thinking**

Now I have all the info I need. Let me write the quant_summary.json atomically.


**Tool call:** `exec`

```json
{
  "command": "python3 << 'EOF'\nimport json\nfrom pathlib import Path\n\nruntime_dir = Path(\"/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16\")\nquantized_dir = Path(\"/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16\")\n\ntotal_bytes = sum(f.stat().st_size for f in quantized_dir.rglob(\"*\") if f.is_file())\nquantized_size_mb = total_bytes / (1024 * 1024)\noriginal_size_mb = 16000.0\ncompression_ratio = original_size_mb / quantized_size_mb\n\noutput_files = sorted([\n    str(f.relative_to(runtime_dir)) \n    for f in runtime_dir.rglob(\"*\") \n    if f.is_file() and 'venv' not in str(f) and '__pycache__' not in str(f)\n])\n\nsummary = {\n    \"model_id\": \"Qwen/WebWorld-8B\",\n    \"scheme\": \"W4A16\",\n    \"method\": \"RTN\",\n    \"export_format\": \"auto_round\",\n    \"device\": \"cuda\",\n    \"quant_num_gpus\": \"1\",\n    \"num_gpus\": \"1\",\n    \"output_dir\": str(runtime_dir),\n    \"runtime_output_dir\": str(runtime_dir),\n    \"quantized_model_dir\": str(quantized_dir),\n    \"status\": \"success\",\n    \"duration_seconds\": 161.0,\n    \"original_size_mb\": original_size_mb,\n    \"quantized_size_mb\": round(quantized_size_mb, 2),\n    \"compression_ratio\": round(compression_ratio, 2),\n    \"errors\": [],\n    \"solutions\": [],\n    \"output_files\": output_files\n}\n\n# Write atomically via temp file\ntmp_path = runtime_dir / \"quant_summary.json.tmp\"\nwith open(tmp_path, \"w\") as f:\n    json.dump(summary, f, indent=2)\ntmp_path.rename(runtime_dir / \"quant_summary.json\")\n\nprint(\"Summary written successfully\")\nprint(json.dumps(summary, indent=2))\nEOF"
}
```

### [2026-05-19 02:55:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Summary written successfully
{
  "model_id": "Qwen/WebWorld-8B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16",
  "status": "success",
  "duration_seconds": 161.0,
  "original_size_mb": 16000.0,
  "quantized_size_mb": 5827.03,
  "compression_ratio": 2.75,
  "errors": [],
  "solutions": [],
  "output_files": [
    "logs/auto.log",
    "logs/quant_exec.log",
    "logs/quant_prompt.txt",
    "quantize.py",
    "request.json"
  ]
}

### [2026-05-19 02:56:01 UTC] ASSISTANT

**Thinking**

Quantization is complete. Let me provide a summary.




<final>
## ✅ Quantization Complete — Qwen/WebWorld-8B

### Configuration
| Parameter | Value |
|-----------|-------|
| Model | Qwen/WebWorld-8B |
| Scheme | W4A16 (INT4 weight, FP16 activation) |
| Method | RTN (iters=0, no calibration training) |
| Format | auto_round |
| Device | cuda (1 GPU) |
| Export path | `/root/.openclaw/workspace/quantized/Qwen_WebWorld-8B-W4A16` |
| Runtime dir | `/root/.openclaw/workspace/quantized/runs/Qwen_WebWorld-8B-W4A16` |

### Results
| Metric | Value |
|--------|-------|
| Status | ✅ success |
| Duration | 161 seconds (~2.7 min) |
| Original size (est.) | 16,000 MB (BF16) |
| Quantized size | 5,827 MB |
| Compression ratio | **2.75x** |
| Peak VRAM | 7.37 GB |
| Peak RAM | 24.91 GB |

### Output Files (runtime dir)
- `quantize.py` — quantization script
- `quant_summary.json` — structured summary
- `logs/quant_exec.log` — full execution log
- `logs/auto.log`, `logs/quant_prompt.txt` — additional logs

### Model Files (quantized output)
- 7 model shards (`model-0000X-of-00007.safetensors`)
- `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`
- `quantization_config.json` — contains autoround config (bits=4, group_size=128, sym=true, iters=0)
- `model.safetensors.index.json` — shard mapping

### Notes
- Quantization used the default NeelNanda/pile-10k calibration dataset (128 samples, filtered to 1,216 after filtering)
- Some layers were skipped (shape not divisible by 32) — this is expected behavior
- The HF_TOKEN warning is benign for public models
</final>
