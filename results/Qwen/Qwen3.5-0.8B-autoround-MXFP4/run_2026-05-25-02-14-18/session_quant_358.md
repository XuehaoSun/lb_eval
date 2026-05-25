# Session: autoeval_quant_358

- **Session ID:** `autoeval_quant_358`
- **Timestamp:** 2026-05-25 01:02:46 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 1: Quantization

### [2026-05-25 01:02:46 UTC] USER

You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_quant/SKILL.md

Model: Qwen/Qwen3.5-0.8B
Quantization: MXFP4 / RTN
Export format: auto_round
Quantized Model Output directory: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4
Quantization execution log path: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/logs/quant_exec.log
Runtime device: cuda
Num gpus: 1

Directory responsibilities:
- Write exported model files to: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4
- Write runtime artifacts such as quant_summary.json, quantize.py, logs, prompts, copied request/session files, and the venv to: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4

CRITICAL SCRIPT REQUIREMENT:
- Before starting quantization, you MUST first generate the quantization script file:
    /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quantize.py
- The file name must be exactly: quantize.py
- The script must be a standalone Python program runnable with:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quantize.py
- The generated quantize.py must focus on the core quantization/export logic only.
- Do NOT put venv creation, pip/uv installation, proxy/bootstrap shell setup, summary generation, or unrelated orchestration into quantize.py.
- In this same OpenClaw task, prepare or reuse the Python environment separately before executing quantize.py.
- In this same OpenClaw task, first write quantize.py, then execute that generated script yourself.
- When you execute quantize.py, you MUST stream stdout/stderr into this log file while still printing output:
    python3 -u /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quantize.py 2>&1 | tee /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/logs/quant_exec.log
- /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json is a final summary artifact, not a progress marker.
- After quantize.py finishes, inspect the exported artifacts and write /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json in a separate finalize step outside quantize.py.
- Do NOT write a success /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json until quantization has finished and the exported artifacts are ready.
- Write /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json atomically via a temporary file and rename/move it into place only at finalize time.
- If quantization fails, still write a minimal failed /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json before exiting non-zero, also atomically.
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

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json - structured summary:
{
  "model_id": "Qwen/Qwen3.5-0.8B",
  "scheme": "MXFP4",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "quant_num_gpus": "1",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4",
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

### [2026-05-25 01:02:51 UTC] ASSISTANT

**Thinking**

Let me start by reading the skill file to understand the quantization process, then proceed with the task.

**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
}
```

### [2026-05-25 01:02:51 UTC] TOOL RESULT
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


### [2026-05-25 01:02:57 UTC] ASSISTANT

**Thinking**

Now I understand the skill. Let me plan the work:

1. **Model**: Qwen/Qwen3.5-0.8B
2. **Scheme**: MXFP4
3. **Method**: RTN (iters=0)
4. **Export format**: auto_round
5. **Output directory**: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4
6. **Runtime artifact directory**: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4
7. **Num gpus**: 1 → device="cuda"
8. **Device**: cuda


9. I need to set up the output directories, establish or reuse the Python environment, create a quantization script, execute it with logging, and then finalize with a summary.

**Tool call:** `exec`

```json
{
  "command": "mkdir -p /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4 /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/logs"
}
```

### [2026-05-25 01:02:57 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-25 01:03:01 UTC] ASSISTANT

**Thinking**

Good. Now let me check if there's a pre-existing venv or model_info.json to reuse.

**Tool call:** `exec`

```json
{
  "command": "echo \"=== Check /root/.venv ===\" && ls -la /root/.venv/bin/python 2>/dev/null && /root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())\" 2>/dev/null || echo \"No /root/.venv or issues\""
}
```

### [2026-05-25 01:03:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

=== Check /root/.venv ===
lrwxrwxrwx 1 root root 73 Apr 24 05:53 /root/.venv/bin/python -> /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
No /root/.venv or issues

### [2026-05-25 01:03:04 UTC] ASSISTANT

**Thinking**

The /root/.venv exists but torch import failed. Let me check what's happening.

**Tool call:** `exec`

```json
{
  "command": "/root/.venv/bin/python -c \"import torch; print('torch:', torch.__version__)\" 2>&1"
}
```

### [2026-05-25 01:03:04 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-25 01:03:07 UTC] ASSISTANT

**Thinking**

/root/.venv doesn't have torch. But the instructions say:
> System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
> python3 -m venv --system-site-packages <path>

So system Python has torch. Let me check the system Python.

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())\" 2>&1"
}
```

### [2026-05-25 01:03:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `1`

Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'torch'

(Command exited with code 1)

### [2026-05-25 01:03:10 UTC] ASSISTANT

**Thinking**

System Python doesn't have torch either. Let me check what Python versions are available and find torch.

**Tool call:** `exec`

```json
{
  "command": "which python3 && python3 --version && find /usr /opt /root -name \"torch\" -type d 2>/dev/null | head -10"
}
```

### [2026-05-25 01:03:10 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.venv/bin/python3
Python 3.12.13

### [2026-05-25 01:03:13 UTC] ASSISTANT

**Thinking**

It seems there's no torch installed anywhere. I need to create a venv with --system-site-packages and then install torch. Let me check if there's a CUDA installation available.

**Tool call:** `exec`

```json
{
  "command": "nvidia-smi 2>&1 | head -20 && echo \"---\" && which nvcc && nvcc --version 2>&1 | head -5"
}
```

### [2026-05-25 01:03:13 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Mon May 25 01:03:13 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:25:00.0 Off |                  Off |
|  0%   34C    P8             17W /  450W |       2MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
---
/usr/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0

### [2026-05-25 01:03:17 UTC] ASSISTANT

**Thinking**

CUDA is available. Now let me create a venv with --system-site-packages and install torch. Since the system Python doesn't have torch either, I'll need to install it. Let me check what pip packages are available.

**Tool call:** `exec`

```json
{
  "command": "pip3 list 2>/dev/null | head -20 && echo \"---\" && pip list 2>/dev/null | head -20"
}
```

### [2026-05-25 01:03:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

---

### [2026-05-25 01:03:19 UTC] ASSISTANT

**Thinking**

Pip is empty or not available. Let me check what's in the system site packages.

**Tool call:** `exec`

```json
{
  "command": "ls /usr/local/lib/python3*/dist-packages/ 2>/dev/null | head -30 && echo \"---\" && ls /usr/lib/python3*/dist-packages/ 2>/dev/null | head -30"
}
```

### [2026-05-25 01:03:19 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

---

### [2026-05-25 01:03:22 UTC] ASSISTANT

**Thinking**

No dist-packages. Let me check more broadly.

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import site; print(site.getsitepackages()); print(site.getusersitepackages())\""
}
```

### [2026-05-25 01:03:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

['/root/.venv/lib/python3.12/site-packages']
/root/.local/lib/python3.12/site-packages

### [2026-05-25 01:03:27 UTC] ASSISTANT

**Thinking**

So the only site-packages is in /root/.venv, and there's no torch. Let me install torch. I'll need to create a venv first and then install torch using uv pip.

Actually, let me re-read the instructions more carefully:

> System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
> python3 -m venv --system-site-packages <path>
> This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
> If /root/.venv exists, reuse /root/.venv before creating a new venv.
> Use uv pip for dependency installation.


> Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment.

So the expectation is that torch+cuda is pre-installed system-wide. But it seems it's not. Let me check if there's a conda or any other python with torch.

**Tool call:** `exec`

```json
{
  "command": "find / -name \"torch\" -type d 2>/dev/null | head -10 && echo \"---\" && find / -name \"auto_round\" -type d 2>/dev/null | head -10"
}
```

### [2026-05-25 01:03:28 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

---

### [2026-05-25 01:03:33 UTC] ASSISTANT

**Thinking**

Neither torch nor auto_round are installed anywhere. I need to install them. Let me create a venv with --system-site-packages and then install the needed packages.

Wait, the instruction says "This ensures the venv inherits torch+cuda" and "Do NOT pip install torch inside the venv." But there's no system torch to inherit. So I think the right approach is:

1. Create a venv with --system-site-packages (as instructed)
2. Install torch using uv pip (since there's no system torch to inherit)

Actually wait - re-reading: "If /root/.venv exists, reuse /root/.venv before creating a new venv."


But /root/.venv doesn't have torch. So I need to create a new venv. Let me create it in the output directory and install torch + auto-round.

Now I'll create the venv and install dependencies:


**Tool call:** `exec`

```json
{
  "command": "python3 -m venv --system-site-packages /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv 2>&1"
}
```

### [2026-05-25 01:03:35 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-05-25 01:03:39 UTC] ASSISTANT

**Thinking**

Now let me install torch with CUDA support and auto-round.

**Tool call:** `exec`

```json
{
  "command": "VENV_PY=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/bin/python\n\n# Install uv if needed\n$VENV_PY -m pip install -U uv 2>&1 | tail -5\n\n# Install torch (CUDA 12.4 compatible)\nuv pip install --python $VENV_PY --index-url https://download.pytorch.org/whl/cu124 torch 2>&1 | tail -5",
  "timeout": 300
}
```

### [2026-05-25 01:03:49 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session young-kelp, pid 574). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-25 01:03:53 UTC] ASSISTANT

**Thinking**

Let me wait for this to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "young-kelp",
  "timeout": 300000
}
```

### [2026-05-25 01:04:09 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

Installing collected packages: uv
Successfully installed uv-0.11.16

[notice] A new release of pip is available: 25.0.1 -> 26.1.1
[notice] To update, run: python -m pip install --upgrade pip
 + setuptools==70.2.0
 + sympy==1.13.1
 + torch==2.6.0+cu124
 + triton==3.2.0
 + typing-extensions==4.15.0

Process exited with code 0.

### [2026-05-25 01:04:14 UTC] ASSISTANT

**Thinking**

Good, torch is installed. Now let me install auto-round.

**Tool call:** `exec`

```json
{
  "command": "VENV_PY=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/bin/python\n\n# Verify torch works\n$VENV_PY -c \"import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())\" 2>&1\n\n# Install auto-round\nuv pip install --python $VENV_PY auto-round 2>&1 | tail -10",
  "timeout": 300
}
```

### [2026-05-25 01:04:21 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:275: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:81.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
torch: 2.6.0+cu124
cuda: True
 + shellingham==1.5.4
 + six==1.17.0
 + tokenizers==0.22.2
 + tqdm==4.67.3
 + transformers==5.9.0
 + typer==0.25.1
 + typing-inspection==0.4.2
 + urllib3==2.7.0
 + xxhash==3.7.0
 + yarl==1.24.2

### [2026-05-25 01:04:26 UTC] ASSISTANT

**Thinking**

Wait, I see transformers 5.9.0 but auto-round is installed without the full list showing. Let me check if auto-round itself was installed.

**Tool call:** `exec`

```json
{
  "command": "VENV_PY=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/bin/python\n$VENV_PY -c \"import auto_round; print('auto_round version:', auto_round.__version__)\" 2>&1"
}
```

### [2026-05-25 01:04:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session vivid-glade, pid 763). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-05-25 01:04:40 UTC] ASSISTANT

**Thinking**

Let me wait for this to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "vivid-glade",
  "timeout": 60000
}
```

### [2026-05-25 01:04:40 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

auto_round version: 0.12.3


Process exited with code 0.

### [2026-05-25 01:04:43 UTC] ASSISTANT

**Thinking**

auto_round 0.12.3 is installed. Now let me also install numpy since torch was complaining about it.

**Tool call:** `exec`

```json
{
  "command": "VENV_PY=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/bin/python\nuv pip install --python $VENV_PY numpy 2>&1 | tail -5",
  "timeout": 120
}
```

### [2026-05-25 01:04:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Using Python 3.12.13 environment at: quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv
Checked 1 package in 40ms

### [2026-05-25 01:04:45 UTC] ASSISTANT

**Thinking**

Hmm, uv pip seems to not actually install. Let me check if numpy is already there.

**Tool call:** `exec`

```json
{
  "command": "VENV_PY=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/bin/python\n$VENV_PY -c \"import numpy; print('numpy:', numpy.__version__)\" 2>&1"
}
```

### [2026-05-25 01:04:46 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

numpy: 2.4.6

### [2026-05-25 01:04:48 UTC] ASSISTANT

**Thinking**

numpy is already there. Now let me also install transformers and accelerate if they're not already.

**Tool call:** `exec`

```json
{
  "command": "VENV_PY=/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/bin/python\n$VENV_PY -c \"import transformers; print('transformers:', transformers.__version__)\" 2>&1\n$VENV_PY -c \"import accelerate; print('accelerate ok')\" 2>&1"
}
```

### [2026-05-25 01:04:51 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

transformers: 5.9.0
accelerate ok

### [2026-05-25 01:05:00 UTC] ASSISTANT

**Thinking**

Everything is set up. Now let me create the quantization script. The requirements are:
- Model: Qwen/Qwen3.5-0.8B
- Scheme: MXFP4
- Method: RTN (iters=0)
- Export format: auto_round
- Output dir: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4
- Runtime device: cuda
- Num gpus: 1 → device="cuda"

Now I'll create the quantization script:


**Tool call:** `write`

```json
{
  "path": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quantize.py",
  "content": "#!/usr/bin/env python3\n\"\"\"\nAuto-Round Quantization Script\nGenerated by auto_quant skill\n\nModel: Qwen/Qwen3.5-0.8B\nScheme: MXFP4\nMethod: RTN (iters=0)\nSamples: 128\nFormat: auto_round\nDevice: cuda (single GPU)\n\"\"\"\n\nimport os\nimport time\n\nstart_time = time.time()\n\n# Configuration\nmodel_name_or_path = \"Qwen/Qwen3.5-0.8B\"\noutput_dir = \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4\"\nscheme = \"MXFP4\"\niters = 0  # RTN mode\nnsamples = 128\nformat_str = \"auto_round\"\nnum_gpus = 1\n\nprint(f\"Loading model: {model_name_or_path}\")\nprint(f\"Scheme: {scheme}\")\nprint(f\"Iters: {iters} (RTN mode)\")\nprint(f\"nsamples: {nsamples}\")\nprint(f\"Format: {format_str}\")\nprint(f\"Device: cuda (single GPU)\")\n\n# Import AutoRound\nfrom auto_round import AutoRound\n\n# Create AutoRound instance\n# Single GPU: use device=\"cuda\"\nar = AutoRound(\n    model_name_or_path,\n    scheme=scheme,\n    iters=iters,\n    nsamples=nsamples,\n    device=\"cuda\",\n)\n\n# Quantize and save\nprint(\"Starting quantization...\")\nar.quantize_and_save(output_dir=output_dir, format=format_str)\n\nend_time = time.time()\nduration = end_time - start_time\nprint(f\"Quantization complete! Output: {output_dir}\")\nprint(f\"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\")\n"
}
```

### [2026-05-25 01:05:00 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 1229 bytes to /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quantize.py

### [2026-05-25 01:05:03 UTC] ASSISTANT

**Thinking**

Now let me execute the quantization script and stream output to the log file.

Now I'll execute the quantization script with output streamed to the log:


**Tool call:** `exec`

```json
{
  "command": "cd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4 && /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/venv/bin/python -u quantize.py 2>&1 | tee logs/quant_exec.log",
  "yieldMs": 30000,
  "background": false,
  "timeout": 1800
}
```

### [2026-05-25 01:05:22 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Loading model: Qwen/Qwen3.5-0.8B
Scheme: MXFP4
Iters: 0 (RTN mode)
nsamples: 128
Format: auto_round
Device: cuda (single GPU)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
404 Client Error. (Request ID: Root=1-6a13a044-2cc037eb20dcc2987213a25f;bf7d6c78-8662-4b03-9020-0e21fad0a6cd)

Entry Not Found for url: https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model_index.json.
[33;1m2026-05-25 01:05:12 WARNING modeling_qwen3_5.py L426: The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d[0m
[33;1m2026-05-25 01:05:12 WARNING base.py L407: `device` is deprecated, please use `device_map` instead[0m
[38;20m2026-05-25 01:05:12 INFO base.py L473: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-05-25 01:05:12 INFO base.py L517: using torch.bfloat16 for quantization tuning[0m
Starting quantization...
[33;1m2026-05-25 01:05:12 WARNING formats.py L166: some layers are skipped quantization (shape not divisible by 32): [0m
[33;1m2026-05-25 01:05:12 WARNING modeling_utils.py L4606: `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.[0m
[33;1m2026-05-25 01:05:12 WARNING base.py L1748: immediate_saving is only supported for int quantization, set to False[0m
[38;20m2026-05-25 01:05:12 INFO base.py L1401: RTN mode detected (iters=0): force blockwise quantization to avoid layer-wise full-model materialization.[0m

  0%|          | 0/24 [00:00<?, ?it/s]
Quantizing model.layers.0:   0%|          | 0/24 [00:00<?, ?it/s][38;20m2026-05-25 01:05:13 INFO offload.py L703: OffloadManager (compressor): tempdir = /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/ar_work_space/offload/compressor_git_plr3[0m
[38;20m2026-05-25 01:05:13 INFO device.py L1692: 'peak_ram': 1.2GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.1:   4%|▍         | 1/24 [00:00<00:12,  1.80it/s][38;20m2026-05-25 01:05:13 INFO device.py L1692: 'peak_ram': 1.24GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.2:   8%|▊         | 2/24 [00:01<00:11,  1.95it/s][38;20m2026-05-25 01:05:14 INFO device.py L1692: 'peak_ram': 1.28GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.3:  12%|█▎        | 3/24 [00:01<00:09,  2.10it/s][38;20m2026-05-25 01:05:14 INFO device.py L1692: 'peak_ram': 1.31GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.4:  17%|█▋        | 4/24 [00:01<00:09,  2.19it/s][38;20m2026-05-25 01:05:15 INFO device.py L1692: 'peak_ram': 1.35GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.5:  21%|██        | 5/24 [00:02<00:08,  2.27it/s][38;20m2026-05-25 01:05:15 INFO device.py L1692: 'peak_ram': 1.39GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.6:  25%|██▌       | 6/24 [00:02<00:07,  2.46it/s][38;20m2026-05-25 01:05:15 INFO device.py L1692: 'peak_ram': 1.43GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.7:  29%|██▉       | 7/24 [00:02<00:06,  2.54it/s][38;20m2026-05-25 01:05:16 INFO device.py L1692: 'peak_ram': 1.47GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.8:  33%|███▎      | 8/24 [00:03<00:06,  2.61it/s][38;20m2026-05-25 01:05:16 INFO device.py L1692: 'peak_ram': 1.51GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.9:  38%|███▊      | 9/24 [00:03<00:05,  2.62it/s][38;20m2026-05-25 01:05:16 INFO device.py L1692: 'peak_ram': 1.51GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.10:  42%|████▏     | 10/24 [00:03<00:05,  2.69it/s][38;20m2026-05-25 01:05:17 INFO device.py L1692: 'peak_ram': 1.55GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.11:  46%|████▌     | 11/24 [00:04<00:04,  2.70it/s][38;20m2026-05-25 01:05:17 INFO device.py L1692: 'peak_ram': 1.59GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.12:  50%|█████     | 12/24 [00:04<00:04,  2.74it/s][38;20m2026-05-25 01:05:17 INFO device.py L1692: 'peak_ram': 1.63GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.13:  54%|█████▍    | 13/24 [00:04<00:03,  2.78it/s][38;20m2026-05-25 01:05:17 INFO device.py L1692: 'peak_ram': 1.67GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.14:  58%|█████▊    | 14/24 [00:04<00:03,  2.80it/s][38;20m2026-05-25 01:05:18 INFO device.py L1692: 'peak_ram': 1.71GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.15:  62%|██████▎   | 15/24 [00:05<00:03,  2.86it/s][38;20m2026-05-25 01:05:18 INFO device.py L1692: 'peak_ram': 1.74GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.16:  67%|██████▋   | 16/24 [00:05<00:02,  2.91it/s][38;20m2026-05-25 01:05:18 INFO device.py L1692: 'peak_ram': 1.78GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.17:  71%|███████   | 17/24 [00:05<00:02,  2.98it/s][38;20m2026-05-25 01:05:18 INFO device.py L1692: 'peak_ram': 1.82GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.18:  75%|███████▌  | 18/24 [00:05<00:01,  3.05it/s][38;20m2026-05-25 01:05:19 INFO device.py L1692: 'peak_ram': 1.86GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.19:  79%|███████▉  | 19/24 [00:06<00:01,  3.11it/s][38;20m2026-05-25 01:05:19 INFO device.py L1692: 'peak_ram': 1.89GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.20:  83%|████████▎ | 20/24 [00:06<00:01,  3.18it/s][38;20m2026-05-25 01:05:19 INFO device.py L1692: 'peak_ram': 1.93GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.21:  88%|████████▊ | 21/24 [00:06<00:00,  3.23it/s][38;20m2026-05-25 01:05:19 INFO device.py L1692: 'peak_ram': 1.97GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.22:  92%|█████████▏| 22/24 [00:06<00:00,  3.29it/s][38;20m2026-05-25 01:05:19 INFO device.py L1692: 'peak_ram': 2.01GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.23:  96%|█████████▌| 23/24 [00:06<00:00,  3.34it/s][38;20m2026-05-25 01:05:20 INFO device.py L1692: 'peak_ram': 2.04GB, 'peak_vram': 0.56GB[0m

Quantizing model.layers.23: 100%|██████████| 24/24 [00:07<00:00,  3.29it/s]

packing:   0%|          | 0/186 [00:00<?, ?it/s]
packing: 100%|██████████| 186/186 [00:00<00:00, 301911.98it/s]
[33;1m2026-05-25 01:05:20 WARNING export_to_nvfp_mxfp.py L249: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4 already exists, this may cause model conflict[0m
[38;20m2026-05-25 01:05:21 INFO missing_tensors.py L236: Found 168 tensor(s) in the source checkpoint that are absent from the saved output (e.g., MTP parameters): model.visual.blocks.[0-11].attn.proj, model.visual.blocks.[0-11].attn.qkv, model.visual.blocks.[0-11].mlp.linear_fc1, model.visual.blocks.[0-11].mlp.linear_fc2, model.visual.blocks.[0-11].norm1, model.visual.blocks.[0-11].norm2, model.visual.merger.linear_fc1, model.visual.merger.linear_fc2, model.visual.merger.norm, model.visual.patch_embed.proj, model.visual.pos_embed, mtp.fc, mtp.layers.0.input_layernorm, mtp.layers.0.mlp.down_proj, mtp.layers.0.mlp.gate_proj, mtp.layers.0.mlp.up_proj, mtp.layers.0.post_attention_layernorm, mtp.layers.0.self_attn.k_norm, mtp.layers.0.self_attn.k_proj, mtp.layers.0.self_attn.o_proj, mtp.layers.0.self_attn.q_norm, mtp.layers.0.self_attn.q_proj, mtp.layers.0.self_attn.v_proj, mtp.norm, mtp.pre_fc_norm_embedding, mtp.pre_fc_norm_hidden. Copying them now...
[0m

Loading missing tensors:   0%|          | 0/1 [00:00<?, ?shard/s]

  model.safetensors-00001-of-00001.safetensors:   0%|          | 0/168 [00:00<?, ?tensor/s][A

                                                                                           [A
Loading missing tensors: 100%|██████████| 1/1 [00:00<00:00, 110.61shard/s]
[38;20m2026-05-25 01:05:21 INFO missing_tensors.py L370: Successfully wrote 168 missing tensor(s) to 'model_extra_tensors.safetensors' in /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4.[0m
[38;20m2026-05-25 01:05:21 INFO device.py L1692: 'peak_ram': 2.04GB, 'peak_vram': 0.56GB[0m
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4
Duration: 17.55 seconds (0.29 minutes)

### [2026-05-25 01:05:28 UTC] ASSISTANT

**Thinking**

The quantization completed successfully. Now I need to:
1. Inspect the exported artifacts
2. Calculate sizes
3. Write quant_summary.json

Let me first look at what was exported.


Quantization completed successfully. Let me inspect the output artifacts:


**Tool call:** `exec`

```json
{
  "command": "echo \"=== Output dir ===\" && ls -la /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/ && echo \"=== Runtime dir ===\" && ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/ && echo \"=== File sizes ===\" && du -sh /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/ && echo \"=== Total runtime artifacts ===\" && du -sh /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/"
}
```

### [2026-05-25 01:05:29 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

=== Output dir ===
total 1011892
drwxr-xr-x 2 root root      4096 May 25 01:05 .
drwxr-xr-x 4 root root        61 May 25 01:02 ..
-rw-r--r-- 1 root root      7755 May 25 01:05 chat_template.jinja
-rw-r--r-- 1 root root      2157 May 25 01:05 config.json
-rw-r--r-- 1 root root       136 May 25 01:05 generation_config.json
-rw-r--r-- 1 root root 773981072 May 25 01:05 model.safetensors
-rw-r--r-- 1 root root     58049 May 25 01:05 model.safetensors.index.json
-rw-r--r-- 1 root root 242109752 May 25 01:05 model_extra_tensors.safetensors
-rw-r--r-- 1 root root       313 May 25 01:05 quantization_config.json
-rw-r--r-- 1 root root  19989325 May 25 01:05 tokenizer.json
-rw-r--r-- 1 root root      1125 May 25 01:05 tokenizer_config.json
=== Runtime dir ===
total 8
drwxr-xr-x 5 root root  114 May 25 01:05 .
drwxr-xr-x 3 root root   45 May 25 01:02 ..
drwxr-xr-x 2 root root   10 May 25 01:05 ar_work_space
drwxr-xr-x 2 root root   84 May 25 01:05 logs
-rw-r--r-- 1 root root 1229 May 25 01:05 quantize.py
-rw-r--r-- 1 root root 1087 May 25 01:02 request.json
drwxr-xr-x 6 root root  132 May 25 01:04 venv
=== File sizes ===
989M	/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/
=== Total runtime artifacts ===
5.4G	/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/

### [2026-05-25 01:05:36 UTC] ASSISTANT

**Thinking**

Let me check the size of the model. The original Qwen3.5-0.8B is about 0.8B parameters, which in BF16 would be about 1.6GB. But I see the model.safetensors is 773MB and model_extra_tensors.safetensors is 242MB. Let me also check what the original model size would be.

Actually, let me think more carefully:
- Qwen/Qwen3.5-0.8B has about 0.8 billion parameters
- In BF16, that's ~1.6GB
- In MXFP4, it should be much smaller


- The model.safetensors is 738MB, which is plausible for MXFP4 compression

I need to get the exact original model size to calculate the compression ratio properly.

**Tool call:** `exec`

```json
{
  "command": "echo \"=== model.safetensors ===\" && ls -lh /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors && echo \"=== model_extra_tensors ===\" && ls -lh /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/model_extra_tensors.safetensors && echo \"=== quantization_config ===\" && cat /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/quantization_config.json && echo \"=== config.json ===\" && cat /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/config.json | python3 -c \"import sys,json; d=json.load(sys.stdin); print('hidden_size:', d.get('hidden_size')); print('num_hidden_layers:', d.get('num_hidden_layers')); print('vocab_size:', d.get('vocab_size'))\""
}
```

### [2026-05-25 01:05:36 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

=== model.safetensors ===
-rw-r--r-- 1 root root 739M May 25 01:05 /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors
=== model_extra_tensors ===
-rw-r--r-- 1 root root 231M May 25 01:05 /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4/model_extra_tensors.safetensors
=== quantization_config ===
{
  "bits": 4,
  "act_bits": 4,
  "data_type": "mx_fp",
  "act_data_type": "mx_fp",
  "group_size": 32,
  "act_group_size": 32,
  "sym": true,
  "act_sym": true,
  "act_dynamic": true,
  "iters": 0,
  "autoround_version": "0.12.3",
  "quant_method": "auto-round",
  "packing_format": "auto_round:llm_compressor"
}=== config.json ===
hidden_size: 1024
num_hidden_layers: 24
vocab_size: 248320

### [2026-05-25 01:05:46 UTC] ASSISTANT

**Thinking**

Now I need to figure out the original model size. Qwen/Qwen3.5-0.8B has:
- 24 layers
- hidden_size: 1024
- vocab_size: 248320

Let me calculate:
- Embeddings: vocab_size * hidden_size * 2 (embedding + output) = 248320 * 1024 * 2 * 2 bytes (BF16) = ~1016 MB
- Attention layers: 24 * (4 * hidden_size^2 + 2 * hidden_size) for QKV + O + FFN
  - QKV: 4 * 1024^2 * 2 bytes = 8 MB per layer
  - O: 2 * 1024^2 * 2 bytes = 4 MB per layer
  - FFN (gate+up+down): 2 * vocab_size * 1024 + 1024 * 1024 * 4... actually this gets complicated

Actually let me just look up the original model size or try to estimate from the total BF16 size.

The main model.safetensors is 739MB (quantized) and model_extra_tensors.safetensors is 231MB (non-quantized, original precision). For MXFP4 compression with 32-element group size, I should calculate the original BF16 size to get an accurate compression ratio rather than relying on HuggingFace's reported 1.78GB.

**Tool call:** `exec`

```json
{
  "command": "# Get original model size from HuggingFace\ncurl -sI \"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model-00001-of-00002.safetensors\" 2>&1 | grep -i \"content-length\\|x-ggpy\\|x-filesize\" || echo \"---\"\ncurl -sI \"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model-00002-of-00002.safetensors\" 2>&1 | grep -i \"content-length\\|x-ggpy\\|x-filesize\" || echo \"---\""
}
```

### [2026-05-25 01:05:46 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

content-length: 15
content-length: 15

### [2026-05-25 01:05:54 UTC] ASSISTANT

**Thinking**

That didn't work - probably the model doesn't exist at that path or needs authentication. Let me try a different approach to calculate the original size.

Actually, for Qwen3.5-0.8B, the original BF16 size can be estimated from the model architecture:
- vocab_size: 248320
- hidden_size: 1024
- num_hidden_layers: 24
- intermediate_size: typically 4 * hidden_size = 4096 for Qwen models

The total parameters are approximately:
- embedding: 248320 * 1024 * 2 (word_embedding + lm_head) ≈ 508 MB
- each layer:
  - attention: QKV projections (3 * 1024 * 1024) + O projection (1024 * 1024) = 4 * 1024 * 1024 * 2 bytes = 8 MB
  - FFN: gate_proj (intermediate_size * hidden_size) + up_proj + down_proj
    For Qwen3, intermediate_size is typically 2730 or around there. Looking at the Qwen2.5-0.5B config, the hidden_size is 896 with an intermediate_size of 896 * 3.5 = 3136. I need to verify the exact intermediate_size from the model config.

**Tool call:** `exec`

```json
{
  "command": "curl -sL \"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json\" 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2))\" 2>&1 | head -50"
}
```

### [2026-05-25 01:05:54 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "architectures": [
    "Qwen3_5ForConditionalGeneration"
  ],
  "image_token_id": 248056,
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "attn_output_gate": true,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3584,
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
      "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,

### [2026-05-25 01:06:06 UTC] ASSISTANT

**Thinking**

Now I have the config. Let me calculate the original BF16 size.

For Qwen3.5-0.8B:
- hidden_size: 1024
- intermediate_size: 3584
- num_hidden_layers: 24
- vocab_size: 248320

The total parameters:
- word_embeddings: 248320 * 1024 = 254,310,880 parameters
- lm_head: 248320 * 1024 = 254,310,880 parameters (often tied)
- Each layer:
  - self_attn: q_proj (1024 * 1024) + k_proj + v_proj + o_proj (1024 * 1024)
  - mlp: gate_proj (3584 * 1024) + up_proj (3584 * 1024) + down_proj (1024 * 3584)
  
For a standard Qwen model, attention uses four fused projections totaling 4,194,304 parameters per layer, and the FFN has three linear layers.

For the MLP, I'm calculating gate_proj (3,670,016 params) + up_proj (3,670,016 params) + down_proj (3,670,016 params) = 11,010,048 params per layer, though I need to verify whether intermediate_size refers to the FFN intermediate dimension or something else for this specific architecture. up_proj and down_proj would be 3584 * 1024 each, with down_proj being the larger 1024 * 3584.

I'm working through the parameter counts more systematically now. The embeddings and LM head are straightforward at roughly 254 MB each in bf16, and I need to account for all 24 layers plus the final norm and any additional components like MoE or other structures. I'm getting tangled up in the matrix dimension math, so let me just look up the actual model size directly from HuggingFace instead of continuing to calculate it manually.

**Tool call:** `exec`

```json
{
  "command": "curl -sL \"https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B\" 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print('sha:', d.get('sha')); print('siblings:', d.get('siblings')); print('modelId:', d.get('modelId'))\" 2>&1"
}
```

### [2026-05-25 01:06:06 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

sha: 2fc06364715b967f1860aea9cf38778875588b17
siblings: [{'rfilename': '.gitattributes'}, {'rfilename': 'LICENSE'}, {'rfilename': 'README.md'}, {'rfilename': 'chat_template.jinja'}, {'rfilename': 'config.json'}, {'rfilename': 'merges.txt'}, {'rfilename': 'model.safetensors-00001-of-00001.safetensors'}, {'rfilename': 'model.safetensors.index.json'}, {'rfilename': 'preprocessor_config.json'}, {'rfilename': 'tokenizer.json'}, {'rfilename': 'tokenizer_config.json'}, {'rfilename': 'video_preprocessor_config.json'}, {'rfilename': 'vocab.json'}]
modelId: Qwen/Qwen3.5-0.8B

### [2026-05-25 01:06:09 UTC] ASSISTANT

**Thinking**

Interesting - the model is a single safetensors file. Let me try to get its size.

**Tool call:** `exec`

```json
{
  "command": "curl -sI \"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors\" 2>&1 | grep -i \"content-length\""
}
```

### [2026-05-25 01:06:09 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

content-length: 1405

### [2026-05-25 01:06:13 UTC] ASSISTANT

**Thinking**

That's only 1405 bytes, which is just the header. The actual model data is in the safetensors file. Let me try a different approach - check the raw URL.Actually, let me just compute the original model size. I'll use the HF API to get the file metadata.

**Tool call:** `exec`

```json
{
  "command": "curl -sL \"https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/model.safetensors.index.json\" 2>/dev/null | python3 -c \"\nimport sys, json\nd = json.load(sys.stdin)\ntotal = 0\nfor k, v in d.get('weight_map', {}).items():\n    print(k, v)\n\" 2>&1 | head -20"
}
```

### [2026-05-25 01:06:13 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

model.language_model.embed_tokens.weight model.safetensors-00001-of-00001.safetensors
model.visual.merger.linear_fc1.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.14.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.12.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.18.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.13.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.2.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.4.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.20.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.1.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.10.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.16.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.17.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.5.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.6.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.8.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.9.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.21.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.22.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors
model.language_model.layers.0.linear_attn.in_proj_qkv.weight model.safetensors-00001-of-00001.safetensors

### [2026-05-25 01:06:19 UTC] ASSISTANT

**Thinking**

I can see it's a Qwen3.5 vision language model. The model includes visual components. Let me try to get the original model size another way - by downloading the safetensors index and checking the total size.

Actually, let me try to calculate the model size from the safetensors header. The single file is `model-00001-of-00001.safetensors` at the root of the repo, not `model.safetensors`.

Wait, the siblings list shows:
- `model.safetensors-00001-of-00001.safetensors`

And I got a content-length of 1405 for that URL. 

So that's just the metadata header. Let me check the HF API for file sizes instead.

**Tool call:** `exec`

```json
{
  "command": "curl -sL \"https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B/tree/main\" 2>/dev/null | python3 -c \"\nimport sys, json\nfiles = json.load(sys.stdin)\nfor f in files:\n    print(f['path'], f.get('size', 'N/A'))\n\" 2>&1 | head -20"
}
```

### [2026-05-25 01:06:19 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

.gitattributes 1570
LICENSE 11544
README.md 61705
chat_template.jinja 7755
config.json 2907
merges.txt 3353259
model.safetensors-00001-of-00001.safetensors 1746942600
model.safetensors.index.json 50900
preprocessor_config.json 390
tokenizer.json 12807982
tokenizer_config.json 16709
video_preprocessor_config.json 385
vocab.json 6722759

### [2026-05-25 01:06:33 UTC] ASSISTANT

**Thinking**

The original model is 1,746,942,600 bytes ≈ 1.63 GB. That's the bf16 model size.

But wait, this is the full model including the visual (vision) encoder. The quantized model I exported only has the language model parts. Let me check what's in the quantized output more carefully.

Actually, looking at the log output:
```
[0m[38;20m2026-05-25 01:05:21 INFO missing_tensors.py L370: Successfully wrote 168 missing tensor(s) to 'model_extra_tensors.safetensors' in /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4.


```

These missing tensors include visual blocks, MTP parameters, and others that weren't quantized, so they're stored separately in model_extra_tensors at 231MB. The main model.safetensors at 739MB represents the quantized MXFP4 language model weights. Looking at the total size, 739MB quantized plus 231MB extra gives me 970MB compared to the original 1.63GB.

The compression ratio works out to roughly 1.68x. But I need to account for the fact that model_extra_tensors is stored in bf16/float32 precision—it's not actually quantized. Looking at what's in that extra file: visual projection layers, MTP module weights, and similar components. The real story here is just the language model quantization savings, since the visual encoder stays in its original precision.

```
model.safetensors (quantized): 739 MB
model_extra_tensors.safetensors (unquantized): 231 MB
Total: 970 MB
Original: 1666 MB (1746942600 bytes)
Compression ratio: 1666/970 = 1.72x
```

Though I should double-check whether the tokenizer files are already factored into that 1666 MB figure or if they're stored separately.

**Tool call:** `exec`

```json
{
  "command": "# Original model size from HuggingFace API\nORIGINAL_SIZE=1746942600\n\n# Quantized model sizes\nMODEL_SIZE=739\nEXTRA_SIZE=231\n# Total = model (quantized) + extra (unquantized but part of original)\nTOTAL_SIZE=$((MODEL_SIZE + EXTRA_SIZE))\n\necho \"Original model size: $ORIGINAL_SIZE bytes ($(echo \"scale=2; $ORIGINAL_SIZE/1024/1024\" | bc) MB)\"\necho \"Quantized model.safetensors: $MODEL_SIZE MB\"\necho \"model_extra_tensors.safetensors: $EXTRA_SIZE MB (unquantized, in original precision)\"\necho \"Total output: $TOTAL_SIZE MB\"\necho \"Compression ratio (total/original): $(echo \"scale=4; $ORIGINAL_SIZE/1024/1024/$TOTAL_SIZE\" | bc)x\"\n\n# Only quantized portion\necho \"Compression ratio (model only/original): $(echo \"scale=4; $ORIGINAL_SIZE/1024/1024/$MODEL_SIZE\" | bc)x\""
}
```

### [2026-05-25 01:06:33 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

sh: 10: bc: not found
Original model size: 1746942600 bytes ( MB)
Quantized model.safetensors: 739 MB
model_extra_tensors.safetensors: 231 MB (unquantized, in original precision)
Total output: 970 MB
sh: 14: bc: not found
Compression ratio (total/original): x
sh: 17: bc: not found
Compression ratio (model only/original): x

### [2026-05-25 01:06:38 UTC] ASSISTANT

**Thinking**

bc is not installed. Let me use python for the calculations.

**Tool call:** `exec`

```json
{
  "command": "python3 << 'EOF'\noriginal_size = 1746942600  # bytes\nmodel_size = 739 * 1024 * 1024  # 739 MB in bytes\nextra_size = 231 * 1024 * 1024  # 231 MB in bytes\n\noriginal_mb = original_size / 1024 / 1024\nmodel_mb = model_size / 1024 / 1024\nextra_mb = extra_size / 1024 / 1024\ntotal_mb = model_mb + extra_mb\n\nprint(f\"Original model size: {original_mb:.2f} MB ({original_size/1024/1024/1024:.4f} GB)\")\nprint(f\"model.safetensors (quantized MXFP4): {model_mb:.2f} MB\")\nprint(f\"model_extra_tensors.safetensors (unquantized): {extra_mb:.2f} MB\")\nprint(f\"Total output: {total_mb:.2f} MB\")\nprint(f\"Compression ratio (total/original): {original_mb/total_mb:.4f}x\")\nprint(f\"Compression ratio (quantized_only/original): {original_mb/model_mb:.4f}x\")\nEOF"
}
```

### [2026-05-25 01:06:38 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Original model size: 1666.01 MB (1.6270 GB)
model.safetensors (quantized MXFP4): 739.00 MB
model_extra_tensors.safetensors (unquantized): 231.00 MB
Total output: 970.00 MB
Compression ratio (total/original): 1.7175x
Compression ratio (quantized_only/original): 2.2544x

### [2026-05-25 01:06:43 UTC] ASSISTANT

**Thinking**

Now I have all the information I need to write the quant_summary.json. Let me compile the information:

- model_id: Qwen/Qwen3.5-0.8B
- scheme: MXFP4
- method: RTN
- export_format: auto_round
- device: cuda
- quant_num_gpus: 1
- num_gpus: 1
- output_dir: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4
- runtime_output_dir: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4


- quantized_model_dir: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4
- status: success
- duration_seconds: 17.55
- original_size_mb: 1666.01
- quantized_size_mb: 970.00 (739 + 231)
- compression_ratio: 1.7175
- errors: []
- solutions: []
- output_files: the files in runtime_output_dir

Now I need to get the complete list of output files in the runtime directory.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/logs/"
}
```

### [2026-05-25 01:06:43 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 72
drwxr-xr-x 2 root root    84 May 25 01:05 .
drwxr-xr-x 5 root root   114 May 25 01:05 ..
-rw-r--r-- 1 root root 52150 May 25 01:06 auto.log
-rw-r--r-- 1 root root  8435 May 25 01:05 quant_exec.log
-rw-r--r-- 1 root root  4913 May 25 01:02 quant_prompt.txt

### [2026-05-25 01:06:51 UTC] ASSISTANT

**Thinking**

Now I have all the information. Let me write the quant_summary.json atomically.

Now I'll write the quant_summary.json:


**Tool call:** `exec`

```json
{
  "command": "python3 << 'PYEOF'\nimport json\nimport os\nimport time\nfrom pathlib import Path\n\noutput_dir = \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4\"\nruntime_output_dir = output_dir\nquantized_model_dir = \"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4\"\n\n# Collect output files from runtime_output_dir\noutput_files = []\nfor f in sorted(Path(runtime_output_dir).rglob(\"*\")):\n    if f.is_file():\n        rel = str(f.relative_to(runtime_output_dir))\n        output_files.append(rel)\n\n# Also list files in quantized_model_dir (model output)\nfor f in sorted(Path(quantized_model_dir).rglob(\"*\")):\n    if f.is_file():\n        rel = str(f.relative_to(quantized_model_dir))\n        output_files.append(f\"../Qwen_Qwen3.5-0.8B-MXFP4/{rel}\")\n\nsummary = {\n    \"model_id\": \"Qwen/Qwen3.5-0.8B\",\n    \"scheme\": \"MXFP4\",\n    \"method\": \"RTN\",\n    \"export_format\": \"auto_round\",\n    \"device\": \"cuda\",\n    \"quant_num_gpus\": \"1\",\n    \"num_gpus\": \"1\",\n    \"output_dir\": runtime_output_dir,\n    \"runtime_output_dir\": runtime_output_dir,\n    \"quantized_model_dir\": quantized_model_dir,\n    \"status\": \"success\",\n    \"duration_seconds\": 17.55,\n    \"original_size_mb\": 1666.01,\n    \"quantized_size_mb\": 970.00,\n    \"compression_ratio\": 1.7175,\n    \"errors\": [],\n    \"solutions\": [],\n    \"output_files\": sorted(set(output_files))\n}\n\n# Write atomically via temp file\ntmp = Path(output_dir) / \"quant_summary.json.tmp\"\nwith open(tmp, \"w\") as f:\n    json.dump(summary, f, indent=2)\ntmp.rename(Path(output_dir) / \"quant_summary.json\")\n\nprint(\"Written:\", Path(output_dir) / \"quant_summary.json\")\nprint(json.dumps(summary, indent=2))\nPYEOF"
}
```

### [2026-05-25 01:06:52 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

python3.12/site-packages/transformers/models/glm/modeling_glm.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm/modular_glm.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4/configuration_glm4.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4/modeling_glm4.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4/modular_glm4.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/configuration_glm46v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/image_processing_glm46v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/image_processing_pil_glm46v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/modeling_glm46v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/modular_glm46v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/processing_glm46v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm46v/video_processing_glm46v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe/configuration_glm4_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe/modeling_glm4_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe/modular_glm4_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/configuration_glm4_moe_lite.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4_moe_lite/modular_glm4_moe_lite.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/configuration_glm4v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/image_processing_glm4v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/image_processing_pil_glm4v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/modeling_glm4v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/modular_glm4v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/processing_glm4v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v/video_processing_glm4v.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/configuration_glm4v_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/modeling_glm4v_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm4v_moe/modular_glm4v_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_image/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_image/configuration_glm_image.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_image/image_processing_glm_image.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_image/image_processing_pil_glm_image.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_image/modeling_glm_image.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_image/modular_glm_image.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_image/processing_glm_image.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_ocr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_ocr/configuration_glm_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_ocr/modeling_glm_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/glm_ocr/modular_glm_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/glmasr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glmasr/configuration_glmasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/glmasr/modeling_glmasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/glmasr/modular_glmasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/glmasr/processing_glmasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/glpn/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/glpn/configuration_glpn.py",
    "venv/lib/python3.12/site-packages/transformers/models/glpn/image_processing_glpn.py",
    "venv/lib/python3.12/site-packages/transformers/models/glpn/image_processing_pil_glpn.py",
    "venv/lib/python3.12/site-packages/transformers/models/glpn/modeling_glpn.py",
    "venv/lib/python3.12/site-packages/transformers/models/got_ocr2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/got_ocr2/configuration_got_ocr2.py",
    "venv/lib/python3.12/site-packages/transformers/models/got_ocr2/image_processing_got_ocr2.py",
    "venv/lib/python3.12/site-packages/transformers/models/got_ocr2/image_processing_pil_got_ocr2.py",
    "venv/lib/python3.12/site-packages/transformers/models/got_ocr2/modeling_got_ocr2.py",
    "venv/lib/python3.12/site-packages/transformers/models/got_ocr2/modular_got_ocr2.py",
    "venv/lib/python3.12/site-packages/transformers/models/got_ocr2/processing_got_ocr2.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt2/configuration_gpt2.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt2/tokenization_gpt2.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_bigcode/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neo/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neo/configuration_gpt_neo.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox/configuration_gpt_neox.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox/modular_gpt_neox.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox/tokenization_gpt_neox.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/configuration_gpt_neox_japanese.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_neox_japanese/tokenization_gpt_neox_japanese.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_oss/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_oss/configuration_gpt_oss.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_oss/modular_gpt_oss.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_sw3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gpt_sw3/tokenization_gpt_sw3.py",
    "venv/lib/python3.12/site-packages/transformers/models/gptj/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/gptj/configuration_gptj.py",
    "venv/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite/configuration_granite.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite/modeling_granite.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite/modular_granite.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite4_vision/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite4_vision/configuration_granite4_vision.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite4_vision/modeling_granite4_vision.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite4_vision/modular_granite4_vision.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite4_vision/processing_granite4_vision.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech/configuration_granite_speech.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech/feature_extraction_granite_speech.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech/modeling_granite_speech.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech/processing_granite_speech.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/configuration_granite_speech_plus.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/modeling_granite_speech_plus.py",
    "venv/lib/python3.12/site-packages/transformers/models/granite_speech_plus/modular_granite_speech_plus.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoe/configuration_granitemoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoe/modeling_granitemoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoe/modular_granitemoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/configuration_granitemoehybrid.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoehybrid/modular_granitemoehybrid.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/configuration_granitemoeshared.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/modeling_granitemoeshared.py",
    "venv/lib/python3.12/site-packages/transformers/models/granitemoeshared/modular_granitemoeshared.py",
    "venv/lib/python3.12/site-packages/transformers/models/grounding_dino/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/grounding_dino/configuration_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/grounding_dino/image_processing_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/grounding_dino/image_processing_pil_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/grounding_dino/modeling_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/grounding_dino/modular_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/grounding_dino/processing_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/groupvit/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/groupvit/configuration_groupvit.py",
    "venv/lib/python3.12/site-packages/transformers/models/groupvit/modeling_groupvit.py",
    "venv/lib/python3.12/site-packages/transformers/models/helium/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/helium/configuration_helium.py",
    "venv/lib/python3.12/site-packages/transformers/models/helium/modeling_helium.py",
    "venv/lib/python3.12/site-packages/transformers/models/helium/modular_helium.py",
    "venv/lib/python3.12/site-packages/transformers/models/herbert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/herbert/tokenization_herbert.py",
    "venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/configuration_hgnet_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/modeling_hgnet_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/hgnet_v2/modular_hgnet_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/hiera/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hiera/configuration_hiera.py",
    "venv/lib/python3.12/site-packages/transformers/models/hiera/modeling_hiera.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/configuration_higgs_audio_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/generation_higgs_audio_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/modeling_higgs_audio_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/modular_higgs_audio_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2/processing_higgs_audio_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/configuration_higgs_audio_v2_tokenizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/higgs_audio_v2_tokenizer/modular_higgs_audio_v2_tokenizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/hrm_text/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hrm_text/configuration_hrm_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/hrm_text/modeling_hrm_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/hrm_text/modular_hrm_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/hubert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hubert/configuration_hubert.py",
    "venv/lib/python3.12/site-packages/transformers/models/hubert/modeling_hubert.py",
    "venv/lib/python3.12/site-packages/transformers/models/hubert/modular_hubert.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/configuration_hunyuan_v1_dense.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_dense/modular_hunyuan_v1_dense.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/configuration_hunyuan_v1_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/hunyuan_v1_moe/modular_hunyuan_v1_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/hy_v3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hy_v3/configuration_hy_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/hy_v3/modeling_hy_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/hy_v3/modular_hy_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/hyperclovax/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/hyperclovax/configuration_hyperclovax.py",
    "venv/lib/python3.12/site-packages/transformers/models/hyperclovax/modeling_hyperclovax.py",
    "venv/lib/python3.12/site-packages/transformers/models/hyperclovax/modular_hyperclovax.py",
    "venv/lib/python3.12/site-packages/transformers/models/ibert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/ibert/configuration_ibert.py",
    "venv/lib/python3.12/site-packages/transformers/models/ibert/modeling_ibert.py",
    "venv/lib/python3.12/site-packages/transformers/models/ibert/quant_modules.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/configuration_idefics.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/image_processing_idefics.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/image_processing_pil_idefics.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/modeling_idefics.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/perceiver.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/processing_idefics.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics/vision.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics2/configuration_idefics2.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics2/image_processing_idefics2.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics2/image_processing_pil_idefics2.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics2/modeling_idefics2.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics2/processing_idefics2.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics3/configuration_idefics3.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics3/image_processing_idefics3.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics3/image_processing_pil_idefics3.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py",
    "venv/lib/python3.12/site-packages/transformers/models/idefics3/processing_idefics3.py",
    "venv/lib/python3.12/site-packages/transformers/models/ijepa/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/ijepa/configuration_ijepa.py",
    "venv/lib/python3.12/site-packages/transformers/models/ijepa/modeling_ijepa.py",
    "venv/lib/python3.12/site-packages/transformers/models/ijepa/modular_ijepa.py",
    "venv/lib/python3.12/site-packages/transformers/models/imagegpt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/imagegpt/configuration_imagegpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/imagegpt/image_processing_imagegpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/imagegpt/image_processing_pil_imagegpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/imagegpt/modeling_imagegpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/informer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/informer/configuration_informer.py",
    "venv/lib/python3.12/site-packages/transformers/models/informer/modeling_informer.py",
    "venv/lib/python3.12/site-packages/transformers/models/informer/modular_informer.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblip/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblip/configuration_instructblip.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblip/modeling_instructblip.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblip/processing_instructblip.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/configuration_instructblipvideo.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/modeling_instructblipvideo.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/modular_instructblipvideo.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/processing_instructblipvideo.py",
    "venv/lib/python3.12/site-packages/transformers/models/instructblipvideo/video_processing_instructblipvideo.py",
    "venv/lib/python3.12/site-packages/transformers/models/internvl/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/internvl/configuration_internvl.py",
    "venv/lib/python3.12/site-packages/transformers/models/internvl/modeling_internvl.py",
    "venv/lib/python3.12/site-packages/transformers/models/internvl/modular_internvl.py",
    "venv/lib/python3.12/site-packages/transformers/models/internvl/processing_internvl.py",
    "venv/lib/python3.12/site-packages/transformers/models/internvl/video_processing_internvl.py",
    "venv/lib/python3.12/site-packages/transformers/models/jais2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/jais2/configuration_jais2.py",
    "venv/lib/python3.12/site-packages/transformers/models/jais2/modeling_jais2.py",
    "venv/lib/python3.12/site-packages/transformers/models/jais2/modular_jais2.py",
    "venv/lib/python3.12/site-packages/transformers/models/jamba/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/jamba/configuration_jamba.py",
    "venv/lib/python3.12/site-packages/transformers/models/jamba/modeling_jamba.py",
    "venv/lib/python3.12/site-packages/transformers/models/jamba/modular_jamba.py",
    "venv/lib/python3.12/site-packages/transformers/models/janus/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/janus/configuration_janus.py",
    "venv/lib/python3.12/site-packages/transformers/models/janus/image_processing_janus.py",
    "venv/lib/python3.12/site-packages/transformers/models/janus/image_processing_pil_janus.py",
    "venv/lib/python3.12/site-packages/transformers/models/janus/modeling_janus.py",
    "venv/lib/python3.12/site-packages/transformers/models/janus/modular_janus.py",
    "venv/lib/python3.12/site-packages/transformers/models/janus/processing_janus.py",
    "venv/lib/python3.12/site-packages/transformers/models/jetmoe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/jetmoe/configuration_jetmoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/jetmoe/modeling_jetmoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/jetmoe/modular_jetmoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/configuration_jina_embeddings_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/modeling_jina_embeddings_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/jina_embeddings_v3/modular_jina_embeddings_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2/configuration_kosmos2.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2/modeling_kosmos2.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2/processing_kosmos2.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/configuration_kosmos2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/image_processing_kosmos2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/image_processing_pil_kosmos2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/modeling_kosmos2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/kosmos2_5/processing_kosmos2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/configuration_kyutai_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/feature_extraction_kyutai_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/modular_kyutai_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/kyutai_speech_to_text/processing_kyutai_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/laguna/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/laguna/configuration_laguna.py",
    "venv/lib/python3.12/site-packages/transformers/models/laguna/modeling_laguna.py",
    "venv/lib/python3.12/site-packages/transformers/models/laguna/modular_laguna.py",
    "venv/lib/python3.12/site-packages/transformers/models/lasr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lasr/configuration_lasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lasr/feature_extraction_lasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lasr/modeling_lasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lasr/modular_lasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lasr/processing_lasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lasr/tokenization_lasr.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlm/configuration_layoutlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/configuration_layoutlmv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/image_processing_layoutlmv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/image_processing_pil_layoutlmv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/modeling_layoutlmv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/processing_layoutlmv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv2/tokenization_layoutlmv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/configuration_layoutlmv3.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/image_processing_layoutlmv3.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/image_processing_pil_layoutlmv3.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/modeling_layoutlmv3.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/processing_layoutlmv3.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutlmv3/tokenization_layoutlmv3.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutxlm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutxlm/configuration_layoutxlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutxlm/modular_layoutxlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutxlm/processing_layoutxlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/layoutxlm/tokenization_layoutxlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/led/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/led/configuration_led.py",
    "venv/lib/python3.12/site-packages/transformers/models/led/modeling_led.py",
    "venv/lib/python3.12/site-packages/transformers/models/levit/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/levit/configuration_levit.py",
    "venv/lib/python3.12/site-packages/transformers/models/levit/image_processing_levit.py",
    "venv/lib/python3.12/site-packages/transformers/models/levit/image_processing_pil_levit.py",
    "venv/lib/python3.12/site-packages/transformers/models/levit/modeling_levit.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2/configuration_lfm2.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2/modeling_lfm2.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2/modular_lfm2.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/configuration_lfm2_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/modeling_lfm2_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_moe/modular_lfm2_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/configuration_lfm2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/image_processing_lfm2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/modeling_lfm2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/modular_lfm2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/lfm2_vl/processing_lfm2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/lightglue/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lightglue/configuration_lightglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/lightglue/image_processing_lightglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/lightglue/image_processing_pil_lightglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/lightglue/modeling_lightglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/lightglue/modular_lightglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/configuration_lighton_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/modeling_lighton_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/modular_lighton_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lighton_ocr/processing_lighton_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lilt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lilt/configuration_lilt.py",
    "venv/lib/python3.12/site-packages/transformers/models/lilt/modeling_lilt.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama/tokenization_llama.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama4/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama4/configuration_llama4.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama4/image_processing_llama4.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama4/modeling_llama4.py",
    "venv/lib/python3.12/site-packages/transformers/models/llama4/processing_llama4.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava/configuration_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava/image_processing_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava/image_processing_pil_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava/modeling_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava/processing_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next/configuration_llava_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_llava_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next/image_processing_pil_llava_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next/modeling_llava_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next/processing_llava_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next_video/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next_video/configuration_llava_next_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next_video/modeling_llava_next_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next_video/modular_llava_next_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next_video/processing_llava_next_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_next_video/video_processing_llava_next_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/configuration_llava_onevision.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/image_processing_llava_onevision.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/image_processing_pil_llava_onevision.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/modeling_llava_onevision.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/modular_llava_onevision.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/processing_llava_onevision.py",
    "venv/lib/python3.12/site-packages/transformers/models/llava_onevision/video_processing_llava_onevision.py",
    "venv/lib/python3.12/site-packages/transformers/models/longcat_flash/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/longcat_flash/configuration_longcat_flash.py",
    "venv/lib/python3.12/site-packages/transformers/models/longcat_flash/modeling_longcat_flash.py",
    "venv/lib/python3.12/site-packages/transformers/models/longcat_flash/modular_longcat_flash.py",
    "venv/lib/python3.12/site-packages/transformers/models/longformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/longformer/configuration_longformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/longt5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/longt5/configuration_longt5.py",
    "venv/lib/python3.12/site-packages/transformers/models/longt5/modeling_longt5.py",
    "venv/lib/python3.12/site-packages/transformers/models/luke/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/luke/configuration_luke.py",
    "venv/lib/python3.12/site-packages/transformers/models/luke/modeling_luke.py",
    "venv/lib/python3.12/site-packages/transformers/models/luke/tokenization_luke.py",
    "venv/lib/python3.12/site-packages/transformers/models/lw_detr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lw_detr/configuration_lw_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lw_detr/modeling_lw_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lw_detr/modular_lw_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/lxmert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/lxmert/configuration_lxmert.py",
    "venv/lib/python3.12/site-packages/transformers/models/lxmert/modeling_lxmert.py",
    "venv/lib/python3.12/site-packages/transformers/models/m2m_100/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/m2m_100/configuration_m2m_100.py",
    "venv/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py",
    "venv/lib/python3.12/site-packages/transformers/models/m2m_100/tokenization_m2m_100.py",
    "venv/lib/python3.12/site-packages/transformers/models/mamba/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mamba/configuration_mamba.py",
    "venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py",
    "venv/lib/python3.12/site-packages/transformers/models/mamba2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mamba2/configuration_mamba2.py",
    "venv/lib/python3.12/site-packages/transformers/models/mamba2/modeling_mamba2.py",
    "venv/lib/python3.12/site-packages/transformers/models/marian/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/marian/configuration_marian.py",
    "venv/lib/python3.12/site-packages/transformers/models/marian/modeling_marian.py",
    "venv/lib/python3.12/site-packages/transformers/models/marian/tokenization_marian.py",
    "venv/lib/python3.12/site-packages/transformers/models/markuplm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/markuplm/configuration_markuplm.py",
    "venv/lib/python3.12/site-packages/transformers/models/markuplm/feature_extraction_markuplm.py",
    "venv/lib/python3.12/site-packages/transformers/models/markuplm/modeling_markuplm.py",
    "venv/lib/python3.12/site-packages/transformers/models/markuplm/processing_markuplm.py",
    "venv/lib/python3.12/site-packages/transformers/models/markuplm/tokenization_markuplm.py",
    "venv/lib/python3.12/site-packages/transformers/models/mask2former/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mask2former/configuration_mask2former.py",
    "venv/lib/python3.12/site-packages/transformers/models/mask2former/image_processing_mask2former.py",
    "venv/lib/python3.12/site-packages/transformers/models/mask2former/image_processing_pil_mask2former.py",
    "venv/lib/python3.12/site-packages/transformers/models/mask2former/modeling_mask2former.py",
    "venv/lib/python3.12/site-packages/transformers/models/mask2former/modular_mask2former.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/configuration_maskformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/configuration_maskformer_swin.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/image_processing_maskformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/image_processing_pil_maskformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/modeling_maskformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/modeling_maskformer_swin.py",
    "venv/lib/python3.12/site-packages/transformers/models/maskformer/modular_maskformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/mbart/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mbart/configuration_mbart.py",
    "venv/lib/python3.12/site-packages/transformers/models/mbart/modeling_mbart.py",
    "venv/lib/python3.12/site-packages/transformers/models/mbart/tokenization_mbart.py",
    "venv/lib/python3.12/site-packages/transformers/models/mbart50/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mbart50/tokenization_mbart50.py",
    "venv/lib/python3.12/site-packages/transformers/models/megatron_bert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/megatron_bert/configuration_megatron_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/megatron_gpt2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py",
    "venv/lib/python3.12/site-packages/transformers/models/metaclip_2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/metaclip_2/configuration_metaclip_2.py",
    "venv/lib/python3.12/site-packages/transformers/models/metaclip_2/modeling_metaclip_2.py",
    "venv/lib/python3.12/site-packages/transformers/models/metaclip_2/modular_metaclip_2.py",
    "venv/lib/python3.12/site-packages/transformers/models/mgp_str/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mgp_str/configuration_mgp_str.py",
    "venv/lib/python3.12/site-packages/transformers/models/mgp_str/modeling_mgp_str.py",
    "venv/lib/python3.12/site-packages/transformers/models/mgp_str/processing_mgp_str.py",
    "venv/lib/python3.12/site-packages/transformers/models/mgp_str/tokenization_mgp_str.py",
    "venv/lib/python3.12/site-packages/transformers/models/mimi/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mimi/configuration_mimi.py",
    "venv/lib/python3.12/site-packages/transformers/models/mimi/modeling_mimi.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/configuration_minicpmv4_6.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/image_processing_minicpmv4_6.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/image_processing_pil_minicpmv4_6.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/modeling_minicpmv4_6.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/modular_minicpmv4_6.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/processing_minicpmv4_6.py",
    "venv/lib/python3.12/site-packages/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax/configuration_minimax.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax/modeling_minimax.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax/modular_minimax.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax_m2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax_m2/configuration_minimax_m2.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax_m2/modeling_minimax_m2.py",
    "venv/lib/python3.12/site-packages/transformers/models/minimax_m2/modular_minimax_m2.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral/configuration_ministral.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral/modeling_ministral.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral/modular_ministral.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral3/configuration_ministral3.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral3/modeling_ministral3.py",
    "venv/lib/python3.12/site-packages/transformers/models/ministral3/modular_ministral3.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral/configuration_mistral.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral/modular_mistral.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral3/configuration_mistral3.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral3/modeling_mistral3.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral3/modular_mistral3.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral4/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral4/configuration_mistral4.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral4/modeling_mistral4.py",
    "venv/lib/python3.12/site-packages/transformers/models/mistral4/modular_mistral4.py",
    "venv/lib/python3.12/site-packages/transformers/models/mixtral/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mixtral/configuration_mixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/mixtral/modeling_mixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/mixtral/modular_mixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/mlcd/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mlcd/configuration_mlcd.py",
    "venv/lib/python3.12/site-packages/transformers/models/mlcd/modeling_mlcd.py",
    "venv/lib/python3.12/site-packages/transformers/models/mlcd/modular_mlcd.py",
    "venv/lib/python3.12/site-packages/transformers/models/mllama/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mllama/configuration_mllama.py",
    "venv/lib/python3.12/site-packages/transformers/models/mllama/image_processing_mllama.py",
    "venv/lib/python3.12/site-packages/transformers/models/mllama/image_processing_pil_mllama.py",
    "venv/lib/python3.12/site-packages/transformers/models/mllama/modeling_mllama.py",
    "venv/lib/python3.12/site-packages/transformers/models/mllama/processing_mllama.py",
    "venv/lib/python3.12/site-packages/transformers/models/mluke/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mluke/tokenization_mluke.py",
    "venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/configuration_mm_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/mm_grounding_dino/modular_mm_grounding_dino.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilebert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilebert/configuration_mobilebert.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilebert/tokenization_mobilebert.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/configuration_mobilenet_v1.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/image_processing_mobilenet_v1.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/image_processing_pil_mobilenet_v1.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/configuration_mobilenet_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/image_processing_mobilenet_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/image_processing_pil_mobilenet_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilenet_v2/modeling_mobilenet_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevit/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevit/configuration_mobilevit.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevit/image_processing_mobilevit.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevit/image_processing_pil_mobilevit.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevit/modeling_mobilevit.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevitv2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevitv2/configuration_mobilevitv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/mobilevitv2/modeling_mobilevitv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert/configuration_modernbert.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert/modeling_modernbert.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert/modular_modernbert.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/configuration_modernbert_decoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/modeling_modernbert_decoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernbert_decoder/modular_modernbert_decoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernvbert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernvbert/configuration_modernvbert.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernvbert/modeling_modernvbert.py",
    "venv/lib/python3.12/site-packages/transformers/models/modernvbert/modular_modernvbert.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine/configuration_moonshine.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine/modeling_moonshine.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine/modular_moonshine.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/configuration_moonshine_streaming.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/modeling_moonshine_streaming.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/modular_moonshine_streaming.py",
    "venv/lib/python3.12/site-packages/transformers/models/moonshine_streaming/processing_moonshine_streaming.py",
    "venv/lib/python3.12/site-packages/transformers/models/moshi/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/moshi/configuration_moshi.py",
    "venv/lib/python3.12/site-packages/transformers/models/moshi/modeling_moshi.py",
    "venv/lib/python3.12/site-packages/transformers/models/mpnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mpnet/configuration_mpnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/mpnet/modeling_mpnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/mpnet/tokenization_mpnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/mpt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mpt/configuration_mpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/mpt/modeling_mpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/mra/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mra/configuration_mra.py",
    "venv/lib/python3.12/site-packages/transformers/models/mra/modeling_mra.py",
    "venv/lib/python3.12/site-packages/transformers/models/mt5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mt5/configuration_mt5.py",
    "venv/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicflamingo/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicflamingo/configuration_musicflamingo.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicflamingo/modeling_musicflamingo.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicflamingo/modular_musicflamingo.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicflamingo/processing_musicflamingo.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen/configuration_musicgen.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen/modeling_musicgen.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen/processing_musicgen.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/configuration_musicgen_melody.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/feature_extraction_musicgen_melody.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/modeling_musicgen_melody.py",
    "venv/lib/python3.12/site-packages/transformers/models/musicgen_melody/processing_musicgen_melody.py",
    "venv/lib/python3.12/site-packages/transformers/models/mvp/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/mvp/configuration_mvp.py",
    "venv/lib/python3.12/site-packages/transformers/models/mvp/modeling_mvp.py",
    "venv/lib/python3.12/site-packages/transformers/models/myt5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/myt5/tokenization_myt5.py",
    "venv/lib/python3.12/site-packages/transformers/models/nanochat/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nanochat/configuration_nanochat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nanochat/modeling_nanochat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nanochat/modular_nanochat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nemotron/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nemotron/configuration_nemotron.py",
    "venv/lib/python3.12/site-packages/transformers/models/nemotron/modeling_nemotron.py",
    "venv/lib/python3.12/site-packages/transformers/models/nemotron_h/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nemotron_h/configuration_nemotron_h.py",
    "venv/lib/python3.12/site-packages/transformers/models/nemotron_h/modeling_nemotron_h.py",
    "venv/lib/python3.12/site-packages/transformers/models/nemotron_h/modular_nemotron_h.py",
    "venv/lib/python3.12/site-packages/transformers/models/nllb/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nllb/tokenization_nllb.py",
    "venv/lib/python3.12/site-packages/transformers/models/nllb_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nllb_moe/configuration_nllb_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/nllb_moe/modeling_nllb_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/nomic_bert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nomic_bert/configuration_nomic_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/nomic_bert/modeling_nomic_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/nomic_bert/modular_nomic_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/nougat/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nougat/configuration_nougat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nougat/image_processing_nougat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nougat/image_processing_pil_nougat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nougat/processing_nougat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nougat/tokenization_nougat.py",
    "venv/lib/python3.12/site-packages/transformers/models/nystromformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/nystromformer/configuration_nystromformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/nystromformer/modeling_nystromformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo/configuration_olmo.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo/modeling_olmo.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo/modular_olmo.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo2/configuration_olmo2.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo2/modeling_olmo2.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo2/modular_olmo2.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo3/configuration_olmo3.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo3/modeling_olmo3.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo3/modular_olmo3.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/configuration_olmo_hybrid.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/modeling_olmo_hybrid.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmo_hybrid/modular_olmo_hybrid.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmoe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmoe/configuration_olmoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmoe/modeling_olmoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/olmoe/modular_olmoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/configuration_omdet_turbo.py",
    "venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/modeling_omdet_turbo.py",
    "venv/lib/python3.12/site-packages/transformers/models/omdet_turbo/processing_omdet_turbo.py",
    "venv/lib/python3.12/site-packages/transformers/models/oneformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/oneformer/configuration_oneformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/oneformer/image_processing_oneformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/oneformer/image_processing_pil_oneformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/oneformer/modeling_oneformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/oneformer/processing_oneformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai/configuration_openai.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai/modeling_openai.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai/tokenization_openai.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/configuration_openai_privacy_filter.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/modeling_openai_privacy_filter.py",
    "venv/lib/python3.12/site-packages/transformers/models/openai_privacy_filter/modular_openai_privacy_filter.py",
    "venv/lib/python3.12/site-packages/transformers/models/opt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/opt/configuration_opt.py",
    "venv/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py",
    "venv/lib/python3.12/site-packages/transformers/models/ovis2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/ovis2/configuration_ovis2.py",
    "venv/lib/python3.12/site-packages/transformers/models/ovis2/image_processing_ovis2.py",
    "venv/lib/python3.12/site-packages/transformers/models/ovis2/image_processing_pil_ovis2.py",
    "venv/lib/python3.12/site-packages/transformers/models/ovis2/modeling_ovis2.py",
    "venv/lib/python3.12/site-packages/transformers/models/ovis2/modular_ovis2.py",
    "venv/lib/python3.12/site-packages/transformers/models/ovis2/processing_ovis2.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlv2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlv2/configuration_owlv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlv2/image_processing_owlv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlv2/image_processing_pil_owlv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlv2/modeling_owlv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlv2/modular_owlv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlv2/processing_owlv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlvit/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlvit/configuration_owlvit.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlvit/image_processing_owlvit.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlvit/image_processing_pil_owlvit.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlvit/modeling_owlvit.py",
    "venv/lib/python3.12/site-packages/transformers/models/owlvit/processing_owlvit.py",
    "venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/configuration_paddleocr_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/image_processing_paddleocr_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/image_processing_pil_paddleocr_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/modeling_paddleocr_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/modular_paddleocr_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/paddleocr_vl/processing_paddleocr_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/paligemma/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/paligemma/configuration_paligemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/paligemma/modeling_paligemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/paligemma/processing_paligemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/configuration_parakeet.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/feature_extraction_parakeet.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/generation_parakeet.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/modeling_parakeet.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/modular_parakeet.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/processing_parakeet.py",
    "venv/lib/python3.12/site-packages/transformers/models/parakeet/tokenization_parakeet.py",
    "venv/lib/python3.12/site-packages/transformers/models/patchtsmixer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/patchtsmixer/configuration_patchtsmixer.py",
    "venv/lib/python3.12/site-packages/transformers/models/patchtsmixer/modeling_patchtsmixer.py",
    "venv/lib/python3.12/site-packages/transformers/models/patchtst/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/patchtst/configuration_patchtst.py",
    "venv/lib/python3.12/site-packages/transformers/models/patchtst/modeling_patchtst.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio/configuration_pe_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio/feature_extraction_pe_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio/modeling_pe_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio/modular_pe_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio/processing_pe_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/configuration_pe_audio_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/modeling_pe_audio_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/modular_pe_audio_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_audio_video/processing_pe_audio_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_video/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_video/configuration_pe_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_video/modeling_pe_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_video/modular_pe_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_video/processing_pe_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pe_video/video_processing_pe_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/pegasus/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pegasus/configuration_pegasus.py",
    "venv/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py",
    "venv/lib/python3.12/site-packages/transformers/models/pegasus/tokenization_pegasus.py",
    "venv/lib/python3.12/site-packages/transformers/models/pegasus_x/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pegasus_x/configuration_pegasus_x.py",
    "venv/lib/python3.12/site-packages/transformers/models/pegasus_x/modeling_pegasus_x.py",
    "venv/lib/python3.12/site-packages/transformers/models/perceiver/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/perceiver/configuration_perceiver.py",
    "venv/lib/python3.12/site-packages/transformers/models/perceiver/image_processing_perceiver.py",
    "venv/lib/python3.12/site-packages/transformers/models/perceiver/image_processing_pil_perceiver.py",
    "venv/lib/python3.12/site-packages/transformers/models/perceiver/modeling_perceiver.py",
    "venv/lib/python3.12/site-packages/transformers/models/perceiver/tokenization_perceiver.py",
    "venv/lib/python3.12/site-packages/transformers/models/perception_lm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/perception_lm/configuration_perception_lm.py",
    "venv/lib/python3.12/site-packages/transformers/models/perception_lm/image_processing_perception_lm.py",
    "venv/lib/python3.12/site-packages/transformers/models/perception_lm/modeling_perception_lm.py",
    "venv/lib/python3.12/site-packages/transformers/models/perception_lm/modular_perception_lm.py",
    "venv/lib/python3.12/site-packages/transformers/models/perception_lm/processing_perception_lm.py",
    "venv/lib/python3.12/site-packages/transformers/models/perception_lm/video_processing_perception_lm.py",
    "venv/lib/python3.12/site-packages/transformers/models/persimmon/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/persimmon/configuration_persimmon.py",
    "venv/lib/python3.12/site-packages/transformers/models/persimmon/modeling_persimmon.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi/configuration_phi.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi/modular_phi.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi3/configuration_phi3.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi3/modeling_phi3.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi3/modular_phi3.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/feature_extraction_phi4_multimodal.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/image_processing_phi4_multimodal.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/modular_phi4_multimodal.py",
    "venv/lib/python3.12/site-packages/transformers/models/phi4_multimodal/processing_phi4_multimodal.py",
    "venv/lib/python3.12/site-packages/transformers/models/phimoe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/phimoe/configuration_phimoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/phimoe/modeling_phimoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/phimoe/modular_phimoe.py",
    "venv/lib/python3.12/site-packages/transformers/models/phobert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/phobert/tokenization_phobert.py",
    "venv/lib/python3.12/site-packages/transformers/models/pi0/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pi0/configuration_pi0.py",
    "venv/lib/python3.12/site-packages/transformers/models/pi0/image_processing_pi0.py",
    "venv/lib/python3.12/site-packages/transformers/models/pi0/modeling_pi0.py",
    "venv/lib/python3.12/site-packages/transformers/models/pi0/modular_pi0.py",
    "venv/lib/python3.12/site-packages/transformers/models/pi0/processing_pi0.py",
    "venv/lib/python3.12/site-packages/transformers/models/pix2struct/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pix2struct/configuration_pix2struct.py",
    "venv/lib/python3.12/site-packages/transformers/models/pix2struct/image_processing_pil_pix2struct.py",
    "venv/lib/python3.12/site-packages/transformers/models/pix2struct/image_processing_pix2struct.py",
    "venv/lib/python3.12/site-packages/transformers/models/pix2struct/modeling_pix2struct.py",
    "venv/lib/python3.12/site-packages/transformers/models/pix2struct/processing_pix2struct.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixio/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixio/configuration_pixio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixio/modeling_pixio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixio/modular_pixio.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixtral/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixtral/configuration_pixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixtral/image_processing_pil_pixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixtral/image_processing_pixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixtral/modeling_pixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/pixtral/processing_pixtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/plbart/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/plbart/configuration_plbart.py",
    "venv/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py",
    "venv/lib/python3.12/site-packages/transformers/models/plbart/modular_plbart.py",
    "venv/lib/python3.12/site-packages/transformers/models/plbart/tokenization_plbart.py",
    "venv/lib/python3.12/site-packages/transformers/models/poolformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/poolformer/configuration_poolformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/poolformer/image_processing_pil_poolformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/poolformer/image_processing_poolformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/poolformer/modeling_poolformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/pop2piano/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pop2piano/configuration_pop2piano.py",
    "venv/lib/python3.12/site-packages/transformers/models/pop2piano/feature_extraction_pop2piano.py",
    "venv/lib/python3.12/site-packages/transformers/models/pop2piano/modeling_pop2piano.py",
    "venv/lib/python3.12/site-packages/transformers/models/pop2piano/processing_pop2piano.py",
    "venv/lib/python3.12/site-packages/transformers/models/pop2piano/tokenization_pop2piano.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/configuration_pp_chart2table.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/image_processing_pil_pp_chart2table.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/image_processing_pp_chart2table.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/modular_pp_chart2table.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_chart2table/processing_pp_chart2table.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/configuration_pp_doclayout_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/image_processing_pp_doclayout_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/modeling_pp_doclayout_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v2/modular_pp_doclayout_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/configuration_pp_doclayout_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/image_processing_pp_doclayout_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_doclayout_v3/modular_pp_doclayout_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/configuration_pp_formulanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/image_processing_pp_formulanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/modeling_pp_formulanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/modular_pp_formulanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_formulanet/processing_pp_formulanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/configuration_pp_lcnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/image_processing_pp_lcnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/modeling_pp_lcnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet/modular_pp_lcnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_lcnet_v3/modular_pp_lcnet_v3.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/configuration_pp_ocrv5_mobile_det.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/modeling_pp_ocrv5_mobile_det.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_det/modular_pp_ocrv5_mobile_det.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/configuration_pp_ocrv5_mobile_rec.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/modeling_pp_ocrv5_mobile_rec.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_mobile_rec/modular_pp_ocrv5_mobile_rec.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/configuration_pp_ocrv5_server_det.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/image_processing_pp_ocrv5_server_det.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/modeling_pp_ocrv5_server_det.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_det/modular_pp_ocrv5_server_det.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/configuration_pp_ocrv5_server_rec.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/image_processing_pp_ocrv5_server_rec.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/modeling_pp_ocrv5_server_rec.py",
    "venv/lib/python3.12/site-packages/transformers/models/pp_ocrv5_server_rec/modular_pp_ocrv5_server_rec.py",
    "venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/configuration_prompt_depth_anything.py",
    "venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/image_processing_pil_prompt_depth_anything.py",
    "venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/image_processing_prompt_depth_anything.py",
    "venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/modeling_prompt_depth_anything.py",
    "venv/lib/python3.12/site-packages/transformers/models/prompt_depth_anything/modular_prompt_depth_anything.py",
    "venv/lib/python3.12/site-packages/transformers/models/prophetnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/prophetnet/configuration_prophetnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/prophetnet/modeling_prophetnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/prophetnet/tokenization_prophetnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt/configuration_pvt.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt/image_processing_pil_pvt.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt/image_processing_pvt.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt/modeling_pvt.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt_v2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt_v2/configuration_pvt_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/pvt_v2/modeling_pvt_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/configuration_qianfan_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/modeling_qianfan_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/modular_qianfan_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/qianfan_ocr/processing_qianfan_ocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2/__pycache__/tokenization_qwen2.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2/configuration_qwen2.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2/modular_qwen2.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2/tokenization_qwen2.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/configuration_qwen2_5_omni.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/configuration_qwen2_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/modeling_qwen2_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_audio/processing_qwen2_audio.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/configuration_qwen2_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/modeling_qwen2_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_moe/modular_qwen2_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/configuration_qwen2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_pil_qwen2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/processing_qwen2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen2_vl/video_processing_qwen2_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3/configuration_qwen3.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3/modular_qwen3.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__pycache__/configuration_qwen3_5.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/__pycache__/modeling_qwen3_5.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/configuration_qwen3_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modular_qwen3_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5/tokenization_qwen3_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/configuration_qwen3_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_moe/modular_qwen3_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_next/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_next/configuration_qwen3_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modeling_qwen3_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_next/modular_qwen3_next.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/configuration_qwen3_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modular_qwen3_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py",
    "venv/lib/python3.12/site-packages/transformers/models/rag/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/rag/configuration_rag.py",
    "venv/lib/python3.12/site-packages/transformers/models/rag/modeling_rag.py",
    "venv/lib/python3.12/site-packages/transformers/models/rag/retrieval_rag.py",
    "venv/lib/python3.12/site-packages/transformers/models/rag/tokenization_rag.py",
    "venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/configuration_recurrent_gemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/reformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/reformer/configuration_reformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/reformer/tokenization_reformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/regnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/regnet/configuration_regnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/regnet/modeling_regnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/rembert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/rembert/configuration_rembert.py",
    "venv/lib/python3.12/site-packages/transformers/models/rembert/modeling_rembert.py",
    "venv/lib/python3.12/site-packages/transformers/models/rembert/tokenization_rembert.py",
    "venv/lib/python3.12/site-packages/transformers/models/resnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/resnet/configuration_resnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/resnet/modeling_resnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/rf_detr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/rf_detr/configuration_rf_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rf_detr/image_processing_rf_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rf_detr/modeling_rf_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rf_detr/modular_rf_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta/configuration_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta/modular_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta/tokenization_roberta_old.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/configuration_roberta_prelayernorm.py",
    "venv/lib/python3.12/site-packages/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py",
    "venv/lib/python3.12/site-packages/transformers/models/roc_bert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/roc_bert/configuration_roc_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/roc_bert/modeling_roc_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/roc_bert/tokenization_roc_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/roformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/roformer/configuration_roformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/roformer/modeling_roformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_roformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/roformer/tokenization_utils.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/configuration_rt_detr_resnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_pil_rt_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/image_processing_rt_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/modeling_rt_detr_resnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr/modular_rt_detr.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/rt_detr_v2/modular_rt_detr_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/rwkv/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/rwkv/configuration_rwkv.py",
    "venv/lib/python3.12/site-packages/transformers/models/rwkv/modeling_rwkv.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam/configuration_sam.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_pil_sam.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam/image_processing_sam.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam/modeling_sam.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam/processing_sam.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2/configuration_sam2.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2/image_processing_sam2.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2/modeling_sam2.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2/modular_sam2.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2/processing_sam2.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2_video/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2_video/configuration_sam2_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2_video/modeling_sam2_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2_video/modular_sam2_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2_video/processing_sam2_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam2_video/video_processing_sam2_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3/configuration_sam3.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3/image_processing_sam3.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3/modeling_sam3.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3/modular_sam3.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3/processing_sam3.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/configuration_sam3_lite_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modeling_sam3_lite_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_lite_text/modular_sam3_lite_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/configuration_sam3_tracker.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modeling_sam3_tracker.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/modular_sam3_tracker.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker/processing_sam3_tracker.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/configuration_sam3_tracker_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modeling_sam3_tracker_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/modular_sam3_tracker_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_tracker_video/processing_sam3_tracker_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_video/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_video/configuration_sam3_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_video/modeling_sam3_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam3_video/processing_sam3_video.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam_hq/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam_hq/configuration_sam_hq.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam_hq/modeling_sam_hq.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam_hq/modular_sam_hq.py",
    "venv/lib/python3.12/site-packages/transformers/models/sam_hq/processing_sam_hq.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/configuration_seamless_m4t.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/modeling_seamless_m4t.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/processing_seamless_m4t.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t/tokenization_seamless_m4t.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py",
    "venv/lib/python3.12/site-packages/transformers/models/seed_oss/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/seed_oss/configuration_seed_oss.py",
    "venv/lib/python3.12/site-packages/transformers/models/seed_oss/modeling_seed_oss.py",
    "venv/lib/python3.12/site-packages/transformers/models/seed_oss/modular_seed_oss.py",
    "venv/lib/python3.12/site-packages/transformers/models/segformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/segformer/configuration_segformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_pil_segformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/segformer/image_processing_segformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/segformer/modeling_segformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/segformer/modular_segformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/seggpt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/seggpt/configuration_seggpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_pil_seggpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/seggpt/image_processing_seggpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/seggpt/modeling_seggpt.py",
    "venv/lib/python3.12/site-packages/transformers/models/sew/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sew/configuration_sew.py",
    "venv/lib/python3.12/site-packages/transformers/models/sew/modeling_sew.py",
    "venv/lib/python3.12/site-packages/transformers/models/sew/modular_sew.py",
    "venv/lib/python3.12/site-packages/transformers/models/sew_d/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/sew_d/configuration_sew_d.py",
    "venv/lib/python3.12/site-packages/transformers/models/sew_d/modeling_sew_d.py",
    "venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/configuration_shieldgemma2.py",
    "venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/modeling_shieldgemma2.py",
    "venv/lib/python3.12/site-packages/transformers/models/shieldgemma2/processing_shieldgemma2.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip/configuration_siglip.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_pil_siglip.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip/image_processing_siglip.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip/processing_siglip.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip/tokenization_siglip.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/configuration_siglip2.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_pil_siglip2.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/image_processing_siglip2.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/modeling_siglip2.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/modular_siglip2.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/processing_siglip2.py",
    "venv/lib/python3.12/site-packages/transformers/models/siglip2/tokenization_siglip2.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanet/configuration_slanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanet/modeling_slanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanet/modular_slanet.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanext/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanext/configuration_slanext.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanext/image_processing_slanext.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanext/modeling_slanext.py",
    "venv/lib/python3.12/site-packages/transformers/models/slanext/modular_slanext.py",
    "venv/lib/python3.12/site-packages/transformers/models/smollm3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/smollm3/configuration_smollm3.py",
    "venv/lib/python3.12/site-packages/transformers/models/smollm3/modeling_smollm3.py",
    "venv/lib/python3.12/site-packages/transformers/models/smollm3/modular_smollm3.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/configuration_smolvlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_pil_smolvlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/image_processing_smolvlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/modeling_smolvlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/modular_smolvlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/processing_smolvlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/smolvlm/video_processing_smolvlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/solar_open/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/solar_open/configuration_solar_open.py",
    "venv/lib/python3.12/site-packages/transformers/models/solar_open/modeling_solar_open.py",
    "venv/lib/python3.12/site-packages/transformers/models/solar_open/modular_solar_open.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_to_text/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_to_text/configuration_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_to_text/feature_extraction_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_to_text/modeling_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_to_text/processing_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/speech_to_text/tokenization_speech_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/models/speecht5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/speecht5/configuration_speecht5.py",
    "venv/lib/python3.12/site-packages/transformers/models/speecht5/feature_extraction_speecht5.py",
    "venv/lib/python3.12/site-packages/transformers/models/speecht5/modeling_speecht5.py",
    "venv/lib/python3.12/site-packages/transformers/models/speecht5/number_normalizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/speecht5/processing_speecht5.py",
    "venv/lib/python3.12/site-packages/transformers/models/speecht5/tokenization_speecht5.py",
    "venv/lib/python3.12/site-packages/transformers/models/splinter/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/splinter/configuration_splinter.py",
    "venv/lib/python3.12/site-packages/transformers/models/splinter/modeling_splinter.py",
    "venv/lib/python3.12/site-packages/transformers/models/splinter/tokenization_splinter.py",
    "venv/lib/python3.12/site-packages/transformers/models/squeezebert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/squeezebert/configuration_squeezebert.py",
    "venv/lib/python3.12/site-packages/transformers/models/squeezebert/modeling_squeezebert.py",
    "venv/lib/python3.12/site-packages/transformers/models/squeezebert/tokenization_squeezebert.py",
    "venv/lib/python3.12/site-packages/transformers/models/stablelm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/stablelm/configuration_stablelm.py",
    "venv/lib/python3.12/site-packages/transformers/models/stablelm/modeling_stablelm.py",
    "venv/lib/python3.12/site-packages/transformers/models/starcoder2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/starcoder2/configuration_starcoder2.py",
    "venv/lib/python3.12/site-packages/transformers/models/starcoder2/modeling_starcoder2.py",
    "venv/lib/python3.12/site-packages/transformers/models/starcoder2/modular_starcoder2.py",
    "venv/lib/python3.12/site-packages/transformers/models/superglue/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/superglue/configuration_superglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_pil_superglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/superglue/image_processing_superglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/superglue/modeling_superglue.py",
    "venv/lib/python3.12/site-packages/transformers/models/superpoint/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/superpoint/configuration_superpoint.py",
    "venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_pil_superpoint.py",
    "venv/lib/python3.12/site-packages/transformers/models/superpoint/image_processing_superpoint.py",
    "venv/lib/python3.12/site-packages/transformers/models/superpoint/modeling_superpoint.py",
    "venv/lib/python3.12/site-packages/transformers/models/swiftformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/swiftformer/configuration_swiftformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/swiftformer/modeling_swiftformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin/configuration_swin.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin/modeling_swin.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin/modular_swin.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin2sr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin2sr/configuration_swin2sr.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_pil_swin2sr.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin2sr/image_processing_swin2sr.py",
    "venv/lib/python3.12/site-packages/transformers/models/swin2sr/modeling_swin2sr.py",
    "venv/lib/python3.12/site-packages/transformers/models/swinv2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/swinv2/configuration_swinv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/swinv2/modeling_swinv2.py",
    "venv/lib/python3.12/site-packages/transformers/models/switch_transformers/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/switch_transformers/configuration_switch_transformers.py",
    "venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modeling_switch_transformers.py",
    "venv/lib/python3.12/site-packages/transformers/models/switch_transformers/modular_switch_transformers.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5/configuration_t5.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5/tokenization_t5.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma/configuration_t5gemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma/modeling_t5gemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma/modular_t5gemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma2/configuration_t5gemma2.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py",
    "venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modular_t5gemma2.py",
    "venv/lib/python3.12/site-packages/transformers/models/table_transformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/table_transformer/configuration_table_transformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/table_transformer/modeling_table_transformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/tapas/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/tapas/configuration_tapas.py",
    "venv/lib/python3.12/site-packages/transformers/models/tapas/modeling_tapas.py",
    "venv/lib/python3.12/site-packages/transformers/models/tapas/tokenization_tapas.py",
    "venv/lib/python3.12/site-packages/transformers/models/textnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/textnet/configuration_textnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_pil_textnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/textnet/image_processing_textnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/textnet/modeling_textnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/configuration_time_series_transformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/time_series_transformer/modeling_time_series_transformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm/configuration_timesfm.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm/modeling_timesfm.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm/modular_timesfm.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/configuration_timesfm2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modeling_timesfm2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesfm2_5/modular_timesfm2_5.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesformer/configuration_timesformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/timesformer/modeling_timesformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/timm_backbone/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/timm_backbone/configuration_timm_backbone.py",
    "venv/lib/python3.12/site-packages/transformers/models/timm_backbone/modeling_timm_backbone.py",
    "venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/configuration_timm_wrapper.py",
    "venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/image_processing_timm_wrapper.py",
    "venv/lib/python3.12/site-packages/transformers/models/timm_wrapper/modeling_timm_wrapper.py",
    "venv/lib/python3.12/site-packages/transformers/models/trocr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/trocr/configuration_trocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/trocr/processing_trocr.py",
    "venv/lib/python3.12/site-packages/transformers/models/tvp/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/tvp/configuration_tvp.py",
    "venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_pil_tvp.py",
    "venv/lib/python3.12/site-packages/transformers/models/tvp/image_processing_tvp.py",
    "venv/lib/python3.12/site-packages/transformers/models/tvp/modeling_tvp.py",
    "venv/lib/python3.12/site-packages/transformers/models/tvp/processing_tvp.py",
    "venv/lib/python3.12/site-packages/transformers/models/udop/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/udop/configuration_udop.py",
    "venv/lib/python3.12/site-packages/transformers/models/udop/modeling_udop.py",
    "venv/lib/python3.12/site-packages/transformers/models/udop/processing_udop.py",
    "venv/lib/python3.12/site-packages/transformers/models/udop/tokenization_udop.py",
    "venv/lib/python3.12/site-packages/transformers/models/umt5/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/umt5/configuration_umt5.py",
    "venv/lib/python3.12/site-packages/transformers/models/umt5/modeling_umt5.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech/configuration_unispeech.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech/modeling_unispeech.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech/modular_unispeech.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/configuration_unispeech_sat.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modeling_unispeech_sat.py",
    "venv/lib/python3.12/site-packages/transformers/models/unispeech_sat/modular_unispeech_sat.py",
    "venv/lib/python3.12/site-packages/transformers/models/univnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/univnet/configuration_univnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/univnet/feature_extraction_univnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/univnet/modeling_univnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/upernet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/upernet/configuration_upernet.py",
    "venv/lib/python3.12/site-packages/transformers/models/upernet/modeling_upernet.py",
    "venv/lib/python3.12/site-packages/transformers/models/uvdoc/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/uvdoc/configuration_uvdoc.py",
    "venv/lib/python3.12/site-packages/transformers/models/uvdoc/image_processing_uvdoc.py",
    "venv/lib/python3.12/site-packages/transformers/models/uvdoc/modeling_uvdoc.py",
    "venv/lib/python3.12/site-packages/transformers/models/uvdoc/modular_uvdoc.py",
    "venv/lib/python3.12/site-packages/transformers/models/vaultgemma/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vaultgemma/configuration_vaultgemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modeling_vaultgemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/vaultgemma/modular_vaultgemma.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/configuration_vibevoice_acoustic_tokenizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/feature_extraction_vibevoice_acoustic_tokenizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modeling_vibevoice_acoustic_tokenizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_acoustic_tokenizer/modular_vibevoice_acoustic_tokenizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/configuration_vibevoice_asr.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modeling_vibevoice_asr.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/modular_vibevoice_asr.py",
    "venv/lib/python3.12/site-packages/transformers/models/vibevoice_asr/processing_vibevoice_asr.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/configuration_video_llama_3.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_pil_video_llama_3.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/image_processing_video_llama_3.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modeling_video_llama_3.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/modular_video_llama_3.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/processing_video_llama_3.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llama_3/video_processing_video_llama_3.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llava/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llava/configuration_video_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llava/image_processing_video_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llava/modeling_video_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llava/processing_video_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/video_llava/video_processing_video_llava.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomae/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomae/configuration_videomae.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_pil_videomae.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomae/image_processing_videomae.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomae/modeling_videomae.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomae/video_processing_videomae.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomt/configuration_videomt.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomt/modeling_videomt.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomt/modular_videomt.py",
    "venv/lib/python3.12/site-packages/transformers/models/videomt/video_processing_videomt.py",
    "venv/lib/python3.12/site-packages/transformers/models/vilt/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vilt/configuration_vilt.py",
    "venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_pil_vilt.py",
    "venv/lib/python3.12/site-packages/transformers/models/vilt/image_processing_vilt.py",
    "venv/lib/python3.12/site-packages/transformers/models/vilt/modeling_vilt.py",
    "venv/lib/python3.12/site-packages/transformers/models/vilt/processing_vilt.py",
    "venv/lib/python3.12/site-packages/transformers/models/vipllava/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vipllava/configuration_vipllava.py",
    "venv/lib/python3.12/site-packages/transformers/models/vipllava/modeling_vipllava.py",
    "venv/lib/python3.12/site-packages/transformers/models/vipllava/modular_vipllava.py",
    "venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py",
    "venv/lib/python3.12/site-packages/transformers/models/visual_bert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/visual_bert/configuration_visual_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/visual_bert/modeling_visual_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit/configuration_vit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_pil_vit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit/image_processing_vit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit/modeling_vit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_mae/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_mae/configuration_vit_mae.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_mae/modeling_vit_mae.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_mae/modular_vit_mae.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_msn/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_msn/configuration_vit_msn.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_msn/modeling_vit_msn.py",
    "venv/lib/python3.12/site-packages/transformers/models/vit_msn/modular_vit_msn.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitdet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitdet/configuration_vitdet.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitdet/modeling_vitdet.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitmatte/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitmatte/configuration_vitmatte.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_pil_vitmatte.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitmatte/image_processing_vitmatte.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitmatte/modeling_vitmatte.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose/configuration_vitpose.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_pil_vitpose.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose/image_processing_vitpose.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose/modeling_vitpose.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py",
    "venv/lib/python3.12/site-packages/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py",
    "venv/lib/python3.12/site-packages/transformers/models/vits/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vits/configuration_vits.py",
    "venv/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py",
    "venv/lib/python3.12/site-packages/transformers/models/vits/tokenization_vits.py",
    "venv/lib/python3.12/site-packages/transformers/models/vivit/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vivit/configuration_vivit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vivit/image_processing_vivit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vivit/modeling_vivit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vivit/modular_vivit.py",
    "venv/lib/python3.12/site-packages/transformers/models/vjepa2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/vjepa2/configuration_vjepa2.py",
    "venv/lib/python3.12/site-packages/transformers/models/vjepa2/modeling_vjepa2.py",
    "venv/lib/python3.12/site-packages/transformers/models/vjepa2/video_processing_vjepa2.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral/configuration_voxtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral/modeling_voxtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral/modular_voxtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral/processing_voxtral.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/configuration_voxtral_realtime.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/feature_extraction_voxtral_realtime.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/modular_voxtral_realtime.py",
    "venv/lib/python3.12/site-packages/transformers/models/voxtral_realtime/processing_voxtral_realtime.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2/configuration_wav2vec2.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2/feature_extraction_wav2vec2.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2/processing_wav2vec2.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2/tokenization_wav2vec2.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/modular_wav2vec2_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/configuration_wav2vec2_conformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_conformer/modular_wav2vec2_conformer.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py",
    "venv/lib/python3.12/site-packages/transformers/models/wavlm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/wavlm/configuration_wavlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/wavlm/modeling_wavlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/wavlm/modular_wavlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/configuration_whisper.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/english_normalizer.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/feature_extraction_whisper.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/generation_whisper.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/processing_whisper.py",
    "venv/lib/python3.12/site-packages/transformers/models/whisper/tokenization_whisper.py",
    "venv/lib/python3.12/site-packages/transformers/models/x_clip/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/x_clip/configuration_x_clip.py",
    "venv/lib/python3.12/site-packages/transformers/models/x_clip/modeling_x_clip.py",
    "venv/lib/python3.12/site-packages/transformers/models/x_clip/modular_x_clip.py",
    "venv/lib/python3.12/site-packages/transformers/models/x_clip/processing_x_clip.py",
    "venv/lib/python3.12/site-packages/transformers/models/xcodec/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xcodec/configuration_xcodec.py",
    "venv/lib/python3.12/site-packages/transformers/models/xcodec/modeling_xcodec.py",
    "venv/lib/python3.12/site-packages/transformers/models/xglm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xglm/configuration_xglm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xglm/tokenization_xglm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm/configuration_xlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm/modeling_xlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm/tokenization_xlm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/configuration_xlm_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modeling_xlm_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/modular_xlm_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta/tokenization_xlm_roberta.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlm_roberta_xl/modular_xlm_roberta_xl.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlnet/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlnet/configuration_xlnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlnet/tokenization_xlnet.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlstm/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlstm/configuration_xlstm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xlstm/modeling_xlstm.py",
    "venv/lib/python3.12/site-packages/transformers/models/xmod/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/xmod/configuration_xmod.py",
    "venv/lib/python3.12/site-packages/transformers/models/xmod/modeling_xmod.py",
    "venv/lib/python3.12/site-packages/transformers/models/yolos/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/yolos/configuration_yolos.py",
    "venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_pil_yolos.py",
    "venv/lib/python3.12/site-packages/transformers/models/yolos/image_processing_yolos.py",
    "venv/lib/python3.12/site-packages/transformers/models/yolos/modeling_yolos.py",
    "venv/lib/python3.12/site-packages/transformers/models/yolos/modular_yolos.py",
    "venv/lib/python3.12/site-packages/transformers/models/yoso/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/yoso/configuration_yoso.py",
    "venv/lib/python3.12/site-packages/transformers/models/yoso/modeling_yoso.py",
    "venv/lib/python3.12/site-packages/transformers/models/youtu/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/youtu/configuration_youtu.py",
    "venv/lib/python3.12/site-packages/transformers/models/youtu/modeling_youtu.py",
    "venv/lib/python3.12/site-packages/transformers/models/youtu/modular_youtu.py",
    "venv/lib/python3.12/site-packages/transformers/models/zamba/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/zamba/configuration_zamba.py",
    "venv/lib/python3.12/site-packages/transformers/models/zamba/modeling_zamba.py",
    "venv/lib/python3.12/site-packages/transformers/models/zamba2/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/zamba2/configuration_zamba2.py",
    "venv/lib/python3.12/site-packages/transformers/models/zamba2/modeling_zamba2.py",
    "venv/lib/python3.12/site-packages/transformers/models/zamba2/modular_zamba2.py",
    "venv/lib/python3.12/site-packages/transformers/models/zoedepth/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/models/zoedepth/configuration_zoedepth.py",
    "venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_pil_zoedepth.py",
    "venv/lib/python3.12/site-packages/transformers/models/zoedepth/image_processing_zoedepth.py",
    "venv/lib/python3.12/site-packages/transformers/models/zoedepth/modeling_zoedepth.py",
    "venv/lib/python3.12/site-packages/transformers/monkey_patching.py",
    "venv/lib/python3.12/site-packages/transformers/optimization.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/any_to_any.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/audio_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/audio_utils.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/automatic_speech_recognition.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/base.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/depth_estimation.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/document_question_answering.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/feature_extraction.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/fill_mask.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/image_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/image_feature_extraction.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/image_segmentation.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/image_text_to_text.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/keypoint_matching.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/mask_generation.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/object_detection.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/pt_utils.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/table_question_answering.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/text_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/text_generation.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/text_to_audio.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/token_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/video_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_audio_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_image_classification.py",
    "venv/lib/python3.12/site-packages/transformers/pipelines/zero_shot_object_detection.py",
    "venv/lib/python3.12/site-packages/transformers/processing_utils.py",
    "venv/lib/python3.12/site-packages/transformers/py.typed",
    "venv/lib/python3.12/site-packages/transformers/pytorch_utils.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/auto.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/base.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_aqlm.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_auto_round.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_awq.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bitnet.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_4bit.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_bnb_8bit.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_compressed_tensors.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_eetq.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fbgemm_fp8.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_finegrained_fp8.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fouroversix.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_fp_quant.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_gptq.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_higgs.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_hqq.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_metal.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_mxfp4.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quanto.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_quark.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_sinq.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_spqr.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_torchao.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizer_vptq.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/__pycache__/quantizers_utils.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/quantizers/auto.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/base.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_aqlm.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_auto_round.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_awq.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bitnet.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_4bit.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_8bit.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_compressed_tensors.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_eetq.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fbgemm_fp8.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_finegrained_fp8.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fouroversix.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_fp_quant.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_gptq.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_higgs.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_hqq.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_metal.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_mxfp4.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quanto.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_quark.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_sinq.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_spqr.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_torchao.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_vptq.py",
    "venv/lib/python3.12/site-packages/transformers/quantizers/quantizers_utils.py",
    "venv/lib/python3.12/site-packages/transformers/safetensors_conversion.py",
    "venv/lib/python3.12/site-packages/transformers/testing_utils.py",
    "venv/lib/python3.12/site-packages/transformers/time_series_utils.py",
    "venv/lib/python3.12/site-packages/transformers/tokenization_mistral_common.py",
    "venv/lib/python3.12/site-packages/transformers/tokenization_python.py",
    "venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py",
    "venv/lib/python3.12/site-packages/transformers/tokenization_utils_sentencepiece.py",
    "venv/lib/python3.12/site-packages/transformers/tokenization_utils_tokenizers.py",
    "venv/lib/python3.12/site-packages/transformers/trainer.py",
    "venv/lib/python3.12/site-packages/transformers/trainer_callback.py",
    "venv/lib/python3.12/site-packages/transformers/trainer_jit_checkpoint.py",
    "venv/lib/python3.12/site-packages/transformers/trainer_optimizer.py",
    "venv/lib/python3.12/site-packages/transformers/trainer_pt_utils.py",
    "venv/lib/python3.12/site-packages/transformers/trainer_seq2seq.py",
    "venv/lib/python3.12/site-packages/transformers/trainer_utils.py",
    "venv/lib/python3.12/site-packages/transformers/training_args.py",
    "venv/lib/python3.12/site-packages/transformers/training_args_seq2seq.py",
    "venv/lib/python3.12/site-packages/transformers/utils/__init__.py",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/auto_docstring.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_parsing_utils.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/chat_template_utils.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/constants.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/doc.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_mistral_common_objects.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_sentencepiece_and_tokenizers_objects.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_torchvision_objects.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/dummy_vision_objects.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/generic.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/hub.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/import_utils.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/kernel_config.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/loading_report.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/logging.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/output_capturing.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/peft_utils.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/quantization_config.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/type_validators.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/__pycache__/versions.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/transformers/utils/attention_visualizer.py",
    "venv/lib/python3.12/site-packages/transformers/utils/auto_docstring.py",
    "venv/lib/python3.12/site-packages/transformers/utils/backbone_utils.py",
    "venv/lib/python3.12/site-packages/transformers/utils/chat_parsing_utils.py",
    "venv/lib/python3.12/site-packages/transformers/utils/chat_template_utils.py",
    "venv/lib/python3.12/site-packages/transformers/utils/constants.py",
    "venv/lib/python3.12/site-packages/transformers/utils/deprecation.py",
    "venv/lib/python3.12/site-packages/transformers/utils/doc.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_detectron2_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_mistral_common_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_music_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_pt_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_sentencepiece_and_tokenizers_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_speech_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_timm_and_torchvision_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_tokenizers_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_torchaudio_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_torchvision_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/dummy_vision_objects.py",
    "venv/lib/python3.12/site-packages/transformers/utils/generic.py",
    "venv/lib/python3.12/site-packages/transformers/utils/hp_naming.py",
    "venv/lib/python3.12/site-packages/transformers/utils/hub.py",
    "venv/lib/python3.12/site-packages/transformers/utils/import_utils.py",
    "venv/lib/python3.12/site-packages/transformers/utils/kernel_config.py",
    "venv/lib/python3.12/site-packages/transformers/utils/loading_report.py",
    "venv/lib/python3.12/site-packages/transformers/utils/logging.py",
    "venv/lib/python3.12/site-packages/transformers/utils/network_logging.py",
    "venv/lib/python3.12/site-packages/transformers/utils/notebook.py",
    "venv/lib/python3.12/site-packages/transformers/utils/output_capturing.py",
    "venv/lib/python3.12/site-packages/transformers/utils/peft_utils.py",
    "venv/lib/python3.12/site-packages/transformers/utils/pytest_helpers.py",
    "venv/lib/python3.12/site-packages/transformers/utils/quantization_config.py",
    "venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2.py",
    "venv/lib/python3.12/site-packages/transformers/utils/sentencepiece_model_pb2_new.py",
    "venv/lib/python3.12/site-packages/transformers/utils/type_validators.py",
    "venv/lib/python3.12/site-packages/transformers/utils/versions.py",
    "venv/lib/python3.12/site-packages/transformers/video_processing_utils.py",
    "venv/lib/python3.12/site-packages/transformers/video_utils.py",
    "venv/lib/python3.12/site-packages/transformers/vision_utils.py",
    "venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/entry_points.txt",
    "venv/lib/python3.12/site-packages/triton-3.2.0.dist-info/top_level.txt",
    "venv/lib/python3.12/site-packages/triton/_C/libproton.so",
    "venv/lib/python3.12/site-packages/triton/_C/libtriton.so",
    "venv/lib/python3.12/site-packages/triton/__init__.py",
    "venv/lib/python3.12/site-packages/triton/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/__pycache__/errors.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/__pycache__/testing.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/_internal_testing.py",
    "venv/lib/python3.12/site-packages/triton/backends/__init__.py",
    "venv/lib/python3.12/site-packages/triton/backends/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/backends/__pycache__/compiler.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/backends/__pycache__/driver.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/compiler.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/backends/amd/__pycache__/driver.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/backends/amd/compiler.py",
    "venv/lib/python3.12/site-packages/triton/backends/amd/driver.c",
    "venv/lib/python3.12/site-packages/triton/backends/amd/driver.py",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_channel_descriptor.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_device_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_atomic.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_bf16.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_bfloat16.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_common.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_complex.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_cooperative_groups.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_fp16.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_gl_interop.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_math_constants.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime_pt_api.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_unsafe_atomics.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_vector_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_math_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_surface_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/amd_warp_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/concepts.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/device_library_decls.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/functional_grid_launch.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/grid_launch_GGL.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/helpers.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_api_trace.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_assert.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_cooperative_groups_helper.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_gcc.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_math_fwd.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_ldg.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_prof_str.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hip_runtime_prof.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/host_defines.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/hsa_helpers.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/macro_based_grid_launch.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/math_fwd.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/ockl_image.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/program_state.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_fetch_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/amd_detail/texture_indirect_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/channel_descriptor.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/device_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/driver_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_bf16.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_bfloat16.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_common.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_complex.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_cooperative_groups.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_deprecated.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_ext.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_fp16.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_gl_interop.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_hcc.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_math_constants.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_profile.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_runtime_api.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_texture_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_vector_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hip_version.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/hiprtc.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/library_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/math_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/surface_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hip/texture_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/Brig.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_common.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_elf.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_kernel_code.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_queue.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/amd_hsa_signal.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_amd_tool.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_api_trace.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_amd.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_finalize.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ext_image.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_aqlprofile.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_loader.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/ext/prof_protocol.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hip_ostream_ops.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hsa_ostream_ops.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/hsa_prof_str.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_ext.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hcc.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hip.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_hsa.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_plugin.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctracer_roctx.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/include/roctracer/roctx.h",
    "venv/lib/python3.12/site-packages/triton/backends/amd/lib/ockl.bc",
    "venv/lib/python3.12/site-packages/triton/backends/amd/lib/ocml.bc",
    "venv/lib/python3.12/site-packages/triton/backends/compiler.py",
    "venv/lib/python3.12/site-packages/triton/backends/driver.py",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/__init__.py",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/compiler.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/__pycache__/driver.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.c",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openacc/cupti_openacc.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/cupti_openmp.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/Openmp/omp-tools.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/builtin_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/channel_descriptor.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/common_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/async.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_reduce.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_scan.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/driver_abi.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/functional.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/helpers.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/info.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/invoke.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/memory.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/partitioning.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/reduce.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/scan.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/details/sync.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/memcpy_async.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/reduce.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cooperative_groups/scan.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/common_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/cudacc_ext.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_double_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/device_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/func_macro.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_config.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_defines.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/host_runtime.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/math_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/mma.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/nvfunctional",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/crt/storage_class.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuComplex.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGL.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaEGLTypedefs.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGL.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaGLTypedefs.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaProfilerTypedefs.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaTypedefs.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAU.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudaVDPAUTypedefs.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_helpers.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_awbarrier_primitives.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_bf16.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_device_runtime_api.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_egl_interop.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp16.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_fp8.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_gl_interop.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_occupancy.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_helpers.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_pipeline_primitives.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_runtime_api.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_stdint.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_surface_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_texture_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cuda_vdpau_interop.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cudart_platform.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_activity_deprecated.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_callbacks.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_checkpoint.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_common.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_driver_cbid.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_events.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_metrics.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_nvtx_cbid.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_pcsampling_util.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_profiler_target.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_result.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_runtime_cbid.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_sass_metrics.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_target.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/cupti_version.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_atomic_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_double_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_launch_parameters.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/device_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/driver_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/fatbinary_section.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaGL_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudaVDPAU_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_gl_interop_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_runtime_api_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cuda_vdpau_interop_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_cudart_removed_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/generated_nvtx_meta.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_config.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/host_defines.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/library_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_constants.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/math_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/mma.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvPTXCompiler.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvfunctional",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_common.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_cuda_host.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_host.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/nvperf_target.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_atomic_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_35_intrinsics.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_indirect_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/surface_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_fetch_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_indirect_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/texture_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_functions.hpp",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/include/vector_types.h",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcheckpoint.so",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.12",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.2024.1.0",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host.so",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_target.so",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti/libpcsamplingutil.so",
    "venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc",
    "venv/lib/python3.12/site-packages/triton/compiler/__init__.py",
    "venv/lib/python3.12/site-packages/triton/compiler/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/compiler/__pycache__/code_generator.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/compiler/__pycache__/compiler.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/compiler/__pycache__/errors.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/compiler/code_generator.py",
    "venv/lib/python3.12/site-packages/triton/compiler/compiler.py",
    "venv/lib/python3.12/site-packages/triton/compiler/errors.py",
    "venv/lib/python3.12/site-packages/triton/compiler/make_launcher.py",
    "venv/lib/python3.12/site-packages/triton/errors.py",
    "venv/lib/python3.12/site-packages/triton/instrumentation/libGPUInstrumentationTestLib.so",
    "venv/lib/python3.12/site-packages/triton/language/__init__.py",
    "venv/lib/python3.12/site-packages/triton/language/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/__pycache__/_utils.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/__pycache__/core.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/__pycache__/math.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/__pycache__/random.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/__pycache__/semantic.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/__pycache__/standard.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/_utils.py",
    "venv/lib/python3.12/site-packages/triton/language/core.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/__init__.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/__init__.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/_experimental_tma.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/libdevice.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/__pycache__/utils.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/_experimental_tma.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/libdevice.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/cuda/utils.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/hip/__init__.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/extra/hip/__pycache__/libdevice.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/language/extra/hip/libdevice.py",
    "venv/lib/python3.12/site-packages/triton/language/extra/libdevice.py",
    "venv/lib/python3.12/site-packages/triton/language/math.py",
    "venv/lib/python3.12/site-packages/triton/language/random.py",
    "venv/lib/python3.12/site-packages/triton/language/semantic.py",
    "venv/lib/python3.12/site-packages/triton/language/standard.py",
    "venv/lib/python3.12/site-packages/triton/profiler/__init__.py",
    "venv/lib/python3.12/site-packages/triton/profiler/flags.py",
    "venv/lib/python3.12/site-packages/triton/profiler/hook.py",
    "venv/lib/python3.12/site-packages/triton/profiler/profile.py",
    "venv/lib/python3.12/site-packages/triton/profiler/proton.py",
    "venv/lib/python3.12/site-packages/triton/profiler/scope.py",
    "venv/lib/python3.12/site-packages/triton/profiler/viewer.py",
    "venv/lib/python3.12/site-packages/triton/runtime/__init__.py",
    "venv/lib/python3.12/site-packages/triton/runtime/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/runtime/__pycache__/autotuner.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/runtime/__pycache__/build.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/runtime/__pycache__/cache.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/runtime/__pycache__/driver.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/runtime/__pycache__/errors.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/runtime/__pycache__/jit.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/runtime/autotuner.py",
    "venv/lib/python3.12/site-packages/triton/runtime/build.py",
    "venv/lib/python3.12/site-packages/triton/runtime/cache.py",
    "venv/lib/python3.12/site-packages/triton/runtime/driver.py",
    "venv/lib/python3.12/site-packages/triton/runtime/errors.py",
    "venv/lib/python3.12/site-packages/triton/runtime/interpreter.py",
    "venv/lib/python3.12/site-packages/triton/runtime/jit.py",
    "venv/lib/python3.12/site-packages/triton/testing.py",
    "venv/lib/python3.12/site-packages/triton/tools/__init__.py",
    "venv/lib/python3.12/site-packages/triton/tools/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/tools/__pycache__/disasm.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/triton/tools/build_extern.py",
    "venv/lib/python3.12/site-packages/triton/tools/compile.c",
    "venv/lib/python3.12/site-packages/triton/tools/compile.h",
    "venv/lib/python3.12/site-packages/triton/tools/compile.py",
    "venv/lib/python3.12/site-packages/triton/tools/disasm.py",
    "venv/lib/python3.12/site-packages/triton/tools/experimental_descriptor.py",
    "venv/lib/python3.12/site-packages/triton/tools/link.py",
    "venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/entry_points.txt",
    "venv/lib/python3.12/site-packages/typer-0.25.1.dist-info/licenses/LICENSE",
    "venv/lib/python3.12/site-packages/typer/.agents/skills/typer/SKILL.md",
    "venv/lib/python3.12/site-packages/typer/__init__.py",
    "venv/lib/python3.12/site-packages/typer/__main__.py",
    "venv/lib/python3.12/site-packages/typer/_completion_classes.py",
    "venv/lib/python3.12/site-packages/typer/_completion_shared.py",
    "venv/lib/python3.12/site-packages/typer/_types.py",
    "venv/lib/python3.12/site-packages/typer/_typing.py",
    "venv/lib/python3.12/site-packages/typer/cli.py",
    "venv/lib/python3.12/site-packages/typer/colors.py",
    "venv/lib/python3.12/site-packages/typer/completion.py",
    "venv/lib/python3.12/site-packages/typer/core.py",
    "venv/lib/python3.12/site-packages/typer/main.py",
    "venv/lib/python3.12/site-packages/typer/models.py",
    "venv/lib/python3.12/site-packages/typer/params.py",
    "venv/lib/python3.12/site-packages/typer/py.typed",
    "venv/lib/python3.12/site-packages/typer/rich_utils.py",
    "venv/lib/python3.12/site-packages/typer/testing.py",
    "venv/lib/python3.12/site-packages/typer/utils.py",
    "venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/typing_extensions-4.15.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.12/site-packages/typing_extensions.py",
    "venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/typing_inspection-0.4.2.dist-info/licenses/LICENSE",
    "venv/lib/python3.12/site-packages/typing_inspection/__init__.py",
    "venv/lib/python3.12/site-packages/typing_inspection/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/typing_inspection/__pycache__/introspection.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/typing_inspection/__pycache__/typing_objects.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/typing_inspection/introspection.py",
    "venv/lib/python3.12/site-packages/typing_inspection/py.typed",
    "venv/lib/python3.12/site-packages/typing_inspection/typing_objects.py",
    "venv/lib/python3.12/site-packages/typing_inspection/typing_objects.pyi",
    "venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/urllib3-2.7.0.dist-info/licenses/LICENSE.txt",
    "venv/lib/python3.12/site-packages/urllib3/__init__.py",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/_base_connection.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/_collections.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/_request_methods.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/_version.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/connection.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/connectionpool.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/exceptions.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/fields.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/filepost.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/poolmanager.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/__pycache__/response.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/_base_connection.py",
    "venv/lib/python3.12/site-packages/urllib3/_collections.py",
    "venv/lib/python3.12/site-packages/urllib3/_request_methods.py",
    "venv/lib/python3.12/site-packages/urllib3/_version.py",
    "venv/lib/python3.12/site-packages/urllib3/connection.py",
    "venv/lib/python3.12/site-packages/urllib3/connectionpool.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/__init__.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/contrib/__pycache__/socks.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/__init__.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/connection.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/emscripten_fetch_worker.js",
    "venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/fetch.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/request.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/emscripten/response.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/pyopenssl.py",
    "venv/lib/python3.12/site-packages/urllib3/contrib/socks.py",
    "venv/lib/python3.12/site-packages/urllib3/exceptions.py",
    "venv/lib/python3.12/site-packages/urllib3/fields.py",
    "venv/lib/python3.12/site-packages/urllib3/filepost.py",
    "venv/lib/python3.12/site-packages/urllib3/http2/__init__.py",
    "venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/http2/__pycache__/probe.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/http2/connection.py",
    "venv/lib/python3.12/site-packages/urllib3/http2/probe.py",
    "venv/lib/python3.12/site-packages/urllib3/poolmanager.py",
    "venv/lib/python3.12/site-packages/urllib3/py.typed",
    "venv/lib/python3.12/site-packages/urllib3/response.py",
    "venv/lib/python3.12/site-packages/urllib3/util/__init__.py",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/connection.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/proxy.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/request.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/response.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/retry.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssl_match_hostname.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/ssltransport.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/timeout.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/url.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/util.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/__pycache__/wait.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/urllib3/util/connection.py",
    "venv/lib/python3.12/site-packages/urllib3/util/proxy.py",
    "venv/lib/python3.12/site-packages/urllib3/util/request.py",
    "venv/lib/python3.12/site-packages/urllib3/util/response.py",
    "venv/lib/python3.12/site-packages/urllib3/util/retry.py",
    "venv/lib/python3.12/site-packages/urllib3/util/ssl_.py",
    "venv/lib/python3.12/site-packages/urllib3/util/ssl_match_hostname.py",
    "venv/lib/python3.12/site-packages/urllib3/util/ssltransport.py",
    "venv/lib/python3.12/site-packages/urllib3/util/timeout.py",
    "venv/lib/python3.12/site-packages/urllib3/util/url.py",
    "venv/lib/python3.12/site-packages/urllib3/util/util.py",
    "venv/lib/python3.12/site-packages/urllib3/util/wait.py",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/licenses/LICENSE-APACHE",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/licenses/LICENSE-MIT",
    "venv/lib/python3.12/site-packages/uv-0.11.16.dist-info/sboms/uv.cyclonedx.json",
    "venv/lib/python3.12/site-packages/uv/__init__.py",
    "venv/lib/python3.12/site-packages/uv/__main__.py",
    "venv/lib/python3.12/site-packages/uv/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/uv/__pycache__/__main__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/uv/__pycache__/_find_uv.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/uv/_find_uv.py",
    "venv/lib/python3.12/site-packages/uv/py.typed",
    "venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.12/site-packages/xxhash-3.7.0.dist-info/top_level.txt",
    "venv/lib/python3.12/site-packages/xxhash/__init__.py",
    "venv/lib/python3.12/site-packages/xxhash/__init__.pyi",
    "venv/lib/python3.12/site-packages/xxhash/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/xxhash/__pycache__/version.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/xxhash/_xxhash.cpython-312-x86_64-linux-gnu.so",
    "venv/lib/python3.12/site-packages/xxhash/py.typed",
    "venv/lib/python3.12/site-packages/xxhash/version.py",
    "venv/lib/python3.12/site-packages/yaml/__init__.py",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/composer.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/constructor.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/cyaml.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/dumper.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/emitter.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/error.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/events.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/loader.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/nodes.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/parser.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/reader.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/representer.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/resolver.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/scanner.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/serializer.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/__pycache__/tokens.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yaml/_yaml.cpython-312-x86_64-linux-gnu.so",
    "venv/lib/python3.12/site-packages/yaml/composer.py",
    "venv/lib/python3.12/site-packages/yaml/constructor.py",
    "venv/lib/python3.12/site-packages/yaml/cyaml.py",
    "venv/lib/python3.12/site-packages/yaml/dumper.py",
    "venv/lib/python3.12/site-packages/yaml/emitter.py",
    "venv/lib/python3.12/site-packages/yaml/error.py",
    "venv/lib/python3.12/site-packages/yaml/events.py",
    "venv/lib/python3.12/site-packages/yaml/loader.py",
    "venv/lib/python3.12/site-packages/yaml/nodes.py",
    "venv/lib/python3.12/site-packages/yaml/parser.py",
    "venv/lib/python3.12/site-packages/yaml/reader.py",
    "venv/lib/python3.12/site-packages/yaml/representer.py",
    "venv/lib/python3.12/site-packages/yaml/resolver.py",
    "venv/lib/python3.12/site-packages/yaml/scanner.py",
    "venv/lib/python3.12/site-packages/yaml/serializer.py",
    "venv/lib/python3.12/site-packages/yaml/tokens.py",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/INSTALLER",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/METADATA",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/RECORD",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/REQUESTED",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/WHEEL",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/licenses/LICENSE",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/licenses/NOTICE",
    "venv/lib/python3.12/site-packages/yarl-1.24.2.dist-info/top_level.txt",
    "venv/lib/python3.12/site-packages/yarl/__init__.py",
    "venv/lib/python3.12/site-packages/yarl/__pycache__/__init__.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yarl/__pycache__/_parse.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yarl/__pycache__/_path.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yarl/__pycache__/_query.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yarl/__pycache__/_quoters.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yarl/__pycache__/_quoting.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yarl/__pycache__/_url.cpython-312.pyc",
    "venv/lib/python3.12/site-packages/yarl/_parse.py",
    "venv/lib/python3.12/site-packages/yarl/_path.py",
    "venv/lib/python3.12/site-packages/yarl/_query.py",
    "venv/lib/python3.12/site-packages/yarl/_quoters.py",
    "venv/lib/python3.12/site-packages/yarl/_quoting.py",
    "venv/lib/python3.12/site-packages/yarl/_quoting_c.cpython-312-x86_64-linux-gnu.so",
    "venv/lib/python3.12/site-packages/yarl/_quoting_c.pyx",
    "venv/lib/python3.12/site-packages/yarl/_quoting_py.py",
    "venv/lib/python3.12/site-packages/yarl/_url.py",
    "venv/lib/python3.12/site-packages/yarl/py.typed",
    "venv/pyvenv.cfg",
    "venv/share/man/man1/isympy.1"
  ]
}

### [2026-05-25 01:07:01 UTC] ASSISTANT

**Thinking**

Something went wrong - the python script printed a massive output. That's strange. Let me check if the quant_summary.json was written successfully.

The output files list included a lot of venv files because it used `rglob("*")` which recurses into the venv directory. I need to fix the output_files list.

Let me verify the quant_summary.json was written and then fix it.

Something went wrong with the output files. Let me verify and fix:


**Tool call:** `exec`

```json
{
  "command": "ls /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json 2>&1 && echo \"---exists---\""
}
```

### [2026-05-25 01:07:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json
---exists---

### [2026-05-25 01:07:05 UTC] ASSISTANT

**Thinking**

The file was written. Let me verify its contents and then rewrite with a proper output_files list.

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"\nimport json\nwith open('/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json') as f:\n    d = json.load(f)\nprint('status:', d['status'])\nprint('output_files count:', len(d.get('output_files', [])))\nprint('first 5:', d.get('output_files', [])[:5])\n\""
}
```

### [2026-05-25 01:07:05 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

status: success
output_files count: 25908
first 5: ['../Qwen_Qwen3.5-0.8B-MXFP4/chat_template.jinja', '../Qwen_Qwen3.5-0.8B-MXFP4/config.json', '../Qwen_Qwen3.5-0.8B-MXFP4/generation_config.json', '../Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors', '../Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors.index.json']

### [2026-05-25 01:07:13 UTC] ASSISTANT

**Thinking**

The quant_summary.json was written but it has 25908 output files (including all venv files). I need to fix this by excluding the venv directory. Let me rewrite it properly.

I need to fix the output_files list (it included venv files). Let me rewrite it properly:


**Tool call:** `exec`

```json
{
  "command": "python3 << 'PYEOF'\nimport json\nfrom pathlib import Path\n\nruntime_dir = Path(\"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4\")\nmodel_dir = Path(\"/root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4\")\n\n# Collect output files from runtime dir, excluding venv\noutput_files = []\nfor f in sorted(runtime_dir.rglob(\"*\")):\n    if f.is_file():\n        rel = str(f.relative_to(runtime_dir))\n        # Skip venv directory\n        if not rel.startswith(\"venv/\"):\n            output_files.append(rel)\n\n# Also list key files in quantized model dir (without path prefix)\nfor f in sorted(model_dir.glob(\"*\")):\n    if f.is_file():\n        output_files.append(f\"../Qwen_Qwen3.5-0.8B-MXFP4/{f.name}\")\n\noutput_files = sorted(set(output_files))\n\nsummary = {\n    \"model_id\": \"Qwen/Qwen3.5-0.8B\",\n    \"scheme\": \"MXFP4\",\n    \"method\": \"RTN\",\n    \"export_format\": \"auto_round\",\n    \"device\": \"cuda\",\n    \"quant_num_gpus\": \"1\",\n    \"num_gpus\": \"1\",\n    \"output_dir\": str(runtime_dir),\n    \"runtime_output_dir\": str(runtime_dir),\n    \"quantized_model_dir\": str(model_dir),\n    \"status\": \"success\",\n    \"duration_seconds\": 17.55,\n    \"original_size_mb\": 1666.01,\n    \"quantized_size_mb\": 970.00,\n    \"compression_ratio\": 1.7175,\n    \"errors\": [],\n    \"solutions\": [],\n    \"output_files\": output_files\n}\n\n# Write atomically via temp file\ntmp = runtime_dir / \"quant_summary.json.tmp\"\nwith open(tmp, \"w\") as f:\n    json.dump(summary, f, indent=2)\ntmp.rename(runtime_dir / \"quant_summary.json\")\n\nprint(f\"Written: {runtime_dir / 'quant_summary.json'}\")\nprint(f\"Output files: {len(output_files)}\")\nfor f in output_files[:10]:\n    print(f\"  {f}\")\nPYEOF"
}
```

### [2026-05-25 01:07:14 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Written: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json
Output files: 15
  ../Qwen_Qwen3.5-0.8B-MXFP4/chat_template.jinja
  ../Qwen_Qwen3.5-0.8B-MXFP4/config.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/generation_config.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors
  ../Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors.index.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/model_extra_tensors.safetensors
  ../Qwen_Qwen3.5-0.8B-MXFP4/quantization_config.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/tokenizer.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/tokenizer_config.json
  logs/auto.log

### [2026-05-25 01:07:18 UTC] ASSISTANT

**Thinking**

The quant_summary.json is now properly written. Let me verify the final state of everything.

Let me verify the final state:


**Tool call:** `exec`

```json
{
  "command": "echo \"=== Final Summary ===\" && python3 -c \"\nimport json\nwith open('/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4/quant_summary.json') as f:\n    d = json.load(f)\nfor k, v in d.items():\n    if k != 'output_files':\n        print(f'{k}: {v}')\nprint(f'output_files ({len(d[\\\"output_files\\\"])} items):')\nfor f in d['output_files']:\n    print(f'  {f}')\n\""
}
```

### [2026-05-25 01:07:19 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

=== Final Summary ===
model_id: Qwen/Qwen3.5-0.8B
scheme: MXFP4
method: RTN
export_format: auto_round
device: cuda
quant_num_gpus: 1
num_gpus: 1
output_dir: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4
runtime_output_dir: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3.5-0.8B-MXFP4
quantized_model_dir: /root/.openclaw/workspace/quantized/Qwen_Qwen3.5-0.8B-MXFP4
status: success
duration_seconds: 17.55
original_size_mb: 1666.01
quantized_size_mb: 970.0
compression_ratio: 1.7175
errors: []
solutions: []
output_files (15 items):
  ../Qwen_Qwen3.5-0.8B-MXFP4/chat_template.jinja
  ../Qwen_Qwen3.5-0.8B-MXFP4/config.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/generation_config.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors
  ../Qwen_Qwen3.5-0.8B-MXFP4/model.safetensors.index.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/model_extra_tensors.safetensors
  ../Qwen_Qwen3.5-0.8B-MXFP4/quantization_config.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/tokenizer.json
  ../Qwen_Qwen3.5-0.8B-MXFP4/tokenizer_config.json
  logs/auto.log
  logs/quant_exec.log
  logs/quant_prompt.txt
  quant_summary.json
  quantize.py
  request.json

### [2026-05-25 01:07:25 UTC] ASSISTANT

**Thinking**

Everything looks good. The quantization was successful. Let me provide a summary of what was done.

**Quantization complete.** Here's a summary of what was done:

---

## ✅ Qwen/Qwen3.5-0.8B — MXFP4 RTN Quantization

**Status:** `success`

### Configuration
| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3.5-0.8B` |
| Scheme | MXFP4 |
| Method | RTN (iters=0, no calibration training) |
| Export Format | `auto_round` |
| Device | `cuda` (single GPU) |
| nsamples | 128 |

### Results
| Metric | Value |
|--------|-------|
| Duration | **17.55 seconds** |
| Original size | 1666.01 MB |
| Quantized size | 970.00 MB |
| **Compression ratio** | **1.72x** |

### Output Files
- **`model.safetensors`** (739 MB) — quantized LM weights in MXFP4 format
- **`model_extra_tensors.safetensors`** (231 MB) — unquantized visual encoder + MTP params preserved separately
- **`quantization_config.json`** — metadata (group_size=32, data_type=mx_fp)
- Tokenizer + config files

### Notes
- Some layers were skipped (shape not divisible by 32) per AutoRound MXFP4 constraints
- AutoRound v0.12.3 with PyTorch 2.6.0+cu124, Transformers 5.9.0
- RTX 4090 GPU used; peak VRAM ~560MB during quantization
