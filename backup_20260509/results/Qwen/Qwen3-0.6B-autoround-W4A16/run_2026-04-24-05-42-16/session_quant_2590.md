# Session: autoeval_quant_2590

- **Session ID:** `autoeval_quant_2590`
- **Timestamp:** 2026-04-24 05:27:10 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 1: Quantization

### [2026-04-24 05:27:10 UTC] USER

You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_quant/SKILL.md

Model: Qwen/Qwen3-0.6B
Quantization: W4A16 / RTN
Export format: auto_round
Quantized Model Output directory: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16
Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16
Runtime device: cuda
Num gpus: 1

Directory responsibilities:
- Write exported model files to: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16
- Write runtime artifacts such as quant_summary.json, quantize.py, logs, prompts, copied request/session files, and the venv to: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16

CRITICAL SCRIPT REQUIREMENT:
- Before starting quantization, you MUST first generate the quantization script file:
    /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quantize.py
- The file name must be exactly: quantize.py
- Run quantization by executing that generated quantize.py script
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

IMPORTANT - After quantization completes (success or failure), you MUST produce:

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quant_summary.json - structured summary:
{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
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

### [2026-04-24 05:27:20 UTC] ASSISTANT

**Thinking**

Let me start by reading the skill instructions to understand how to proceed with the quantization task.

**Tool call:** `read`

```json
{
  "filePath": "/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
}
```

### [2026-04-24 05:27:20 UTC] TOOL RESULT
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


### [2026-04-24 05:27:33 UTC] ASSISTANT

**Thinking**

Now I have the skill instructions. Let me analyze the task:

1. Model: Qwen/Qwen3-0.6B
2. Quantization: W4A16 / RTN
3. Export format: auto_round
4. Output directory: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16
5. Runtime artifact directory: /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16
6. Runtime device: cuda
7. Num gpus: 1

RTN means "Round-to-Nearest" which uses iters=0 (no training).

Let me start by:
1. Creating the runtime output directory
2. Setting up the environment (checking for existing venvs)
3. Generating the quantization script
4. Running the quantization
5. Creating the summary

Let me first check for existing venvs and the model info file.

**Tool call:** `exec`

```json
{
  "command": "mkdir -p /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/logs\nmkdir -p /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\nls -la /root/.venv/bin/python 2>/dev/null || echo \"No /root/.venv\"\nls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/"
}
```

### [2026-04-24 05:27:33 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

No /root/.venv
total 16
drwxr-xr-x 3 root root 4096 Apr 24 05:27 .
drwxr-xr-x 3 root root 4096 Apr 24 05:27 ..
drwxr-xr-x 2 root root 4096 Apr 24 05:27 logs
-rw-r--r-- 1 root root 1004 Apr 24 05:27 request.json

### [2026-04-24 05:27:36 UTC] ASSISTANT

**Thinking**

No /root/.venv exists. I need to create a venv. Let me create the venv in the runtime output directory.

**Tool call:** `exec`

```json
{
  "command": "python3 -m venv --system-site-packages /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv 2>&1"
}
```

### [2026-04-24 05:27:38 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-24 05:27:46 UTC] ASSISTANT

**Thinking**

Good, venv created. Now install auto-round. Let me install from GitHub as suggested in the skill.

**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/python auto-round 2>&1 | tail -20"
}
```

### [2026-04-24 05:27:46 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

sh: 1: uv: not found

### [2026-04-24 05:27:53 UTC] ASSISTANT

**Thinking**

uv is not found, I need to install it first or use pip directly. Let me check if pip is available.

**Tool call:** `exec`

```json
{
  "command": "which pip pip3 2>/dev/null; /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/python -m pip --version 2>&1 | head -5"
}
```

### [2026-04-24 05:27:53 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/usr/bin/pip
/usr/bin/pip3
pip 22.0.2 from /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/pip (python 3.10)

### [2026-04-24 05:27:56 UTC] ASSISTANT

**Thinking**

Let me install uv first, then use it for the remaining installations.

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/python -m pip install -U uv 2>&1 | tail -5"
}
```

### [2026-04-24 05:28:06 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session plaid-otter, pid 2648). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-04-24 05:28:11 UTC] ASSISTANT

**Thinking**

Let me wait for this to finish.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "plaid-otter",
  "timeout": 30000
}
```

### [2026-04-24 05:28:11 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

  Downloading uv-0.11.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (24.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24.9/24.9 MB 8.1 MB/s eta 0:00:00
WARNING: Ignoring invalid distribution -ympy (/usr/local/lib/python3.10/dist-packages)
Installing collected packages: uv
Successfully installed uv-0.11.7


Process exited with code 0.

### [2026-04-24 05:28:14 UTC] ASSISTANT

**Thinking**

Good, uv is installed. Now install auto-round.

**Tool call:** `exec`

```json
{
  "command": "uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/python auto-round 2>&1 | tail -20",
  "yieldMs": 60000
}
```

### [2026-04-24 05:28:14 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

sh: 1: uv: not found

### [2026-04-24 05:28:17 UTC] ASSISTANT

**Thinking**

The uv binary was installed in the venv but it's not in PATH. Let me use the full path.

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/uv pip install --python /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/python auto-round 2>&1 | tail -20",
  "yieldMs": 120000
}
```

### [2026-04-24 05:30:17 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `running`

Command still running (session amber-daisy, pid 2657). Use process (list/poll/log/write/kill/clear/remove) for follow-up.

### [2026-04-24 05:30:22 UTC] ASSISTANT

**Thinking**

Let me wait for it to complete.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "amber-daisy",
  "timeout": 120000
}
```

### [2026-04-24 05:32:22 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `running`

(no new output)

Process still running.

### [2026-04-24 05:32:27 UTC] ASSISTANT

**Thinking**

Let me wait more.

**Tool call:** `process`

```json
{
  "action": "poll",
  "sessionId": "amber-daisy",
  "timeout": 180000
}
```

### [2026-04-24 05:32:50 UTC] TOOL RESULT
**Tool:** `process` | **Status:** `completed` | **Exit code:** `0`

+ regex==2026.4.4
 + requests==2.33.1
 + rich==15.0.0
 + safetensors==0.7.0
 + shellingham==1.5.4
 + six==1.17.0
 + sympy==1.14.0
 + threadpoolctl==3.6.0
 + tokenizers==0.22.2
 + torch==2.11.0
 + tqdm==4.67.3
 + transformers==5.6.2
 + triton==3.6.0
 + typer==0.24.2
 + typing-extensions==4.15.0
 + typing-inspection==0.4.2
 + tzdata==2026.1
 + urllib3==2.6.3
 + xxhash==3.6.0
 + yarl==1.23.0

Process exited with code 0.

### [2026-04-24 05:32:55 UTC] ASSISTANT

**Thinking**

Good, auto-round is installed. Now let me check torch and generate the quantization script.

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/python -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\""
}
```

### [2026-04-24 05:32:58 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
torch: 2.11.0+cu130 cuda: False

### [2026-04-24 05:33:01 UTC] ASSISTANT

**Thinking**

CUDA is not available. But wait, the system has an NVIDIA driver that's too old. Let me check if there's a GPU available via nvidia-smi.

**Tool call:** `exec`

```json
{
  "command": "nvidia-smi 2>&1 | head -30"
}
```

### [2026-04-24 05:33:02 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Fri Apr 24 05:33:01 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.20             Driver Version: 570.133.20     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-PCIE-40GB          On  |   00000000:01:00.0 Off |                    0 |
| N/A   28C    P0             30W /  250W |      14MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

### [2026-04-24 05:33:08 UTC] ASSISTANT

**Thinking**

CUDA is available (driver 570.133.20 with CUDA 12.8), but torch says CUDA isn't available due to a version mismatch. The issue is that the installed torch was built with CUDA 13.0 but the driver only supports up to 12.8.

Wait, actually the warning says "found version 12080" which is the driver version (570.133.20 → 570.133.20 / 100 = 5.70.13320). No wait, the driver version is 570.133.20 and the CUDA version is 12.8. 

Let me check: the torch is built with CUDA 13.0 (cu130) but the system driver only supports up to CUDA 12.8. 

I should try upgrading PyTorch to a version that's compatible with CUDA 12.8, or see if I can use the system's existing PyTorch installation instead of relying on the one that came with the venv.

**Tool call:** `exec`

```json
{
  "command": "pip show torch 2>&1; pip3 show torch 2>&1 | head -5"
}
```

### [2026-04-24 05:33:11 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

WARNING: Ignoring invalid distribution -ympy (/usr/local/lib/python3.10/dist-packages)
Name: torch
Version: 2.10.0
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: 
Author: 
Author-email: PyTorch Team <packages@pytorch.org>
License: BSD-3-Clause
Location: /usr/local/lib/python3.10/dist-packages
Requires: cuda-bindings, filelock, fsspec, jinja2, networkx, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-cufile-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-cusparselt-cu12, nvidia-nccl-cu12, nvidia-nvjitlink-cu12, nvidia-nvshmem-cu12, nvidia-nvtx-cu12, sympy, triton, typing-extensions
Required-by: accelerate, auto-round, compressed-tensors, flashinfer-python, quack-kernels, torch_c_dlpack_ext, torchaudio, torchvision, vllm, xgrammar
WARNING: Ignoring invalid distribution -ympy (/usr/local/lib/python3.10/dist-packages)
Name: torch
Version: 2.10.0
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page:

### [2026-04-24 05:33:17 UTC] ASSISTANT

**Thinking**

The system Python has torch 2.10.0 with CUDA 12.6 support. Let me use that instead. I should recreate the venv to use the system site packages properly. Let me check if I can use the system Python with the venv.

Actually, I already created the venv with `--system-site-packages`, so it should have access to the system packages. But the venv has its own torch installation (2.11.0+cu130) which was installed as a dependency of auto-round and doesn't work with CUDA.

Let me try using the system Python directly instead of the venv for the quantization, since the system has torch 2.10.0 with proper CUDA support.


I'll need to check what the auto-round package requires to ensure compatibility.

**Tool call:** `exec`

```json
{
  "command": "python3 -c \"import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\""
}
```

### [2026-04-24 05:33:19 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

torch: 2.10.0+cu128 cuda: True

### [2026-04-24 05:33:47 UTC] ASSISTANT

**Thinking**

The system Python has torch 2.10.0 with CUDA 12.8 support (cuda is available). The venv has torch 2.11.0 with CUDA 13.0 but it can't find the driver (driver too old for CUDA 13).

Let me use the system Python instead, or fix the venv to use the system torch. Since the skill says to use `--system-site-packages`, and torch is already available in the system Python with working CUDA, I should use the system Python for the quantization.

Actually, let me re-create the venv without `--system-site-packages` so it properly uses the system torch from `/usr/local/lib/python3.10/dist-packages`. Or better, just use the system Python directly.


Since the system Python already has all the necessary packages like torch 2.10.0 with CUDA 12.8, transformers, and accelerate, I'll bypass the venv entirely and run the quantization directly with the system Python.

**Tool call:** `write`

```json
{
  "content": "#!/usr/bin/env python3\n\"\"\"\nAuto-Round Quantization Script\nGenerated by auto_quant skill\n\nModel: Qwen/Qwen3-0.6B\nOutput: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\nScheme: W4A16\nIterations: 0 (RTN mode)\nSamples: 128\nFormat: auto_round\n\"\"\"\n\nimport sys\nimport os\nimport time\nimport json\nfrom pathlib import Path\n\n# Ensure we use system Python with working CUDA\nsys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')\n\nstart_time = time.time()\nerrors = []\nsolutions = []\n\nprint(f\"Python: {sys.executable}\")\nprint(f\"PyTorch: checking...\")\n\ntry:\n    import torch\n    print(f\"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}\")\nexcept Exception as e:\n    errors.append(f\"torch import failed: {e}\")\n    print(f\"ERROR: {e}\")\n\ntry:\n    from auto_round import AutoRound\n    print(\"auto_round imported successfully\")\nexcept Exception as e:\n    errors.append(f\"auto_round import failed: {e}\")\n    print(f\"ERROR: {e}\")\n\nif errors:\n    print(\"FATAL: Cannot proceed due to import errors\")\n    sys.exit(1)\n\n# Configuration\nmodel_name_or_path = \"Qwen/Qwen3-0.6B\"\noutput_dir = \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\"\nruntime_output_dir = \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16\"\nscheme = \"W4A16\"\niters = 0  # RTN mode\nnsamples = 128\nformat_str = \"auto_round\"\nnum_gpus = 1\n\nautoround_kwargs = {\"device\": \"cuda\"} if num_gpus <= 1 else {\"device_map\": \"auto\"}\n\nprint(f\"\\n=== Quantization Config ===\")\nprint(f\"Model: {model_name_or_path}\")\nprint(f\"Scheme: {scheme}\")\nprint(f\"Iters: {iters} (RTN)\")\nprint(f\"nsamples: {nsamples}\")\nprint(f\"Format: {format_str}\")\nprint(f\"Device args: {autoround_kwargs}\")\nprint(f\"Output: {output_dir}\")\nprint(f\"=============================\\n\")\n\ntry:\n    print(\"Creating AutoRound instance...\")\n    ar = AutoRound(\n        model_name_or_path,\n        scheme=scheme,\n        iters=iters,\n        nsamples=nsamples,\n        **autoround_kwargs,\n    )\n    print(\"AutoRound instance created\")\n\n    print(\"Starting quantization and save...\")\n    ar.quantize_and_save(output_dir=output_dir, format=format_str)\n    print(\"Quantization complete!\")\n\nexcept Exception as e:\n    errors.append(str(e))\n    import traceback\n    errors.append(traceback.format_exc())\n    print(f\"ERROR during quantization: {e}\")\n    traceback.print_exc()\n\nend_time = time.time()\nduration = end_time - start_time\n\n# Calculate sizes\noriginal_size = None\nquantized_size = None\ncompression_ratio = None\n\ntry:\n    # Get original model size (downloaded from HuggingFace)\n    from transformers import AutoConfig\n    config = AutoConfig.from_pretrained(model_name_or_path)\n    hidden_size = getattr(config, 'hidden_size', 0)\n    num_layers = getattr(config, 'num_hidden_layers', 0)\n    vocab_size = getattr(config, 'vocab_size', 0)\n    # Rough estimate: params in billions * 2 bytes (BF16)\n    # Qwen3-0.6B has ~0.6B params\n    original_size = 0.6 * 2  # ~1.2 GB for BF16\n    print(f\"Estimated original model size: {original_size:.2f} GB\")\nexcept Exception as e:\n    print(f\"Could not estimate original size: {e}\")\n\n# Check quantized output\nquantized_path = Path(output_dir)\nif quantized_path.exists():\n    total_size = 0\n    for f in quantized_path.rglob(\"*\"):\n        if f.is_file():\n            total_size += f.stat().st_size\n    quantized_size = total_size / (1024**3)\n    print(f\"Quantized model size: {quantized_size:.4f} GB\")\n    if original_size:\n        compression_ratio = original_size / quantized_size\n\n# Collect output files\noutput_files = []\nruntime_path = Path(runtime_output_dir)\nif runtime_path.exists():\n    for f in sorted(runtime_path.rglob(\"*\")):\n        if f.is_file() and not f.name.endswith('.pyc') and '__pycache__' not in str(f):\n            output_files.append(str(f))\n\n# Generate quant_summary.json\nsummary = {\n    \"model_id\": \"Qwen/Qwen3-0.6B\",\n    \"scheme\": \"W4A16\",\n    \"method\": \"RTN\",\n    \"export_format\": \"auto_round\",\n    \"device\": \"cuda\",\n    \"num_gpus\": \"1\",\n    \"output_dir\": output_dir,\n    \"runtime_output_dir\": runtime_output_dir,\n    \"quantized_model_dir\": output_dir,\n    \"status\": \"success\" if not errors else \"failed\",\n    \"duration_seconds\": duration,\n    \"original_size_mb\": original_size * 1024 if original_size else None,\n    \"quantized_size_mb\": quantized_size * 1024 if quantized_size else None,\n    \"compression_ratio\": compression_ratio,\n    \"errors\": errors,\n    \"solutions\": solutions,\n    \"output_files\": output_files\n}\n\nsummary_path = Path(runtime_output_dir) / \"quant_summary.json\"\nwith open(summary_path, \"w\") as f:\n    json.dump(summary, f, indent=2)\nprint(f\"\\nSummary written to: {summary_path}\")\nprint(json.dumps(summary, indent=2))\n\nif errors:\n    sys.exit(1)",
  "filePath": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quantize.py"
}
```

### [2026-04-24 05:33:47 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 4665 bytes to /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quantize.py

### [2026-04-24 05:33:52 UTC] ASSISTANT

**Thinking**

Now let me run the quantization script. I'll use the system Python with CUDA support.

**Tool call:** `exec`

```json
{
  "command": "cd /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16 && python3 quantize.py 2>&1",
  "yieldMs": 300000
}
```

### [2026-04-24 05:35:07 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

n_text_dual_encoder.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/visual_bert/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/visual_bert/configuration_visual_bert.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/visual_bert/modeling_visual_bert.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit/configuration_vit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit/image_processing_pil_vit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit/image_processing_vit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit/modeling_vit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit_mae/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit_mae/configuration_vit_mae.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit_mae/modeling_vit_mae.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit_msn/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit_msn/configuration_vit_msn.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vit_msn/modeling_vit_msn.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitdet/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitdet/configuration_vitdet.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitdet/modeling_vitdet.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitmatte/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitmatte/configuration_vitmatte.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitmatte/image_processing_pil_vitmatte.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitmatte/image_processing_vitmatte.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitmatte/modeling_vitmatte.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose/configuration_vitpose.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose/image_processing_pil_vitpose.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose/image_processing_vitpose.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose/modeling_vitpose.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose_backbone/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vits/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vits/configuration_vits.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vits/modeling_vits.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vits/tokenization_vits.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vivit/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vivit/configuration_vivit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vivit/image_processing_vivit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vivit/modeling_vivit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vjepa2/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vjepa2/configuration_vjepa2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vjepa2/modeling_vjepa2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/vjepa2/video_processing_vjepa2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral/configuration_voxtral.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral/modeling_voxtral.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral/modular_voxtral.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral/processing_voxtral.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral_realtime/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral_realtime/configuration_voxtral_realtime.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral_realtime/feature_extraction_voxtral_realtime.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral_realtime/modeling_voxtral_realtime.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral_realtime/modular_voxtral_realtime.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/voxtral_realtime/processing_voxtral_realtime.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2/configuration_wav2vec2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2/feature_extraction_wav2vec2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2/modeling_wav2vec2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2/processing_wav2vec2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2/tokenization_wav2vec2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_bert/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_bert/modular_wav2vec2_bert.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_conformer/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_conformer/configuration_wav2vec2_conformer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_conformer/modular_wav2vec2_conformer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_phoneme/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_with_lm/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wavlm/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wavlm/configuration_wavlm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wavlm/modeling_wavlm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/wavlm/modular_wavlm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/configuration_whisper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/english_normalizer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/feature_extraction_whisper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/generation_whisper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/modeling_whisper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/processing_whisper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/whisper/tokenization_whisper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/x_clip/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/x_clip/configuration_x_clip.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/x_clip/modeling_x_clip.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/x_clip/modular_x_clip.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/x_clip/processing_x_clip.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xcodec/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xcodec/configuration_xcodec.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xcodec/modeling_xcodec.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xglm/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xglm/configuration_xglm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xglm/tokenization_xglm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm/configuration_xlm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm/modeling_xlm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm/tokenization_xlm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta/configuration_xlm_roberta.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta/modeling_xlm_roberta.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta/modular_xlm_roberta.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta/tokenization_xlm_roberta.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta_xl/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlm_roberta_xl/modular_xlm_roberta_xl.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlnet/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlnet/configuration_xlnet.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlnet/tokenization_xlnet.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlstm/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlstm/configuration_xlstm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xlstm/modeling_xlstm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xmod/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xmod/configuration_xmod.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/xmod/modeling_xmod.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yolos/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yolos/configuration_yolos.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yolos/image_processing_pil_yolos.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yolos/image_processing_yolos.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yolos/modeling_yolos.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yolos/modular_yolos.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yoso/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yoso/configuration_yoso.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/yoso/modeling_yoso.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/youtu/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/youtu/configuration_youtu.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/youtu/modeling_youtu.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/youtu/modular_youtu.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zamba/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zamba/configuration_zamba.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zamba/modeling_zamba.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zamba2/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zamba2/configuration_zamba2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zamba2/modeling_zamba2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zamba2/modular_zamba2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zoedepth/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zoedepth/configuration_zoedepth.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zoedepth/image_processing_pil_zoedepth.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zoedepth/image_processing_zoedepth.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/models/zoedepth/modeling_zoedepth.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/monkey_patching.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/optimization.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/any_to_any.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/audio_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/audio_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/automatic_speech_recognition.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/base.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/depth_estimation.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/document_question_answering.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/feature_extraction.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/fill_mask.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/image_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/image_feature_extraction.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/image_segmentation.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/image_text_to_text.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/keypoint_matching.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/mask_generation.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/object_detection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/pt_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/table_question_answering.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/text_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/text_generation.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/text_to_audio.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/token_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/video_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/zero_shot_audio_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/zero_shot_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/zero_shot_image_classification.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pipelines/zero_shot_object_detection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/processing_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/py.typed",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/pytorch_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/auto.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/base.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_aqlm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_auto_round.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_awq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_bitnet.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_bnb_4bit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_bnb_8bit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_compressed_tensors.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_eetq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_fbgemm_fp8.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_finegrained_fp8.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_fouroversix.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_fp_quant.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_gptq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_higgs.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_hqq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_metal.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_mxfp4.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_quanto.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_quark.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_sinq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_spqr.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_torchao.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizer_vptq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/quantizers/quantizers_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/safetensors_conversion.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/testing_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/time_series_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/tokenization_mistral_common.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/tokenization_python.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/tokenization_utils_base.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/tokenization_utils_sentencepiece.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/tokenization_utils_tokenizers.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/trainer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/trainer_callback.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/trainer_jit_checkpoint.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/trainer_optimizer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/trainer_pt_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/trainer_seq2seq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/trainer_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/training_args.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/training_args_seq2seq.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/attention_visualizer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/auto_docstring.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/backbone_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/chat_parsing_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/chat_template_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/constants.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/deprecation.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/doc.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_detectron2_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_mistral_common_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_music_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_pt_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_sentencepiece_and_tokenizers_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_speech_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_timm_and_torchvision_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_tokenizers_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_torchaudio_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_torchvision_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/dummy_vision_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/generic.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/hp_naming.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/hub.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/import_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/kernel_config.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/loading_report.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/logging.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/metrics.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/network_logging.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/notebook.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/output_capturing.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/peft_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/pytest_helpers.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/quantization_config.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/sentencepiece_model_pb2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/sentencepiece_model_pb2_new.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/type_validators.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/utils/versions.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/video_processing_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers/video_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/entry_points.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/transformers-5.6.2.dist-info/top_level.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/FileCheck",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/_C/libproton.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/_C/libtriton/linear_layout.pyi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/_C/libtriton.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/_filecheck.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/_internal_testing.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/compiler.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/driver.c",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/driver.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_channel_descriptor.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_device_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_atomic.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_common.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_gl_interop.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_runtime_pt_api.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_unsafe_atomics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_hip_vector_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_math_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_surface_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_warp_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/amd_warp_sync_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/device_library_decls.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/hip_assert.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/hip_fp16_math_fwd.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/hip_ldg.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/hip_prof_str.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/hip_runtime_prof.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/host_defines.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/math_fwd.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/ockl_image.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/texture_fetch_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/amd_detail/texture_indirect_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/channel_descriptor.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/driver_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/hip_common.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/hip_deprecated.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/hip_runtime.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/hip_runtime_api.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/hip_texture_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/hip_vector_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/hip_version.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/library_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/linker_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/surface_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hip/texture_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hipblas-common/hipblas-common.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hsa/amd_hsa_kernel_code.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hsa/hsa.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hsa/hsa_ext_amd.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hsa/hsa_ext_image.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_loader.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/hsa/hsa_ven_amd_pc_sampling.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/roctracer/ext/prof_protocol.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/roctracer/roctracer.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/roctracer/roctracer_ext.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/roctracer/roctracer_hip.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/roctracer/roctracer_roctx.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/include/roctracer/roctx.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/lib/asanrtl.bc",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ockl.bc",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/amd/lib/ocml.bc",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/compiler.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/driver.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/bin/cuobjdump",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/bin/nvdisasm",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas-blackwell",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/compiler.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/driver.c",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/driver.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/Openacc/cupti_openacc.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/Openmp/cupti_openmp.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/Openmp/omp-tools.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/builtin_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/channel_descriptor.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/common_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/async.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_reduce.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/coalesced_scan.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/driver_abi.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/functional.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/helpers.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/info.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/invoke.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/memory.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/partitioning.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/reduce.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/scan.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/details/sync.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/memcpy_async.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/reduce.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups/scan.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cooperative_groups.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/common_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/cudacc_ext.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/device_double_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/device_double_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/device_fp128_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/device_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/device_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/func_macro.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/host_config.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/host_defines.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/host_runtime.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/math_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/math_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/mma.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/mma.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/nvfunctional",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_100_rt.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_100_rt.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_70_rt.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_80_rt.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/sm_90_rt.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/crt/storage_class.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuComplex.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaEGL.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaEGLTypedefs.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaGL.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaGLTypedefs.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaProfilerTypedefs.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaTypedefs.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaVDPAU.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudaVDPAUTypedefs.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_awbarrier.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_awbarrier_helpers.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_awbarrier_primitives.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_bf16.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_bf16.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_device_runtime_api.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_egl_interop.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp16.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp16.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp4.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp4.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp6.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp6.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp8.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_fp8.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_gl_interop.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_occupancy.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_pipeline.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_pipeline_helpers.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_pipeline_primitives.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_runtime.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_runtime_api.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_stdint.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_surface_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_texture_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cuda_vdpau_interop.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cudart_platform.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_activity.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_activity_deprecated.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_callbacks.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_checkpoint.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_common.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_driver_cbid.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_events.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_metrics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_nvtx_cbid.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_pcsampling.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_pcsampling_util.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_pmsampling.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_profiler_host.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_profiler_target.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_range_profiler.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_result.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_runtime_cbid.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_sass_metrics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_target.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/cupti_version.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/device_atomic_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/device_atomic_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/device_double_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/device_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/device_launch_parameters.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/device_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/driver_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/driver_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/fatbinary_section.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_cudaGL_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_cudaVDPAU_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_cuda_gl_interop_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_cuda_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_cuda_runtime_api_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_cuda_vdpau_interop_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_cudart_removed_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/generated_nvtx_meta.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/host_config.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/host_defines.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/library_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/math_constants.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/math_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/mma.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/nvPTXCompiler.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/nvfunctional",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/nvperf_common.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/nvperf_cuda_host.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/nvperf_host.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/nvperf_target.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_20_atomic_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_20_intrinsics.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_30_intrinsics.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_32_atomic_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_32_intrinsics.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_35_atomic_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_35_intrinsics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_60_atomic_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/sm_61_intrinsics.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/surface_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/surface_indirect_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/surface_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/texture_fetch_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/texture_indirect_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/texture_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/vector_functions.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/vector_functions.hpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/include/vector_types.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libcheckpoint.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.12",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.2025.1.1",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libcupti_static.a",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_host_static.a",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libnvperf_target.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libpcsamplingutil.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/libdevice.10.bc",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/compiler/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/compiler/code_generator.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/compiler/compiler.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/compiler/errors.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/compiler/make_launcher.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/errors.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/_compiler.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/_runtime.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/amd/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/amd/gfx1250.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/_core.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/_layouts.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/_math.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/_semantic.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/_standard.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/_layouts.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/_ops.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/cdna3/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/cdna4/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/cdna4/async_copy.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/gfx1250/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/gfx1250/async_copy.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/gfx1250/mbarrier.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/gfx1250/tdm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/rdna3/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/amd/rdna4/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/extra/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/ampere/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/ampere/async_copy.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/ampere/mbarrier.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/blackwell/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/blackwell/float2.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/blackwell/tma.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/hopper/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/hopper/mbarrier.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/language/nvidia/hopper/tma.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/nvidia/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/nvidia/blackwell.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/experimental/gluon/nvidia/hopper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/instrumentation/libGPUInstrumentationTestLib.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/instrumentation/libPrintLoadStoreMemSpaces.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/knobs.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/core.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/cuda/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/cuda/gdc.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/cuda/libdevice.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/cuda/utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/hip/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/hip/libdevice.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/hip/utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/extra/libdevice.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/math.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/random.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/semantic.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/standard.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/language/target_info.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/context.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/flags.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/hooks/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/hooks/hook.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/hooks/instrumentation.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/hooks/launch.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/language.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/mode.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/profile.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/proton.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/scope.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/specs.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/state.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/profiler/viewer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/_allocation.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/_async_compile.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/autotuner.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/build.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/cache.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/driver.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/errors.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/interpreter.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/runtime/jit.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/testing.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/build_extern.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/compile.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/disasm.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/extra/cuda/compile.c",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/extra/cuda/compile.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/extra/hip/compile.cpp",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/extra/hip/compile.h",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/link.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/mxfp.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/ragged_tma.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/tensor_descriptor.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/triton_to_gluon_translater/translator.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton/tools/triton_to_gluon_translater/translator_helpers.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/entry_points.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/triton-3.6.0.dist-info/top_level.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/__main__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/_completion_classes.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/_completion_shared.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/_types.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/_typing.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/cli.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/colors.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/completion.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/core.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/main.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/models.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/params.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/py.typed",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/rich_utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/testing.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer/utils.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer-0.24.2.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer-0.24.2.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer-0.24.2.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer-0.24.2.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer-0.24.2.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer-0.24.2.dist-info/entry_points.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typer-0.24.2.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_extensions-4.15.0.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_extensions-4.15.0.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_extensions-4.15.0.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_extensions-4.15.0.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_extensions-4.15.0.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_extensions-4.15.0.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_extensions.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection/introspection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection/py.typed",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection/typing_objects.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection/typing_objects.pyi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection-0.4.2.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection-0.4.2.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection-0.4.2.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection-0.4.2.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection-0.4.2.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/typing_inspection-0.4.2.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Abidjan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Accra",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Addis_Ababa",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Algiers",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Asmara",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Asmera",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Bamako",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Bangui",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Banjul",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Bissau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Blantyre",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Brazzaville",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Bujumbura",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Cairo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Casablanca",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Ceuta",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Conakry",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Dakar",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Dar_es_Salaam",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Djibouti",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Douala",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/El_Aaiun",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Freetown",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Gaborone",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Harare",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Johannesburg",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Juba",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Kampala",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Khartoum",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Kigali",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Kinshasa",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Lagos",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Libreville",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Lome",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Luanda",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Lubumbashi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Lusaka",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Malabo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Maputo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Maseru",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Mbabane",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Mogadishu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Monrovia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Nairobi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Ndjamena",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Niamey",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Nouakchott",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Ouagadougou",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Porto-Novo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Sao_Tome",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Timbuktu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Tripoli",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Tunis",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/Windhoek",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Africa/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Adak",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Anchorage",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Anguilla",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Antigua",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Araguaina",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Buenos_Aires",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Catamarca",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/ComodRivadavia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Cordoba",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Jujuy",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/La_Rioja",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Mendoza",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Rio_Gallegos",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Salta",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/San_Juan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/San_Luis",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Tucuman",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/Ushuaia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Argentina/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Aruba",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Asuncion",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Atikokan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Atka",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Bahia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Bahia_Banderas",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Barbados",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Belem",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Belize",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Blanc-Sablon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Boa_Vista",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Bogota",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Boise",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Buenos_Aires",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Cambridge_Bay",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Campo_Grande",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Cancun",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Caracas",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Catamarca",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Cayenne",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Cayman",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Chicago",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Chihuahua",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Ciudad_Juarez",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Coral_Harbour",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Cordoba",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Costa_Rica",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Coyhaique",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Creston",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Cuiaba",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Curacao",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Danmarkshavn",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Dawson",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Dawson_Creek",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Denver",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Detroit",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Dominica",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Edmonton",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Eirunepe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/El_Salvador",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Ensenada",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Fort_Nelson",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Fort_Wayne",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Fortaleza",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Glace_Bay",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Godthab",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Goose_Bay",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Grand_Turk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Grenada",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Guadeloupe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Guatemala",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Guayaquil",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Guyana",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Halifax",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Havana",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Hermosillo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Indianapolis",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Knox",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Marengo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Petersburg",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Tell_City",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Vevay",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Vincennes",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/Winamac",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indiana/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Indianapolis",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Inuvik",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Iqaluit",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Jamaica",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Jujuy",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Juneau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Kentucky/Louisville",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Kentucky/Monticello",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Kentucky/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Knox_IN",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Kralendijk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/La_Paz",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Lima",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Los_Angeles",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Louisville",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Lower_Princes",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Maceio",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Managua",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Manaus",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Marigot",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Martinique",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Matamoros",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Mazatlan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Mendoza",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Menominee",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Merida",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Metlakatla",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Mexico_City",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Miquelon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Moncton",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Monterrey",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Montevideo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Montreal",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Montserrat",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Nassau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/New_York",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Nipigon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Nome",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Noronha",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/North_Dakota/Beulah",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/North_Dakota/Center",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/North_Dakota/New_Salem",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/North_Dakota/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Nuuk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Ojinaga",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Panama",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Pangnirtung",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Paramaribo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Phoenix",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Port-au-Prince",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Port_of_Spain",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Porto_Acre",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Porto_Velho",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Puerto_Rico",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Punta_Arenas",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Rainy_River",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Rankin_Inlet",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Recife",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Regina",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Resolute",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Rio_Branco",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Rosario",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Santa_Isabel",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Santarem",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Santiago",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Santo_Domingo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Sao_Paulo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Scoresbysund",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Shiprock",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Sitka",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/St_Barthelemy",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/St_Johns",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/St_Kitts",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/St_Lucia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/St_Thomas",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/St_Vincent",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Swift_Current",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Tegucigalpa",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Thule",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Thunder_Bay",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Tijuana",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Toronto",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Tortola",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Vancouver",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Virgin",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Whitehorse",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Winnipeg",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Yakutat",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/Yellowknife",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/America/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Casey",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Davis",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/DumontDUrville",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Macquarie",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Mawson",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/McMurdo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Palmer",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Rothera",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/South_Pole",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Syowa",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Troll",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/Vostok",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Antarctica/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Arctic/Longyearbyen",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Arctic/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Aden",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Almaty",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Amman",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Anadyr",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Aqtau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Aqtobe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Ashgabat",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Ashkhabad",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Atyrau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Baghdad",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Bahrain",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Baku",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Bangkok",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Barnaul",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Beirut",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Bishkek",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Brunei",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Calcutta",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Chita",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Choibalsan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Chongqing",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Chungking",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Colombo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Dacca",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Damascus",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Dhaka",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Dili",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Dubai",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Dushanbe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Famagusta",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Gaza",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Harbin",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Hebron",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Ho_Chi_Minh",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Hong_Kong",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Hovd",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Irkutsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Istanbul",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Jakarta",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Jayapura",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Jerusalem",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kabul",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kamchatka",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Karachi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kashgar",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kathmandu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Katmandu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Khandyga",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kolkata",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Krasnoyarsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kuala_Lumpur",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kuching",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Kuwait",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Macao",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Macau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Magadan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Makassar",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Manila",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Muscat",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Nicosia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Novokuznetsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Novosibirsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Omsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Oral",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Phnom_Penh",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Pontianak",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Pyongyang",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Qatar",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Qostanay",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Qyzylorda",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Rangoon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Riyadh",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Saigon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Sakhalin",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Samarkand",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Seoul",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Shanghai",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Singapore",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Srednekolymsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Taipei",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Tashkent",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Tbilisi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Tehran",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Tel_Aviv",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Thimbu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Thimphu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Tokyo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Tomsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Ujung_Pandang",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Ulaanbaatar",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Ulan_Bator",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Urumqi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Ust-Nera",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Vientiane",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Vladivostok",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Yakutsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Yangon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Yekaterinburg",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/Yerevan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Asia/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Azores",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Bermuda",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Canary",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Cape_Verde",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Faeroe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Faroe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Jan_Mayen",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Madeira",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Reykjavik",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/South_Georgia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/St_Helena",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/Stanley",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Atlantic/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/ACT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Adelaide",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Brisbane",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Broken_Hill",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Canberra",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Currie",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Darwin",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Eucla",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Hobart",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/LHI",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Lindeman",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Lord_Howe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Melbourne",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/NSW",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/North",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Perth",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Queensland",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/South",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Sydney",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Tasmania",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Victoria",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/West",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/Yancowinna",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Australia/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Brazil/Acre",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Brazil/DeNoronha",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Brazil/East",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Brazil/West",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Brazil/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/CET",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/CST6CDT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Atlantic",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Central",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Eastern",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Mountain",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Newfoundland",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Pacific",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Saskatchewan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/Yukon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Canada/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Chile/Continental",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Chile/EasterIsland",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Chile/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Cuba",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/EET",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/EST",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/EST5EDT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Egypt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Eire",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+0",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+1",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+10",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+11",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+12",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+2",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+3",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+4",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+5",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+6",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+7",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+8",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT+9",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-0",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-1",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-10",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-11",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-12",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-13",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-14",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-2",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-3",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-4",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-5",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-6",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-7",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-8",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT-9",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/GMT0",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/Greenwich",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/UCT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/UTC",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/Universal",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/Zulu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Etc/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Amsterdam",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Andorra",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Astrakhan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Athens",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Belfast",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Belgrade",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Berlin",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Bratislava",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Brussels",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Bucharest",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Budapest",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Busingen",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Chisinau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Copenhagen",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Dublin",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Gibraltar",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Guernsey",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Helsinki",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Isle_of_Man",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Istanbul",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Jersey",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Kaliningrad",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Kiev",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Kirov",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Kyiv",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Lisbon",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Ljubljana",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/London",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Luxembourg",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Madrid",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Malta",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Mariehamn",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Minsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Monaco",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Moscow",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Nicosia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Oslo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Paris",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Podgorica",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Prague",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Riga",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Rome",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Samara",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/San_Marino",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Sarajevo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Saratov",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Simferopol",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Skopje",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Sofia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Stockholm",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Tallinn",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Tirane",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Tiraspol",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Ulyanovsk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Uzhgorod",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Vaduz",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Vatican",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Vienna",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Vilnius",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Volgograd",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Warsaw",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Zagreb",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Zaporozhye",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/Zurich",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Europe/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Factory",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/GB",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/GB-Eire",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/GMT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/GMT+0",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/GMT-0",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/GMT0",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Greenwich",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/HST",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Hongkong",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Iceland",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Antananarivo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Chagos",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Christmas",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Cocos",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Comoro",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Kerguelen",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Mahe",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Maldives",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Mauritius",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Mayotte",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/Reunion",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Indian/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Iran",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Israel",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Jamaica",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Japan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Kwajalein",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Libya",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/MET",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/MST",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/MST7MDT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Mexico/BajaNorte",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Mexico/BajaSur",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Mexico/General",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Mexico/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/NZ",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/NZ-CHAT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Navajo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/PRC",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/PST8PDT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Apia",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Auckland",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Bougainville",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Chatham",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Chuuk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Easter",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Efate",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Enderbury",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Fakaofo",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Fiji",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Funafuti",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Galapagos",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Gambier",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Guadalcanal",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Guam",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Honolulu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Johnston",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Kanton",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Kiritimati",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Kosrae",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Kwajalein",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Majuro",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Marquesas",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Midway",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Nauru",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Niue",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Norfolk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Noumea",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Pago_Pago",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Palau",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Pitcairn",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Pohnpei",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Ponape",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Port_Moresby",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Rarotonga",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Saipan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Samoa",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Tahiti",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Tarawa",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Tongatapu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Truk",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Wake",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Wallis",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/Yap",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Pacific/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Poland",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Portugal",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/ROC",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/ROK",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Singapore",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Turkey",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/UCT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Alaska",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Aleutian",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Arizona",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Central",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/East-Indiana",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Eastern",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Hawaii",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Indiana-Starke",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Michigan",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Mountain",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Pacific",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/Samoa",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/US/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/UTC",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Universal",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/W-SU",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/WET",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/Zulu",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/iso3166.tab",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/leapseconds",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/tzdata.zi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/zone.tab",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/zone1970.tab",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zoneinfo/zonenow.tab",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata/zones",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/licenses/licenses/LICENSE_APACHE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/tzdata-2026.1.dist-info/top_level.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/_base_connection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/_collections.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/_request_methods.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/_version.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/connection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/connectionpool.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/emscripten/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/emscripten/connection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/emscripten/emscripten_fetch_worker.js",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/emscripten/fetch.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/emscripten/request.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/emscripten/response.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/pyopenssl.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/contrib/socks.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/exceptions.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/fields.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/filepost.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/http2/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/http2/connection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/http2/probe.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/poolmanager.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/py.typed",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/response.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/connection.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/proxy.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/request.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/response.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/retry.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/ssl_.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/ssl_match_hostname.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/ssltransport.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/timeout.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/url.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/util.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3/util/wait.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3-2.6.3.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3-2.6.3.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3-2.6.3.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3-2.6.3.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3-2.6.3.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/urllib3-2.6.3.dist-info/licenses/LICENSE.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv/__main__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv/_find_uv.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv/py.typed",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/licenses/LICENSE-APACHE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/licenses/LICENSE-MIT",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/uv-0.11.7.dist-info/sboms/uv.cyclonedx.json",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash/__init__.pyi",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash/_xxhash.cpython-310-x86_64-linux-gnu.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash/py.typed",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash/version.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash-3.6.0.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash-3.6.0.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash-3.6.0.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash-3.6.0.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash-3.6.0.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash-3.6.0.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/xxhash-3.6.0.dist-info/top_level.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/_yaml.cpython-310-x86_64-linux-gnu.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/composer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/constructor.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/cyaml.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/dumper.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/emitter.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/error.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/events.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/loader.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/nodes.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/parser.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/reader.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/representer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/resolver.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/scanner.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/serializer.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yaml/tokens.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/__init__.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_parse.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_path.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_query.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_quoters.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_quoting.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_quoting_c.cpython-310-x86_64-linux-gnu.so",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_quoting_c.pyx",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_quoting_py.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/_url.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl/py.typed",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/INSTALLER",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/METADATA",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/RECORD",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/REQUESTED",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/WHEEL",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/licenses/LICENSE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/licenses/NOTICE",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/lib/python3.10/site-packages/yarl-1.23.0.dist-info/top_level.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/pyvenv.cfg",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/share/man/man1/isympy.1"
  ]
}

### [2026-04-24 05:35:18 UTC] ASSISTANT

**Thinking**

The output seems truncated, but I can see it's listing many files from transformers, triton, etc. - this looks like the output of running `python3 quantize.py` but it got cut off. Let me check if there's a quant_summary.json and the output directory.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/ 2>&1"
}
```

### [2026-04-24 05:35:18 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 3632
drwxr-xr-x 4 root root    4096 Apr 24 05:35 .
drwxr-xr-x 3 root root    4096 Apr 24 05:27 ..
drwxr-xr-x 2 root root    4096 Apr 24 05:27 logs
-rw-r--r-- 1 root root 3687794 Apr 24 05:35 quant_summary.json
-rw-r--r-- 1 root root    4665 Apr 24 05:33 quantize.py
-rw-r--r-- 1 root root    1004 Apr 24 05:27 request.json
drwxr-xr-x 6 root root    4096 Apr 24 05:32 venv

### [2026-04-24 05:35:23 UTC] ASSISTANT

**Thinking**

The quantize.py ran but let me check the actual output - it seems like it generated a summary. Let me look at the summary to see if it succeeded.

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quant_summary.json 2>&1 | head -30"
}
```

### [2026-04-24 05:35:23 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "runtime_output_dir": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "status": "success",
  "duration_seconds": 71.89238858222961,
  "original_size_mb": 1228.8,
  "quantized_size_mb": 526.0521745681763,
  "compression_ratio": 2.3358899732877116,
  "errors": [],
  "solutions": [],
  "output_files": [
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/logs/auto.log",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/logs/quant_prompt.txt",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quantize.py",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/request.json",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/.lock",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/Activate.ps1",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/accelerate",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/accelerate-config",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/accelerate-estimate-memory",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/accelerate-launch",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/accelerate-merge-weights",
    "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/venv/bin/activate",

### [2026-04-24 05:35:31 UTC] ASSISTANT

**Thinking**

The quantization was successful! Let me verify the output model directory to make sure the model files are there.

**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/ 2>&1"
}
```

### [2026-04-24 05:35:31 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

total 538708
drwxr-xr-x  2 root root      4096 Apr 24 05:35 .
drwxr-xr-x 14 root root      4096 Apr 24 05:27 ..
-rw-r--r--  1 root root      4168 Apr 24 05:35 chat_template.jinja
-rw-r--r--  1 root root      1657 Apr 24 05:35 config.json
-rw-r--r--  1 root root       213 Apr 24 05:35 generation_config.json
-rw-r--r--  1 root root 540176136 Apr 24 05:35 model.safetensors
-rw-r--r--  1 root root       196 Apr 24 05:35 quantization_config.json
-rw-r--r--  1 root root  11422650 Apr 24 05:35 tokenizer.json
-rw-r--r--  1 root root       665 Apr 24 05:35 tokenizer_config.json

### [2026-04-24 05:35:42 UTC] ASSISTANT

**Thinking**

The quantization was successful. The model files are in `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/`:
- model.safetensors (540MB - the quantized model)
- config.json, tokenizer files, etc.

And the runtime artifacts including quant_summary.json are in `/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/`.

The quantization succeeded with:
- Status: success
- Original size: ~1229 MB (BF16)
- Quantized size: ~526 MB (W4A16)
- Compression ratio: ~2.34x
- Duration: ~72 seconds
- Method: RTN (iters=0)

Let me update the output_files in quant_summary.json to only include relevant files (not the venv which has thousands of files). Let me generate a cleaner summary.



Quantization completed successfully. Let me verify and clean up the output files list in the summary:


**Tool call:** `exec`

```json
{
  "command": "ls -la /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/ 2>&1 | grep -v '^total'"
}
```

### [2026-04-24 05:35:42 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

drwxr-xr-x  2 root root      4096 Apr 24 05:35 .
drwxr-xr-x 14 root root      4096 Apr 24 05:27 ..
-rw-r--r--  1 root root      4168 Apr 24 05:35 chat_template.jinja
-rw-r--r--  1 root root      1657 Apr 24 05:35 config.json
-rw-r--r--  1 root root       213 Apr 24 05:35 generation_config.json
-rw-r--r--  1 root root 540176136 Apr 24 05:35 model.safetensors
-rw-r--r--  1 root root       196 Apr 24 05:35 quantization_config.json
-rw-r--r--  1 root root  11422650 Apr 24 05:35 tokenizer.json
-rw-r--r--  1 root root       665 Apr 24 05:35 tokenizer_config.json

### [2026-04-24 05:35:54 UTC] ASSISTANT

**Thinking**

The quantization was successful. The output directory contains all the expected files:
- model.safetensors (the quantized model weights, ~515 MB)
- config.json, generation_config.json, tokenizer files
- quantization_config.json (AutoRound config)
- chat_template.jinja

The compression ratio is ~2.34x (from ~1.2GB BF16 to ~526MB W4A16).

Let me update the quant_summary.json to only list the key output files (not the entire venv).



Let me update the summary with cleaner output files:


**Tool call:** `write`

```json
{
  "content": "{\n  \"model_id\": \"Qwen/Qwen3-0.6B\",\n  \"scheme\": \"W4A16\",\n  \"method\": \"RTN\",\n  \"export_format\": \"auto_round\",\n  \"device\": \"cuda\",\n  \"num_gpus\": \"1\",\n  \"output_dir\": \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\",\n  \"runtime_output_dir\": \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16\",\n  \"quantized_model_dir\": \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\",\n  \"status\": \"success\",\n  \"duration_seconds\": 71.89,\n  \"original_size_mb\": 1228.8,\n  \"quantized_size_mb\": 526.05,\n  \"compression_ratio\": 2.34,\n  \"errors\": [],\n  \"solutions\": [],\n  \"output_files\": [\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/model.safetensors\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantization_config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/generation_config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/tokenizer.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/tokenizer_config.json\",\n    \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/chat_template.jinja\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quantize.py\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quant_summary.json\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/logs/auto.log\",\n    \"/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/logs/quant_prompt.txt\"\n  ]\n}",
  "filePath": "/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quant_summary.json"
}
```

### [2026-04-24 05:35:54 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 1538 bytes to /root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quant_summary.json

### [2026-04-24 05:36:02 UTC] ASSISTANT

**Thinking**

Done! Let me provide a summary to the user.



**Quantization Complete!**

| Field | Value |
|-------|-------|
| **Model** | Qwen/Qwen3-0.6B |
| **Scheme** | W4A16 (INT4 weight, FP16 activation) |
| **Method** | RTN (Round-To-Nearest) |
| **Export Format** | auto_round |
| **Status** | ✅ success |
| **Duration** | 71.89 seconds |
| **Original Size** | 1228.8 MB (BF16) |
| **Quantized Size** | 526.05 MB |
| **Compression Ratio** | 2.34x |

**Output files:**
- `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/model.safetensors` (515 MB)
- `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/config.json`
- `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantization_config.json`
- `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/tokenizer.json`
- `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/tokenizer_config.json`
- `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/chat_template.jinja`
- `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/generation_config.json`

**Runtime artifacts:**
- `/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quantize.py`
- `/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/quant_summary.json`
- `/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16/logs/`
