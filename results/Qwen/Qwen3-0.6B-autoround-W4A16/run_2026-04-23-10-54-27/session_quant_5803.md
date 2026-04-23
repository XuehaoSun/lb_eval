# Session: autoeval_quant_5803

- **Session ID:** `autoeval_quant_5803`
- **Timestamp:** 2026-04-23 10:50:33 UTC
- **Working Dir:** `/root/.openclaw/workspace`

## Step 1: Quantization

### [2026-04-23 10:50:33 UTC] USER

You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: /root/.openclaw/workspace/skills/auto_quant/SKILL.md

Model: Qwen/Qwen3-0.6B
Quantization: W4A16 / RTN
Export format: auto_round
Quantized Model Output directory: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16
Runtime device: cuda
Num gpus: 1

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.

IMPORTANT - After quantization completes (success or failure), you MUST produce:

/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quant_summary.json - structured summary:
{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "original_size_mb": <float or null>,
  "quantized_size_mb": <float or null>,
  "compression_ratio": <float or null>,
  "errors": [<list of error strings>],
  "solutions": [<list of solution strings>],
  "output_files": [<list of file paths in output_dir>]
}

Write as valid JSON. If quantization fails, still write quant_summary.json with status=failed.

### [2026-04-23 10:50:41 UTC] ASSISTANT

**Thinking**

Let me start by reading the SKILL.md file to understand the quantization instructions, then proceed with the task.

**Tool call:** `read`

```json
{
  "path": "/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
}
```

### [2026-04-23 10:50:41 UTC] TOOL RESULT
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
| `device_map` | Device to use | No | `0` (GPU 0) |

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
    venv_pip  = f"{venv_path}/bin/pip"
    venv_py   = f"{venv_path}/bin/python"
    print(f"✅ Reusing shared venv from auto_run: {venv_path}")
    # → Skip Steps 2.1-2.2, go directly to Step 3
else:
    print("ℹ️  No model_info.json found, will create standalone venv in output_dir")
    venv_path = "{output_dir}/venv"
    venv_pip  = f"{venv_path}/bin/pip"
    venv_py   = f"{venv_path}/bin/python"
    # → Continue with Steps 2.1-2.2 below
```

### Create Isolated Virtual Environment

**Only run the steps below if model_info.json was NOT found above.**

```bash
# Create output directory
mkdir -p {output_dir}
mkdir -p {output_dir}/logs

# Create virtual environment
python3 -m venv {output_dir}/venv

# Activate and upgrade pip
{output_dir}/venv/bin/pip install -U pip setuptools wheel
```

### Install Auto-Round

**Option A: From local source (editable - allows source modifications)**
```bash
# Copy source if needed
cp -r /storage/lkk/auto-round {output_dir}/auto-round-src

# Install in editable mode
{output_dir}/venv/bin/pip install -e {output_dir}/auto-round-src
```

**Option B: From GitHub**
```bash
{output_dir}/venv/bin/pip install git+https://github.com/intel/auto-round.git
```

**Option C: From PyPI**
```bash
{output_dir}/venv/bin/pip install auto-round
```

### Install Additional Dependencies

```bash
# Core dependencies (usually auto-installed)
{output_dir}/venv/bin/pip install torch transformers accelerate datasets

# For specific formats
{output_dir}/venv/bin/pip install compressed-tensors  # For better compression
{output_dir}/venv/bin/pip install llama-cpp-python   # For GGUF inference
{output_dir}/venv/bin/pip install gptqmodel           # For GPTQ inference
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

# Optional: specify device
device_map = "0"  # or "auto", "cuda", "cpu", "0,1,2,3"

print(f"Loading model: {{model_name_or_path}}")
print(f"Scheme: {{scheme}}")
print(f"Iters: {{iters}}")
print(f"nsamples: {{nsamples}}")
print(f"Format: {{format_str}}")

# Create AutoRound instance
ar = AutoRound(
    model_name_or_path,
    scheme=scheme,
    iters=iters,
    nsamples=nsamples,
    device_map=device_map,
    # Optional optimizations
    # enable_torch_compile=True,  # Faster quantization (PyTorch 2.6+)
    # low_gpu_mem_usage=True,    # Lower VRAM, ~30% slower
    # disable_opt_rtn=True,      # For GGUF: use pure RTN
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
{venv}/bin/pip install --upgrade auto-round

# Or from source
{venv}/bin/pip install -e /path/to/auto-round --force-reinstall

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
    device_map="auto",            # Let AutoRound handle device
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
    device_map="0,1,2,3",  # Multiple GPUs
```

**CLI alternatives:**
```bash
# Use light recipe
auto-round-light --model ... --scheme W4A16

# Low memory mode
auto-round --model ... --scheme W4A16 --low_gpu_mem_usage
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
{venv}/bin/pip install transformers>=4.35.0
{venv}/bin/pip install transformers==4.40.0

# Upgrade torch (PyTorch 2.6+ recommended for auto-round)
{venv}/bin/pip install torch>=2.5.0
{venv}/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch

# Reinstall auto-round dependencies
{venv}/bin/pip install -r /path/to/auto-round/requirements.txt
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
pip show torch transformers auto-round
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
python3 -m venv /storage/quantized/llama-8b-w4a16/venv
/storage/quantized/llama-8b-w4a16/venv/bin/pip install -U pip setuptools wheel
/storage/quantized/llama-8b-w4a16/venv/bin/pip install -e /storage/lkk/auto-round
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
    device_map="0",
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
| Multiple GPUs | `device_map="0,1,2,3"` |
| 2-bit quantization | Use `W2A16` + best recipe |
| MXFP4/MXFP8 | Research only, no kernel |

---

## Notes

- **VRAM Requirements**: ~10-20GB for 8B models, ~20-40GB for 70B models
- **Time**: ~10-20 min for 8B model with default settings
- **MXFP4/MXFP8**: Research features, no real inference kernels
- **Gated models**: May require HuggingFace token
- **Editable install**: Install auto-round in editable mode to modify source if needed


### [2026-04-23 10:50:47 UTC] ASSISTANT

**Thinking**

Now I understand the task. Let me proceed with quantizing the Qwen3-0.6B model using Intel Auto-Round with W4A16 scheme and RTN method (iters=0).

Let me:
1. Create the output directory
2. Set up the virtual environment
3. Install auto-round
4. Create and run the quantization script
5. Generate the summary JSON

Let me start by checking if there's a shared workspace with model_info.json.

**Tool call:** `exec`

```json
{
  "command": "mkdir -p /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/logs"
}
```

**Tool call:** `exec`

```json
{
  "command": "python3 -m venv --system-site-packages /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv"
}
```

### [2026-04-23 10:50:47 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-23 10:50:50 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(no output)

### [2026-04-23 10:50:54 UTC] ASSISTANT

**Thinking**

Good, venv created. Now let me install auto-round and dependencies.

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv/bin/pip install -U pip setuptools wheel 2>&1 | tail -5"
}
```

### [2026-04-23 10:50:59 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Attempting uninstall: pip
    Found existing installation: pip 22.0.2
    Uninstalling pip-22.0.2:
      Successfully uninstalled pip-22.0.2
Successfully installed pip-26.0.1 setuptools-82.0.1 wheel-0.47.0

### [2026-04-23 10:51:04 UTC] ASSISTANT

**Thinking**

Good. Now let me install auto-round from PyPI.

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv/bin/pip install auto-round 2>&1 | tail -10"
}
```

### [2026-04-23 10:51:05 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic->auto-round) (0.7.0)
Requirement already satisfied: pydantic-core==2.46.3 in /usr/local/lib/python3.10/dist-packages (from pydantic->auto-round) (2.46.3)
Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.10/dist-packages (from pydantic->auto-round) (0.4.2)
Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.10/dist-packages (from typer->transformers>=4.38->auto-round) (8.3.2)
Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer->transformers>=4.38->auto-round) (1.5.4)
Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.10/dist-packages (from typer->transformers>=4.38->auto-round) (15.0.0)
Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.10/dist-packages (from typer->transformers>=4.38->auto-round) (0.0.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=12.3.0->typer->transformers>=4.38->auto-round) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=12.3.0->typer->transformers>=4.38->auto-round) (2.20.0)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->transformers>=4.38->auto-round) (0.1.2)

### [2026-04-23 10:51:08 UTC] ASSISTANT

**Thinking**

Looks like auto-round is installed. Let me check what version and verify the import works.

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv/bin/pip show auto-round 2>&1"
}
```

### [2026-04-23 10:51:08 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Name: auto-round
Version: 0.12.2
Summary: Repository of AutoRound: Advanced Weight-Only Quantization Algorithm for LLMs
Home-page: https://github.com/intel/auto-round
Author: Intel AIPT Team
Author-email: wenhua.cheng@intel.com, weiwei1.zhang@intel.com, heng.guo@intel.com
License: Apache 2.0
Location: /usr/local/lib/python3.10/dist-packages
Requires: accelerate, datasets, numpy, py-cpuinfo, pydantic, threadpoolctl, torch, tqdm, transformers
Required-by:

### [2026-04-23 10:51:25 UTC] ASSISTANT

**Thinking**

Good, auto-round 0.12.2 is installed. Now let me write and run the quantization script. For RTN, we use iters=0.

Now let me write the quantization script:


**Tool call:** `write`

```json
{
  "path": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantize_script.py",
  "content": "#!/usr/bin/env python3\n\"\"\"\nAuto-Round Quantization Script\nModel: Qwen/Qwen3-0.6B\nScheme: W4A16\nMethod: RTN (iters=0)\nFormat: auto_round\nDevice: cuda\n\"\"\"\n\nimport time\nimport json\nimport os\nfrom pathlib import Path\n\nstart_time = time.time()\nerrors = []\nsolutions = []\n\noutput_dir = \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\"\nmodel_path = \"Qwen/Qwen3-0.6B\"\nscheme = \"W4A16\"\niters = 0  # RTN mode\nnsamples = 128\nformat_str = \"auto_round\"\ndevice_map = \"cuda\"\n\nprint(f\"Loading model: {model_path}\")\nprint(f\"Scheme: {scheme}\")\nprint(f\"Iters: {iters} (RTN mode)\")\nprint(f\"nsamples: {nsamples}\")\nprint(f\"Format: {format_str}\")\nprint(f\"Device: {device_map}\")\n\ntry:\n    from auto_round import AutoRound\n\n    ar = AutoRound(\n        model_name_or_path=model_path,\n        scheme=scheme,\n        iters=iters,\n        nsamples=nsamples,\n        device_map=device_map,\n    )\n\n    print(\"Starting quantization...\")\n    ar.quantize_and_save(output_dir=output_dir, format=format_str)\n\n    print(f\"Quantization complete! Output: {output_dir}\")\n\nexcept Exception as e:\n    errors.append(str(e))\n    import traceback\n    errors.append(traceback.format_exc())\n    print(f\"Error during quantization: {e}\")\n    print(traceback.format_exc())\n\n# Generate summary\nend_time = time.time()\nduration = end_time - start_time\n\n# Collect output files\noutput_path = Path(output_dir)\noutput_files = []\nquantized_model_dir = None\n\nif output_path.exists():\n    for f in sorted(output_path.rglob(\"*\")):\n        if f.is_file() and not f.name.endswith(('.pyc', '.pyo', '__pycache__')):\n            size = f.stat().st_size\n            output_files.append(str(f.relative_to(output_path)))\n            if f.is_dir() and f.name == \"quantized_model\":\n                quantized_model_dir = str(f)\n\n# Calculate sizes\noriginal_size_mb = None\nquantized_size_mb = None\ncompression_ratio = None\n\ntry:\n    import subprocess\n    result = subprocess.run(\n        [\"python3\", \"-c\", \n         f\"from transformers import AutoModel; \"\n         f\"model = AutoModel.from_pretrained('{model_path}', trust_remote_code=True); \"\n         f\"total = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024; \"\n         f\"print(f'{{total:.2f}}')\"],\n        capture_output=True, text=True, timeout=60\n    )\n    if result.returncode == 0:\n        original_size_mb = float(result.stdout.strip())\nexcept:\n    pass\n\nquant_dir = output_path / \"quantized_model\"\nif quant_dir.exists():\n    total_q = 0\n    for f in quant_dir.rglob(\"*\"):\n        if f.is_file():\n            total_q += f.stat().st_size\n    quantized_size_mb = total_q / 1024 / 1024\n\nif original_size_mb and quantized_size_mb:\n    compression_ratio = original_size_mb / quantized_size_mb\n\nsummary = {\n    \"model_id\": model_path,\n    \"scheme\": scheme,\n    \"method\": \"RTN\",\n    \"export_format\": format_str,\n    \"device\": device_map,\n    \"num_gpus\": \"1\",\n    \"output_dir\": output_dir,\n    \"quantized_model_dir\": quantized_model_dir or output_dir,\n    \"status\": \"failed\" if errors else \"success\",\n    \"duration_seconds\": round(duration, 2),\n    \"original_size_mb\": original_size_mb,\n    \"quantized_size_mb\": quantized_size_mb,\n    \"compression_ratio\": round(compression_ratio, 2) if compression_ratio else None,\n    \"errors\": errors,\n    \"solutions\": solutions,\n    \"output_files\": output_files\n}\n\nsummary_path = output_path / \"quant_summary.json\"\nwith open(summary_path, \"w\") as f:\n    json.dump(summary, f, indent=2)\n\nprint(f\"\\nSummary written to: {summary_path}\")\nprint(json.dumps(summary, indent=2))"
}
```

### [2026-04-23 10:51:25 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 3523 bytes to /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantize_script.py

### [2026-04-23 10:51:30 UTC] ASSISTANT

**Thinking**

Now let me run the quantization script. I'll set the timeout to 30 minutes.

**Tool call:** `exec`

```json
{
  "command": "cd /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16 && /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv/bin/python quantize_script.py 2>&1",
  "yieldMs": 600000,
  "timeout": 1800
}
```

### [2026-04-23 10:51:39 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Loading model: Qwen/Qwen3-0.6B
Scheme: W4A16
Iters: 0 (RTN mode)
nsamples: 128
Format: auto_round
Device: cuda
Error during quantization: AutoRound.__new__() missing 1 required positional argument: 'model'
Traceback (most recent call last):
  File "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantize_script.py", line 38, in <module>
    ar = AutoRound(
TypeError: AutoRound.__new__() missing 1 required positional argument: 'model'


Summary written to: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quant_summary.json
{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "status": "failed",
  "duration_seconds": 3.47,
  "original_size_mb": 1136.88,
  "quantized_size_mb": null,
  "compression_ratio": null,
  "errors": [
    "AutoRound.__new__() missing 1 required positional argument: 'model'",
    "Traceback (most recent call last):\n  File \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantize_script.py\", line 38, in <module>\n    ar = AutoRound(\nTypeError: AutoRound.__new__() missing 1 required positional argument: 'model'\n"
  ],
  "solutions": [],
  "output_files": [
    "logs/auto.log",
    "logs/quant_prompt.txt",
    "quantize_script.py",
    "request.json",
    "venv/bin/Activate.ps1",
    "venv/bin/activate",
    "venv/bin/activate.csh",
    "venv/bin/activate.fish",
    "venv/bin/pip",
    "venv/bin/pip3",
    "venv/bin/pip3.10",
    "venv/bin/python",
    "venv/bin/python3",
    "venv/bin/python3.10",
    "venv/bin/wheel",
    "venv/lib/python3.10/site-packages/_distutils_hack/__init__.py",
    "venv/lib/python3.10/site-packages/_distutils_hack/override.py",
    "venv/lib/python3.10/site-packages/distutils-precedence.pth",
    "venv/lib/python3.10/site-packages/pip/__init__.py",
    "venv/lib/python3.10/site-packages/pip/__main__.py",
    "venv/lib/python3.10/site-packages/pip/__pip-runner__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/build_env.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cache.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/autocompletion.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/base_command.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/cmdoptions.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/command_context.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/index_command.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/main.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/main_parser.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/parser.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/progress_bars.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/req_command.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/spinners.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/status_codes.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/cache.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/check.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/completion.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/configuration.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/debug.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/download.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/freeze.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/hash.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/help.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/index.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/inspect.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/install.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/list.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/lock.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/search.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/show.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/uninstall.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/configuration.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/installed.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/sdist.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/collector.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/package_finder.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/sources.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/_distutils.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/_sysconfig.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/main.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/_json.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/_compat.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/_dists.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/_envs.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/pkg_resources.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/candidate.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/direct_url.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/format_control.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/index.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/installation_report.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/link.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/release_control.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/scheme.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/search_scope.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/selection_prefs.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/target_python.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/auth.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/cache.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/download.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/lazy_wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/session.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/utils.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/xmlrpc.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/build_tracker.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/metadata.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/metadata_editable.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/wheel_editable.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/check.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/freeze.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/install/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/install/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/prepare.py",
    "venv/lib/python3.10/site-packages/pip/_internal/pyproject.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/constructors.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/pep723.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_dependency_group.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_file.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_install.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_set.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_uninstall.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/legacy/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/legacy/resolver.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/candidates.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/factory.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/provider.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/reporter.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/requirements.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/resolver.py",
    "venv/lib/python3.10/site-packages/pip/_internal/self_outdated_check.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/_jaraco_text.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/_log.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/appdirs.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/compat.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/compatibility_tags.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/datetime.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/deprecation.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/direct_url_helpers.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/egg_link.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/entrypoints.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/filesystem.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/filetypes.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/glibc.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/hashes.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/logging.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/misc.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/packaging.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/pylock.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/retry.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/subprocess.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/temp_dir.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/unpacking.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/urls.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/virtualenv.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/bazaar.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/git.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/mercurial.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/subversion.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/versioncontrol.py",
    "venv/lib/python3.10/site-packages/pip/_internal/wheel_builder.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/README.rst",
    "venv/lib/python3.10/site-packages/pip/_vendor/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/_cmd.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/adapter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/cache.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/caches/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/caches/file_cache.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/caches/redis_cache.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/controller.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/filewrapper.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/heuristics.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/serialize.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/wrapper.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/cacert.pem",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/core.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_implementation.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_lint_dependency_groups.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_pip_wrapper.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_toml_compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/resources.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/scripts.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/t32.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/t64-arm.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/t64.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/util.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/w32.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/w64-arm.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/w64.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/distro.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/LICENSE.md",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/codec.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/core.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/idnadata.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/intranges.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/package_data.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/uts46data.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/COPYING",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/ext.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/fallback.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/LICENSE.APACHE",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/LICENSE.BSD",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_elffile.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_manylinux.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_musllinux.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_parser.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_structures.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_tokenizer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/licenses/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/licenses/_spdx.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/markers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/metadata.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/pylock.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/requirements.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/specifiers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/tags.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/utils.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/version.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pkg_resources/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/pkg_resources/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/android.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/api.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/macos.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/unix.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/version.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/windows.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/console.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/filter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/filters/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/formatter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/formatters/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/formatters/_mapping.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexers/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexers/_mapping.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexers/python.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/modeline.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/plugin.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/regexopt.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/scanner.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/sphinxext.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/style.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/styles/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/styles/_mapping.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/token.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/unistring.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/util.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_impl.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/__version__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/_internal_utils.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/adapters.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/api.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/auth.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/certs.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/cookies.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/help.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/hooks.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/models.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/packages.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/sessions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/status_codes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/structures.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/utils.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/providers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/reporters.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/abstract.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/criterion.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/resolution.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/structs.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_cell_widths.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_emoji_codes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_emoji_replace.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_export_format.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_extension.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_fileno.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_inspect.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_log_render.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_loop.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_null_file.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_palettes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_pick.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_ratio.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_spinners.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_stack.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_timer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_win32_console.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_windows.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_windows_renderer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_wrap.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/abc.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/align.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/ansi.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/bar.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/box.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/cells.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/color.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/color_triplet.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/columns.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/console.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/constrain.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/containers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/control.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/default_styles.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/diagnose.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/emoji.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/errors.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/file_proxy.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/filesize.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/highlighter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/json.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/jupyter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/layout.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/live.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/live_render.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/logging.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/markup.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/measure.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/padding.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/pager.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/palette.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/panel.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/pretty.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/progress.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/progress_bar.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/prompt.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/protocol.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/region.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/repr.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/rule.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/scope.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/screen.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/segment.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/spinner.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/status.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/style.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/styled.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/syntax.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/table.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/terminal_theme.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/text.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/theme.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/themes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/traceback.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/tree.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/_parser.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/_re.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/_types.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/_writer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_api.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_macos.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_openssl.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_ssl_constants.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_windows.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/_collections.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/_version.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/connection.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/connectionpool.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_appengine_environ.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_securetransport/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_securetransport/bindings.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_securetransport/low_level.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/appengine.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/ntlmpool.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/pyopenssl.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/securetransport.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/socks.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/fields.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/filepost.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/backports/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/backports/makefile.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/backports/weakref_finalize.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/six.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/poolmanager.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/request.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/connection.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/proxy.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/queue.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/request.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/response.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/retry.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/ssl_.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/ssl_match_hostname.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/ssltransport.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/timeout.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/url.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/wait.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/vendor.txt",
    "venv/lib/python3.10/site-packages/pip/py.typed",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/AUTHORS.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/cachecontrol/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/certifi/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/dependency_groups/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/distlib/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/distro/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/idna/LICENSE.md",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/msgpack/COPYING",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE.APACHE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE.BSD",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/pkg_resources/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/platformdirs/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/pygments/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/pyproject_hooks/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/requests/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/resolvelib/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/rich/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/tomli/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/tomli_w/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/truststore/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/urllib3/LICENSE.txt",
    "venv/lib/python3.10/site-packages/setuptools/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_core_metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_discovery.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_log.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_macos_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_modified.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_msvccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/ccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/cmd.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/_framework_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/bdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/bdist_dumb.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/bdist_rpm.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/check.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/clean.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/config.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_data.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_headers.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_lib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compat/numpy.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/base.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/cygwin.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/msvc.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_base.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_cygwin.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_mingw.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_msvc.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_unix.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/unix.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/zos.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/core.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/cygwinccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/debug.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/dep_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/dir_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/dist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/extension.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/fancy_getopt.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/file_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/filelist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/log.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/spawn.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/sysconfig.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/support.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_bdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_bdist_dumb.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_bdist_rpm.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_check.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_clean.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_cmd.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_config_cmd.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_core.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_dir_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_dist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_extension.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_file_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_filelist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_data.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_headers.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_lib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_log.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_modified.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_spawn.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_sysconfig.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_text_file.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_version.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_versionpredicate.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/unix_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/text_file.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/unixccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/version.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/versionpredicate.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/zosccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_entry_points.py",
    "venv/lib/python3.10/site-packages/setuptools/_imp.py",
    "venv/lib/python3.10/site-packages/setuptools/_importlib.py",
    "venv/lib/python3.10/site-packages/setuptools/_itertools.py",
    "venv/lib/python3.10/site-packages/setuptools/_normalization.py",
    "venv/lib/python3.10/site-packages/setuptools/_path.py",
    "venv/lib/python3.10/site-packages/setuptools/_reqs.py",
    "venv/lib/python3.10/site-packages/setuptools/_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_shutil.py",
    "venv/lib/python3.10/site-packages/setuptools/_static.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/.lock",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/autoasync.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/autocommand.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/automain.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/autoparse.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/__main__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/compat/py38.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_adapters.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_collections.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_functools.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_itertools.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_meta.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_text.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_typing.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/compat/py311.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/diagnose.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/context/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/context/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/functools/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/functools/__init__.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/functools/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/Lorem ipsum.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/layouts.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/show-newlines.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/strip-prefix.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/to-dvorak.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/to-qwerty.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/__init__.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/more.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/more.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/recipes.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/recipes.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_elffile.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_manylinux.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_musllinux.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_parser.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_structures.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_tokenizer.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/licenses/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/licenses/_spdx.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/markers.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/pylock.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/requirements.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/specifiers.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/tags.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/utils.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/version.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.APACHE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.BSD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/__main__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/android.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/api.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/macos.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/unix.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/version.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/windows.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/_parser.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/_re.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/_types.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/__main__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/convert.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/pack.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/tags.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/unpack.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_setuptools_logging.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/macosx_libfile.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/wheelfile.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/licenses/LICENSE.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/_functools.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/overlay.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/py310.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/py313.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/glob.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/build_meta.py",
    "venv/lib/python3.10/site-packages/setuptools/cli-32.exe",
    "venv/lib/python3.10/site-packages/setuptools/cli-64.exe",
    "venv/lib/python3.10/site-packages/setuptools/cli-arm64.exe",
    "venv/lib/python3.10/site-packages/setuptools/cli.exe",
    "venv/lib/python3.10/site-packages/setuptools/command/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/command/_requirestxt.py",
    "venv/lib/python3.10/site-packages/setuptools/command/alias.py",
    "venv/lib/python3.10/site-packages/setuptools/command/bdist_egg.py",
    "venv/lib/python3.10/site-packages/setuptools/command/bdist_rpm.py",
    "venv/lib/python3.10/site-packages/setuptools/command/bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/command/develop.py",
    "venv/lib/python3.10/site-packages/setuptools/command/dist_info.py",
    "venv/lib/python3.10/site-packages/setuptools/command/easy_install.py",
    "venv/lib/python3.10/site-packages/setuptools/command/editable_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/command/egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install_egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install_lib.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/command/rotate.py",
    "venv/lib/python3.10/site-packages/setuptools/command/saveopts.py",
    "venv/lib/python3.10/site-packages/setuptools/command/sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/command/setopt.py",
    "venv/lib/python3.10/site-packages/setuptools/command/test.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py310.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py311.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py312.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/config/NOTICE",
    "venv/lib/python3.10/site-packages/setuptools/config/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_apply_pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/NOTICE",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/error_reporting.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/extra_validations.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/fastjsonschema_exceptions.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/fastjsonschema_validations.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/formats.py",
    "venv/lib/python3.10/site-packages/setuptools/config/distutils.schema.json",
    "venv/lib/python3.10/site-packages/setuptools/config/expand.py",
    "venv/lib/python3.10/site-packages/setuptools/config/pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/config/setupcfg.py",
    "venv/lib/python3.10/site-packages/setuptools/config/setuptools.schema.json",
    "venv/lib/python3.10/site-packages/setuptools/depends.py",
    "venv/lib/python3.10/site-packages/setuptools/discovery.py",
    "venv/lib/python3.10/site-packages/setuptools/dist.py",
    "venv/lib/python3.10/site-packages/setuptools/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/extension.py",
    "venv/lib/python3.10/site-packages/setuptools/glob.py",
    "venv/lib/python3.10/site-packages/setuptools/gui-32.exe",
    "venv/lib/python3.10/site-packages/setuptools/gui-64.exe",
    "venv/lib/python3.10/site-packages/setuptools/gui-arm64.exe",
    "venv/lib/python3.10/site-packages/setuptools/gui.exe",
    "venv/lib/python3.10/site-packages/setuptools/installer.py",
    "venv/lib/python3.10/site-packages/setuptools/launch.py",
    "venv/lib/python3.10/site-packages/setuptools/launcher manifest.xml",
    "venv/lib/python3.10/site-packages/setuptools/logging.py",
    "venv/lib/python3.10/site-packages/setuptools/modified.py",
    "venv/lib/python3.10/site-packages/setuptools/monkey.py",
    "venv/lib/python3.10/site-packages/setuptools/msvc.py",
    "venv/lib/python3.10/site-packages/setuptools/namespaces.py",
    "venv/lib/python3.10/site-packages/setuptools/script (dev).tmpl",
    "venv/lib/python3.10/site-packages/setuptools/script.tmpl",
    "venv/lib/python3.10/site-packages/setuptools/tests/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/downloads/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/downloads/preload.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/setupcfg_examples.txt",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_apply_pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_expand.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_pyprojecttoml_dynamic_deps.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_setupcfg.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/contexts.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/environment.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/fixtures.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/indexes/test_links_priority/external.html",
    "venv/lib/python3.10/site-packages/setuptools/tests/indexes/test_links_priority/simple/foobar/index.html",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/helpers.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/test_pbr.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/test_pip_install_sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/mod_with_constant.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/namespaces.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/script-with-bom.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_bdist_deprecations.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_bdist_egg.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_meta.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_config_discovery.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_core_metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_depends.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_develop.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_dist.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_dist_info.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_distutils_adoption.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_editable_install.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_extern.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_find_packages.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_find_py_modules.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_glob.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_logging.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_manifest.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_namespaces.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_setopt.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_setuptools.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_shutil_wrapper.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_unicode_utils.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_virtualenv.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_warnings.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_windows_wrappers.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/text.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/textwrap.py",
    "venv/lib/python3.10/site-packages/setuptools/unicode_utils.py",
    "venv/lib/python3.10/site-packages/setuptools/version.py",
    "venv/lib/python3.10/site-packages/setuptools/warnings.py",
    "venv/lib/python3.10/site-packages/setuptools/wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/windows_support.py",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/wheel/__init__.py",
    "venv/lib/python3.10/site-packages/wheel/__main__.py",
    "venv/lib/python3.10/site-packages/wheel/_bdist_wheel.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/__init__.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/convert.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/info.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/pack.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/tags.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/unpack.py",
    "venv/lib/python3.10/site-packages/wheel/_metadata.py",
    "venv/lib/python3.10/site-packages/wheel/_setuptools_logging.py",
    "venv/lib/python3.10/site-packages/wheel/bdist_wheel.py",
    "venv/lib/python3.10/site-packages/wheel/macosx_libfile.py",
    "venv/lib/python3.10/site-packages/wheel/metadata.py",
    "venv/lib/python3.10/site-packages/wheel/wheelfile.py",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/licenses/LICENSE.txt",
    "venv/pyvenv.cfg"
  ]
}

### [2026-04-23 10:51:47 UTC] ASSISTANT

**Thinking**

The error is that `AutoRound.__new__()` requires a `model` argument. The API seems to have changed in version 0.12.2. Let me check the auto-round API to understand the correct usage.


The API signature changed in this version. Let me check the actual AutoRound API:


**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv/bin/python -c \"from auto_round import AutoRound; import inspect; print(inspect.signature(AutoRound.__init__))\" 2>&1"
}
```

### [2026-04-23 10:51:51 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(self, /, *args, **kwargs)

### [2026-04-23 10:51:57 UTC] ASSISTANT

**Thinking**

The `__init__` signature shows `self, /, *args, **kwargs` which means the AutoRound class uses `__new__` and then `__init__` with different arguments. Let me check the `__new__` signature.

**Tool call:** `exec`

```json
{
  "command": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv/bin/python -c \"from auto_round import AutoRound; import inspect; print(inspect.signature(AutoRound.__new__))\" 2>&1"
}
```

### [2026-04-23 10:52:02 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

(cls, model: 'Union[torch.nn.Module, str]', tokenizer=None, platform: 'str' = 'hf', scheme: 'Union[str, dict, QuantizationScheme, AutoScheme]' = 'W4A16', layer_config: 'dict[str, Union[str, dict, QuantizationScheme]]' = None, dataset: 'Union[str, list, tuple, torch.utils.data.DataLoader]' = 'NeelNanda/pile-10k', iters: 'int' = 200, seqlen: 'int' = 2048, nsamples: 'int' = 128, batch_size: 'int' = 8, gradient_accumulate_steps: 'int' = 1, low_gpu_mem_usage: 'bool' = False, device_map: 'Union[str, torch.device, int, dict]' = 0, enable_torch_compile: 'bool' = False, seed: 'int' = 42, enable_adam: 'bool' = False, extra_config: 'ExtraConfig' = None, enable_alg_ext: 'bool' = False, disable_opt_rtn: 'bool | None' = None, low_cpu_mem_usage: 'bool' = True, **kwargs) -> 'BaseCompressor'

### [2026-04-23 10:52:19 UTC] ASSISTANT

**Thinking**

Now I see - the `model` argument is the first positional argument, and it's required. The API changed significantly. I need to pass the model as the first positional argument. Let me check what type of model is expected - can I pass a string (model path) or do I need to load the model first?

The signature shows `model: 'Union[torch.nn.Module, str]'` so I can pass a string path. Let me update the script.

Now I see — `model` is the first positional argument. Let me fix the script:


**Tool call:** `write`

```json
{
  "path": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantize_script.py",
  "content": "#!/usr/bin/env python3\n\"\"\"\nAuto-Round Quantization Script\nModel: Qwen/Qwen3-0.6B\nScheme: W4A16\nMethod: RTN (iters=0)\nFormat: auto_round\nDevice: cuda\n\"\"\"\n\nimport time\nimport json\nimport os\nfrom pathlib import Path\n\nstart_time = time.time()\nerrors = []\nsolutions = []\n\noutput_dir = \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\"\nmodel_path = \"Qwen/Qwen3-0.6B\"\nscheme = \"W4A16\"\niters = 0  # RTN mode\nnsamples = 128\nformat_str = \"auto_round\"\ndevice_map = \"cuda\"\n\nprint(f\"Loading model: {model_path}\")\nprint(f\"Scheme: {scheme}\")\nprint(f\"Iters: {iters} (RTN mode)\")\nprint(f\"nsamples: {nsamples}\")\nprint(f\"Format: {format_str}\")\nprint(f\"Device: {device_map}\")\n\ntry:\n    from auto_round import AutoRound\n\n    ar = AutoRound(\n        model_path,  # model as first positional arg\n        tokenizer=None,\n        scheme=scheme,\n        iters=iters,\n        nsamples=nsamples,\n        device_map=device_map,\n    )\n\n    print(\"Starting quantization...\")\n    ar.quantize_and_save(output_dir=output_dir, format=format_str)\n\n    print(f\"Quantization complete! Output: {output_dir}\")\n\nexcept Exception as e:\n    errors.append(str(e))\n    import traceback\n    errors.append(traceback.format_exc())\n    print(f\"Error during quantization: {e}\")\n    print(traceback.format_exc())\n\n# Generate summary\nend_time = time.time()\nduration = end_time - start_time\n\n# Collect output files\noutput_path = Path(output_dir)\noutput_files = []\nquantized_model_dir = None\n\nif output_path.exists():\n    for f in sorted(output_path.rglob(\"*\")):\n        if f.is_file() and not f.name.endswith(('.pyc', '.pyo', '__pycache__')):\n            size = f.stat().st_size\n            output_files.append(str(f.relative_to(output_path)))\n            if f.is_dir() and f.name == \"quantized_model\":\n                quantized_model_dir = str(f)\n\n# Calculate sizes\noriginal_size_mb = 1136.88  # known from first attempt\n\nquant_dir = output_path / \"quantized_model\"\nquantized_size_mb = None\nif quant_dir.exists():\n    total_q = 0\n    for f in quant_dir.rglob(\"*\"):\n        if f.is_file():\n            total_q += f.stat().st_size\n    quantized_size_mb = total_q / 1024 / 1024\n\ncompression_ratio = None\nif original_size_mb and quantized_size_mb:\n    compression_ratio = original_size_mb / quantized_size_mb\n\nsummary = {\n    \"model_id\": model_path,\n    \"scheme\": scheme,\n    \"method\": \"RTN\",\n    \"export_format\": format_str,\n    \"device\": device_map,\n    \"num_gpus\": \"1\",\n    \"output_dir\": output_dir,\n    \"quantized_model_dir\": quantized_model_dir or output_dir,\n    \"status\": \"failed\" if errors else \"success\",\n    \"duration_seconds\": round(duration, 2),\n    \"original_size_mb\": original_size_mb,\n    \"quantized_size_mb\": quantized_size_mb,\n    \"compression_ratio\": round(compression_ratio, 2) if compression_ratio else None,\n    \"errors\": errors,\n    \"solutions\": solutions,\n    \"output_files\": output_files\n}\n\nsummary_path = output_path / \"quant_summary.json\"\nwith open(summary_path, \"w\") as f:\n    json.dump(summary, f, indent=2)\n\nprint(f\"\\nSummary written to: {summary_path}\")\nprint(json.dumps(summary, indent=2))"
}
```

### [2026-04-23 10:52:19 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 3074 bytes to /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantize_script.py

### [2026-04-23 10:52:24 UTC] ASSISTANT

**Thinking**

Let me also clear the old logs that were created from the previous failed run.

**Tool call:** `exec`

```json
{
  "command": "rm -rf /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantized_model /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/logs /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/request.json 2>/dev/null; echo \"cleaned\""
}
```

### [2026-04-23 10:52:24 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

cleaned

### [2026-04-23 10:52:29 UTC] ASSISTANT

**Thinking**

Now let's run the quantization script again.

**Tool call:** `exec`

```json
{
  "command": "cd /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16 && /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/venv/bin/python quantize_script.py 2>&1",
  "yieldMs": 600000,
  "timeout": 1800
}
```

### [2026-04-23 10:53:38 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading model: Qwen/Qwen3-0.6B
Scheme: W4A16
Iters: 0 (RTN mode)
nsamples: 128
Format: auto_round
Device: cuda
404 Client Error. (Request ID: Root=1-69e9f8da-21789cc246bf1b791509bc65;10a8aeec-35c7-4be7-bc19-1a297cf2a8b2)

Entry Not Found for url: https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model_index.json.
Loading weights:   0%|          | 0/311 [00:00<?, ?it/s]Loading weights: 100%|██████████| 311/311 [00:00<00:00, 4005.36it/s]
[38;20m2026-04-23 10:52:40 INFO base.py L473: `enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy.[0m
[38;20m2026-04-23 10:52:40 INFO base.py L517: using torch.bfloat16 for quantization tuning[0m
[33;1m2026-04-23 10:52:40 WARNING formats.py L166: some layers are skipped quantization (shape not divisible by 32): [0m
[33;1m2026-04-23 10:52:40 WARNING modeling_utils.py L4435: `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.[0m
[38;20m2026-04-23 10:52:41 INFO base.py L1150: start to compute imatrix[0m
Starting quantization...
  0%|          | 0/28 [00:00<?, ?it/s]Quantizing model.layers.0:   0%|          | 0/28 [00:00<?, ?it/s][38;20m2026-04-23 10:52:55 INFO device.py L1692: 'peak_ram': 2.19GB, 'peak_vram': 1.98GB[0m
Quantizing model.layers.0:   4%|▎         | 1/28 [00:02<00:57,  2.12s/it]Quantizing model.layers.1:   4%|▎         | 1/28 [00:02<00:57,  2.12s/it][38;20m2026-04-23 10:52:57 INFO device.py L1692: 'peak_ram': 2.23GB, 'peak_vram': 2.0GB[0m
Quantizing model.layers.1:   7%|▋         | 2/28 [00:03<00:47,  1.81s/it]Quantizing model.layers.2:   7%|▋         | 2/28 [00:03<00:47,  1.81s/it][38;20m2026-04-23 10:52:58 INFO device.py L1692: 'peak_ram': 2.26GB, 'peak_vram': 2.01GB[0m
Quantizing model.layers.2:  11%|█         | 3/28 [00:05<00:42,  1.71s/it]Quantizing model.layers.3:  11%|█         | 3/28 [00:05<00:42,  1.71s/it][38;20m2026-04-23 10:53:00 INFO device.py L1692: 'peak_ram': 2.3GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.3:  14%|█▍        | 4/28 [00:06<00:40,  1.67s/it]Quantizing model.layers.4:  14%|█▍        | 4/28 [00:06<00:40,  1.67s/it][38;20m2026-04-23 10:53:02 INFO device.py L1692: 'peak_ram': 2.34GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.4:  18%|█▊        | 5/28 [00:08<00:37,  1.64s/it]Quantizing model.layers.5:  18%|█▊        | 5/28 [00:08<00:37,  1.64s/it][38;20m2026-04-23 10:53:03 INFO device.py L1692: 'peak_ram': 2.37GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.5:  21%|██▏       | 6/28 [00:10<00:35,  1.62s/it]Quantizing model.layers.6:  21%|██▏       | 6/28 [00:10<00:35,  1.62s/it][38;20m2026-04-23 10:53:05 INFO device.py L1692: 'peak_ram': 2.41GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.6:  25%|██▌       | 7/28 [00:11<00:33,  1.61s/it]Quantizing model.layers.7:  25%|██▌       | 7/28 [00:11<00:33,  1.61s/it][38;20m2026-04-23 10:53:06 INFO device.py L1692: 'peak_ram': 2.44GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.7:  29%|██▊       | 8/28 [00:13<00:31,  1.60s/it]Quantizing model.layers.8:  29%|██▊       | 8/28 [00:13<00:31,  1.60s/it][38;20m2026-04-23 10:53:08 INFO device.py L1692: 'peak_ram': 2.48GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.8:  32%|███▏      | 9/28 [00:14<00:30,  1.59s/it]Quantizing model.layers.9:  32%|███▏      | 9/28 [00:14<00:30,  1.59s/it][38;20m2026-04-23 10:53:09 INFO device.py L1692: 'peak_ram': 2.51GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.9:  36%|███▌      | 10/28 [00:16<00:28,  1.56s/it]Quantizing model.layers.10:  36%|███▌      | 10/28 [00:16<00:28,  1.56s/it][38;20m2026-04-23 10:53:11 INFO device.py L1692: 'peak_ram': 2.55GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.10:  39%|███▉      | 11/28 [00:17<00:25,  1.49s/it]Quantizing model.layers.11:  39%|███▉      | 11/28 [00:17<00:25,  1.49s/it][38;20m2026-04-23 10:53:12 INFO device.py L1692: 'peak_ram': 2.59GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.11:  43%|████▎     | 12/28 [00:18<00:23,  1.45s/it]Quantizing model.layers.12:  43%|████▎     | 12/28 [00:18<00:23,  1.45s/it][38;20m2026-04-23 10:53:14 INFO device.py L1692: 'peak_ram': 2.63GB, 'peak_vram': 2.07GB[0m
Quantizing model.layers.12:  46%|████▋     | 13/28 [00:20<00:21,  1.44s/it]Quantizing model.layers.13:  46%|████▋     | 13/28 [00:20<00:21,  1.44s/it][38;20m2026-04-23 10:53:15 INFO device.py L1692: 'peak_ram': 2.66GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.13:  50%|█████     | 14/28 [00:21<00:19,  1.42s/it]Quantizing model.layers.14:  50%|█████     | 14/28 [00:21<00:19,  1.42s/it][38;20m2026-04-23 10:53:16 INFO device.py L1692: 'peak_ram': 2.7GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.14:  54%|█████▎    | 15/28 [00:23<00:18,  1.40s/it]Quantizing model.layers.15:  54%|█████▎    | 15/28 [00:23<00:18,  1.40s/it][38;20m2026-04-23 10:53:18 INFO device.py L1692: 'peak_ram': 2.74GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.15:  57%|█████▋    | 16/28 [00:24<00:16,  1.38s/it]Quantizing model.layers.16:  57%|█████▋    | 16/28 [00:24<00:16,  1.38s/it][38;20m2026-04-23 10:53:19 INFO device.py L1692: 'peak_ram': 2.78GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.16:  61%|██████    | 17/28 [00:25<00:15,  1.41s/it]Quantizing model.layers.17:  61%|██████    | 17/28 [00:25<00:15,  1.41s/it][38;20m2026-04-23 10:53:20 INFO device.py L1692: 'peak_ram': 2.8GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.17:  64%|██████▍   | 18/28 [00:27<00:13,  1.39s/it]Quantizing model.layers.18:  64%|██████▍   | 18/28 [00:27<00:13,  1.39s/it][38;20m2026-04-23 10:53:22 INFO device.py L1692: 'peak_ram': 2.84GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.18:  68%|██████▊   | 19/28 [00:28<00:12,  1.38s/it]Quantizing model.layers.19:  68%|██████▊   | 19/28 [00:28<00:12,  1.38s/it][38;20m2026-04-23 10:53:23 INFO device.py L1692: 'peak_ram': 2.88GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.19:  71%|███████▏  | 20/28 [00:29<00:10,  1.37s/it]Quantizing model.layers.20:  71%|███████▏  | 20/28 [00:29<00:10,  1.37s/it][38;20m2026-04-23 10:53:25 INFO device.py L1692: 'peak_ram': 2.92GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.20:  75%|███████▌  | 21/28 [00:31<00:09,  1.36s/it]Quantizing model.layers.21:  75%|███████▌  | 21/28 [00:31<00:09,  1.36s/it][38;20m2026-04-23 10:53:26 INFO device.py L1692: 'peak_ram': 2.95GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.21:  79%|███████▊  | 22/28 [00:32<00:08,  1.35s/it]Quantizing model.layers.22:  79%|███████▊  | 22/28 [00:32<00:08,  1.35s/it][38;20m2026-04-23 10:53:27 INFO device.py L1692: 'peak_ram': 2.99GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.22:  82%|████████▏ | 23/28 [00:34<00:06,  1.35s/it]Quantizing model.layers.23:  82%|████████▏ | 23/28 [00:34<00:06,  1.35s/it][38;20m2026-04-23 10:53:29 INFO device.py L1692: 'peak_ram': 3.03GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.23:  86%|████████▌ | 24/28 [00:35<00:05,  1.34s/it]Quantizing model.layers.24:  86%|████████▌ | 24/28 [00:35<00:05,  1.34s/it][38;20m2026-04-23 10:53:30 INFO device.py L1692: 'peak_ram': 3.06GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.24:  89%|████████▉ | 25/28 [00:36<00:04,  1.34s/it]Quantizing model.layers.25:  89%|████████▉ | 25/28 [00:36<00:04,  1.34s/it][38;20m2026-04-23 10:53:31 INFO device.py L1692: 'peak_ram': 3.1GB, 'peak_vram': 2.1GB[0m
Quantizing model.layers.25:  93%|█████████▎| 26/28 [00:38<00:02,  1.34s/it]Quantizing model.layers.26:  93%|█████████▎| 26/28 [00:38<00:02,  1.34s/it][38;20m2026-04-23 10:53:33 INFO device.py L1692: 'peak_ram': 3.14GB, 'peak_vram': 2.16GB[0m
Quantizing model.layers.26:  96%|█████████▋| 27/28 [00:39<00:01,  1.34s/it]Quantizing model.layers.27:  96%|█████████▋| 27/28 [00:39<00:01,  1.34s/it][38;20m2026-04-23 10:53:34 INFO device.py L1692: 'peak_ram': 3.17GB, 'peak_vram': 2.16GB[0m
Quantizing model.layers.27: 100%|██████████| 28/28 [00:40<00:00,  1.34s/it]Quantizing model.layers.27: 100%|██████████| 28/28 [00:40<00:00,  1.45s/it]
[38;20m2026-04-23 10:53:35 INFO shard_writer.py L250: model has been saved to /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/[0m
[38;20m2026-04-23 10:53:37 INFO device.py L1692: 'peak_ram': 3.17GB, 'peak_vram': 2.16GB[0m
Quantization complete! Output: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16

Summary written to: /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quant_summary.json
{
  "model_id": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN",
  "export_format": "auto_round",
  "device": "cuda",
  "num_gpus": "1",
  "output_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "quantized_model_dir": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16",
  "status": "success",
  "duration_seconds": 68.04,
  "original_size_mb": 1136.88,
  "quantized_size_mb": null,
  "compression_ratio": null,
  "errors": [],
  "solutions": [],
  "output_files": [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "quant_summary.json",
    "quantization_config.json",
    "quantize_script.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "venv/bin/Activate.ps1",
    "venv/bin/activate",
    "venv/bin/activate.csh",
    "venv/bin/activate.fish",
    "venv/bin/pip",
    "venv/bin/pip3",
    "venv/bin/pip3.10",
    "venv/bin/python",
    "venv/bin/python3",
    "venv/bin/python3.10",
    "venv/bin/wheel",
    "venv/lib/python3.10/site-packages/_distutils_hack/__init__.py",
    "venv/lib/python3.10/site-packages/_distutils_hack/override.py",
    "venv/lib/python3.10/site-packages/distutils-precedence.pth",
    "venv/lib/python3.10/site-packages/pip/__init__.py",
    "venv/lib/python3.10/site-packages/pip/__main__.py",
    "venv/lib/python3.10/site-packages/pip/__pip-runner__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/build_env.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cache.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/autocompletion.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/base_command.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/cmdoptions.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/command_context.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/index_command.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/main.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/main_parser.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/parser.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/progress_bars.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/req_command.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/spinners.py",
    "venv/lib/python3.10/site-packages/pip/_internal/cli/status_codes.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/cache.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/check.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/completion.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/configuration.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/debug.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/download.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/freeze.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/hash.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/help.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/index.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/inspect.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/install.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/list.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/lock.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/search.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/show.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/uninstall.py",
    "venv/lib/python3.10/site-packages/pip/_internal/commands/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/configuration.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/installed.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/sdist.py",
    "venv/lib/python3.10/site-packages/pip/_internal/distributions/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/collector.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/package_finder.py",
    "venv/lib/python3.10/site-packages/pip/_internal/index/sources.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/_distutils.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/_sysconfig.py",
    "venv/lib/python3.10/site-packages/pip/_internal/locations/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/main.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/_json.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/_compat.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/_dists.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/importlib/_envs.py",
    "venv/lib/python3.10/site-packages/pip/_internal/metadata/pkg_resources.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/candidate.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/direct_url.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/format_control.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/index.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/installation_report.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/link.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/release_control.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/scheme.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/search_scope.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/selection_prefs.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/target_python.py",
    "venv/lib/python3.10/site-packages/pip/_internal/models/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/auth.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/cache.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/download.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/lazy_wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/session.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/utils.py",
    "venv/lib/python3.10/site-packages/pip/_internal/network/xmlrpc.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/build_tracker.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/metadata.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/metadata_editable.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/build/wheel_editable.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/check.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/freeze.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/install/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/install/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/operations/prepare.py",
    "venv/lib/python3.10/site-packages/pip/_internal/pyproject.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/constructors.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/pep723.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_dependency_group.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_file.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_install.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_set.py",
    "venv/lib/python3.10/site-packages/pip/_internal/req/req_uninstall.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/legacy/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/legacy/resolver.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/base.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/candidates.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/factory.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/provider.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/reporter.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/requirements.py",
    "venv/lib/python3.10/site-packages/pip/_internal/resolution/resolvelib/resolver.py",
    "venv/lib/python3.10/site-packages/pip/_internal/self_outdated_check.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/_jaraco_text.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/_log.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/appdirs.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/compat.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/compatibility_tags.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/datetime.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/deprecation.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/direct_url_helpers.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/egg_link.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/entrypoints.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/filesystem.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/filetypes.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/glibc.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/hashes.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/logging.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/misc.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/packaging.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/pylock.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/retry.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/subprocess.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/temp_dir.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/unpacking.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/urls.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/virtualenv.py",
    "venv/lib/python3.10/site-packages/pip/_internal/utils/wheel.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/bazaar.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/git.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/mercurial.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/subversion.py",
    "venv/lib/python3.10/site-packages/pip/_internal/vcs/versioncontrol.py",
    "venv/lib/python3.10/site-packages/pip/_internal/wheel_builder.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/README.rst",
    "venv/lib/python3.10/site-packages/pip/_vendor/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/_cmd.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/adapter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/cache.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/caches/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/caches/file_cache.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/caches/redis_cache.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/controller.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/filewrapper.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/heuristics.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/serialize.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/cachecontrol/wrapper.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/cacert.pem",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/core.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/certifi/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_implementation.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_lint_dependency_groups.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_pip_wrapper.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/_toml_compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/dependency_groups/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/resources.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/scripts.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/t32.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/t64-arm.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/t64.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/util.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/w32.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/w64-arm.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distlib/w64.exe",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/distro.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/distro/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/LICENSE.md",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/codec.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/core.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/idnadata.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/intranges.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/package_data.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/idna/uts46data.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/COPYING",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/ext.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/msgpack/fallback.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/LICENSE.APACHE",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/LICENSE.BSD",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_elffile.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_manylinux.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_musllinux.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_parser.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_structures.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/_tokenizer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/licenses/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/licenses/_spdx.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/markers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/metadata.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/pylock.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/requirements.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/specifiers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/tags.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/utils.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/packaging/version.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pkg_resources/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/pkg_resources/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/android.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/api.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/macos.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/unix.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/version.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/platformdirs/windows.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/console.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/filter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/filters/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/formatter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/formatters/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/formatters/_mapping.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexers/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexers/_mapping.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/lexers/python.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/modeline.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/plugin.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/regexopt.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/scanner.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/sphinxext.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/style.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/styles/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/styles/_mapping.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/token.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/unistring.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pygments/util.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_impl.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/__version__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/_internal_utils.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/adapters.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/api.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/auth.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/certs.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/compat.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/cookies.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/help.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/hooks.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/models.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/packages.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/sessions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/status_codes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/structures.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/requests/utils.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/providers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/reporters.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/abstract.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/criterion.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/resolvers/resolution.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/resolvelib/structs.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/__main__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_cell_widths.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_emoji_codes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_emoji_replace.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_export_format.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_extension.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_fileno.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_inspect.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_log_render.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_loop.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_null_file.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_palettes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_pick.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_ratio.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_spinners.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_stack.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_timer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_win32_console.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_windows.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_windows_renderer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/_wrap.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/abc.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/align.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/ansi.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/bar.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/box.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/cells.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/color.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/color_triplet.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/columns.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/console.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/constrain.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/containers.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/control.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/default_styles.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/diagnose.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/emoji.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/errors.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/file_proxy.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/filesize.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/highlighter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/json.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/jupyter.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/layout.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/live.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/live_render.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/logging.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/markup.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/measure.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/padding.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/pager.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/palette.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/panel.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/pretty.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/progress.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/progress_bar.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/prompt.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/protocol.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/region.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/repr.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/rule.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/scope.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/screen.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/segment.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/spinner.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/status.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/style.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/styled.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/syntax.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/table.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/terminal_theme.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/text.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/theme.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/themes.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/traceback.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/rich/tree.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/_parser.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/_re.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/_types.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/_writer.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/tomli_w/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/LICENSE",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_api.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_macos.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_openssl.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_ssl_constants.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/_windows.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/truststore/py.typed",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/_collections.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/_version.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/connection.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/connectionpool.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_appengine_environ.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_securetransport/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_securetransport/bindings.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/_securetransport/low_level.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/appengine.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/ntlmpool.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/pyopenssl.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/securetransport.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/contrib/socks.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/exceptions.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/fields.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/filepost.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/backports/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/backports/makefile.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/backports/weakref_finalize.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/packages/six.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/poolmanager.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/request.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/response.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/__init__.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/connection.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/proxy.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/queue.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/request.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/response.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/retry.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/ssl_.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/ssl_match_hostname.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/ssltransport.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/timeout.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/url.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/urllib3/util/wait.py",
    "venv/lib/python3.10/site-packages/pip/_vendor/vendor.txt",
    "venv/lib/python3.10/site-packages/pip/py.typed",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/AUTHORS.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/cachecontrol/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/certifi/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/dependency_groups/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/distlib/LICENSE.txt",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/distro/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/idna/LICENSE.md",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/msgpack/COPYING",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE.APACHE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/packaging/LICENSE.BSD",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/pkg_resources/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/platformdirs/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/pygments/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/pyproject_hooks/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/requests/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/resolvelib/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/rich/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/tomli/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/tomli_w/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/truststore/LICENSE",
    "venv/lib/python3.10/site-packages/pip-26.0.1.dist-info/licenses/src/pip/_vendor/urllib3/LICENSE.txt",
    "venv/lib/python3.10/site-packages/setuptools/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_core_metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_discovery.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_log.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_macos_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_modified.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/_msvccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/ccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/cmd.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/_framework_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/bdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/bdist_dumb.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/bdist_rpm.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/build_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/check.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/clean.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/config.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_data.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_headers.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_lib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/command/sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compat/numpy.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/base.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/cygwin.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/msvc.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_base.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_cygwin.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_mingw.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_msvc.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/tests/test_unix.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/unix.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/zos.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/core.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/cygwinccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/debug.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/dep_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/dir_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/dist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/extension.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/fancy_getopt.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/file_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/filelist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/log.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/spawn.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/sysconfig.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/support.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_bdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_bdist_dumb.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_bdist_rpm.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_build_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_check.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_clean.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_cmd.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_config_cmd.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_core.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_dir_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_dist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_extension.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_file_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_filelist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_data.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_headers.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_lib.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_log.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_modified.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_spawn.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_sysconfig.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_text_file.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_version.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/test_versionpredicate.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/tests/unix_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/text_file.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/unixccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/util.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/version.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/versionpredicate.py",
    "venv/lib/python3.10/site-packages/setuptools/_distutils/zosccompiler.py",
    "venv/lib/python3.10/site-packages/setuptools/_entry_points.py",
    "venv/lib/python3.10/site-packages/setuptools/_imp.py",
    "venv/lib/python3.10/site-packages/setuptools/_importlib.py",
    "venv/lib/python3.10/site-packages/setuptools/_itertools.py",
    "venv/lib/python3.10/site-packages/setuptools/_normalization.py",
    "venv/lib/python3.10/site-packages/setuptools/_path.py",
    "venv/lib/python3.10/site-packages/setuptools/_reqs.py",
    "venv/lib/python3.10/site-packages/setuptools/_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/_shutil.py",
    "venv/lib/python3.10/site-packages/setuptools/_static.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/.lock",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/autoasync.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/autocommand.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/automain.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/autoparse.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/autocommand-2.2.2.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/__main__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/compat/py38.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/backports.tarfile-1.2.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_adapters.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_collections.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_compat.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_functools.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_itertools.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_meta.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_text.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/_typing.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/compat/py311.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/diagnose.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/importlib_metadata-8.7.1.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/context/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/context/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/functools/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/functools/__init__.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/functools/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/Lorem ipsum.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/layouts.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/show-newlines.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/strip-prefix.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/to-dvorak.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/to-qwerty.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco.text-4.0.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_context-6.1.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/jaraco_functools-4.4.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/__init__.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/more.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/more.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/recipes.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/recipes.pyi",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/more_itertools-10.8.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_elffile.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_manylinux.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_musllinux.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_parser.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_structures.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/_tokenizer.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/licenses/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/licenses/_spdx.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/markers.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/pylock.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/requirements.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/specifiers.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/tags.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/utils.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging/version.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.APACHE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/packaging-26.0.dist-info/licenses/LICENSE.BSD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/__main__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/android.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/api.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/macos.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/unix.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/version.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs/windows.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/platformdirs-4.4.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/_parser.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/_re.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/_types.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli/py.typed",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/tomli-2.4.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/__main__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/convert.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/pack.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/tags.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_commands/unpack.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/_setuptools_logging.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/macosx_libfile.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel/wheelfile.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/wheel-0.46.3.dist-info/licenses/LICENSE.txt",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/_functools.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/overlay.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/py310.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/compat/py313.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp/glob.py",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools/_vendor/zipp-3.23.0.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/setuptools/archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/build_meta.py",
    "venv/lib/python3.10/site-packages/setuptools/cli-32.exe",
    "venv/lib/python3.10/site-packages/setuptools/cli-64.exe",
    "venv/lib/python3.10/site-packages/setuptools/cli-arm64.exe",
    "venv/lib/python3.10/site-packages/setuptools/cli.exe",
    "venv/lib/python3.10/site-packages/setuptools/command/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/command/_requirestxt.py",
    "venv/lib/python3.10/site-packages/setuptools/command/alias.py",
    "venv/lib/python3.10/site-packages/setuptools/command/bdist_egg.py",
    "venv/lib/python3.10/site-packages/setuptools/command/bdist_rpm.py",
    "venv/lib/python3.10/site-packages/setuptools/command/bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/command/build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/command/develop.py",
    "venv/lib/python3.10/site-packages/setuptools/command/dist_info.py",
    "venv/lib/python3.10/site-packages/setuptools/command/easy_install.py",
    "venv/lib/python3.10/site-packages/setuptools/command/editable_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/command/egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install_egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install_lib.py",
    "venv/lib/python3.10/site-packages/setuptools/command/install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/command/rotate.py",
    "venv/lib/python3.10/site-packages/setuptools/command/saveopts.py",
    "venv/lib/python3.10/site-packages/setuptools/command/sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/command/setopt.py",
    "venv/lib/python3.10/site-packages/setuptools/command/test.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py310.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py311.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py312.py",
    "venv/lib/python3.10/site-packages/setuptools/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/config/NOTICE",
    "venv/lib/python3.10/site-packages/setuptools/config/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_apply_pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/NOTICE",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/error_reporting.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/extra_validations.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/fastjsonschema_exceptions.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/fastjsonschema_validations.py",
    "venv/lib/python3.10/site-packages/setuptools/config/_validate_pyproject/formats.py",
    "venv/lib/python3.10/site-packages/setuptools/config/distutils.schema.json",
    "venv/lib/python3.10/site-packages/setuptools/config/expand.py",
    "venv/lib/python3.10/site-packages/setuptools/config/pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/config/setupcfg.py",
    "venv/lib/python3.10/site-packages/setuptools/config/setuptools.schema.json",
    "venv/lib/python3.10/site-packages/setuptools/depends.py",
    "venv/lib/python3.10/site-packages/setuptools/discovery.py",
    "venv/lib/python3.10/site-packages/setuptools/dist.py",
    "venv/lib/python3.10/site-packages/setuptools/errors.py",
    "venv/lib/python3.10/site-packages/setuptools/extension.py",
    "venv/lib/python3.10/site-packages/setuptools/glob.py",
    "venv/lib/python3.10/site-packages/setuptools/gui-32.exe",
    "venv/lib/python3.10/site-packages/setuptools/gui-64.exe",
    "venv/lib/python3.10/site-packages/setuptools/gui-arm64.exe",
    "venv/lib/python3.10/site-packages/setuptools/gui.exe",
    "venv/lib/python3.10/site-packages/setuptools/installer.py",
    "venv/lib/python3.10/site-packages/setuptools/launch.py",
    "venv/lib/python3.10/site-packages/setuptools/launcher manifest.xml",
    "venv/lib/python3.10/site-packages/setuptools/logging.py",
    "venv/lib/python3.10/site-packages/setuptools/modified.py",
    "venv/lib/python3.10/site-packages/setuptools/monkey.py",
    "venv/lib/python3.10/site-packages/setuptools/msvc.py",
    "venv/lib/python3.10/site-packages/setuptools/namespaces.py",
    "venv/lib/python3.10/site-packages/setuptools/script (dev).tmpl",
    "venv/lib/python3.10/site-packages/setuptools/script.tmpl",
    "venv/lib/python3.10/site-packages/setuptools/tests/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/compat/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/compat/py39.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/downloads/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/downloads/preload.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/setupcfg_examples.txt",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_apply_pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_expand.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_pyprojecttoml.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_pyprojecttoml_dynamic_deps.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/config/test_setupcfg.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/contexts.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/environment.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/fixtures.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/indexes/test_links_priority/external.html",
    "venv/lib/python3.10/site-packages/setuptools/tests/indexes/test_links_priority/simple/foobar/index.html",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/__init__.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/helpers.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/test_pbr.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/integration/test_pip_install_sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/mod_with_constant.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/namespaces.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/script-with-bom.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_archive_util.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_bdist_deprecations.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_bdist_egg.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_bdist_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_clib.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_ext.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_meta.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_build_py.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_config_discovery.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_core_metadata.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_depends.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_develop.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_dist.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_dist_info.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_distutils_adoption.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_editable_install.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_egg_info.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_extern.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_find_packages.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_find_py_modules.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_glob.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_install_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_logging.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_manifest.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_namespaces.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_scripts.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_sdist.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_setopt.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_setuptools.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_shutil_wrapper.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_unicode_utils.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_virtualenv.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_warnings.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/test_windows_wrappers.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/text.py",
    "venv/lib/python3.10/site-packages/setuptools/tests/textwrap.py",
    "venv/lib/python3.10/site-packages/setuptools/unicode_utils.py",
    "venv/lib/python3.10/site-packages/setuptools/version.py",
    "venv/lib/python3.10/site-packages/setuptools/warnings.py",
    "venv/lib/python3.10/site-packages/setuptools/wheel.py",
    "venv/lib/python3.10/site-packages/setuptools/windows_support.py",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/licenses/LICENSE",
    "venv/lib/python3.10/site-packages/setuptools-82.0.1.dist-info/top_level.txt",
    "venv/lib/python3.10/site-packages/wheel/__init__.py",
    "venv/lib/python3.10/site-packages/wheel/__main__.py",
    "venv/lib/python3.10/site-packages/wheel/_bdist_wheel.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/__init__.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/convert.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/info.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/pack.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/tags.py",
    "venv/lib/python3.10/site-packages/wheel/_commands/unpack.py",
    "venv/lib/python3.10/site-packages/wheel/_metadata.py",
    "venv/lib/python3.10/site-packages/wheel/_setuptools_logging.py",
    "venv/lib/python3.10/site-packages/wheel/bdist_wheel.py",
    "venv/lib/python3.10/site-packages/wheel/macosx_libfile.py",
    "venv/lib/python3.10/site-packages/wheel/metadata.py",
    "venv/lib/python3.10/site-packages/wheel/wheelfile.py",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/INSTALLER",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/METADATA",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/RECORD",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/REQUESTED",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/WHEEL",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/entry_points.txt",
    "venv/lib/python3.10/site-packages/wheel-0.47.0.dist-info/licenses/LICENSE.txt",
    "venv/pyvenv.cfg"
  ]
}

### [2026-04-23 10:53:49 UTC] ASSISTANT

**Thinking**

The quantization succeeded. The model was quantized and saved. Let me now update the summary JSON with the correct quantized model size. The quantized model was saved to `output_dir` directly (not to a `quantized_model` subdirectory), with `model.safetensors` being the main weight file.

Let me calculate the actual quantized model size.

Quantization succeeded. Let me update the summary with accurate sizes:


**Tool call:** `exec`

```json
{
  "command": "ls -lh /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/*.safetensors /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/*.json 2>/dev/null | head -20"
}
```

### [2026-04-23 10:53:49 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

-rw-r--r-- 1 root root 1.7K Apr 23 10:53 /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/config.json
-rw-r--r-- 1 root root  213 Apr 23 10:53 /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/generation_config.json
-rw-r--r-- 1 root root 516M Apr 23 10:53 /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/model.safetensors
-rw-r--r-- 1 root root  74K Apr 23 10:53 /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quant_summary.json
-rw-r--r-- 1 root root  196 Apr 23 10:53 /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantization_config.json
-rw-r--r-- 1 root root  11M Apr 23 10:53 /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/tokenizer.json
-rw-r--r-- 1 root root  665 Apr 23 10:53 /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/tokenizer_config.json

### [2026-04-23 10:53:57 UTC] ASSISTANT

**Thinking**

So the quantized model.safetensors is 516MB. The original model was approximately 1136.88 MB in BF16. So the compression ratio is 1136.88 / 516 = ~2.2x.

Let me now rewrite the summary JSON with the correct values.

**Tool call:** `write`

```json
{
  "path": "/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quant_summary.json",
  "content": "{\n  \"model_id\": \"Qwen/Qwen3-0.6B\",\n  \"scheme\": \"W4A16\",\n  \"method\": \"RTN\",\n  \"export_format\": \"auto_round\",\n  \"device\": \"cuda\",\n  \"num_gpus\": \"1\",\n  \"output_dir\": \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\",\n  \"quantized_model_dir\": \"/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16\",\n  \"status\": \"success\",\n  \"duration_seconds\": 68.04,\n  \"original_size_mb\": 1136.88,\n  \"quantized_size_mb\": 516.0,\n  \"compression_ratio\": 2.2,\n  \"errors\": [],\n  \"solutions\": [],\n  \"output_files\": [\n    \"chat_template.jinja\",\n    \"config.json\",\n    \"generation_config.json\",\n    \"model.safetensors\",\n    \"quant_summary.json\",\n    \"quantization_config.json\",\n    \"quantize_script.py\",\n    \"tokenizer.json\",\n    \"tokenizer_config.json\"\n  ]\n}"
}
```

### [2026-04-23 10:53:57 UTC] TOOL RESULT
**Tool:** `write`

Successfully wrote 746 bytes to /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quant_summary.json

### [2026-04-23 10:54:01 UTC] ASSISTANT

**Thinking**

Let me verify the quantization config and confirm the model structure.

**Tool call:** `exec`

```json
{
  "command": "cat /root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/quantization_config.json"
}
```

### [2026-04-23 10:54:01 UTC] TOOL RESULT
**Tool:** `exec` | **Status:** `completed` | **Exit code:** `0`

{
  "bits": 4,
  "data_type": "int",
  "group_size": 128,
  "sym": true,
  "iters": 0,
  "autoround_version": "0.12.2",
  "quant_method": "auto-round",
  "packing_format": "auto_round:auto_gptq"
}

### [2026-04-23 10:54:11 UTC] ASSISTANT

**Thinking**

The quantization is confirmed as W4A16 (4-bit weight, activation in bf16), with group size 128, symmetric quantization, using RTN (iters=0), and the export format is auto_round with auto_gptq packing.

Let me now also verify the output files are correct.

Quantization completed successfully. Here's the summary:

**Qwen/Qwen3-0.6B — W4A16 / RTN / auto_round**

| Metric | Value |
|--------|-------|
| Original size | 1136.88 MB |
| Quantized size | 516.0 MB |
| Compression | **2.2x** |
| Duration | 68.04 seconds |
| Status | ✅ success |

**Output files in** `/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16/`:
- `model.safetensors` (516 MB) — quantized weights
- `config.json`, `generation_config.json`, `quantization_config.json` — model configs
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` — tokenizer
- `quant_summary.json` — quantization report

**Quantization config:** 4-bit INT weights, group_size=128, symmetric, RTN mode (iters=0), auto_round+auto_gptq packing format.

`quant_summary.json` has been written to the output directory.
