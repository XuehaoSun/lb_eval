---
name: error_analysis
description: Analyze quantization/evaluation pipeline failure logs — trace root causes, diagnose issues, and propose concrete fixes like a senior engineer.
metadata:
  openclaw:
    emoji: "🔍"
    skillKey: error-analysis
    requires:
      bins: [python3, git, pip]
      env: []
      config: []
---

# Error Analysis — Expert Debugging for Auto-Quantization Pipeline

You are diagnosing failures in a pipeline that takes HuggingFace models through:
```
setup_env → quantize (auto-round) → evaluate (lm_eval) → upload (HuggingFace Hub)
```

Your job is to **think like a senior engineer**: trace the actual root cause, not just pattern-match the error message. Below is the methodology.

---

## Part 1: How to Read a Failure Log

### 1.1 Start From the Bottom

Always read **backwards**. The last exception is where the program died. Example:

```
Exception: expected `,` or `}` at line 1 column 9
```

This is the SYMPTOM. Now trace UP the traceback to find the CAUSE.

### 1.2 Follow the Call Stack Upward

```
File "quantize.py", line 130, in quantize
    tokenizer = AutoTokenizer.from_pretrained(...)
File "tokenization_utils_base.py", line 1931, in _from_pretrained
    init_kwargs = cls.convert_to_native_format(**init_kwargs)
File "tokenization_utils_tokenizers.py", line 116, in convert_to_native_format
    local_kwargs["tokenizer_object"] = TokenizerFast.from_file(fast_tokenizer_file)
```

**What you learn:** The quantize script calls `AutoTokenizer.from_pretrained()` → transformers tries to load a fast tokenizer → `TokenizerFast.from_file()` fails parsing JSON.

**Root cause:** The `tokenizer.json` file on HuggingFace is malformed JSON. This is the MODEL AUTHOR's fault, not ours.

### 1.3 Handle Chained Exceptions

When you see:
```
The above exception was the direct cause of the following exception:
```
or:
```
During handling of the above exception, another exception occurred:
```

Always trace to the **ORIGINAL** (topmost) exception — that's the real cause.

### 1.4 Look for Version Information in the Log

Before the traceback, the log often contains environment info:
```
transformers==4.52.4
auto_round==0.6.3
torch==2.7.1+cu126
```

**Write these down.** They tell you whether a version incompatibility is possible.

### 1.5 Look for Context BEFORE the Error

The 20-30 lines before `Traceback` often reveal:
- Which model was being loaded
- Which layer was being quantized (progress bar)
- Memory usage at failure time (peak_ram, peak_vram)
- HTTP requests that failed (404, 504)

---

## Part 2: Diagnostic Decision Trees

### 2.1 Tokenizer Failures

**Symptom:** Error in `TokenizerFast.from_file()` or `AutoTokenizer.from_pretrained()`

**Decision tree:**
1. Error is `expected , or }` or `invalid JSON` → tokenizer.json is **corrupt on HuggingFace**
   - Verify: `curl -s https://huggingface.co/{model}/raw/main/tokenizer.json | python3 -m json.tool`
   - If invalid JSON → **NOT FIXABLE BY US.** Model author must fix their upload.
   - Workaround: Try `use_fast=False` (uses slow tokenizer, may work if tokenizer_config.json is OK)
2. Error is `Tokenizer class 'XXXTokenizer' does not exist` → model uses custom tokenizer
   - Check: Does model need `trust_remote_code=True`?
   - Check: Is the custom tokenizer class defined in the model repo?
   - Fix: Add `trust_remote_code=True` to `from_pretrained()` call
3. Error is `Can't load tokenizer... doesn't appear to have a file named tokenizer.json` → model has no fast tokenizer
   - Fix: Use `use_fast=False`

### 2.2 Model Loading Failures

**Symptom:** Error in `AutoModelForCausalLM.from_pretrained()` or `from_config()`

**Decision tree:**
1. `unexpected keyword argument` → transformers version too old for this model
   - Check: What transformers version does the model's `config.json` require?
   - Fix: `pip install -U transformers`
   - Verify: `python3 -c "import transformers; print(transformers.__version__)"`
2. `Could not import module` / `modeling_xxx.py` → custom model code fails
   - Check: Read the model's custom `modeling_*.py` on HuggingFace
   - Check: Does it import packages we don't have? (`flash_attn`, `mamba_ssm`, etc.)
   - Fix: `pip install <missing_package>` then ensure `trust_remote_code=True`
3. `size mismatch for ...` → checkpoint incompatible with config
   - This means model author uploaded mismatched weights. **NOT FIXABLE.**
4. `assert processor is not None` / `image_processor` → multimodal model
   - This pipeline is text-only. Multimodal models are **NOT SUPPORTED.**
   - Report as unsupported architecture, skip.

### 2.3 Quantization Failures (auto-round internal)

**Symptom:** Error inside `auto_round/` package code

**Decision tree:**
1. `shape not divisible by group_size` / `cannot reshape` → layer dimensions incompatible
   - This happens with unusual architectures (e.g., MoE with non-standard expert sizes)
   - Fix: Try `--group_size -1` (per-channel) or skip specific layers
   - Check: `python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('{model}'); print(c)"` to see architecture
2. `IndexError` in `_sampling_inputs` → calibration data format issue
   - Check: Is the model expecting chat format vs plain text?
   - Fix: Try different calibration dataset
3. `KeyError` in layer processing → model has non-standard layer names
   - Check: What are the actual layer names? `python3 -c "from transformers import AutoModel; m=AutoModel.from_pretrained('{model}', device_map='meta'); print([n for n,_ in m.named_modules()][:20])"`
   - Fix: May need `--layer_config` or auto-round upgrade
4. Error in `compressors/mllm/` → multimodal compressor shouldn't be used for text model
   - Check model architecture — if it's not multimodal, this is an auto-round routing bug
   - Fix: `pip install -U auto-round` (from main branch)

### 2.4 dtype / Tensor Errors

**Symptom:** `RuntimeError: expected mat1 and mat2 to have the same dtype`

**Diagnosis:**
1. Check: Which dtypes? (e.g., `BFloat16 != Half`)
2. Where: Usually in the evaluation phase after quantization
3. Why: Quantized model saved in float16, but some layers kept bfloat16 (or vice versa)
4. Fix: Add `--model_dtype float16` to the quantize command, OR cast during eval
5. Verify: `python3 -c "import torch; from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('{quant_model}', device_map='auto'); print(set(p.dtype for p in m.parameters()))"`

**IMPORTANT:** If the error is `BFloat16 != Half` in a MoE model's router layer (`gate`, `router`), it's because the router weights weren't quantized and kept their original dtype while attention layers were cast to float16. Fix: ensure all layers use the same dtype after quantization.

### 2.5 Out of Memory

**Symptom:** `CUDA out of memory`, `OutOfMemoryError`, process killed by signal 9, exit code 137

**Diagnosis process:**
1. Check RAM progression in the log: look for `peak_ram` and `peak_vram` lines
   - If peak_vram grows linearly → normal quantization memory growth
   - If peak_vram jumps suddenly → a specific layer is too large
2. Calculate expected memory:
   - Model param count × 2 bytes (fp16) = minimum GPU memory for weights
   - Quantization needs ~2-3x that (weights + activations + calibration data)
   - Evaluation needs ~1.5x model size (weights + KV cache)
3. Fix priority:
   - First: `--low_gpu_mem_usage` flag in quantize
   - Second: Reduce `--nsamples 128` → `--nsamples 64`
   - Third: Reduce `--seqlen 2048` → `--seqlen 512`
   - Last resort: Multi-GPU or skip model

### 2.6 Network/Hub Errors

**Symptom:** `504 Gateway Time-out`, `ConnectionError`, `SSLError`

**Key insight:** Check if this is during MODEL DOWNLOAD or DATASET DOWNLOAD:
- If `huggingface.co/api/models/` → model download issue → retry
- If `huggingface.co/api/datasets/cais/mmlu/` → evaluation dataset download issue → retry with `HF_DATASETS_OFFLINE=1` if cached

**IMPORTANT:** 504 on mmlu dataset is VERY COMMON in our pipeline. It's a HuggingFace CDN overload issue. The fix is always just RETRY. If it happens 3+ times, check if we can pre-cache datasets.

### 2.7 Process Killed (Signal 9/137)

**Symptom:** No traceback, just `Killed` or process exits with code 137

**Diagnosis:**
1. Check `dmesg | grep -i "oom"` — was it the OOM killer?
2. Check the last progress indicator before death — which layer/step was it on?
3. Check system memory at death: if `peak_ram` was climbing toward system limit, it's OOM
4. If no OOM evidence → could be pipeline timeout (check if there's a timeout setting)

---

## Part 3: Environment Investigation Commands

When you need to verify your hypothesis, run these:

```bash
# Package versions
pip show transformers auto-round torch tokenizers lm-eval | grep -E "^(Name|Version)"

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# CUDA version
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}')"

# Model architecture quick check
python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('{MODEL_ID}', trust_remote_code=True)
print(f'arch: {c.architectures}')
print(f'hidden: {c.hidden_size}, layers: {c.num_hidden_layers}')
print(f'vocab: {c.vocab_size}')
"

# Tokenizer validity check
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('{MODEL_ID}', use_fast=False)
print(f'OK: {type(t).__name__}, vocab_size={t.vocab_size}')
"

# Check available disk/memory
df -h / && free -h

# Check if model exists on HuggingFace
curl -sI "https://huggingface.co/api/models/{MODEL_ID}" | head -3
```

---

## Part 4: Fix Execution and Verification

### 4.1 Fix Strategy Priority

Always try fixes in this order (least invasive first):
1. **Configuration change** — add a flag, change a parameter
2. **Package upgrade** — `pip install -U <package>`
3. **Workaround** — alternative code path (e.g., `use_fast=False`)
4. **Code patch** — modify a source file (last resort, document clearly)

### 4.2 Before Applying a Fix

Ask yourself:
- Will this fix ONLY this error, or could it break something else?
- Is this fix reproducible? (If another model hits the same error, will this fix apply?)
- Am I treating the symptom or the cause?

### 4.3 After Applying a Fix

Verify with a minimal smoke test before re-running the full phase:
```bash
# For tokenizer fixes:
python3 -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('{MODEL}', use_fast=False); print('OK')"

# For model loading fixes:
python3 -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{MODEL}', device_map='meta', trust_remote_code=True); print('OK')"

# For dependency fixes:
python3 -c "import {package}; print(f'{package.__name__}=={package.__version__}')"
```

### 4.4 Patching Model Custom Code (CRITICAL TECHNIQUE)

When a model uses `trust_remote_code=True`, HuggingFace downloads the model's custom Python files to a local cache. **YOU CAN AND SHOULD modify this cached code to fix bugs.**

**Where the code lives:**
```
~/.cache/huggingface/modules/transformers_modules/{org}/{model_name_encoded}/{commit_hash}/
```

Example: For `Quazim0t0/Escarda-86M-Base`, the path would be:
```
~/.cache/huggingface/modules/transformers_modules/Quazim0t0/Escarda_hyphen_86M_hyphen_Base/{hash}/model_v2.py
```

Note: HuggingFace encodes special characters — hyphens become `_hyphen_`, dots become `_dot_`.

**How to find the exact path:**
```bash
# From the traceback — the path is RIGHT THERE in the error:
# File "/root/.cache/huggingface/modules/transformers_modules/Quazim0t0/Escarda_hyphen_86M_hyphen_Base/cf953b.../model_v2.py", line 147
# Just read that path directly!

# Or search by model name:
find ~/.cache/huggingface/modules/transformers_modules -name "*.py" | grep -i "{model_name}"
```

**Common dtype fixes in model custom code:**

The most common bug is mixed dtypes. Model authors write `.float()` or hardcode dtypes:
```python
# BUG (original):
h = h + torch.matmul(compressed[:, k:k+valid_len, :].float(), proj.t())
# proj is bfloat16, .float() makes compressed float32 → CRASH

# FIX:
h = h + torch.matmul(compressed[:, k:k+valid_len, :].to(proj.dtype), proj.t())
```

**General pattern for dtype fixes:**
```python
# Replace .float() or .half() with .to(reference_tensor.dtype)
# Replace hardcoded torch.float32 with the actual model dtype
# Use .to(self.weight.dtype) or .to(hidden_states.dtype) to match existing tensors
```

**Other common model code bugs you can fix:**
```python
# Missing device handling:
# BUG: tensor = torch.zeros(size)  ← goes to CPU
# FIX: tensor = torch.zeros(size, device=hidden_states.device, dtype=hidden_states.dtype)

# Incompatible attention mask shape:
# BUG: mask = torch.ones(seq_len, seq_len)
# FIX: mask = torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype)

# Missing return value that newer transformers expects:
# BUG: return hidden_states
# FIX: return hidden_states, None, None  # (hidden, present_kv, aux_loss)
```

**Workflow for patching model code:**
1. Read the traceback — identify the EXACT file and line number
2. Read that file: `cat {path_from_traceback}`
3. Understand the bug (usually dtype mismatch or missing device)
4. Apply minimal fix with `sed` or by editing the file directly
5. Verify: `python3 -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{MODEL}', device_map='cpu', trust_remote_code=True); print('Load OK')"`
6. Let the pipeline re-run the quantize phase

**This is NOT "modifying the model" — it's fixing a bug in the author's Python code so quantization can proceed. The model weights are unchanged.**

### 4.5 When NOT to Fix

**Stop and report** (do NOT attempt a fix) when:
- The model is multimodal and our pipeline is text-only
- The model is private/gated and we don't have access
- Fixing would require downgrading PyTorch (NEVER do this)
- The model architecture is fundamentally incompatible (e.g., no standard forward() signature)
- The bug is in the model's WEIGHTS (not code) — corrupt safetensors, mismatched shapes

---

## Part 5: Analyzing Unknown Errors

When the error doesn't match any known pattern:

### Step 1: Identify the Component

Look at the traceback file paths:
- `auto_round/` → auto-round library
- `transformers/` → HuggingFace transformers
- `torch/` → PyTorch itself
- `lm_eval/` → evaluation framework
- Model's custom code (`transformers_modules/`) → model-specific code

### Step 2: Determine if It's a Bug or a Limitation

- **Bug**: The code should handle this case but doesn't (regression, edge case)
  - Evidence: It worked before, or similar models work fine
  - Fix: Upgrade the package, or patch the specific function
- **Limitation**: The code was never designed for this case
  - Evidence: The architecture is unusual, or the model does something non-standard
  - Fix: May not be fixable. Report upstream.

### Step 3: Search for Known Issues

```bash
# Search auto-round issues
curl -s "https://api.github.com/search/issues?q=repo:intel/auto-round+{ERROR_KEYWORD}" | python3 -c "import json,sys; [print(f'#{i[\"number\"]}: {i[\"title\"]}') for i in json.load(sys.stdin).get('items',[])]"

# Search transformers issues
curl -s "https://api.github.com/search/issues?q=repo:huggingface/transformers+{ERROR_KEYWORD}" | python3 -c "import json,sys; [print(f'#{i[\"number\"]}: {i[\"title\"]}') for i in json.load(sys.stdin).get('items',[])]"
```

### Step 4: Form a Hypothesis and Test It

Don't guess. Form a specific, testable hypothesis:
- BAD: "Maybe it's a version issue"
- GOOD: "The `convert_to_native_format` function was added in transformers 4.50. If our version is older, it would fail here. Let me check: `pip show transformers`"

### Step 5: Suggest Taxonomy Update

If this is genuinely a new error class, suggest adding it:
```
New category suggestion: {name}
Signatures: ["{regex_pattern}"]
Root cause: {why this happens}
Fix strategy: {what to do}
```

---

## Part 6: Common Patterns I've Seen Repeatedly

### Pattern: "Tokenizer JSON parse error"
- `expected , or } at line 1 column 9`
- 90% of the time: model author uploaded a corrupt `tokenizer.json`
- Workaround: `AutoTokenizer.from_pretrained(model, use_fast=False)`
- If slow tokenizer also fails → model is broken, skip it

### Pattern: "MoE model dtype mismatch during eval"
- `BFloat16 != Half` in `modeling_mixtral.py` router
- Cause: Router/gate weights stay bfloat16 while quantized layers become float16
- Fix: Force dtype during eval: `model.to(torch.float16)` before running lm_eval

### Pattern: "504 on MMLU dataset"
- HuggingFace CDN overload — happens in waves
- Always just retry. If persistent (3+ failures), pre-cache: `datasets.load_dataset("cais/mmlu", "all")` separately

### Pattern: "Custom model code imports missing package"
- The model's `modeling_*.py` does `from flash_attn import ...`
- Our env doesn't have it, but it's optional for inference
- Fix: `pip install flash-attn --no-build-isolation` OR check if there's a fallback path in the model code

### Pattern: "Custom model code dtype bug during calibration"
- `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16`
- Traceback shows `transformers_modules/{org}/{model}/model_*.py` — it's MODEL CODE, not ours
- Cause: Model author hardcodes `.float()` or `.half()` in forward pass, but other tensors are bfloat16
- **FIX (you CAN do this):**
  1. Read the file path from the traceback (it's in `~/.cache/huggingface/modules/transformers_modules/...`)
  2. Open that file, find the line
  3. Replace `.float()` with `.to(other_tensor.dtype)` — use the OTHER operand's dtype as reference
  4. Example: `torch.matmul(x.float(), proj.t())` → `torch.matmul(x.to(proj.dtype), proj.t())`
- Verification: `python3 -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{MODEL}', device_map='cpu', trust_remote_code=True); print('OK')"`

### Pattern: "Killed during layer N quantization, RAM climbing"
- Model is too large for system RAM (not GPU — RAM!)
- Check: Was `peak_ram` approaching system total?
- Fix: `--low_gpu_mem_usage` offloads layers to disk during quantization

### Pattern: "Regular expression group error in custom model"
- `re/_parser.py addgroup` error → model's custom code uses invalid regex
- This is in the model's custom code at `~/.cache/huggingface/modules/transformers_modules/...`
- **FIX**: Find the regex in the model file and fix it (usually an unescaped special char)
- Read the traceback to find the exact file and line, then patch it

### Pattern: "`assert processor is not None`"
- Model is multimodal (VL/audio) but loaded in text pipeline
- NOT FIXABLE. Our pipeline is text-only.
- Report as unsupported architecture.

---

## Part 7: Output Format

After analysis, output a JSON diagnosis:

```json
{
  "category": "tokenizer_error|transformers_incompatible|autoround_internal|pytorch_cuda|dtype_mismatch|out_of_memory|multimodal_unsupported|missing_dependency|dataset_error|eval_framework|network_error|model_unavailable|process_killed|unknown",
  "phase": "quantize|evaluate|setup_env|upload",
  "key_error": "<exact final error line from the log>",
  "root_cause": "<detailed explanation: what happened, why, and how you traced it>",
  "traceback_analysis": "<walk through the call stack explaining what each frame means>",
  "retryable": true|false,
  "fix_available": true|false,
  "suggested_fix": "<exact commands to run, in order>",
  "fix_verification": "<command to verify the fix worked before full re-run>",
  "workaround": "<alternative approach if primary fix is risky>",
  "affected_component": "auto_round|transformers|tokenizers|torch|lm_eval|model_code|infrastructure|model_data",
  "severity": "critical|high|medium|low",
  "confidence": 0.0-1.0,
  "versions_involved": {
    "auto_round": "<version from log or pip>",
    "transformers": "<version from log or pip>",
    "torch": "<version from log or pip>"
  },
  "similar_known_issues": ["<GitHub issue URLs if found>"],
  "new_taxonomy_suggestion": null | {"name": "...", "signatures": ["..."], "fix_strategy": "..."},
  "community_summary": "<2-3 sentence summary suitable for a public issue report>"
}
```

---

## Constraints (HARD RULES)

- **NEVER** downgrade PyTorch — it breaks CUDA/pipeline. Not negotiable.
- **NEVER** modify evaluation tasks, scoring criteria, or expected output format.
- **NEVER** skip quantization layers to hide errors (that produces invalid models).
- **NEVER** guess without evidence. If you're unsure, say confidence < 0.5.
- **ALWAYS** show your reasoning chain (traceback analysis → hypothesis → verification).
- **ALWAYS** try the minimal fix first. Don't `pip install -U` everything.
- **ALWAYS** provide a verification command so the fix can be tested quickly.
