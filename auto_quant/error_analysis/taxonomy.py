# Error Classification Taxonomy for Auto-Quantization Pipeline
#
# Used by openclaw agent to categorize, diagnose, and propose fixes for pipeline failures.
# Each category has: signature patterns, root cause analysis guide, and fix strategy.

TAXONOMY = {
    # ══════════════════════════════════════════════════════════════
    # Category 1: AutoRound Code Issues
    # ══════════════════════════════════════════════════════════════
    "autoround_internal_error": {
        "description": "Bug or limitation in AutoRound library code itself",
        "signatures": [
            r"auto_round.*Error",
            r"auto_round.*assert",
            r"shape not divisible by",
            r"autoround.*RuntimeError",
            r"quantize\.py.*AutoRound",
        ],
        "root_cause_guide": (
            "Check auto_round version vs model architecture compatibility. "
            "Look at layer shapes, quantization config parsing, and export logic. "
            "Compare with supported model list in auto_round documentation."
        ),
        "fix_strategy": "upgrade_autoround_or_patch",
        "retryable": True,
        "workaround_hints": [
            "pip install auto-round@git+https://github.com/intel/auto-round.git@main",
            "Try scheme=W8A16 if W4A16 fails on specific layers",
            "Use --low_gpu_mem_usage for large models",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 2: Transformers / HuggingFace Issues
    # ══════════════════════════════════════════════════════════════
    "transformers_incompatible": {
        "description": "Incompatibility between transformers version and model requirements",
        "signatures": [
            r"(?:ImportError|AttributeError|TypeError).*transformers",
            r"transformers.*(?:ImportError|AttributeError|TypeError)",
            r"AutoModelForCausalLM.*unexpected keyword",
            r"does not appear to have a file named config\.json",
            r"trust_remote_code.*(?:Error|must be)",
            r"Could not import module.*modeling",
            r"is not a valid model identifier",
            r"not a local folder and is not a valid model identifier",
        ],
        "root_cause_guide": (
            "Check if model requires newer transformers version. "
            "Look for custom modeling code (trust_remote_code=True). "
            "Check if model repo was deleted or made private on HuggingFace."
        ),
        "fix_strategy": "upgrade_transformers_or_configure",
        "retryable": True,
        "workaround_hints": [
            "pip install -U transformers",
            "Set trust_remote_code=True in model loading",
            "Check if model is still available on HuggingFace Hub",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 3: Tokenizer Issues
    # ══════════════════════════════════════════════════════════════
    "tokenizer_error": {
        "description": "Tokenizer loading or parsing failure",
        "signatures": [
            r"tokenizer.*Error",
            r"Tokenizer.*does not exist",
            r"Tokenizer class.*not.*imported",
            r"TokenizerFast\.from_file",
            r"expected.*at line.*column",
            r"tokenizer\.json",
            r"tokenizer_config\.json",
            r"Can't load tokenizer",
        ],
        "root_cause_guide": (
            "Check if tokenizer.json is malformed on HuggingFace Hub. "
            "Check if model uses custom tokenizer class needing trust_remote_code. "
            "Clear HF cache and retry. Check tokenizers library version."
        ),
        "fix_strategy": "fix_tokenizer_loading",
        "retryable": True,
        "workaround_hints": [
            "pip install -U tokenizers transformers",
            "Clear ~/.cache/huggingface/hub/models--{org}--{model}/",
            "Use use_fast=False for problematic tokenizers",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 4: PyTorch / CUDA / Kernel Issues
    # ══════════════════════════════════════════════════════════════
    "pytorch_cuda_error": {
        "description": "PyTorch version, CUDA driver, or kernel incompatibility",
        "signatures": [
            r"NVIDIA driver.*too old",
            r"CUDA.*not available",
            r"RuntimeError.*CUDA",
            r"torch.*not compiled with CUDA",
            r"cudaErrorNoKernelImageForDevice",
            r"no kernel image is available",
            r"CUBLAS_STATUS",
            r"cuDNN",
        ],
        "root_cause_guide": (
            "CRITICAL: Do NOT downgrade PyTorch (breaks other pipeline stages). "
            "Check CUDA driver version vs PyTorch CUDA version. "
            "Check GPU compute capability vs compiled kernels."
        ),
        "fix_strategy": "environment_config_only",
        "retryable": False,
        "workaround_hints": [
            "This is an environment issue - cannot be fixed by agent",
            "Requires driver update or PyTorch version matching CUDA driver",
            "Report as infrastructure issue",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 5: dtype Mismatch
    # ══════════════════════════════════════════════════════════════
    "dtype_mismatch": {
        "description": "Tensor dtype incompatibility during quantization or evaluation",
        "signatures": [
            r"BFloat16.*Half",
            r"Half.*BFloat16",
            r"expected.*same dtype.*BFloat16.*Half",
            r"expected.*same dtype.*Half.*BFloat16",
            r"RuntimeError.*mat1 and mat2.*same dtype",
            r"bfloat16.*float16",
            r"cannot safely cast",
        ],
        "root_cause_guide": (
            "Check if model uses bfloat16 but quantization or eval expects float16. "
            "Check --dtype parameter in auto-round call. "
            "Check if GPU supports bfloat16 (requires Ampere+)."
        ),
        "fix_strategy": "adjust_dtype_config",
        "retryable": True,
        "workaround_hints": [
            "Add --model_dtype float16 to auto-round command",
            "Set torch_dtype=torch.float16 in model loading",
            "For eval: use --dtype float16 in lm_eval",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 6: Out of Memory
    # ══════════════════════════════════════════════════════════════
    "out_of_memory": {
        "description": "GPU VRAM or system RAM exhausted",
        "signatures": [
            r"CUDA out of memory",
            r"OutOfMemoryError",
            r"torch\.cuda\.OutOfMemoryError",
            r"Killed.*signal 9",
            r"Cannot allocate memory",
            r"MemoryError",
        ],
        "root_cause_guide": (
            "Check model size vs available GPU memory. "
            "Check if --low_gpu_mem_usage is enabled. "
            "Check batch_size and nsamples settings."
        ),
        "fix_strategy": "reduce_memory_usage",
        "retryable": True,
        "workaround_hints": [
            "Add --low_gpu_mem_usage to auto-round",
            "Reduce nsamples (e.g., 64 or 32)",
            "Reduce seqlen (e.g., 512)",
            "Use device_map='auto' for multi-GPU sharding",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 7: Multimodal / Unsupported Architecture
    # ══════════════════════════════════════════════════════════════
    "multimodal_unsupported": {
        "description": "Model is multimodal (vision/audio) and not supported by text-only pipeline",
        "signatures": [
            r"processor should not be None",
            r"image_processor",
            r"Can't load.*image processor",
            r"MultiModal",
            r"VisionModel",
            r"Qwen.*VL",
            r"video_processor",
        ],
        "root_cause_guide": (
            "This model contains vision/audio components that the text-only "
            "quantization pipeline cannot handle. This is NOT fixable by the agent."
        ),
        "fix_strategy": "not_fixable",
        "retryable": False,
        "workaround_hints": [
            "Skip this model - requires multimodal quantization support",
            "Report as unsupported architecture",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 8: Missing Dependencies
    # ══════════════════════════════════════════════════════════════
    "missing_dependency": {
        "description": "Required Python package or system library not installed",
        "signatures": [
            r"ModuleNotFoundError",
            r"ImportError",
            r"No module named",
            r"cannot import name",
            r"pkg_resources.*not found",
        ],
        "root_cause_guide": (
            "Check which package is missing. "
            "Check if it's a model-specific requirement (custom code). "
            "Check requirements.txt or setup.py of the model repo."
        ),
        "fix_strategy": "install_missing_package",
        "retryable": True,
        "workaround_hints": [
            "pip install <missing_package>",
            "Check model card for installation requirements",
            "pip install -r requirements.txt from model repo",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 9: Dataset / Calibration Issues
    # ══════════════════════════════════════════════════════════════
    "dataset_error": {
        "description": "Calibration dataset loading or format issues",
        "signatures": [
            r"Columns.*not in the dataset",
            r"attention_mask.*not in",
            r"Dataset.*Error",
            r"datasets.*ConnectionError",
            r"FileNotFoundError.*dataset",
        ],
        "root_cause_guide": (
            "Check if calibration dataset format matches model tokenizer output. "
            "Some models don't produce attention_mask. "
            "Check network connectivity to HuggingFace datasets."
        ),
        "fix_strategy": "adjust_dataset_config",
        "retryable": True,
        "workaround_hints": [
            "Try --dataset 'NeelNanda/pile-10k' as alternative",
            "Disable attention_mask requirement if model doesn't use it",
            "Check HuggingFace API status for dataset availability",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 10: Evaluation Framework Issues
    # ══════════════════════════════════════════════════════════════
    "eval_framework_error": {
        "description": "lm_eval harness or evaluation task issues",
        "signatures": [
            r"lm_eval.*Error",
            r"harness.*Error",
            r"task.*not found",
            r"torch_compile.*unexpected keyword",
            r"504 Gateway Time-out.*datasets",
            r"Server error.*huggingface\.co/api/datasets",
        ],
        "root_cause_guide": (
            "Check lm_eval version compatibility with model wrapper. "
            "Check if eval tasks are available (network issues). "
            "Check if quantized model wrapper accepts all lm_eval kwargs."
        ),
        "fix_strategy": "fix_eval_config",
        "retryable": True,
        "workaround_hints": [
            "Retry (may be transient HuggingFace 504 error)",
            "pip install -U lm_eval",
            "Check if model's __init__ accepts torch_compile kwarg",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 11: Network / Hub Connectivity
    # ══════════════════════════════════════════════════════════════
    "network_error": {
        "description": "Network timeout or HuggingFace Hub connectivity issues",
        "signatures": [
            r"ConnectionError",
            r"TimeoutError",
            r"HTTPError.*5\d\d",
            r"504 Gateway Time-out",
            r"Connection reset by peer",
            r"SSLError",
        ],
        "root_cause_guide": (
            "Check if this is a transient HuggingFace API issue. "
            "Check proxy/VPN settings. "
            "May resolve on retry."
        ),
        "fix_strategy": "retry_or_wait",
        "retryable": True,
        "workaround_hints": [
            "Wait and retry (transient server issue)",
            "Set HF_HUB_DOWNLOAD_TIMEOUT=300",
            "Check HuggingFace status page",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 12: Model Deleted / Unavailable
    # ══════════════════════════════════════════════════════════════
    "model_unavailable": {
        "description": "Model removed from HuggingFace Hub or made private",
        "signatures": [
            r"404 Client Error",
            r"Repository Not Found",
            r"does not appear to have a file named",
            r"is not a valid model identifier",
            r"We couldn't connect to.*to download",
            r"gated.*access",
            r"Access to model.*is restricted",
        ],
        "root_cause_guide": (
            "Model has been deleted, made private, or is gated. "
            "This is NOT fixable by the agent."
        ),
        "fix_strategy": "not_fixable",
        "retryable": False,
        "workaround_hints": [
            "Model no longer available - mark as permanently failed",
            "Contact model author if gated access is needed",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # Category 13: Process Killed (OOM Killer, Timeout)
    # ══════════════════════════════════════════════════════════════
    "process_killed": {
        "description": "Process terminated by OS (OOM killer) or pipeline timeout",
        "signatures": [
            r"Killed",
            r"signal 9",
            r"SIGKILL",
            r"exit code 137",
            r"exit code -9",
        ],
        "root_cause_guide": (
            "Process was killed by OS OOM killer or exceeded pipeline time limit. "
            "Check system RAM usage and model size. "
            "Check if quantization is stuck (infinite loop)."
        ),
        "fix_strategy": "reduce_resource_usage",
        "retryable": True,
        "workaround_hints": [
            "Use --low_gpu_mem_usage to reduce peak memory",
            "Reduce nsamples and seqlen",
            "Check if model fits in available RAM",
        ],
    },
}


def classify_error(log_text: str) -> tuple[str, dict]:
    """Classify an error log into a taxonomy category.

    Returns (category_name, category_dict) or ("unknown", {...}) if no match.
    """
    import re

    for category, info in TAXONOMY.items():
        for pattern in info["signatures"]:
            if re.search(pattern, log_text, re.IGNORECASE):
                return category, info

    return "unknown", {
        "description": "Unclassified error - requires manual analysis",
        "signatures": [],
        "root_cause_guide": "No matching pattern found. Manual log inspection required.",
        "fix_strategy": "manual_investigation",
        "retryable": None,
        "workaround_hints": ["Inspect full log for error context"],
    }
