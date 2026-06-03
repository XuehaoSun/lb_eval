#!/usr/bin/env python3
"""Pre-flight dependency check for model quantization.

Proactively detects and installs model-specific dependencies BEFORE quantization,
avoiding reliance on the agent fix loop for common dependency issues.

Checks performed:
  1. Model config.json → transformers_version requirement
  2. Model repo → requirements.txt (if exists)
  3. Model auto_map → try importing custom code, install missing deps
  4. Known model_type → architecture-specific deps (e.g., mamba needs mamba-ssm)

Usage:
    python preflight_deps.py --model <model_id> [--install]

Exit codes:
    0 — all dependencies satisfied
    1 — missing dependencies (printed to stdout), --install not set
    2 — install attempted but failed
"""

import argparse
import importlib
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [preflight] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Known model_type → extra packages mapping
# These are architectures that need specific packages beyond transformers
KNOWN_DEPS = {
    "mamba": ["mamba-ssm", "causal-conv1d"],
    "mamba2": ["mamba-ssm", "causal-conv1d"],
    "rwkv": ["rwkv"],
    "rwkv5": ["rwkv"],
    "persimmon": ["einops"],
    "phi": ["einops"],
    "phi3": ["einops"],
    "stablelm": ["einops"],
    "gpt_bigcode": ["einops"],
    "cohere": ["einops"],
    "dbrx": ["einops"],
    "jamba": ["mamba-ssm"],
    "zamba": ["mamba-ssm"],
    "recurrentgemma": ["einops"],
}


def get_model_config(model_id: str) -> dict:
    """Download and parse model config.json from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "config.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not download config.json: {e}")
        return {}


def get_repo_requirements(model_id: str) -> list[str]:
    """Check if model repo has a requirements.txt and parse it."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "requirements.txt")
        with open(path) as f:
            reqs = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    reqs.append(line)
            return reqs
    except Exception:
        return []


def check_transformers_version(config: dict) -> list[str]:
    """Check if installed transformers meets model's requirement."""
    required = config.get("transformers_version")
    if not required:
        return []

    try:
        import transformers
        from packaging.version import Version

        installed = Version(transformers.__version__)
        needed = Version(required)

        if installed < needed:
            return [f"transformers>={required}"]
    except Exception:
        pass

    return []


def check_known_deps(config: dict) -> list[str]:
    """Check known model_type → package mappings."""
    model_type = config.get("model_type", "")
    deps = KNOWN_DEPS.get(model_type, [])

    missing = []
    for pkg in deps:
        module_name = pkg.replace("-", "_")
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pkg)

    return missing


def check_custom_code_deps(model_id: str, config: dict) -> list[str]:
    """Try loading custom code and catch ImportErrors to find missing deps."""
    auto_map = config.get("auto_map", {})
    if not auto_map:
        return []

    # Try to load the tokenizer/model to trigger ImportErrors
    missing = []
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except ImportError as e:
        # Extract package name from ImportError
        msg = str(e)
        if "No module named" in msg:
            module = msg.split("No module named")[-1].strip().strip("'\"")
            base_pkg = module.split(".")[0]
            missing.append(base_pkg)
    except Exception:
        pass

    return missing


def install_packages(packages: list[str]) -> bool:
    """Attempt to install packages via uv pip."""
    if not packages:
        return True

    cmd = ["uv", "pip", "install"] + packages
    logger.info(f"Installing: {' '.join(packages)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"uv pip install failed:\n{result.stderr[-500:]}")
        return False

    logger.info("Installation successful")
    return True


def preflight(model_id: str, do_install: bool = False) -> int:
    """Run all pre-flight dependency checks.

    Returns:
        0 if all deps satisfied, 1 if missing (no install), 2 if install failed
    """
    logger.info(f"Pre-flight check for: {model_id}")

    config = get_model_config(model_id)
    if not config:
        logger.warning("Could not load config, skipping pre-flight (will rely on agent)")
        return 0

    model_type = config.get("model_type", "unknown")
    auto_map = config.get("auto_map", {})
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  auto_map: {'yes' if auto_map else 'no'}")
    logger.info(f"  transformers_version: {config.get('transformers_version', 'not specified')}")

    # Collect all missing deps
    all_missing = []

    # Check 1: transformers version
    tf_deps = check_transformers_version(config)
    if tf_deps:
        logger.info(f"  [!] transformers upgrade needed: {tf_deps}")
        all_missing.extend(tf_deps)

    # Check 2: requirements.txt from repo
    repo_reqs = get_repo_requirements(model_id)
    if repo_reqs:
        logger.info(f"  [i] repo requirements.txt found: {repo_reqs}")
        all_missing.extend(repo_reqs)

    # Check 3: known model_type deps
    known = check_known_deps(config)
    if known:
        logger.info(f"  [!] known deps for {model_type}: {known}")
        all_missing.extend(known)

    # Check 4: custom code import check
    custom = check_custom_code_deps(model_id, config)
    if custom:
        logger.info(f"  [!] custom code missing deps: {custom}")
        all_missing.extend(custom)

    # Deduplicate
    all_missing = list(dict.fromkeys(all_missing))

    if not all_missing:
        logger.info("  ✓ All dependencies satisfied")
        return 0

    logger.info(f"  Missing dependencies: {all_missing}")

    if do_install:
        success = install_packages(all_missing)
        if not success:
            return 2
        # Re-check after install
        tf_still = check_transformers_version(config)
        if tf_still:
            logger.warning(f"  transformers still outdated after install: {tf_still}")
            return 2
        logger.info("  ✓ All dependencies installed successfully")
        return 0
    else:
        # Just report
        for dep in all_missing:
            print(dep)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-flight model dependency check")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--install", action="store_true",
                        help="Automatically install missing dependencies")
    args = parser.parse_args()

    sys.exit(preflight(args.model, args.install))
