#!/usr/bin/env python3
"""
upload_model_hf.py — Upload a quantized model to HuggingFace Hub.

Supports multi-token failover: tries each provided HF token in order
until one succeeds. Useful when accounts have space/quota limits.

Usage:
    python3 upload_model_hf.py /path/to/quantized_model Qwen3-8B-autoround-W4A16 \
        --tokens "hf_token1,hf_token2" \
        --orgs "Intel,kaitchup" \
        --summary-json /path/to/summary.json

Environment variables (alternative to CLI args):
    HF_TOKENS       — comma-separated HuggingFace API tokens
    HF_UPLOAD_ORGS  — comma-separated org/user names
    HTTP_PROXY / HTTPS_PROXY — proxy settings
"""

import argparse
import json
import os
import sys
import time


def check_huggingface_hub():
    """Ensure huggingface_hub is installed."""
    try:
        from huggingface_hub import HfApi
        return True
    except ImportError:
        return False


def get_account_info(api, token):
    """Get the username associated with a token."""
    try:
        info = api.whoami(token=token)
        return info.get("name", None)
    except Exception:
        return None


def try_upload(api, token, org, repo_name, model_path):
    """Attempt to create repo and upload model. Returns repo_url on success."""
    repo_id = f"{org}/{repo_name}" if org else repo_name

    print(f"  [HF] Trying upload to: {repo_id}")

    # Create repo (or confirm it exists)
    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    print(f"  [HF] Repo ready: {repo_id}")

    # Determine files to exclude (venv, cache, etc.)
    ignore_patterns = [
        "*.pyc",
        "__pycache__",
        "venv/**",
        ".venv/**",
        "*.egg-info",
        ".git/**",
        "session_*.jsonl",
        "session_*.md",
    ]

    # Upload all model files
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        token=token,
        repo_type="model",
        commit_message=f"Upload quantized model {repo_name}",
        ignore_patterns=ignore_patterns,
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"  [HF] Upload successful: {repo_url}")
    return repo_url


def main():
    parser = argparse.ArgumentParser(
        description="Upload quantized model to HuggingFace with multi-token failover"
    )
    parser.add_argument("model_path", help="Path to the quantized model directory")
    parser.add_argument("repo_name", help="Target repo name (e.g., Qwen3-8B-autoround-W4A16)")
    parser.add_argument(
        "--tokens",
        help="Comma-separated HF tokens (or set HF_TOKENS env var)",
        default=os.environ.get("HF_TOKENS", ""),
    )
    parser.add_argument(
        "--orgs",
        help="Comma-separated org/user names (or set HF_UPLOAD_ORGS env var)",
        default=os.environ.get("HF_UPLOAD_ORGS", ""),
    )
    parser.add_argument(
        "--summary-json",
        help="Path to summary.json — will be updated with the uploaded repo URL",
    )
    args = parser.parse_args()

    # Validate model path
    if not os.path.isdir(args.model_path):
        print(f"ERROR: Model path is not a directory: {args.model_path}")
        sys.exit(1)

    # Check dependency
    if not check_huggingface_hub():
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    from huggingface_hub import HfApi

    # Parse tokens and orgs
    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    orgs = [o.strip() for o in args.orgs.split(",") if o.strip()]

    if not tokens:
        print("ERROR: No HF tokens provided. Set HF_TOKENS env var or use --tokens.")
        sys.exit(1)

    # Build (token, org) pairs
    pairs = []
    for i, token in enumerate(tokens):
        if i < len(orgs):
            org = orgs[i]
        elif orgs:
            org = orgs[-1]
        else:
            org = None
        pairs.append((token, org))

    api = HfApi()
    repo_url = None
    errors = []

    print(f"[HF] Uploading model: {args.repo_name}")
    print(f"[HF] Source path: {args.model_path}")
    print(f"[HF] Trying {len(pairs)} account(s)...")

    for idx, (token, org) in enumerate(pairs, 1):
        # If no org specified, detect from token
        if not org:
            org = get_account_info(api, token)
            if not org:
                print(f"  [HF] Token #{idx}: could not determine account, skipping")
                errors.append(f"Token #{idx}: auth failed")
                continue

        try:
            repo_url = try_upload(api, token, org, args.repo_name, args.model_path)
            break
        except Exception as e:
            err_msg = f"Token #{idx} ({org}): {e}"
            print(f"  [HF] FAILED — {err_msg}")
            errors.append(err_msg)
            continue

    if not repo_url:
        print("[HF] ERROR: All upload attempts failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Update summary.json with the uploaded repo URL
    if args.summary_json and os.path.exists(args.summary_json):
        try:
            with open(args.summary_json) as f:
                summary = json.load(f)
            summary["hf_repo"] = repo_url
            summary["upload_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(args.summary_json, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[HF] Updated {args.summary_json} with hf_repo={repo_url}")
        except Exception as e:
            print(f"[HF] Warning: could not update summary.json: {e}")

    # Print machine-readable output for caller scripts
    print(f"REPO_URL={repo_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
