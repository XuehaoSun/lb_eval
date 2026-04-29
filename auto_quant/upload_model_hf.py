#!/usr/bin/env python3
"""
Upload a quantized model to Hugging Face Hub.

Features:
- multi-token / multi-account failover
- shared ledger backed by a shared git repo (recommended for multi-machine use)
- local usage ledger fallback
- quota-based account selection
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import tempfile
import time
from pathlib import Path

from hf_shared_ledger import Candidate, SharedLedger

IGNORE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    "venv/**",
    ".venv/**",
    "*.egg-info",
    ".git/**",
    "session_*.jsonl",
    "session_*.md",
    "quant_summary.json",
    "summary.json",
    "accuracy.json",
    "request.json",
    "logs/**",
    "lm_eval_results/**",
    "*.log",
    "*_prompt.txt",
    "hf_account_usage.json",
    "quantize_script.py",
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gib_bytes(value_gb: float) -> int:
    return int(value_gb * (1024**3))


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def str_to_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def check_huggingface_hub() -> bool:
    try:
        from huggingface_hub import HfApi  # noqa: F401

        return True
    except ImportError:
        return False


def get_account_info(api, token: str) -> str | None:
    try:
        info = api.whoami(token=token)
        return info.get("name")
    except Exception:
        return None


def should_ignore(relative_path: str) -> bool:
    name = Path(relative_path).name
    return any(
        fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(name, pattern)
        for pattern in IGNORE_PATTERNS
    )


def get_folder_size_bytes(folder_path: str) -> int:
    root = Path(folder_path)
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if should_ignore(relative_path):
            continue
        total += path.stat().st_size
    return total


def normalize_account_id(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def build_candidates(api, tokens: list[str], orgs: list[str], account_ids: list[str], quota_bytes: int) -> list[dict]:
    candidates: list[dict] = []
    seen_ids: set[str] = set()
    for index, token in enumerate(tokens):
        if index < len(orgs):
            org = orgs[index]
        elif orgs:
            org = orgs[-1]
        else:
            org = get_account_info(api, token)
            if not org:
                print(f"  [HF] Token #{index + 1}: could not determine account, skipping")
                continue

        if index < len(account_ids) and account_ids[index].strip():
            account_id = normalize_account_id(account_ids[index].strip())
        else:
            account_id = normalize_account_id(org)

        if account_id in seen_ids:
            suffix = 2
            base = account_id
            while f"{base}-{suffix}" in seen_ids:
                suffix += 1
            account_id = f"{base}-{suffix}"
        seen_ids.add(account_id)

        candidates.append(
            {
                "account_id": account_id,
                "org": org,
                "token": token,
                "quota_bytes": quota_bytes,
            }
        )
    return candidates


def ensure_local_account_record(state: dict, candidate: dict) -> dict:
    accounts = state.setdefault("accounts", {})
    account = accounts.setdefault(
        candidate["account_id"],
        {
            "account_id": candidate["account_id"],
            "org": candidate["org"],
            "quota_bytes": candidate["quota_bytes"],
            "used_bytes": 0,
            "remaining_bytes": candidate["quota_bytes"],
            "last_upload_at": None,
            "repos": {},
        },
    )
    account["org"] = candidate["org"]
    account.setdefault("quota_bytes", candidate["quota_bytes"])
    account.setdefault("used_bytes", 0)
    account.setdefault("repos", {})
    account.setdefault("last_upload_at", None)
    account["remaining_bytes"] = max(0, account["quota_bytes"] - account["used_bytes"])
    return account


def load_local_usage_state(path: Path, candidates: list[dict]) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
    else:
        state = {"version": 1, "updated_at": utc_now(), "accounts": {}}

    state.setdefault("version", 1)
    state.setdefault("accounts", {})
    for candidate in candidates:
        ensure_local_account_record(state, candidate)
    return state


def save_local_usage_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    for account in state.get("accounts", {}).values():
        account["remaining_bytes"] = max(0, account["quota_bytes"] - account["used_bytes"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def rank_local_candidates(candidates: list[dict], state: dict, repo_name: str, upload_size_bytes: int) -> list[dict]:
    ranked = []
    for order, candidate in enumerate(candidates):
        account = state["accounts"][candidate["account_id"]]
        repo_id = f"{candidate['org']}/{repo_name}"
        existing_repo_bytes = int(account["repos"].get(repo_id, {}).get("size_bytes", 0))
        reserve_bytes = max(0, upload_size_bytes - existing_repo_bytes)
        remaining = max(0, account["quota_bytes"] - account["used_bytes"])
        remaining_after = remaining - reserve_bytes
        ranked.append(
            {
                "order": order,
                "candidate": candidate,
                "repo_id": repo_id,
                "used_bytes": account["used_bytes"],
                "remaining_bytes": remaining,
                "existing_repo_bytes": existing_repo_bytes,
                "reserve_bytes": reserve_bytes,
                "remaining_after_reservation": remaining_after,
                "fits": remaining_after >= 0,
            }
        )

    ranked.sort(key=lambda item: (0 if item["fits"] else 1, -item["remaining_after_reservation"], item["order"]))
    return ranked


def update_local_usage_state(state: dict, candidate: dict, repo_id: str, upload_size_bytes: int, source_path: str) -> dict:
    account = state["accounts"][candidate["account_id"]]
    previous_size = int(account["repos"].get(repo_id, {}).get("size_bytes", 0))
    account["used_bytes"] = max(0, account["used_bytes"] - previous_size + upload_size_bytes)
    account["repos"][repo_id] = {
        "size_bytes": upload_size_bytes,
        "source_path": source_path,
        "updated_at": utc_now(),
    }
    account["last_upload_at"] = utc_now()
    account["remaining_bytes"] = max(0, account["quota_bytes"] - account["used_bytes"])
    return account


def try_upload(api, token: str, org: str, repo_name: str, model_path: str) -> tuple[str, str]:
    repo_id = f"{org}/{repo_name}" if org else repo_name

    print(f"  [HF] Trying upload to: {repo_id}")
    api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True, private=False)
    print(f"  [HF] Repo ready: {repo_id}")

    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        token=token,
        repo_type="model",
        commit_message=f"Upload quantized model {repo_name}",
        ignore_patterns=IGNORE_PATTERNS,
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"  [HF] Upload successful: {repo_url}")
    return repo_id, repo_url


def update_summary_json(summary_json: str | None, fields: dict) -> None:
    if not summary_json or not os.path.exists(summary_json):
        return
    try:
        with open(summary_json, encoding="utf-8") as handle:
            summary = json.load(handle)
        summary.update(fields)
        with open(summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"[HF] Updated {summary_json} with upload metadata")
    except Exception as exc:
        print(f"[HF] Warning: could not update summary.json: {exc}")


def print_candidate_ranking(prefix: str, ranking: list[dict]) -> None:
    print(prefix)
    for item in ranking:
        if "candidate" in item:
            account_id = item["candidate"]["account_id"]
            org = item["candidate"]["org"]
        else:
            account_id = item["account_id"]
            org = item["org"]
        fit_status = "fit" if item["fits"] else "insufficient"
        print(
            "  - "
            f"{account_id} ({org}): "
            f"used={format_gb(item['used_bytes'])}, "
            f"remaining={format_gb(item['remaining_bytes'])}, "
            f"reserve={format_gb(item['reserve_bytes'])}, "
            f"after={format_gb(item['remaining_after_reservation'])}, "
            f"{fit_status}"
        )


def upload_with_shared_ledger(api, args, candidates: list[dict], upload_size_bytes: int) -> tuple[str, str, dict]:
    ledger = SharedLedger(
        repo=args.shared_ledger_repo,
        clone_dir=args.shared_ledger_clone_dir,
        branch=args.shared_ledger_branch,
        token=args.shared_ledger_token or None,
        git_user_name=args.shared_ledger_git_user_name,
        git_user_email=args.shared_ledger_git_user_email,
        reservation_ttl_seconds=args.shared_ledger_ttl_seconds,
    )

    remaining = candidates[:]
    errors: list[str] = []

    while remaining:
        reservation = ledger.reserve_account(
            [Candidate(account_id=item["account_id"], org=item["org"], quota_bytes=item["quota_bytes"]) for item in remaining],
            repo_name=args.repo_name,
            estimated_size_bytes=upload_size_bytes,
        )
        print_candidate_ranking("[HF] Shared-ledger account ranking:", reservation.get("ranking", []))
        selected = next(item for item in remaining if item["account_id"] == reservation["account_id"])

        try:
            repo_id, repo_url = try_upload(api, selected["token"], selected["org"], args.repo_name, args.model_path)
            ledger.commit_reservation(reservation, actual_size_bytes=upload_size_bytes, repo_url=repo_url)
            update_summary_json(
                args.summary_json,
                {
                    "hf_repo": repo_url,
                    "hf_account": selected["org"],
                    "hf_account_id": selected["account_id"],
                    "hf_shared_ledger_enabled": True,
                    "hf_shared_ledger_repo": args.shared_ledger_repo,
                    "hf_shared_ledger_clone_dir": args.shared_ledger_clone_dir,
                    "hf_shared_ledger_reservation_id": reservation["reservation_id"],
                    "hf_remaining_gb": round(reservation["remaining_after_reservation_bytes"] / (1024**3), 2),
                    "upload_time": utc_now(),
                },
            )
            return repo_id, repo_url, selected
        except Exception as exc:
            try:
                ledger.release_reservation(reservation, f"upload_failed: {exc}")
            except Exception as release_exc:
                print(f"[HF] Warning: could not release reservation {reservation['reservation_id']}: {release_exc}")
            err_msg = f"{selected['account_id']} ({selected['org']}): {exc}"
            print(f"  [HF] FAILED — {err_msg}")
            errors.append(err_msg)
            remaining = [item for item in remaining if item["account_id"] != selected["account_id"]]

    print("[HF] ERROR: All shared-ledger upload attempts failed:")
    for error in errors:
        print(f"  - {error}")
    raise RuntimeError("all shared-ledger upload attempts failed")


def upload_with_local_ledger(api, args, candidates: list[dict], upload_size_bytes: int) -> tuple[str, str, dict]:
    usage_file = Path(args.usage_file).resolve()
    state = load_local_usage_state(usage_file, candidates)
    save_local_usage_state(usage_file, state)

    print(f"[HF] Usage ledger: {usage_file}")
    ranking = rank_local_candidates(candidates, state, args.repo_name, upload_size_bytes)
    print_candidate_ranking("[HF] Local-ledger account ranking:", ranking)

    errors: list[str] = []
    for item in ranking:
        candidate = item["candidate"]
        if not item["fits"]:
            print(
                f"  [HF] Candidate {candidate['account_id']} ({candidate['org']}): "
                "ledger says remaining space is insufficient, trying anyway as fallback"
            )
        try:
            repo_id, repo_url = try_upload(api, candidate["token"], candidate["org"], args.repo_name, args.model_path)
            account = update_local_usage_state(state, candidate, repo_id, upload_size_bytes, args.model_path)
            save_local_usage_state(usage_file, state)
            update_summary_json(
                args.summary_json,
                {
                    "hf_repo": repo_url,
                    "hf_account": candidate["org"],
                    "hf_account_id": candidate["account_id"],
                    "hf_shared_ledger_enabled": False,
                    "hf_usage_file": str(usage_file),
                    "hf_remaining_gb": round(account["remaining_bytes"] / (1024**3), 2),
                    "upload_time": utc_now(),
                },
            )
            print(
                f"[HF] Local ledger updated for {candidate['account_id']}: "
                f"used={format_gb(account['used_bytes'])}, "
                f"remaining={format_gb(account['remaining_bytes'])}"
            )
            return repo_id, repo_url, candidate
        except Exception as exc:
            err_msg = f"{candidate['account_id']} ({candidate['org']}): {exc}"
            print(f"  [HF] FAILED — {err_msg}")
            errors.append(err_msg)

    print("[HF] ERROR: All local-ledger upload attempts failed:")
    for error in errors:
        print(f"  - {error}")
    raise RuntimeError("all local-ledger upload attempts failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload quantized model to HuggingFace with shared/local ledger")
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
        "--account-ids",
        help="Optional comma-separated stable account ids for ledger tracking",
        default=os.environ.get("HF_ACCOUNT_IDS", ""),
    )
    parser.add_argument(
        "--summary-json",
        help="Path to summary.json / quant_summary.json to update after upload",
    )
    parser.add_argument(
        "--usage-file",
        default=os.environ.get(
            "HF_USAGE_FILE", str(Path(__file__).resolve().with_name("hf_account_usage.json"))
        ),
        help="Local JSON ledger file for per-account storage usage tracking",
    )
    parser.add_argument(
        "--capacity-gb",
        type=float,
        default=float(os.environ.get("HF_ACCOUNT_CAPACITY_GB", "100")),
        help="Per-account storage capacity in GiB for the ledger",
    )
    parser.add_argument(
        "--shared-ledger-enabled",
        default=os.environ.get("HF_SHARED_LEDGER_ENABLED", ""),
        help="Enable shared ledger mode (true/false). Defaults to true when repo is configured.",
    )
    parser.add_argument(
        "--shared-ledger-repo",
        default=os.environ.get("HF_SHARED_LEDGER_REPO", ""),
        help="HF dataset repo id / URL / local git path used as the shared ledger backend",
    )
    parser.add_argument(
        "--shared-ledger-token",
        default=os.environ.get("HF_SHARED_LEDGER_TOKEN", ""),
        help="Token used to pull/push the shared ledger repo",
    )
    parser.add_argument(
        "--shared-ledger-branch",
        default=os.environ.get("HF_SHARED_LEDGER_BRANCH", "main"),
        help="Branch used by the shared ledger repo",
    )
    parser.add_argument(
        "--shared-ledger-clone-dir",
        default=os.environ.get("HF_SHARED_LEDGER_CLONE_DIR")
        or str(Path(__file__).resolve().parent / "hf_shared_ledger"),
        help="Local clone/cache dir for the shared ledger repo",
    )
    parser.add_argument(
        "--shared-ledger-ttl-seconds",
        type=int,
        default=int(os.environ.get("HF_SHARED_LEDGER_RESERVATION_TTL_SECONDS", "7200")),
        help="Reservation TTL in seconds for the shared ledger",
    )
    parser.add_argument(
        "--shared-ledger-git-user-name",
        default=os.environ.get("HF_SHARED_LEDGER_GIT_USER_NAME", "hf-ledger-bot"),
        help="git user.name for shared-ledger commits",
    )
    parser.add_argument(
        "--shared-ledger-git-user-email",
        default=os.environ.get("HF_SHARED_LEDGER_GIT_USER_EMAIL", "hf-ledger-bot@local"),
        help="git user.email for shared-ledger commits",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"ERROR: Model path is not a directory: {args.model_path}")
        return 1

    if args.capacity_gb <= 0:
        print("ERROR: --capacity-gb must be > 0")
        return 1

    if args.shared_ledger_ttl_seconds <= 0:
        print("ERROR: --shared-ledger-ttl-seconds must be > 0")
        return 1

    if not check_huggingface_hub():
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    from huggingface_hub import HfApi

    tokens = [token.strip() for token in args.tokens.split(",") if token.strip()]
    orgs = [org.strip() for org in args.orgs.split(",") if org.strip()]
    account_ids = [item.strip() for item in args.account_ids.split(",") if item.strip()]
    if not tokens:
        print("ERROR: No HF tokens provided. Set HF_TOKENS env var or use --tokens.")
        return 1

    api = HfApi()
    quota_bytes = gib_bytes(args.capacity_gb)
    candidates = build_candidates(api, tokens, orgs, account_ids, quota_bytes)
    if not candidates:
        print("ERROR: No valid HuggingFace accounts available.")
        return 1

    # Fall back to an appropriate token when no dedicated ledger token is set.
    # GitHub repos → use GIT_TOKEN; HF repos → use the first HF upload token.
    if not args.shared_ledger_token:
        ledger_repo = (args.shared_ledger_repo or "").strip()
        if "github.com" in ledger_repo:
            args.shared_ledger_token = os.environ.get("GIT_TOKEN", "")
        elif tokens:
            args.shared_ledger_token = tokens[0]

    upload_size_bytes = get_folder_size_bytes(args.model_path)
    shared_enabled = str_to_bool(
        args.shared_ledger_enabled,
        default=bool(args.shared_ledger_repo.strip()),
    )

    print(f"[HF] Uploading model: {args.repo_name}")
    print(f"[HF] Source path: {args.model_path}")
    print(f"[HF] Estimated upload size: {format_gb(upload_size_bytes)}")
    print(f"[HF] Accounts available: {len(candidates)}")

    try:
        if shared_enabled:
            if not args.shared_ledger_repo.strip():
                print("ERROR: shared ledger enabled but HF_SHARED_LEDGER_REPO is empty")
                return 1
            _, repo_url, _ = upload_with_shared_ledger(api, args, candidates, upload_size_bytes)
        else:
            _, repo_url, _ = upload_with_local_ledger(api, args, candidates, upload_size_bytes)
    except Exception as exc:
        print(f"[HF] ERROR: {exc}")
        return 1

    print(f"REPO_URL={repo_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
