#!/usr/bin/env python3
"""
Upload auto_quant result artifacts into the lb_eval GitHub repository.

Artifacts copied from the runtime output directory:
- quant_summary.json / summary.json
- accuracy.json
- lm_eval_results/
- quantize.py
- logs/
- session_*.jsonl
- session_*.md

Layout:
results/<org>/<artifact_name>/results_<timestamp>.json
results/<org>/<artifact_name>/run_<timestamp>/...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def looks_like_quantized_artifact(model_short: str) -> bool:
    lowered = model_short.lower()
    markers = (
        "-autoround-",
        "-gptq",
        "-awq",
        ".gguf",
        "-gguf",
        "llm-compressor",
    )
    return any(marker in lowered for marker in markers)


def run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def clone_repo(repo_url: str, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", repo_url, str(target_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git clone failed")


def ensure_git_config(repo_dir: Path, user_name: str, user_email: str) -> None:
    run_git(["config", "user.name", user_name], repo_dir)
    run_git(["config", "user.email", user_email], repo_dir)


def build_auth_url(repo_url: str, token: str | None) -> str:
    if not token:
        return repo_url
    parts = urlsplit(repo_url)
    if parts.scheme != "https":
        return repo_url
    netloc = f"x-access-token:{token}@{parts.netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def copy_file(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def copy_tree(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists() or not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    copied.append(str(dst))


def detect_artifact_name(model_id: str, scheme: str, quant_summary: dict | None) -> str:
    if quant_summary:
        hf_repo = quant_summary.get("hf_repo")
        if isinstance(hf_repo, str) and hf_repo.strip():
            return hf_repo.rstrip("/").rsplit("/", 1)[-1]

    model_short = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    if looks_like_quantized_artifact(model_short):
        return sanitize_name(model_short)
    return sanitize_name(f"{model_short}-autoround-{scheme}")


def resolve_repo_dir(repo_dir_arg: str, clone_dir_arg: str, repo_url: str, token: str | None) -> Path:
    repo_dir_raw = (repo_dir_arg or "").strip()
    clone_dir_raw = (clone_dir_arg or "").strip()
    default_clone_dir = Path(__file__).resolve().parent / "lb_eval"

    if repo_dir_raw:
        repo_dir = Path(repo_dir_raw).resolve()
    else:
        repo_dir = Path(clone_dir_raw).resolve() if clone_dir_raw else default_clone_dir.resolve()

    if (repo_dir / ".git").exists():
        return repo_dir

    if repo_dir.exists():
        if any(repo_dir.iterdir()):
            raise RuntimeError(f"target repo dir exists but is not a git repo: {repo_dir}")
    else:
        repo_dir.mkdir(parents=True, exist_ok=True)

    if not repo_url:
        raise RuntimeError("GIT_REPO is required when the local git repo is missing")

    auth_url = build_auth_url(repo_url, token)
    print(f"[github-upload] Cloning repo into: {repo_dir}")
    shutil.rmtree(repo_dir)
    clone_repo(auth_url, repo_dir)
    if auth_url != repo_url:
        run_git(["remote", "set-url", "origin", repo_url], repo_dir)
    return repo_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload result artifacts to the lb_eval GitHub repo")
    parser.add_argument("runtime_output_dir", help="Directory containing quant/eval runtime artifacts")
    parser.add_argument("model_id", help="Original model id, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--pipeline", default="", help="Pipeline label: auto_quant or auto_eval")
    parser.add_argument("--scheme", default="W4A16", help="Quantization scheme label")
    parser.add_argument("--quant-num-gpus", default="", help="Quantization GPU count")
    parser.add_argument("--eval-num-gpus", default="", help="Evaluation GPU count")
    parser.add_argument(
        "--model-output-dir",
        default="",
        help="Optional model directory associated with this runtime output",
    )
    parser.add_argument(
        "--repo-dir",
        default=os.environ.get(
            "GIT_RESULTS_REPO_DIR",
            "",
        ),
        help="Local clone of the lb_eval git repo. If empty or missing, the repo is cloned automatically.",
    )
    parser.add_argument(
        "--clone-dir",
        default=os.environ.get("GIT_RESULTS_CLONE_DIR") or str(Path(__file__).resolve().parent / "lb_eval"),
        help="Clone destination used when --repo-dir is empty or missing",
    )
    parser.add_argument(
        "--git-repo",
        default=os.environ.get("GIT_REPO", ""),
        help="Remote git repo URL used for authenticated pull/push when token is provided",
    )
    parser.add_argument(
        "--git-token",
        default=os.environ.get("GIT_TOKEN", ""),
        help="GitHub token used for authenticated push",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("GIT_BRANCH", ""),
        help="Branch to update. Defaults to current branch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare artifact files locally but do not create a commit or push",
    )
    parser.add_argument(
        "--git-user-name",
        default=os.environ.get("GIT_USER_NAME", "lb-eval-bot"),
        help="git config user.name for generated commits",
    )
    parser.add_argument(
        "--git-user-email",
        default=os.environ.get("GIT_USER_EMAIL", "lb-eval-bot@local"),
        help="git config user.email for generated commits",
    )
    args = parser.parse_args()

    runtime_output_dir = Path(args.runtime_output_dir).resolve()
    if not runtime_output_dir.is_dir():
        print(f"ERROR: runtime output directory not found: {runtime_output_dir}")
        return 1

    model_output_dir = None
    if args.model_output_dir.strip():
        model_output_dir = Path(args.model_output_dir).resolve()

    try:
        repo_dir = resolve_repo_dir(args.repo_dir, args.clone_dir, args.git_repo, args.git_token or None)
    except Exception as exc:
        print(f"ERROR: could not prepare git repo: {exc}")
        return 1

    quant_summary_path = runtime_output_dir / "quant_summary.json"
    summary_path = runtime_output_dir / "summary.json"
    accuracy_path = runtime_output_dir / "accuracy.json"
    quantize_script_path = runtime_output_dir / "quantize.py"
    legacy_quantize_script_path = runtime_output_dir / "quantize_script.py"
    lm_eval_results_dir = runtime_output_dir / "lm_eval_results"
    logs_dir = runtime_output_dir / "logs"
    quant_summary = load_json(quant_summary_path) or load_json(summary_path)
    accuracy = load_json(accuracy_path)

    artifact_name = detect_artifact_name(args.model_id, args.scheme, quant_summary)
    org = args.model_id.split("/", 1)[0] if "/" in args.model_id else "unknown"
    timestamp = file_timestamp()

    branch = args.branch
    if not branch:
        branch = run_git(["branch", "--show-current"], repo_dir).stdout.strip() or "main"

    print(f"[github-upload] repo dir: {repo_dir}")
    print(f"[github-upload] model id: {args.model_id}")
    print(f"[github-upload] artifact name: {artifact_name}")
    print(f"[github-upload] result org: {org}")
    print(f"[github-upload] branch: {branch}")
    ensure_git_config(repo_dir, args.git_user_name, args.git_user_email)
    print(f"[github-upload] git user: {args.git_user_name} <{args.git_user_email}>")
    if not args.dry_run:
        remote_url = args.git_repo or run_git(["remote", "get-url", "origin"], repo_dir).stdout.strip()
        auth_url = build_auth_url(remote_url, args.git_token or None)
        pull_target = auth_url if args.git_token else "origin"
        push_target = auth_url if args.git_token else "origin"

        pull_args = ["pull", "--rebase", pull_target, branch]
        pull_result = run_git(pull_args, repo_dir, check=False)
        if pull_result.returncode != 0:
            print(f"[github-upload] ERROR: git pull failed:\n{pull_result.stderr.strip()}")
            return 1

    model_result_dir = repo_dir / "results" / org / artifact_name
    run_dir = model_result_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    copy_file(quant_summary_path, run_dir / "quant_summary.json", copied)
    copy_file(summary_path, run_dir / "summary.json", copied)
    copy_file(accuracy_path, run_dir / "accuracy.json", copied)
    if quantize_script_path.is_file():
        copy_file(quantize_script_path, run_dir / "quantize.py", copied)
    elif legacy_quantize_script_path.is_file():
        copy_file(legacy_quantize_script_path, run_dir / "quantize.py", copied)
    copy_tree(lm_eval_results_dir, run_dir / "lm_eval_results", copied)
    copy_tree(logs_dir, run_dir / "logs", copied)

    for path in sorted(runtime_output_dir.glob("session_*.jsonl")):
        copy_file(path, run_dir / path.name, copied)
    for path in sorted(runtime_output_dir.glob("session_*.md")):
        copy_file(path, run_dir / path.name, copied)

    aggregate = {
        "pipeline": args.pipeline or ("auto_quant" if quant_summary else "auto_eval"),
        "model_id": args.model_id,
        "artifact_name": artifact_name,
        "generated_at": utc_now(),
        "source_runtime_dir": str(runtime_output_dir),
        "source_model_dir": str(model_output_dir) if model_output_dir else None,
        "run_dir": str(run_dir.relative_to(repo_dir)),
        "quant_summary": quant_summary,
        "accuracy": accuracy,
        "copied_files": [str(Path(path).relative_to(repo_dir)) for path in copied],
    }
    if aggregate["pipeline"] == "auto_quant":
        aggregate["quant_num_gpus"] = args.quant_num_gpus or (
            str(quant_summary.get("quant_num_gpus") or quant_summary.get("num_gpus"))
            if isinstance(quant_summary, dict) and (quant_summary.get("quant_num_gpus") or quant_summary.get("num_gpus")) is not None
            else None
        )
        aggregate["eval_num_gpus"] = args.eval_num_gpus or (
            str(accuracy.get("eval_num_gpus") or accuracy.get("num_gpus"))
            if isinstance(accuracy, dict) and (accuracy.get("eval_num_gpus") or accuracy.get("num_gpus")) is not None
            else None
        )
    else:
        aggregate["num_gpus"] = args.eval_num_gpus or args.quant_num_gpus or (
            str(accuracy.get("num_gpus"))
            if isinstance(accuracy, dict) and accuracy.get("num_gpus") is not None
            else None
        )
    aggregate_path = model_result_dir / f"results_{timestamp}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    copied.append(str(aggregate_path))

    rel_paths = [str(Path(path).relative_to(repo_dir)) for path in copied]
    if not rel_paths:
        print("[github-upload] No artifacts found to upload.")
        return 0

    if args.dry_run:
        print("[github-upload] Dry run prepared artifacts:")
        for rel_path in rel_paths:
            print(f"  - {rel_path}")
        return 0

    run_git(["add", *rel_paths], repo_dir)
    diff_result = run_git(["diff", "--cached", "--quiet"], repo_dir, check=False)
    if diff_result.returncode == 0:
        print("[github-upload] No changes to commit.")
        return 0

    commit_message = (
        f"Add auto_quant artifacts for {artifact_name}\n\n"
        "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    )
    commit_result = run_git(["commit", "-m", commit_message], repo_dir, check=False)
    if commit_result.returncode != 0:
        print(f"[github-upload] ERROR: git commit failed:\n{commit_result.stderr.strip()}")
        return 1

    push_args = ["push", push_target, f"HEAD:{branch}"]
    push_result = run_git(push_args, repo_dir, check=False)
    if push_result.returncode != 0:
        print(f"[github-upload] ERROR: git push failed:\n{push_result.stderr.strip()}")
        return 1

    print("[github-upload] Uploaded artifacts:")
    for rel_path in rel_paths:
        print(f"  - {rel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
