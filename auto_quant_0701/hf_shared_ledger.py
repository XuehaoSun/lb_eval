#!/usr/bin/env python3
"""
Shared append-only ledger for Hugging Face upload account reservations.

Design goals:
- shared across machines via a git-backed repository (typically a HF dataset repo)
- append-only events to reduce merge conflicts
- local file lock to serialize same-machine writers
- reservation TTL so crashed workers do not block capacity forever
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit


@dataclass
class Candidate:
    account_id: str
    org: str
    quota_bytes: int


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_remote_url(repo: str, token: str | None = None) -> str:
    repo = (repo or "").strip()
    if not repo:
        raise ValueError("shared ledger repo is empty")

    if "://" in repo or repo.startswith("/") or repo.startswith("."):
        remote = repo
    else:
        remote = f"https://huggingface.co/datasets/{repo}"

    if not token:
        return remote

    parts = urlsplit(remote)
    if parts.scheme != "https":
        return remote
    netloc = f"hf-ledger:{token}@{parts.netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def clean_remote_url(repo: str) -> str:
    return build_remote_url(repo, token=None)


def run_git(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
    )


def ensure_git_config(repo_dir: Path, user_name: str, user_email: str) -> None:
    run_git(["config", "user.name", user_name], repo_dir)
    run_git(["config", "user.email", user_email], repo_dir)


def clone_repo(remote_url: str, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    result = run_git(["clone", remote_url, str(target_dir)], check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git clone failed")


@contextlib.contextmanager
def file_lock(lock_path: Path) -> Iterable[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return 0.0


class SharedLedger:
    def __init__(
        self,
        *,
        repo: str,
        clone_dir: str,
        branch: str = "main",
        token: str | None = None,
        git_user_name: str = "hf-ledger-bot",
        git_user_email: str = "hf-ledger-bot@local",
        reservation_ttl_seconds: int = 7200,
        max_retries: int = 5,
    ) -> None:
        self.repo = repo
        self.clone_dir = Path(clone_dir).resolve()
        self.branch = branch or "main"
        self.token = token
        self.git_user_name = git_user_name
        self.git_user_email = git_user_email
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.max_retries = max_retries
        self.lock_path = self.clone_dir.parent / f"{self.clone_dir.name}.lock"

    def reserve_account(
        self,
        candidates: list[Candidate],
        *,
        repo_name: str,
        estimated_size_bytes: int,
    ) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with file_lock(self.lock_path):
                    repo_dir = self._prepare_repo()
                    state = self._compute_state(repo_dir, candidates)
                    ranked = self._rank_candidates(state, candidates, repo_name, estimated_size_bytes)
                    if not ranked:
                        raise RuntimeError("no candidates available for reservation")

                    selected = ranked[0]
                    reservation = self._build_reservation_event(selected, repo_name, estimated_size_bytes)
                    event_rel_path = self._write_event(repo_dir, reservation)
                    self._commit_and_push(
                        repo_dir,
                        [event_rel_path],
                        f"Reserve HF upload space for {selected['repo_id']}",
                    )
                    reservation["attempt"] = attempt
                    reservation["ledger_repo_dir"] = str(repo_dir)
                    reservation["ranking"] = ranked
                    return reservation
            except Exception as exc:
                last_error = exc
                self._reset_clone()
        raise RuntimeError(f"failed to reserve shared ledger slot: {last_error}")

    def commit_reservation(
        self,
        reservation: dict,
        *,
        actual_size_bytes: int,
        repo_url: str,
    ) -> dict:
        event = {
            "version": 1,
            "event_type": "commit",
            "event_id": str(uuid.uuid4()),
            "reservation_id": reservation["reservation_id"],
            "account_id": reservation["account_id"],
            "org": reservation["org"],
            "repo_id": reservation["repo_id"],
            "repo_url": repo_url,
            "actual_bytes": actual_size_bytes,
            "created_at": utc_now(),
        }
        return self._write_followup_event(event, f"Commit HF upload space for {reservation['repo_id']}")

    def release_reservation(self, reservation: dict, reason: str) -> dict:
        event = {
            "version": 1,
            "event_type": "release",
            "event_id": str(uuid.uuid4()),
            "reservation_id": reservation["reservation_id"],
            "account_id": reservation["account_id"],
            "org": reservation["org"],
            "repo_id": reservation["repo_id"],
            "reason": reason,
            "created_at": utc_now(),
        }
        return self._write_followup_event(event, f"Release HF upload space for {reservation['repo_id']}")

    def _write_followup_event(self, event: dict, commit_message: str) -> dict:
        last_error: Exception | None = None
        for _ in range(1, self.max_retries + 1):
            try:
                with file_lock(self.lock_path):
                    repo_dir = self._prepare_repo()
                    event_rel_path = self._write_event(repo_dir, event)
                    self._commit_and_push(repo_dir, [event_rel_path], commit_message)
                    event["ledger_repo_dir"] = str(repo_dir)
                    return event
            except Exception as exc:
                last_error = exc
                self._reset_clone()
        raise RuntimeError(f"failed to write shared ledger event: {last_error}")

    def _prepare_repo(self) -> Path:
        clean_remote = clean_remote_url(self.repo)
        auth_remote = build_remote_url(self.repo, self.token)

        if not (self.clone_dir / ".git").exists():
            if self.clone_dir.exists():
                if any(self.clone_dir.iterdir()):
                    raise RuntimeError(f"clone dir exists but is not a git repo: {self.clone_dir}")
                shutil.rmtree(self.clone_dir)
            clone_repo(auth_remote, self.clone_dir)
            if auth_remote != clean_remote:
                run_git(["remote", "set-url", "origin", clean_remote], self.clone_dir)

        ensure_git_config(self.clone_dir, self.git_user_name, self.git_user_email)

        pull_target = auth_remote if self.token else "origin"
        pull_result = run_git(["pull", "--rebase", pull_target, self.branch], self.clone_dir, check=False)
        if pull_result.returncode != 0:
            raise RuntimeError(pull_result.stderr.strip() or pull_result.stdout.strip() or "git pull failed")

        return self.clone_dir

    def _reset_clone(self) -> None:
        if self.clone_dir.exists():
            shutil.rmtree(self.clone_dir)

    def _event_path(self, repo_dir: Path, event_type: str, event_id: str) -> Path:
        now = time.gmtime()
        day = time.strftime("%Y-%m-%d", now)
        ts = time.strftime("%Y-%m-%dT%H-%M-%SZ", now)
        return repo_dir / "events" / day / f"{ts}_{event_type}_{event_id}.json"

    def _write_event(self, repo_dir: Path, event: dict) -> str:
        path = self._event_path(repo_dir, event["event_type"], event["event_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(event, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(path.relative_to(repo_dir))

    def _commit_and_push(self, repo_dir: Path, rel_paths: list[str], commit_message: str) -> None:
        run_git(["add", *rel_paths], repo_dir)
        diff_result = run_git(["diff", "--cached", "--quiet"], repo_dir, check=False)
        if diff_result.returncode == 0:
            return

        commit_result = run_git(["commit", "-m", commit_message], repo_dir, check=False)
        if commit_result.returncode != 0:
            raise RuntimeError(commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed")

        auth_remote = build_remote_url(self.repo, self.token)
        push_target = auth_remote if self.token else "origin"
        push_result = run_git(["push", push_target, f"HEAD:{self.branch}"], repo_dir, check=False)
        if push_result.returncode != 0:
            raise RuntimeError(push_result.stderr.strip() or push_result.stdout.strip() or "git push failed")

    def _iter_events(self, repo_dir: Path) -> Iterable[dict]:
        events_dir = repo_dir / "events"
        if not events_dir.exists():
            return []
        records: list[dict] = []
        for path in sorted(events_dir.rglob("*.json")):
            try:
                with path.open(encoding="utf-8") as handle:
                    event = json.load(handle)
                records.append(event)
            except Exception:
                continue
        records.sort(key=lambda item: (item.get("created_at", ""), item.get("event_id", "")))
        return records

    def _compute_state(self, repo_dir: Path, candidates: list[Candidate]) -> dict:
        now_ts = time.time()
        reservations: dict[str, dict] = {}
        released: set[str] = set()
        committed_reservations: set[str] = set()
        repo_sizes: dict[str, dict[str, dict]] = {candidate.account_id: {} for candidate in candidates}

        for event in self._iter_events(repo_dir):
            event_type = event.get("event_type")
            if event_type == "reserve":
                reservations[event["reservation_id"]] = event
            elif event_type == "release":
                released.add(event["reservation_id"])
            elif event_type == "commit":
                committed_reservations.add(event["reservation_id"])
                account_id = event["account_id"]
                repo_id = event["repo_id"]
                repo_sizes.setdefault(account_id, {})[repo_id] = {
                    "actual_bytes": int(event.get("actual_bytes", 0)),
                    "created_at": event.get("created_at"),
                }

        accounts = {}
        for candidate in candidates:
            used_bytes = sum(repo["actual_bytes"] for repo in repo_sizes.get(candidate.account_id, {}).values())
            reserved_bytes = 0
            active_reservations = []
            for reservation in reservations.values():
                if reservation.get("account_id") != candidate.account_id:
                    continue
                if reservation["reservation_id"] in released or reservation["reservation_id"] in committed_reservations:
                    continue
                expires_at = parse_timestamp(reservation.get("expires_at"))
                if expires_at and expires_at < now_ts:
                    continue
                reserved_bytes += int(reservation.get("reserve_bytes", 0))
                active_reservations.append(reservation)

            accounts[candidate.account_id] = {
                "account_id": candidate.account_id,
                "org": candidate.org,
                "quota_bytes": candidate.quota_bytes,
                "used_bytes": used_bytes,
                "reserved_bytes": reserved_bytes,
                "remaining_bytes": candidate.quota_bytes - used_bytes - reserved_bytes,
                "repos": repo_sizes.get(candidate.account_id, {}),
                "active_reservations": active_reservations,
            }
        return {"accounts": accounts}

    def _rank_candidates(
        self,
        state: dict,
        candidates: list[Candidate],
        repo_name: str,
        estimated_size_bytes: int,
    ) -> list[dict]:
        ranked = []
        for order, candidate in enumerate(candidates):
            account = state["accounts"][candidate.account_id]
            repo_id = f"{candidate.org}/{repo_name}"
            existing_repo_bytes = int(account["repos"].get(repo_id, {}).get("actual_bytes", 0))
            reserve_bytes = max(0, estimated_size_bytes - existing_repo_bytes)
            remaining_after = account["remaining_bytes"] - reserve_bytes
            ranked.append(
                {
                    "order": order,
                    "account_id": candidate.account_id,
                    "org": candidate.org,
                    "repo_id": repo_id,
                    "quota_bytes": candidate.quota_bytes,
                    "used_bytes": account["used_bytes"],
                    "reserved_bytes": account["reserved_bytes"],
                    "remaining_bytes": account["remaining_bytes"],
                    "existing_repo_bytes": existing_repo_bytes,
                    "reserve_bytes": reserve_bytes,
                    "remaining_after_reservation": remaining_after,
                    "fits": remaining_after >= 0,
                }
            )

        ranked.sort(
            key=lambda item: (
                0 if item["fits"] else 1,
                -item["remaining_after_reservation"],
                item["order"],
            )
        )
        return ranked

    def _build_reservation_event(self, selected: dict, repo_name: str, estimated_size_bytes: int) -> dict:
        reservation_id = str(uuid.uuid4())
        created_at = utc_now()
        expires_at = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + self.reservation_ttl_seconds)
        )
        return {
            "version": 1,
            "event_type": "reserve",
            "event_id": str(uuid.uuid4()),
            "reservation_id": reservation_id,
            "account_id": selected["account_id"],
            "org": selected["org"],
            "repo_id": selected["repo_id"],
            "repo_name": repo_name,
            "estimated_bytes": estimated_size_bytes,
            "existing_repo_bytes": selected["existing_repo_bytes"],
            "reserve_bytes": selected["reserve_bytes"],
            "created_at": created_at,
            "expires_at": expires_at,
            "worker": {
                "host": socket.gethostname(),
                "pid": os.getpid(),
            },
            "remaining_before_bytes": selected["remaining_bytes"],
            "remaining_after_reservation_bytes": selected["remaining_after_reservation"],
            "fits": selected["fits"],
        }
