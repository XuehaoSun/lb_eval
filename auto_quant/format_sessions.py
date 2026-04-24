#!/usr/bin/env python3
"""
Format OpenClaw session JSONL logs into Markdown.

Usage:
    python3 format_sessions.py session_quant_2816.jsonl
    python3 format_sessions.py session_quant_2816.jsonl session_eval_2816.jsonl
    python3 format_sessions.py /path/to/output_dir
    python3 format_sessions.py /path/to/output_dir --output-dir /tmp/md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


def iter_jsonl_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[format_sessions] WARNING: {path}:{line_no}: invalid JSON: {exc}",
                    file=sys.stderr,
                )


def format_timestamp(timestamp: str | None) -> str:
    if not timestamp:
        return "unknown"
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except ValueError:
        return timestamp


def stringify_json(data: object) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False, indent=2)


def format_content_item(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return stringify_json(item)

    item_type = item.get("type", "")
    if item_type == "text":
        return item.get("text", "")
    if item_type == "thinking":
        thinking = item.get("thinking", "")
        return f"**Thinking**\n\n{thinking}"
    if item_type == "toolCall":
        name = item.get("name", "unknown")
        arguments = stringify_json(item.get("arguments", {}))
        return f"**Tool call:** `{name}`\n\n```json\n{arguments}\n```"
    if item_type == "toolResult":
        content = format_content(item.get("content", []))
        return f"**Tool result**\n\n{content}"
    if item_type == "image":
        source = item.get("source", {})
        return f"[image] {source.get('url', 'unknown')}"
    return stringify_json(item)


def format_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [format_content_item(item) for item in content]
        return "\n\n".join(part for part in parts if part)
    return format_content_item(content)


def role_title(role: str) -> str:
    mapping = {
        "user": "USER",
        "assistant": "ASSISTANT",
        "toolResult": "TOOL RESULT",
    }
    return mapping.get(role, role.upper() if role else "UNKNOWN")


def format_message(record: dict) -> str:
    message = record.get("message", {})
    role = message.get("role", "unknown")
    timestamp = format_timestamp(record.get("timestamp"))
    lines = [f"### [{timestamp}] {role_title(role)}"]

    if role == "toolResult":
        tool_name = message.get("toolName", "unknown")
        details = message.get("details") or {}
        meta = [f"**Tool:** `{tool_name}`"]
        if "status" in details:
            meta.append(f"**Status:** `{details['status']}`")
        if "exitCode" in details:
            meta.append(f"**Exit code:** `{details['exitCode']}`")
        lines.append(" | ".join(meta))
        body = format_content(message.get("content", []))
        if body:
            lines.append("")
            lines.append(body)
        return "\n".join(lines)

    content = message.get("content", [])
    body = format_content(content)
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def detect_step(session_id: str) -> str:
    lowered = session_id.lower()
    if "quant" in lowered:
        return "Step 1: Quantization"
    if "eval" in lowered:
        return "Step 2: Evaluation"
    return "Session"


def collect_input_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            dir_files = sorted(path.glob("session_*.jsonl"))
            if not dir_files:
                dir_files = sorted(path.glob("*.jsonl"))
            files.extend(dir_files)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(file_path)
    return deduped


def render_session(path: Path) -> str:
    session_meta: dict = {"id": path.stem, "timestamp": None, "cwd": None}
    messages: list[dict] = []

    for record in iter_jsonl_records(path):
        record_type = record.get("type")
        if record_type == "session":
            session_meta.update(
                {
                    "id": record.get("id", session_meta["id"]),
                    "timestamp": record.get("timestamp"),
                    "cwd": record.get("cwd"),
                }
            )
        elif record_type == "message":
            messages.append(record)

    lines = [
        f"# Session: {session_meta['id']}",
        "",
        f"- **Session ID:** `{session_meta['id']}`",
        f"- **Timestamp:** {format_timestamp(session_meta.get('timestamp'))}",
        f"- **Working Dir:** `{session_meta.get('cwd') or 'unknown'}`",
        "",
        f"## {detect_step(session_meta['id'])}",
        "",
    ]

    if not messages:
        lines.append("_No messages found._")
    else:
        for message in messages:
            lines.append(format_message(message))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Format OpenClaw session JSONL logs into Markdown.")
    parser.add_argument("inputs", nargs="+", help="JSONL files or directories containing session JSONL files")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory for generated Markdown files. Defaults to each input file's directory.",
    )
    args = parser.parse_args()

    input_files = collect_input_files(args.inputs)
    if not input_files:
        print("[format_sessions] ERROR: no JSONL files found.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_files:
        markdown = render_session(input_path)
        destination = (
            output_dir / f"{input_path.stem}.md"
            if output_dir
            else input_path.with_suffix(".md")
        )
        destination.write_text(markdown, encoding="utf-8")
        print(f"[format_sessions] Wrote {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
