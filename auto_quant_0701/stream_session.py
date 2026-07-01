#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def shorten(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "thinking":
                    continue
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part).strip()
    return json.dumps(content, ensure_ascii=False)


def role_text_summary(role: str, message: dict) -> str | None:
    content = message.get("content", [])
    if not isinstance(content, list):
        text = content_text(content).strip()
        return f"{role}: {shorten(text, 1500)}" if text else None

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            rendered = json.dumps(item, ensure_ascii=False).strip()
            if rendered:
                parts.append(rendered)
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = str(item.get("text", "")).strip()
            if text:
                parts.append(text)
        elif item_type == "thinking" and role == "assistant":
            thinking = str(item.get("thinking", "")).strip()
            if thinking:
                parts.append(f"[thinking]\n{thinking}")

    if not parts:
        return None
    return f"{role}:\n{shorten(chr(10).join(parts), 2000)}"


def tool_call_summary(message: dict) -> str | None:
    content = message.get("content", [])
    if not isinstance(content, list):
        return None

    summaries: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "toolCall":
            continue
        name = item.get("name", "unknown")
        arguments = item.get("arguments", {})
        if name == "read":
            target = arguments.get("file") if isinstance(arguments, dict) else None
            summaries.append(f"tool call: read {target or ''}".strip())
            continue
        if name == "process":
            action = arguments.get("action", "unknown") if isinstance(arguments, dict) else "unknown"
            session_id = arguments.get("sessionId", "") if isinstance(arguments, dict) else ""
            summaries.append(f"tool call: process action={action} session={session_id}".strip())
            continue
        if name == "exec":
            command = arguments.get("command", "") if isinstance(arguments, dict) else ""
            summaries.append(f"tool call: exec\n{shorten(str(command), 1200)}")
            continue
        summaries.append(f"tool call: {name}")

    if not summaries:
        return None
    return "\n\n".join(summaries)


def tool_result_summary(message: dict, last_outputs: dict[str, str]) -> str | None:
    tool_name = message.get("toolName", "unknown")
    details = message.get("details") or {}
    tool_call_id = message.get("toolCallId") or f"{tool_name}:{details.get('sessionId', '')}"
    prefix = f"tool result: {tool_name}"

    if tool_name == "read":
        return prefix

    text = ""
    if tool_name == "process":
        text = str(details.get("tail") or details.get("aggregated") or content_text(message.get("content", [])) or "")
    elif tool_name == "exec":
        text = str(details.get("aggregated") or content_text(message.get("content", [])) or "")
    else:
        text = content_text(message.get("content", []))

    text = text.strip()
    if not text:
        status = details.get("status")
        exit_code = details.get("exitCode")
        meta = []
        if status:
            meta.append(f"status={status}")
        if exit_code is not None:
            meta.append(f"exit={exit_code}")
        return f"{prefix} {' '.join(meta)}".strip()

    previous = last_outputs.get(tool_call_id, "")
    if previous and text.startswith(previous):
        text = text[len(previous):].lstrip()
    last_outputs[tool_call_id] = str(details.get("tail") or details.get("aggregated") or content_text(message.get("content", [])) or "").strip()

    if not text:
        return None

    meta = []
    if details.get("status"):
        meta.append(f"status={details['status']}")
    if details.get("exitCode") is not None:
        meta.append(f"exit={details['exitCode']}")

    header = prefix
    if meta:
        header += " " + " ".join(meta)
    return f"{header}\n{shorten(text, 4000)}"


def emit(label: str, text: str) -> None:
    if not text:
        return
    sys.stdout.write(f"[session:{label}] {text}\n")
    sys.stdout.flush()


def follow(path: Path, label: str, poll_interval: float) -> int:
    start = time.time()
    while not path.exists():
        if time.time() - start > 30:
            emit(label, f"waiting for session file: {path}")
            start = time.time()
        time.sleep(poll_interval)

    emit(label, f"following session file: {path}")
    last_outputs: dict[str, str] = {}

    with path.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                time.sleep(poll_interval)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("type") != "message":
                continue

            message = record.get("message", {})
            role = message.get("role")
            if role == "user":
                summary = role_text_summary("user", message)
                if summary:
                    emit(label, summary)
            if role == "assistant":
                role_summary = role_text_summary("assistant", message)
                if role_summary:
                    emit(label, role_summary)
                summary = tool_call_summary(message)
                if summary:
                    emit(label, summary)
            elif role == "toolResult":
                summary = tool_result_summary(message, last_outputs)
                if summary:
                    emit(label, summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Follow OpenClaw session JSONL and print incremental summaries.")
    parser.add_argument("session_file", help="Path to session JSONL file")
    parser.add_argument("--label", default="openclaw", help="Short label used in log prefixes")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between polls")
    args = parser.parse_args()

    try:
        return follow(Path(args.session_file), args.label, args.poll_interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
