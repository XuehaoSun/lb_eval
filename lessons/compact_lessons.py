#!/usr/bin/env python3
"""Compact BitLesson files: merge duplicates, summarize, prune stale entries.

Usage:
    python compact_lessons.py [lessons_dir]

Merges lessons with similar error keywords (Jaccard similarity >= 0.6),
combining their verified_count and source_tasks.
"""

import json
import sys
from pathlib import Path

MAX_LESSONS_BEFORE_COMPACT = 50
SIMILARITY_THRESHOLD = 0.6


def keyword_similarity(kw1: list, kw2: list) -> float:
    """Jaccard similarity between keyword sets."""
    s1 = set(k.lower() for k in kw1 if k)
    s2 = set(k.lower() for k in kw2 if k)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def compact_file(filepath: Path) -> None:
    """Compact a single JSONL lessons file."""
    lessons = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lessons.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(lessons) < MAX_LESSONS_BEFORE_COMPACT:
        print(f"  {filepath.name}: {len(lessons)} lessons (below threshold, skip)")
        return

    # Group by similar error_keywords
    groups: list[list[dict]] = []
    used: set[int] = set()

    for i, lesson in enumerate(lessons):
        if i in used:
            continue
        group = [lesson]
        used.add(i)
        kw_i = lesson.get("error_keywords", [])

        for j in range(i + 1, len(lessons)):
            if j in used:
                continue
            kw_j = lessons[j].get("error_keywords", [])
            if keyword_similarity(kw_i, kw_j) >= SIMILARITY_THRESHOLD:
                group.append(lessons[j])
                used.add(j)
        groups.append(group)

    # Merge each group
    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Pick the one with highest verified_count as base
        base = max(group, key=lambda x: x.get("verified_count", 1))
        all_tasks: list[str] = []
        all_refs: list[str] = []
        total_verified = 0

        for item in group:
            total_verified += item.get("verified_count", 1)
            tasks = item.get("source_tasks", [])
            if not tasks:
                # Legacy single field
                task = item.get("source_task", "")
                if task:
                    tasks = [task]
            all_tasks.extend(tasks)
            all_refs.extend(item.get("log_refs", []))

        base["verified_count"] = total_verified
        base["source_tasks"] = list(dict.fromkeys(t for t in all_tasks if t))[:10]
        base["log_refs"] = list(dict.fromkeys(r for r in all_refs if r))[:5]
        base.pop("source_task", None)

        # Merge tracebacks: keep the longest one
        tracebacks = [item.get("error_traceback", "") for item in group]
        base["error_traceback"] = max(tracebacks, key=len)

        merged.append(base)

    # Write back
    with open(filepath, "w") as f:
        for lesson in merged:
            f.write(json.dumps(lesson, ensure_ascii=False) + "\n")

    print(f"  {filepath.name}: {len(lessons)} -> {len(merged)} lessons (compacted)")


def main():
    lessons_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not lessons_dir.is_dir():
        print(f"ERROR: {lessons_dir} is not a directory")
        sys.exit(1)

    print(f"Compacting lessons in {lessons_dir}/")
    for jsonl_file in sorted(lessons_dir.glob("*.jsonl")):
        compact_file(jsonl_file)
    print("Done.")


if __name__ == "__main__":
    main()
