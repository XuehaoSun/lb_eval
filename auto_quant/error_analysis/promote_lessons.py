#!/usr/bin/env python3
"""L3 self-learning: promote recurring agent-labeled errors into a learned taxonomy overlay.

The curated TAXONOMY (taxonomy.py) is measured by PRECISION — it only classifies what it is
sure about. Long-tail coverage (RECALL) is the agent's job. This tool closes the loop: when the
deterministic taxonomy repeatedly says "unknown" but the fix-loop agent keeps assigning the SAME
semantic ERROR_CLASS to that error, we learn a conservative regex signature for it and write it to
`learned_signatures.json` — a SEPARATE overlay that taxonomy.classify_error() consults only AFTER
curated matching fails. This never corrupts the curated set and stays fully auditable.

    # inspect proposals (safe, default):
    python3 promote_lessons.py --lessons-dir ../lessons

    # write/update the overlay:
    python3 promote_lessons.py --lessons-dir ../lessons --apply

Promotion gates (all must hold), tuned for precision over recall:
  * taxonomy could NOT classify the error (error_category == "unknown")   → genuine gap
  * the agent assigned a stable, non-generic snake_case class             → real semantic label
  * the class recurs >= THRESHOLD times across lessons                     → not a one-off
  * a distinctive signature (>= MIN_LEN chars, >= 2 alpha words) can be    → precise, not broad
    derived from the recurring error text
  * the derived signature still yields "unknown" under the CURRENT curated → no shadowing/dupes
    taxonomy
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Import the curated classifier so we can (a) confirm errors are genuinely "unknown" and
# (b) make sure a learned signature won't shadow a curated category.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import classify_error  # noqa: E402

DEFAULT_THRESHOLD = 3
DEFAULT_MIN_SIG_LEN = 16

# Agent class tokens too generic to ever be a useful learned category.
_GENERIC_CLASSES = {
    "unknown", "error", "unfixable", "fixable", "failure", "failed",
    "runtime_error", "exception", "misc", "other", "n_a", "na", "none",
}

# Phrases that are too broad to promote as a signature (would over-match).
_GENERIC_PHRASES = {
    "error", "exception", "failed", "traceback", "runtime error",
    "value error", "type error", "assertion error", "the above exception",
    "during handling", "most recent call last",
}

_DIGIT_TOKEN = "\x00NUM\x00"


def _normalize(sig: str) -> str:
    """Normalize an error signature so variable parts (numbers/paths/hex/quotes) don't
    prevent recurrence matching. Numbers become a placeholder later turned into \\d+."""
    s = sig.strip()
    s = re.sub(r"0x[0-9a-fA-F]+", _DIGIT_TOKEN, s)          # hex addresses
    s = re.sub(r"/[^\s'\"]+", " ", s)                        # absolute paths
    s = re.sub(r"\b\d+\b", _DIGIT_TOKEN, s)                  # bare integers
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_regex(normalized: str) -> str:
    """Turn a normalized skeleton into a conservative regex (literals escaped, numbers -> \\d+)."""
    parts = normalized.split(_DIGIT_TOKEN)
    return r"\d+".join(re.escape(p) for p in parts)


def _alpha_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", text))


def _longest_common_substring(strings: list[str]) -> str:
    """Longest substring common to ALL strings (on normalized text)."""
    if not strings:
        return ""
    shortest = min(strings, key=len)
    best = ""
    for i in range(len(shortest)):
        for j in range(len(shortest), i + len(best), -1):
            cand = shortest[i:j]
            if all(cand in s for s in strings):
                if len(cand) > len(best):
                    best = cand
                break
    return best.strip()


def derive_signature(signatures: list[str], min_len: int) -> tuple[str, str] | None:
    """Derive a (regex, example) from a group of recurring error signatures, or None.

    Strategy: if one normalized signature dominates the group, use it (highest precision);
    otherwise fall back to the longest common substring shared by all members.
    """
    cleaned = [s for s in signatures if s and s.strip()]
    if not cleaned:
        return None
    normed = [_normalize(s) for s in cleaned]

    # 1. Dominant identical normalized signature?
    counts: dict[str, int] = defaultdict(int)
    for n in normed:
        counts[n] += 1
    dominant, dom_count = max(counts.items(), key=lambda kv: kv[1])
    if dom_count >= max(2, len(normed) // 2 + 1):
        skeleton = dominant
    else:
        # 2. Longest common substring across the whole group.
        skeleton = _longest_common_substring(normed)

    skeleton = skeleton.strip(" :-|")
    display = skeleton.replace(_DIGIT_TOKEN, "N")
    if len(display) < min_len:
        return None
    if _alpha_word_count(display) < 2:
        return None
    if display.strip().lower() in _GENERIC_PHRASES:
        return None

    regex = _to_regex(skeleton)
    # Pick a real example that this regex actually matches.
    example = ""
    for s in cleaned:
        try:
            if re.search(regex, s, re.IGNORECASE):
                example = s[:150]
                break
        except re.error:
            return None
    return regex, (example or cleaned[0][:150])


def load_lessons(lessons_dir: Path) -> list[dict]:
    lessons = []
    for fpath in sorted(lessons_dir.glob("*.jsonl")):
        try:
            for line in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    lessons.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            continue
    return lessons


def build_proposals(lessons: list[dict], threshold: int, min_len: int) -> list[dict]:
    # Group candidate lessons by the agent's semantic class.
    groups: dict[str, list[dict]] = defaultdict(list)
    for les in lessons:
        taxo = (les.get("error_category") or "").strip().lower()
        agent = (les.get("agent_category") or "").strip().lower()
        # Only learn where the deterministic taxonomy genuinely could NOT classify it.
        if taxo and taxo != "unknown":
            continue
        if not agent or agent in _GENERIC_CLASSES:
            continue
        groups[agent].append(les)

    proposals = []
    for agent_class, group in sorted(groups.items()):
        if len(group) < threshold:
            continue
        sigs = [les.get("error_signature", "") for les in group]
        derived = derive_signature(sigs, min_len)
        if not derived:
            continue
        regex, example = derived

        # No shadowing: the example must STILL be unknown under the curated taxonomy.
        if classify_error(example)[0] != "unknown":
            continue

        phases = sorted({les.get("phase", "") for les in group if les.get("phase")})
        times = sorted(les.get("timestamp", "") for les in group if les.get("timestamp"))
        root_causes = [les.get("agent_root_cause", "") for les in group if les.get("agent_root_cause")]
        proposals.append({
            "category": agent_class,
            "signature": regex,
            "description": (root_causes[0][:200] if root_causes else f"Learned category '{agent_class}'"),
            "source_count": len(group),
            "example": example,
            "phases": phases,
            "first_seen": times[0] if times else "",
            "last_seen": times[-1] if times else "",
            "fix_strategy": "agent_investigation",
            "retryable": None,
            "learned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
    return proposals


def merge_overlay(existing: list[dict], proposals: list[dict]) -> list[dict]:
    """Merge proposals into the existing overlay, keyed by (category, signature)."""
    by_key = {(e["category"], e["signature"]): e for e in existing}
    for p in proposals:
        key = (p["category"], p["signature"])
        if key in by_key:
            # Refresh count/last_seen, keep original first_seen.
            old = by_key[key]
            old["source_count"] = max(old.get("source_count", 0), p["source_count"])
            old["last_seen"] = p["last_seen"] or old.get("last_seen", "")
            old["example"] = p["example"]
        else:
            by_key[key] = p
    return list(by_key.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lessons-dir", required=True, type=Path, help="Directory of lessons/*.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "learned_signatures.json",
                    help="Overlay file to write (default: error_analysis/learned_signatures.json)")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                    help=f"Min recurrences to promote (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--min-signature-len", type=int, default=DEFAULT_MIN_SIG_LEN,
                    help=f"Min derived signature length (default {DEFAULT_MIN_SIG_LEN})")
    ap.add_argument("--apply", action="store_true", help="Write the overlay (default: dry-run print only)")
    args = ap.parse_args()

    if not args.lessons_dir.is_dir():
        print(f"[error] lessons-dir not found: {args.lessons_dir}", file=sys.stderr)
        return 2

    lessons = load_lessons(args.lessons_dir)
    proposals = build_proposals(lessons, args.threshold, args.min_signature_len)

    print(f"[promote] scanned {len(lessons)} lessons → {len(proposals)} promotable categor(ies) "
          f"(threshold={args.threshold})\n")
    if not proposals:
        print("  No categories met the promotion gates. Nothing to do.")
        return 0

    for p in proposals:
        print(f"  • {p['category']}  (x{p['source_count']}, phases={p['phases']})")
        print(f"      signature: {p['signature']}")
        print(f"      example:   {p['example']}")
        print()

    if not args.apply:
        print("[dry-run] Re-run with --apply to write these to the learned overlay.")
        return 0

    existing = []
    if args.out.exists():
        try:
            data = json.loads(args.out.read_text(encoding="utf-8"))
            existing = data.get("signatures", data) if isinstance(data, dict) else data
        except ValueError:
            existing = []

    merged = merge_overlay(existing, proposals)
    payload = {
        "version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": "Auto-learned overlay (L3). Consulted by taxonomy.classify_error ONLY after curated matching fails. Safe to prune/edit by hand.",
        "signatures": merged,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[apply] wrote {len(merged)} learned signature(s) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
