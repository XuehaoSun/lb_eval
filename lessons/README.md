# BitLesson — Accumulated Experience Store

This directory stores quantization pipeline lessons learned from agent fix attempts.

## Format

Each `.jsonl` file corresponds to a pipeline phase:
- `setup_env.jsonl` — Environment installation issues and fixes
- `quantize.jsonl` — Quantization failures and solutions
- `evaluate.jsonl` — Evaluation problems and resolutions
- `global.jsonl` — Cross-phase general lessons

## Lesson Schema

```json
{
  "id": "lesson-20260528133000",
  "timestamp": "2026-05-28T13:30:00Z",
  "phase": "quantize",
  "error_signature": "1-line summary of the error",
  "error_traceback": "Full traceback (last 50 lines)",
  "error_keywords": ["keyword1", "keyword2"],
  "model": "org/model-name",
  "scheme": "W4A16",
  "method": "RTN",
  "solution": "Brief description of what fixed it",
  "status": "fixed|still_failing|drift",
  "verified_count": 1,
  "source_tasks": ["org/model_W4A16_RTN"]
}
```

## How It Works

1. Pipeline runs → phase script fails
2. `search_lessons()` finds matching past fixes by keyword similarity
3. Matching lessons are injected into agent prompt (signature + solution only)
4. Agent attempts fix → phase re-runs
5. Outcome saved as new lesson (`save_lesson()`)
6. Lessons pushed to git → available for next task

## Compaction

When a file exceeds 50 entries, `compact_lessons.py` merges similar entries
(Jaccard keyword similarity ≥ 0.6), combining verified_count and source_tasks.
