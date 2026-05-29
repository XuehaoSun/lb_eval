#!/bin/bash
# agent_fix_loop.sh — Sourceable library for the agent-assisted fix loop.
#
# Provides:
#   agent_fix_loop <phase_name> <script_path> [args...]
#   save_lesson <phase> <error_context> <status> <solution_note>
#   search_lessons <phase> <error_text>
#   maybe_compact_lessons
#   push_lessons_to_git
#
# Required environment:
#   RUN_OUTPUT_DIR    — base output dir for this run
#   LESSONS_DIR       — path to lessons/ directory (git tracked)
#   MAX_FIX_ATTEMPTS  — max agent retry attempts (default: 3)
#   MODEL_ID, SCHEME, METHOD — for lesson metadata

# Guard against double-source
[[ -n "${_AGENT_FIX_LOOP_SOURCED:-}" ]] && return 0
_AGENT_FIX_LOOP_SOURCED=1

MAX_FIX_ATTEMPTS="${MAX_FIX_ATTEMPTS:-10}"
LESSONS_DIR="${LESSONS_DIR:-${LB_EVAL_REPO_DIR:-$(dirname "$0")/../lessons}}"

# ═══════════════════════════════════════════════════════════════════
# agent_fix_loop — run a phase script, retry with agent on failure
# ═══════════════════════════════════════════════════════════════════
agent_fix_loop() {
    local phase_name="$1"
    local script_path="$2"
    shift 2
    local script_args=("$@")

    local max_attempts="${MAX_FIX_ATTEMPTS}"
    local attempt=0
    local phase_log="${RUN_OUTPUT_DIR}/logs/${phase_name}.log"
    local fix_log_dir="${RUN_OUTPUT_DIR}/logs/agent_fixes/${phase_name}"
    mkdir -p "$(dirname "${phase_log}")" "${fix_log_dir}"

    # First execution (deterministic script)
    log_step "Phase: ${phase_name}"
    bash "${script_path}" "${script_args[@]}" 2>&1 | tee "${phase_log}"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_ok "${phase_name} succeeded"
        return 0
    fi

    log_warn "${phase_name} failed (exit=${exit_code}), entering agent fix loop"

    # Fix loop
    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        log_step "Agent fix attempt ${attempt}/${max_attempts} for ${phase_name}"

        # 1. Extract error context
        local error_tail
        error_tail=$(tail -100 "${phase_log}")

        # 2. Check for drift (same error repeating)
        if [ $attempt -gt 1 ]; then
            local prev_sig
            prev_sig=$(echo "${error_tail}" | grep -i "error\|exception\|failed" | head -1 | cut -c1-100)
            local prev_retry_log="${fix_log_dir}/retry_$((attempt - 1)).log"
            if [ -f "${prev_retry_log}" ]; then
                local prev_err
                prev_err=$(tail -100 "${prev_retry_log}" | grep -i "error\|exception\|failed" | head -1 | cut -c1-100)
                if [ -n "${prev_sig}" ] && [ "${prev_sig}" == "${prev_err}" ]; then
                    log_warn "Drift detected: same error repeating. Aborting fix loop."
                    save_lesson "${phase_name}" "${error_tail}" "drift" "Same error repeated ${attempt} times"
                    break
                fi
            fi
        fi

        # 3. Load all lessons for agent context
        local lessons=""
        if [ -d "${LESSONS_DIR}" ]; then
            lessons=$(load_all_lessons 2>/dev/null || true)
        fi
        if [ -n "${lessons}" ]; then
            log_info "Loaded lessons for agent (let agent decide relevance)"
        else
            log_info "No lessons available"
        fi

        # 4. Build agent prompt
        local fix_prompt
        fix_prompt=$(build_fix_prompt "${phase_name}" "${error_tail}" "${lessons}")

        # 5. Save prompt for audit
        local prompt_file="${fix_log_dir}/prompt_${attempt}.txt"
        printf '%s\n' "${fix_prompt}" > "${prompt_file}"

        # 6. Call OpenClaw agent
        local agent_log="${fix_log_dir}/attempt_${attempt}.log"
        run_openclaw_fix "${fix_prompt}" "${agent_log}" || true

        # 7. Re-run phase script to verify
        log_info "Re-running ${phase_name} after agent fix..."
        local retry_log="${fix_log_dir}/retry_${attempt}.log"
        bash "${script_path}" "${script_args[@]}" 2>&1 | tee "${retry_log}"
        exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            log_ok "${phase_name} fixed on attempt ${attempt}"
            save_lesson "${phase_name}" "${error_tail}" "fixed" "Agent fixed on attempt ${attempt}"
            return 0
        fi

        phase_log="${retry_log}"
        save_lesson "${phase_name}" "${error_tail}" "still_failing" "Attempt ${attempt} did not resolve"
    done

    log_error "${phase_name} failed after ${max_attempts} fix attempts"
    return 1
}

# ═══════════════════════════════════════════════════════════════════
# build_fix_prompt — construct the agent prompt for fixing a phase
# ═══════════════════════════════════════════════════════════════════
build_fix_prompt() {
    local phase="$1"
    local error="$2"
    local lessons="$3"

    local lessons_section=""
    if [ -n "${lessons}" ]; then
        lessons_section="## Historical Lessons (from past runs — decide which are relevant):
${lessons}
Review the lessons above and apply the most relevant fix for the current error."
    else
        lessons_section="## Historical Lessons:
No lessons available yet."
    fi

    cat <<PROMPT
You are fixing a failed "${phase}" phase in the quantization pipeline.

## Error Output (last 100 lines):
${error}

${lessons_section}

## Your Task:
1. First output a brief FIX_PLAN (3 lines max) describing what you will do
2. Then execute the fix (modify files, install packages, adjust parameters)
3. The phase script will be re-run after your fix to verify

## Constraints:
- Do NOT reinstall or downgrade torch (it will break CUDA)
- Do NOT modify the evaluation tasks or expected output format
- Keep fixes minimal and targeted
- If you need to install a package, use: pip install <package>
- If unsupported model architecture, try: pip install -U auto-round transformers
- Working directory: ${RUN_OUTPUT_DIR}
PROMPT
}

# ═══════════════════════════════════════════════════════════════════
# run_openclaw_fix — call OpenClaw agent with the fix prompt
# ═══════════════════════════════════════════════════════════════════
run_openclaw_fix() {
    local prompt="$1"
    local log_file="$2"

    if ! command -v openclaw >/dev/null 2>&1; then
        log_warn "openclaw not found, skipping agent fix"
        echo "openclaw not available" > "${log_file}"
        return 1
    fi

    local timeout="${AGENT_TIMEOUT:-600}"
    local session_id="fix_${phase_name:-unknown}_$$_$(date +%s)"

    log_info "Calling openclaw agent (session=${session_id}, timeout=${timeout}s)..."
    timeout "${timeout}" openclaw agent --local \
        --session-id "${session_id}" \
        --message "${prompt}" \
        --timeout "${timeout}" \
        2>&1 | tee "${log_file}" || {
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo "[TIMEOUT] Agent exceeded ${timeout}s" >> "${log_file}"
            log_warn "Agent timed out after ${timeout}s"
        fi
    }

    return 0
}

# ═══════════════════════════════════════════════════════════════════
# save_lesson — persist a lesson to the JSONL file
# ═══════════════════════════════════════════════════════════════════
save_lesson() {
    local phase="$1"
    local error_context="$2"
    local status="$3"
    local solution_note="$4"

    local lessons_file="${LESSONS_DIR}/${phase}.jsonl"
    mkdir -p "${LESSONS_DIR}"

    # Pass error_context via env var (not stdin, which conflicts with heredoc)
    LESSON_ERROR_CONTEXT="${error_context}" python3 - "${phase}" "${status}" "${solution_note}" "${MODEL_ID:-unknown}" "${SCHEME:-W4A16}" "${METHOD:-RTN}" "${lessons_file}" <<'PYEOF'
import json
import sys
import os
import datetime
import re

phase = sys.argv[1]
status = sys.argv[2]
solution_note = sys.argv[3]
model_id = sys.argv[4]
scheme = sys.argv[5]
method = sys.argv[6]
lessons_file = sys.argv[7]

error_context = os.environ.get("LESSON_ERROR_CONTEXT", "")

# Extract signature (first error/exception line)
error_signature = ""
for line in error_context.splitlines():
    lower = line.lower()
    if any(kw in lower for kw in ("error", "exception", "failed", "traceback")):
        error_signature = line.strip()[:150]
        break
if not error_signature:
    error_signature = error_context.strip().splitlines()[-1][:150] if error_context.strip() else "unknown error"

# Extract keywords
words = re.findall(r'[a-zA-Z]{4,}', error_signature.lower())
keywords = list(dict.fromkeys(words))[:5]  # unique, ordered

# Full traceback (last 50 lines)
traceback_lines = error_context.strip().splitlines()[-50:]
error_traceback = "\n".join(traceback_lines)

lesson = {
    "id": f"lesson-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": phase,
    "error_signature": error_signature,
    "error_traceback": error_traceback,
    "error_keywords": keywords,
    "model": model_id,
    "scheme": scheme,
    "method": method,
    "solution": solution_note,
    "status": status,
    "verified_count": 1,
    "source_tasks": [f"{model_id}_{scheme}_{method}"],
}

with open(lessons_file, "a") as f:
    f.write(json.dumps(lesson, ensure_ascii=False) + "\n")

print(f"[lesson] Saved: [{status}] {error_signature[:80]}")
PYEOF
}

# ═══════════════════════════════════════════════════════════════════
# load_all_lessons — load all lessons as text for agent to decide relevance
# ═══════════════════════════════════════════════════════════════════
load_all_lessons() {
    [ ! -d "${LESSONS_DIR}" ] && return 0

    python3 - "${LESSONS_DIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

lessons_dir = Path(sys.argv[1])
lessons = []

for fpath in sorted(lessons_dir.glob("*.jsonl")):
    try:
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                lesson = json.loads(line)
                lessons.append(lesson)
    except (FileNotFoundError, json.JSONDecodeError):
        continue

# Deduplicate by error_signature
seen = set()
unique = []
for les in lessons:
    sig = les.get("error_signature", "")
    if sig not in seen:
        seen.add(sig)
        unique.append(les)

# Sort by verified_count (most reliable first), cap at 10 to avoid huge prompts
unique.sort(key=lambda x: x.get("verified_count", 0), reverse=True)
for i, les in enumerate(unique[:10], 1):
    status = les.get("status", "unknown")
    verified = les.get("verified_count", 0)
    phase = les.get("phase", "?")
    sig = les.get("error_signature", "")[:120]
    solution = les.get("solution", "")
    notes = les.get("notes", "")
    print(f"Lesson {i} [phase={phase}, status={status}, verified={verified}x]:")
    print(f"  Error: {sig}")
    print(f"  Solution: {solution}")
    if notes:
        print(f"  Notes: {notes}")
    print()
PYEOF
}

# ═══════════════════════════════════════════════════════════════════
# maybe_compact_lessons — compact if > 50 entries
# ═══════════════════════════════════════════════════════════════════
maybe_compact_lessons() {
    local compact_script="${LESSONS_DIR}/compact_lessons.py"
    [ ! -f "${compact_script}" ] && return 0

    for f in "${LESSONS_DIR}"/*.jsonl; do
        [ ! -f "$f" ] && continue
        local count
        count=$(wc -l < "$f")
        if [ "$count" -gt 50 ]; then
            log_info "Compacting lessons (${count} entries in $(basename "$f"))..."
            python3 "${compact_script}" "${LESSONS_DIR}"
            break
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════
# push_lessons_to_git — commit and push lessons
# ═══════════════════════════════════════════════════════════════════
push_lessons_to_git() {
    maybe_compact_lessons

    local repo_dir="${LB_EVAL_REPO_DIR:-}"
    [ -z "${repo_dir}" ] && return 0
    [ ! -d "${repo_dir}/.git" ] && return 0

    cd "${repo_dir}"
    git add --force lessons/ 2>/dev/null || true
    if ! git diff --cached --quiet lessons/ 2>/dev/null; then
        git commit -m "lessons: update from ${MODEL_ID:-unknown} ${SCHEME:-} ${METHOD:-}" || true
        # Use token-authenticated URL if available
        local push_url="origin"
        if [[ -n "${GIT_TOKEN:-}" && -n "${GIT_REPO:-}" ]]; then
            push_url="${GIT_REPO/https:\/\//https://x-access-token:${GIT_TOKEN}@}"
        fi
        git push "${push_url}" "${GIT_BRANCH:-main}" 2>/dev/null || log_warn "Failed to push lessons"
    fi
    cd - > /dev/null
}
