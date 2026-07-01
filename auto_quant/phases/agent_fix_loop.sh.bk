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
            # Extract agent's fix summary (first lines containing FIX_PLAN or actual commands)
            local fix_summary=""
            if [ -f "${agent_log}" ]; then
                fix_summary=$(grep -A3 "FIX_PLAN\|Fix applied\|Installing\|pip install\|Changing\|Setting" "${agent_log}" | head -5 | tr '\n' '; ')
            fi
            fix_summary="${fix_summary:-Agent fixed on attempt ${attempt}}"
            save_lesson "${phase_name}" "${error_tail}" "fixed" "${fix_summary}"
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
1. READ the traceback carefully — identify the EXACT file and line that caused the error
2. Determine if the fault is in: auto-round code, transformers, model's custom code, or environment
3. Output a brief FIX_PLAN (3 lines max) describing what you will do
4. Execute the fix, then the phase will be re-run to verify

## Key Technique: Patching Model Custom Code

If the traceback shows files in \`~/.cache/huggingface/modules/transformers_modules/\`, that is the
MODEL'S CUSTOM CODE that was downloaded from HuggingFace. **YOU CAN AND SHOULD EDIT THESE FILES.**

Common fixes for model custom code:
- dtype mismatch (\`.float()\` mixed with bfloat16): Replace \`.float()\` with \`.to(other_tensor.dtype)\`
- Missing device: Add \`device=hidden_states.device\` to tensor creation
- Invalid regex: Fix the regex pattern in the model file
- Missing imports: Add the import or install the package

Example: If you see:
  File "/root/.cache/huggingface/modules/transformers_modules/Org/Model/hash/model.py", line 147
    h = h + torch.matmul(compressed[:, k:k+valid_len, :].float(), proj.t())
  RuntimeError: expected m1 and m2 to have the same dtype

Fix: Edit that file, change \`.float()\` to \`.to(proj.dtype)\`

## Constraints:
- Do NOT reinstall or downgrade torch (it will break CUDA)
- Do NOT modify the evaluation tasks or expected output format
- Keep fixes minimal and targeted — change only what's needed
- If you need to install a package, use: pip install <package>
- If unsupported model architecture (multimodal/VL), report and stop
- Working directory: ${RUN_OUTPUT_DIR}
- Model: ${MODEL_ID}
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
    local sessions_dir="${OPENCLAW_SESSIONS_DIR:-/root/.openclaw/agents/main/sessions}"
    local session_file="${sessions_dir}/${session_id}.jsonl"

    log_info "Calling openclaw agent (session=${session_id}, timeout=${timeout}s)..."
    log_info "  Session file: ${session_file}"

    # Background progress reporter — prints elapsed time + session size every 30s
    local _progress_pid=""
    (
        local _start=$SECONDS
        while true; do
            sleep 30
            local elapsed=$(( SECONDS - _start ))
            local session_lines=0
            [[ -f "${session_file}" ]] && session_lines=$(wc -l < "${session_file}" 2>/dev/null || echo 0)
            log_info "  [agent running ${elapsed}s] session: ${session_lines} messages"
        done
    ) &
    _progress_pid=$!

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

    # Stop progress reporter
    if [[ -n "${_progress_pid}" ]]; then
        kill "${_progress_pid}" 2>/dev/null || true
        wait "${_progress_pid}" 2>/dev/null || true
    fi

    # Print session summary to auto.log
    if [[ -f "${session_file}" ]]; then
        local msg_count tool_count
        msg_count=$(grep -c '"type":"message"\|"type": "message"' "${session_file}" 2>/dev/null || echo 0)
        tool_count=$(grep -c '"tool_use"\|"tool_call"' "${session_file}" 2>/dev/null || echo 0)
        log_info "  Agent session complete: ${msg_count} messages, ${tool_count} tool calls"
    fi

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
print(f"[lesson]   Solution: {solution_note}")
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
                # Only load lessons with actionable solutions (fixed, verified, or seed lessons)
                if lesson.get("status") in ("fixed", "seed", "verified"):
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
    verified = les.get("verified_count", 0)
    phase = les.get("phase", "?")
    sig = les.get("error_signature", "")[:120]
    solution = les.get("solution", "")
    notes = les.get("notes", "")
    print(f"Lesson {i} [phase={phase}, verified={verified}x]:")
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

    local lessons_dir="${LESSONS_DIR:-}"
    [ -z "${lessons_dir}" ] && return 0
    [ ! -d "${lessons_dir}" ] && return 0

    # Check if any lessons exist to push
    local has_lessons=false
    for f in "${lessons_dir}"/*.jsonl; do
        [ -f "$f" ] && has_lessons=true && break
    done
    [ "${has_lessons}" = false ] && return 0

    # Need GIT_TOKEN and GIT_REPO to push
    if [[ -z "${GIT_TOKEN:-}" || -z "${GIT_REPO:-}" ]]; then
        log_warn "push_lessons: GIT_TOKEN or GIT_REPO not set, skipping"
        return 0
    fi

    local branch="${GIT_BRANCH:-main}"
    local auth_url="${GIT_REPO/https:\/\//https://x-access-token:${GIT_TOKEN}@}"
    local tmp_clone="${RUN_OUTPUT_DIR}/.lessons_push_tmp"

    # Clone fresh (shallow, only the branch we need)
    rm -rf "${tmp_clone}"
    log_info "push_lessons: cloning repo for lessons push..."
    if ! git clone --depth 1 --branch "${branch}" "${auth_url}" "${tmp_clone}" 2>/dev/null; then
        log_warn "push_lessons: git clone failed"
        return 0
    fi

    # Copy lessons into the clone
    mkdir -p "${tmp_clone}/auto_quant/lessons"
    cp -f "${lessons_dir}"/*.jsonl "${tmp_clone}/auto_quant/lessons/" 2>/dev/null || true

    # Commit and push
    cd "${tmp_clone}"
    git config user.name "${GIT_USER_NAME:-auto-pipeline}"
    git config user.email "${GIT_USER_EMAIL:-auto@pipeline.local}"
    git add --force auto_quant/lessons/ 2>/dev/null || true

    if ! git diff --cached --quiet auto_quant/lessons/ 2>/dev/null; then
        git commit -m "lessons: update from ${MODEL_ID:-unknown} ${SCHEME:-} ${METHOD:-}" || true
        git push origin "${branch}" 2>/dev/null && log_ok "push_lessons: pushed successfully" || log_warn "push_lessons: git push failed"
    else
        log_info "push_lessons: no changes to push"
    fi

    cd - > /dev/null
    rm -rf "${tmp_clone}"
}
