#!/bin/bash
# test_lesson_system.sh — Unit tests for the lesson load/save/agent-fix flow
#
# Tests:
#   1. load_all_lessons correctly reads and deduplicates lessons
#   2. save_lesson correctly writes JSONL entry with proper fields
#   3. agent_fix_loop triggers lesson load on phase failure
#   4. agent_fix_loop saves lesson on successful fix
#   5. agent_fix_loop saves lesson on drift detection
#   6. End-to-end: fail → load lessons → fix → save new lesson
#
# Usage: bash tests/test_lesson_system.sh

set -euo pipefail

# ═══ Test framework ═══
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
FAIL_MESSAGES=()

pass() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_RUN=$((TESTS_RUN + 1))
    echo "  ✅ PASS: $1"
}

fail() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TESTS_RUN=$((TESTS_RUN + 1))
    FAIL_MESSAGES+=("$1: $2")
    echo "  ❌ FAIL: $1 — $2"
}

assert_eq() {
    local expected="$1" actual="$2" msg="$3"
    if [ "${expected}" = "${actual}" ]; then
        pass "${msg}"
    else
        fail "${msg}" "expected='${expected}', actual='${actual}'"
    fi
}

assert_contains() {
    local haystack="$1" needle="$2" msg="$3"
    if echo "${haystack}" | grep -qF "${needle}"; then
        pass "${msg}"
    else
        fail "${msg}" "output does not contain '${needle}'"
    fi
}

assert_file_exists() {
    local path="$1" msg="$2"
    if [ -f "${path}" ]; then
        pass "${msg}"
    else
        fail "${msg}" "file not found: ${path}"
    fi
}

assert_file_line_count() {
    local path="$1" expected="$2" msg="$3"
    local actual
    actual=$(wc -l < "${path}" 2>/dev/null || echo 0)
    if [ "${actual}" -eq "${expected}" ]; then
        pass "${msg}"
    else
        fail "${msg}" "expected ${expected} lines, got ${actual}"
    fi
}

# ═══ Setup test environment ═══
TEST_DIR=$(mktemp -d)
trap "rm -rf ${TEST_DIR}" EXIT

export RUN_OUTPUT_DIR="${TEST_DIR}/output"
export LESSONS_DIR="${TEST_DIR}/lessons"
export LB_EVAL_REPO_DIR=""  # disable git push in tests
export MAX_FIX_ATTEMPTS=3
export MODEL_ID="test-model/Qwen3-0.6B"
export SCHEME="W4A16"
export METHOD="RTN"

mkdir -p "${RUN_OUTPUT_DIR}/logs" "${LESSONS_DIR}"

# Stub log functions (agent_fix_loop.sh expects these)
log_step() { echo "[STEP] $*"; }
log_ok() { echo "[OK] $*"; }
log_warn() { echo "[WARN] $*"; }
log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*"; }
export -f log_step log_ok log_warn log_info log_error

# Source the library under test
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/phases/agent_fix_loop.sh"

echo "═══════════════════════════════════════════════"
echo " Lesson System Unit Tests"
echo "═══════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 1: save_lesson creates valid JSONL entry
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 1: save_lesson creates valid JSONL entry"

save_lesson "quantize" "RuntimeError: CUDA out of memory" "fixed" "Reduced batch size from 16 to 8"

assert_file_exists "${LESSONS_DIR}/quantize.jsonl" "Lesson file created"

# Validate JSON structure
ENTRY=$(tail -1 "${LESSONS_DIR}/quantize.jsonl")
PHASE=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['phase'])")
STATUS=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'])")
SOLUTION=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['solution'])")
ERROR_SIG=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['error_signature'])")
MODEL=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['model'])")

assert_eq "quantize" "${PHASE}" "phase field correct"
assert_eq "fixed" "${STATUS}" "status field correct"
assert_eq "Reduced batch size from 16 to 8" "${SOLUTION}" "solution field correct"
assert_contains "${ERROR_SIG}" "CUDA out of memory" "error_signature extracted"
assert_eq "test-model/Qwen3-0.6B" "${MODEL}" "model field correct"

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 2: save_lesson handles multiline error context
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 2: save_lesson handles multiline error context"

MULTI_ERROR="Traceback (most recent call last):
  File \"quantize.py\", line 45, in <module>
    autoround.quantize()
  File \"/root/.venv/lib/python3.12/site-packages/auto_round/autoround.py\", line 312
RuntimeError: CUDA error: device-side assert triggered"

save_lesson "quantize" "${MULTI_ERROR}" "still_failing" "Attempt 1 did not resolve"

LINE_COUNT=$(wc -l < "${LESSONS_DIR}/quantize.jsonl")
assert_eq "2" "${LINE_COUNT}" "Two entries now in quantize.jsonl"

ENTRY2=$(tail -1 "${LESSONS_DIR}/quantize.jsonl")
SIG2=$(echo "${ENTRY2}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['error_signature'])")
assert_contains "${SIG2}" "Traceback" "Signature from multiline error"

# Verify traceback stored
TB=$(echo "${ENTRY2}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['error_traceback'])")
assert_contains "${TB}" "device-side assert triggered" "Full traceback preserved"

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 3: load_all_lessons reads and formats lessons
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 3: load_all_lessons reads and formats lessons"

# Add a lesson to another phase file
save_lesson "evaluate" "AssertionError: acc=0 for task hellaswag" "fixed" "Re-ran with correct num_fewshot"

OUTPUT=$(load_all_lessons)
assert_contains "${OUTPUT}" "CUDA out of memory" "Lesson 1 from quantize.jsonl loaded"
assert_contains "${OUTPUT}" "acc=0" "Lesson from evaluate.jsonl loaded"
assert_contains "${OUTPUT}" "phase=quantize" "Phase tag shown"
assert_contains "${OUTPUT}" "phase=evaluate" "Phase tag shown for evaluate"

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 4: load_all_lessons deduplicates by error_signature
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 4: load_all_lessons deduplicates by error_signature"

# Save same error again (should be deduplicated)
save_lesson "quantize" "RuntimeError: CUDA out of memory" "fixed" "Different solution"

OUTPUT=$(load_all_lessons)
CUDA_COUNT=$(echo "${OUTPUT}" | grep -c "CUDA out of memory" || true)
assert_eq "1" "${CUDA_COUNT}" "Duplicate error_signature deduplicated"

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 5: load_all_lessons caps at 10 entries
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 5: load_all_lessons caps at 10 entries"

# Create 15 unique lessons
for i in $(seq 1 15); do
    save_lesson "stress" "UniqueError${i}: something broke #${i}" "fixed" "Solution ${i}"
done

OUTPUT=$(load_all_lessons)
LESSON_COUNT=$(echo "${OUTPUT}" | grep -c "^Lesson " || true)
if [ "${LESSON_COUNT}" -le 10 ]; then
    pass "Capped at ≤10 lessons (got ${LESSON_COUNT})"
else
    fail "Cap at 10" "Got ${LESSON_COUNT} lessons"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 6: agent_fix_loop triggers lessons on failure & saves on fix
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 6: agent_fix_loop E2E — fail, load lessons, fix, save lesson"

# Clean lessons for this test
rm -rf "${LESSONS_DIR}"/*
mkdir -p "${LESSONS_DIR}"

# Seed one lesson
save_lesson "quantize" "ImportError: cannot import auto_round" "fixed" "pip install auto-round"

# Create a phase script that fails once then succeeds
PHASE_SCRIPT="${TEST_DIR}/fake_quantize.sh"
COUNTER_FILE="${TEST_DIR}/run_counter"
echo "0" > "${COUNTER_FILE}"

cat > "${PHASE_SCRIPT}" <<'SCRIPT'
#!/bin/bash
COUNT=$(cat "${COUNTER_FILE}")
COUNT=$((COUNT + 1))
echo "${COUNT}" > "${COUNTER_FILE}"

if [ "${COUNT}" -le 1 ]; then
    echo "Starting quantization..."
    echo "Loading model..."
    echo "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    exit 1
fi

echo "Quantization completed successfully"
exit 0
SCRIPT
chmod +x "${PHASE_SCRIPT}"
export COUNTER_FILE

# Mock openclaw (agent stub that does nothing but "fix")
MOCK_OPENCLAW="${TEST_DIR}/openclaw"
cat > "${MOCK_OPENCLAW}" <<'MOCK'
#!/bin/bash
echo "[AGENT] Analyzing error..."
echo "[AGENT] Fix applied: reduced memory usage"
MOCK
chmod +x "${MOCK_OPENCLAW}"
export PATH="${TEST_DIR}:${PATH}"

# Run agent_fix_loop
FIX_OUTPUT=$(agent_fix_loop "quantize" "${PHASE_SCRIPT}" 2>&1)

# Verify:
# a) Loop detected failure and entered fix mode
assert_contains "${FIX_OUTPUT}" "failed" "Detected initial failure"
# b) Lessons were loaded
assert_contains "${FIX_OUTPUT}" "Loaded lessons" "Lessons loaded for agent"
# c) Fix succeeded
assert_contains "${FIX_OUTPUT}" "fixed on attempt 1" "Fix succeeded on attempt 1"
# d) A new lesson was saved
FINAL_LESSONS=$(cat "${LESSONS_DIR}/quantize.jsonl")
assert_contains "${FINAL_LESSONS}" "CUDA out of memory" "New lesson saved with error signature"
assert_contains "${FINAL_LESSONS}" "Fix applied" "Fix solution extracted from agent log"

# e) Prompt file was created with lessons injected
PROMPT_FILE="${RUN_OUTPUT_DIR}/logs/agent_fixes/quantize/prompt_1.txt"
assert_file_exists "${PROMPT_FILE}" "Agent prompt saved"
PROMPT_CONTENT=$(cat "${PROMPT_FILE}")
assert_contains "${PROMPT_CONTENT}" "Historical Lessons" "Prompt contains lessons section"
assert_contains "${PROMPT_CONTENT}" "cannot import auto_round" "Seed lesson injected into prompt"

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 7: agent_fix_loop detects drift and saves drift lesson
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 7: agent_fix_loop detects drift (same error repeating)"

rm -rf "${LESSONS_DIR}"/*
mkdir -p "${LESSONS_DIR}"

# Phase script that always fails with same error
ALWAYS_FAIL="${TEST_DIR}/always_fail.sh"
cat > "${ALWAYS_FAIL}" <<'SCRIPT'
#!/bin/bash
echo "Loading model..."
echo "Error: cannot allocate tensor of size 4GB"
exit 1
SCRIPT
chmod +x "${ALWAYS_FAIL}"

# Run — should hit drift detection
DRIFT_OUTPUT=$(agent_fix_loop "quantize" "${ALWAYS_FAIL}" 2>&1 || true)

assert_contains "${DRIFT_OUTPUT}" "Drift detected" "Drift detection triggered"

# Verify drift lesson saved
if [ -f "${LESSONS_DIR}/quantize.jsonl" ]; then
    DRIFT_LESSON=$(cat "${LESSONS_DIR}/quantize.jsonl")
    assert_contains "${DRIFT_LESSON}" "drift" "Drift lesson saved with status=drift"
else
    fail "Drift lesson file" "quantize.jsonl not created"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 8: build_fix_prompt includes lessons when available
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 8: build_fix_prompt formats correctly"

rm -rf "${LESSONS_DIR}"/*
mkdir -p "${LESSONS_DIR}"
save_lesson "evaluate" "ValueError: invalid num_fewshot" "fixed" "Set num_fewshot=5"

LESSONS_TEXT=$(load_all_lessons)
PROMPT=$(build_fix_prompt "evaluate" "ValueError: task failed" "${LESSONS_TEXT}")

assert_contains "${PROMPT}" 'fixing a failed "evaluate" phase' "Prompt mentions phase"
assert_contains "${PROMPT}" "ValueError: task failed" "Prompt contains error"
assert_contains "${PROMPT}" "Historical Lessons" "Prompt has lessons section"
assert_contains "${PROMPT}" "invalid num_fewshot" "Lesson content in prompt"

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 9: build_fix_prompt handles no lessons gracefully
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 9: build_fix_prompt with no lessons"

PROMPT_EMPTY=$(build_fix_prompt "setup_env" "pip install failed" "")
assert_contains "${PROMPT_EMPTY}" "No lessons available yet" "Shows no-lessons message"

echo ""

# ═══════════════════════════════════════════════════════════════════
# TEST 10: save_lesson preserves keywords and metadata
# ═══════════════════════════════════════════════════════════════════
echo "▶ Test 10: save_lesson metadata fields"

rm -rf "${LESSONS_DIR}"/*
mkdir -p "${LESSONS_DIR}"

export SCHEME="NVFP4"
export METHOD="GPTQ"
save_lesson "quantize" "AutoRoundError: unsupported scheme NVFP4_WRONG" "fixed" "Fixed scheme name"

ENTRY=$(tail -1 "${LESSONS_DIR}/quantize.jsonl")
SCHEME_VAL=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['scheme'])")
METHOD_VAL=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['method'])")
KEYWORDS=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join(d['error_keywords']))")
SOURCE=$(echo "${ENTRY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['source_tasks'][0])")

assert_eq "NVFP4" "${SCHEME_VAL}" "scheme metadata preserved"
assert_eq "GPTQ" "${METHOD_VAL}" "method metadata preserved"
assert_contains "${KEYWORDS}" "autorounderror" "Keywords extracted from signature"
assert_contains "${SOURCE}" "NVFP4_GPTQ" "source_tasks includes scheme+method"

echo ""

# ═══ Summary ═══
echo "═══════════════════════════════════════════════"
echo " Results: ${TESTS_PASSED}/${TESTS_RUN} passed, ${TESTS_FAILED} failed"
echo "═══════════════════════════════════════════════"

if [ ${TESTS_FAILED} -gt 0 ]; then
    echo ""
    echo "Failures:"
    for msg in "${FAIL_MESSAGES[@]}"; do
        echo "  • ${msg}"
    done
    exit 1
fi

echo ""
echo "All tests passed! ✅"
exit 0
