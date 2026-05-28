#!/usr/bin/env bash
# auto_v3.sh — Phases-based quantization pipeline (v3)
#
# Architecture:
#   Phase 1: setup_env.sh     (deterministic environment install)
#   Phase 2: quantize.py      (deterministic quantization with recipes)
#   Phase 3: evaluate.sh      (deterministic evaluation, hf/vllm backend)
#   Phase 4: upload           (reuse existing upload_model_hf.py + upload_results_github.py)
#
#   On failure: agent_fix_loop attempts repair via OpenClaw agent
#
# Usage:
#   bash auto_v3.sh <task_json_file> [options]
#
# Options:
#   --skip-upload      Skip all uploads
#   --skip-agent       Skip agent fix loop (fail immediately on error)
#   --dry-run          Print resolved configuration and exit
#   -h, --help         Show this help

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASES_DIR="${SCRIPT_DIR}/phases"

# ═══ Global log capture ═══
# Capture entire pipeline stdout+stderr to auto.log for full traceability
_AUTO_LOG="${SCRIPT_DIR}/output/.auto_v3_$$.log"
mkdir -p "$(dirname "${_AUTO_LOG}")"
exec > >(tee -a "${_AUTO_LOG}") 2>&1

# ═══ Colors ═══
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
fi

log_info()  { echo -e "${CYAN}[auto_v3]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[auto_v3]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[auto_v3]${NC} $*"; }
log_error() { echo -e "${RED}[auto_v3]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${CYAN}═══════ $* ═══════${NC}\n"; }

# ═══ Load config ═══
if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
    source "${SCRIPT_DIR}/config.env"
fi

# ═══ Source agent fix loop library ═══
source "${PHASES_DIR}/agent_fix_loop.sh"

# ═══ Parse arguments ═══
TASK_JSON=""
SKIP_UPLOAD=false
SKIP_AGENT=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-upload)  SKIP_UPLOAD=true; shift ;;
        --skip-agent)   SKIP_AGENT=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash auto_v3.sh <task_json_file> [--skip-upload] [--skip-agent] [--dry-run]"
            exit 0 ;;
        *)
            if [[ -z "$TASK_JSON" ]]; then
                TASK_JSON="$1"
            fi
            shift ;;
    esac
done

if [[ -z "$TASK_JSON" ]]; then
    log_error "No task JSON file specified"
    echo "Usage: bash auto_v3.sh <task_json_file>"
    exit 1
fi

# Resolve JSON path
if [[ ! -f "$TASK_JSON" ]] && [[ -f "${SCRIPT_DIR}/${TASK_JSON}" ]]; then
    TASK_JSON="${SCRIPT_DIR}/${TASK_JSON}"
fi
if [[ ! -f "$TASK_JSON" ]]; then
    log_error "Task JSON not found: $TASK_JSON"
    exit 1
fi

# ═══ Parse task JSON ═══
eval "$(python3 - "$TASK_JSON" <<'PYEOF'
import json
import sys

with open(sys.argv[1]) as f:
    task = json.load(f)

# Extract fields with defaults
model = task.get("model", "")
scheme = task.get("scheme", task.get("quant_type", "W4A16"))
method = task.get("method", "RTN")
export_format = task.get("export_format", "auto_round")
auto_round_ref = task.get("auto_round_ref", "latest")
transformers_ref = task.get("transformers_ref", "auto")
request_filename = task.get("request_filename", "")

# Normalize method from iters
iters = task.get("iters", None)
if iters is not None:
    method = "RTN" if int(iters) == 0 else "TUNING"

print(f'MODEL_ID="{model}"')
print(f'SCHEME="{scheme}"')
print(f'METHOD="{method}"')
print(f'EXPORT_FORMAT="{export_format}"')
print(f'AUTO_ROUND_REF="{auto_round_ref}"')
print(f'TRANSFORMERS_REF="{transformers_ref}"')
print(f'REQUEST_FILENAME="{request_filename}"')
PYEOF
)"

# ═══ Derive variables ═══
case "${EXPORT_FORMAT}" in
    auto_round)      EVAL_BACKEND="hf" ;;
    llm_compressor)  EVAL_BACKEND="vllm" ;;
    *)               EVAL_BACKEND="hf" ;;
esac

case "${METHOD}" in
    RTN)    ITERS=0;   METHOD_SUFFIX="RTN" ;;
    TUNING) ITERS=200; METHOD_SUFFIX="Tuning" ;;
    *)      ITERS=0;   METHOD_SUFFIX="${METHOD}" ;;
esac

# Use config.env defaults where task JSON didn't override
DEVICE="${DEVICE:-cuda}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
EVAL_TASKS="${EVAL_TASKS:-piqa,mmlu,hellaswag}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_GPUS="${NUM_GPUS:-1}"

# Output directories
MODEL_SHORT="${MODEL_ID#*/}"
HF_REPO_NAME="${MODEL_SHORT}-AutoRound-${SCHEME}-${METHOD_SUFFIX}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
RUNTIME_OUTPUT_BASE_DIR="${RUNTIME_OUTPUT_BASE_DIR:-${OUTPUT_DIR}/runs}"
RUN_OUTPUT_DIR="${RUNTIME_OUTPUT_BASE_DIR}/${HF_REPO_NAME}"
QUANTIZED_MODEL_DIR="${RUN_OUTPUT_DIR}/quantized_model"
EVAL_OUTPUT_DIR="${RUN_OUTPUT_DIR}/lm_eval_results"
LOG_DIR="${RUN_OUTPUT_DIR}/logs"

# lb_eval repo
LB_EVAL_REPO_DIR="${GIT_RESULTS_REPO_DIR:-${SCRIPT_DIR}/lb_eval}"
LESSONS_DIR="${LB_EVAL_REPO_DIR}/lessons"
GIT_BRANCH="${GIT_BRANCH:-main}"

# Export for child scripts
export MODEL_ID SCHEME METHOD ITERS EXPORT_FORMAT EVAL_BACKEND
export AUTO_ROUND_REF TRANSFORMERS_REF
export DEVICE DEVICE_INDEX EVAL_TASKS EVAL_BATCH_SIZE NUM_GPUS
export RUN_OUTPUT_DIR QUANTIZED_MODEL_DIR EVAL_OUTPUT_DIR
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export LB_EVAL_REPO_DIR LESSONS_DIR GIT_BRANCH
export REQUEST_FILENAME

mkdir -p "${RUN_OUTPUT_DIR}" "${LOG_DIR}" "${LESSONS_DIR}"

# Relocate global auto.log into the proper log directory
if [[ -f "${_AUTO_LOG}" ]]; then
    mv "${_AUTO_LOG}" "${LOG_DIR}/auto.log" 2>/dev/null || true
    _AUTO_LOG="${LOG_DIR}/auto.log"
    exec > >(tee -a "${_AUTO_LOG}") 2>&1
fi

# ═══ Dry run ═══
if [[ "$DRY_RUN" == "true" ]]; then
    log_step "DRY RUN — Resolved Configuration"
    echo "  MODEL_ID:         ${MODEL_ID}"
    echo "  SCHEME:           ${SCHEME}"
    echo "  METHOD:           ${METHOD} (iters=${ITERS})"
    echo "  EXPORT_FORMAT:    ${EXPORT_FORMAT}"
    echo "  EVAL_BACKEND:     ${EVAL_BACKEND}"
    echo "  AUTO_ROUND_REF:   ${AUTO_ROUND_REF}"
    echo "  TRANSFORMERS_REF: ${TRANSFORMERS_REF}"
    echo "  RUN_OUTPUT_DIR:   ${RUN_OUTPUT_DIR}"
    echo "  QUANTIZED_MODEL:  ${QUANTIZED_MODEL_DIR}"
    echo "  EVAL_OUTPUT:      ${EVAL_OUTPUT_DIR}"
    echo "  LESSONS_DIR:      ${LESSONS_DIR}"
    echo "  SKIP_UPLOAD:      ${SKIP_UPLOAD}"
    echo "  SKIP_AGENT:       ${SKIP_AGENT}"
    exit 0
fi

# ═══ Pull latest lessons ═══
if [[ -d "${LB_EVAL_REPO_DIR}/.git" ]]; then
    cd "${LB_EVAL_REPO_DIR}"
    git pull --rebase 2>/dev/null || log_warn "git pull failed (non-fatal)"
    cd - > /dev/null
fi

# ═══ Copy task JSON for reference ═══
cp "${TASK_JSON}" "${RUN_OUTPUT_DIR}/request.json" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════
log_step "Pipeline: ${MODEL_ID} | ${SCHEME}/${METHOD}/${EXPORT_FORMAT}"
PIPELINE_START=$(date +%s)
FAILED_STEPS=()

# --- Phase 1: Environment Setup ---
if [[ "$SKIP_AGENT" == "true" ]]; then
    bash "${PHASES_DIR}/setup_env.sh" 2>&1 | tee "${LOG_DIR}/setup_env.log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "setup_env failed (no agent retry)"
        FAILED_STEPS+=("setup_env")
    fi
else
    agent_fix_loop "setup_env" "${PHASES_DIR}/setup_env.sh" || {
        FAILED_STEPS+=("setup_env")
        log_error "setup_env failed after all fix attempts"
    }
fi

# --- Phase 2: Quantization ---
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    if [[ "$SKIP_AGENT" == "true" ]]; then
        bash "${PHASES_DIR}/quantize_wrapper.sh" 2>&1 | tee "${LOG_DIR}/quantize.log"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            FAILED_STEPS+=("quantize")
        fi
    else
        agent_fix_loop "quantize" "${PHASES_DIR}/quantize_wrapper.sh" || {
            FAILED_STEPS+=("quantize")
        }
    fi
fi

# --- Phase 3: Evaluation ---
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    if [[ "$SKIP_AGENT" == "true" ]]; then
        bash "${PHASES_DIR}/evaluate.sh" "${QUANTIZED_MODEL_DIR}" 2>&1 | tee "${LOG_DIR}/evaluate.log"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            FAILED_STEPS+=("evaluate")
        fi
    else
        agent_fix_loop "evaluate" "${PHASES_DIR}/evaluate.sh" "${QUANTIZED_MODEL_DIR}" || {
            FAILED_STEPS+=("evaluate")
        }
    fi
fi

# ═══ Determine pipeline status ═══
PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
    PIPELINE_STATUS="Finished"
    log_ok "Pipeline completed successfully in ${PIPELINE_DURATION}s"
else
    PIPELINE_STATUS="Failed"
    log_error "Pipeline failed at: ${FAILED_STEPS[*]} (${PIPELINE_DURATION}s)"
fi

# ═══ Collect OpenClaw session logs ═══
# Copy .jsonl session files from the openclaw sessions directory into RUN_OUTPUT_DIR,
# then format them to human-readable .md (matching old pipeline behavior)
OPENCLAW_SESSIONS_DIR="${OPENCLAW_SESSIONS_DIR:-/root/.openclaw/agents/main/sessions}"
if [[ -d "${OPENCLAW_SESSIONS_DIR}" ]]; then
    _session_count=0
    for _jsonl in "${OPENCLAW_SESSIONS_DIR}"/*.jsonl; do
        [[ -f "$_jsonl" ]] || continue
        # Only copy sessions created during this pipeline run (mtime > PIPELINE_START)
        if [[ $(stat -c %Y "$_jsonl" 2>/dev/null || echo 0) -ge ${PIPELINE_START} ]]; then
            cp "$_jsonl" "${RUN_OUTPUT_DIR}/" 2>/dev/null && ((_session_count++)) || true
        fi
    done
    if [[ $_session_count -gt 0 ]]; then
        log_info "Collected ${_session_count} openclaw session(s)"
        # Format sessions to Markdown for human readability
        FORMATTER="${SCRIPT_DIR}/format_sessions.py"
        if [[ -f "${FORMATTER}" ]]; then
            python3 "${FORMATTER}" "${RUN_OUTPUT_DIR}"/session_*.jsonl 2>/dev/null || true
        fi
    fi
fi

# ═══ Generate Report (before upload so it gets included) ═══
log_info "Generating run report..."
python3 "${PHASES_DIR}/generate_report.py" "${RUN_OUTPUT_DIR}" || log_warn "Report generation failed (non-fatal)"

# ═══ Phase 4: Upload ═══
if [[ "$SKIP_UPLOAD" != "true" ]]; then
    log_step "Upload Results"

    # 4a: Upload quantized model to HF Hub
    if [[ -d "${QUANTIZED_MODEL_DIR}" ]] && [[ "$PIPELINE_STATUS" == "Finished" ]]; then
        log_info "Uploading quantized model to HuggingFace Hub..."
        python3 "${SCRIPT_DIR}/upload_model_hf.py" \
            "${QUANTIZED_MODEL_DIR}" \
            "${HF_REPO_NAME}" \
            --tokens "${HF_TOKENS:-}" \
            --orgs "${HF_UPLOAD_ORGS:-}" \
            --account-ids "${HF_ACCOUNT_IDS:-}" \
            --summary-json "${RUN_OUTPUT_DIR}/quant_summary.json" \
            --accuracy-json "${RUN_OUTPUT_DIR}/accuracy.json" \
            --usage-file "${HF_USAGE_FILE:-}" \
            --capacity-gb "${HF_ACCOUNT_CAPACITY_GB:-1000}" \
            --shared-ledger-enabled "${HF_SHARED_LEDGER_ENABLED:-false}" \
            --shared-ledger-repo "${HF_SHARED_LEDGER_REPO:-}" \
            --shared-ledger-token "${HF_SHARED_LEDGER_TOKEN:-}" \
            --shared-ledger-branch "${HF_SHARED_LEDGER_BRANCH:-main}" \
            2>&1 | tee "${LOG_DIR}/upload_hf.log" || log_warn "HF upload failed"
    fi

    # 4b: Upload results to lb_eval GitHub
    log_info "Uploading results to lb_eval GitHub..."
    python3 "${SCRIPT_DIR}/upload_results_github.py" \
        "${RUN_OUTPUT_DIR}" \
        "${MODEL_ID}" \
        --scheme "${SCHEME}" \
        --model-output-dir "${QUANTIZED_MODEL_DIR}" \
        --repo-dir "${LB_EVAL_REPO_DIR}" \
        --git-repo "${GIT_REPO:-}" \
        --git-token "${GIT_TOKEN:-}" \
        --request-filename "${REQUEST_FILENAME:-}" \
        --git-user-name "${GIT_USER_NAME:-auto-pipeline}" \
        --git-user-email "${GIT_USER_EMAIL:-auto@pipeline.local}" \
        2>&1 | tee "${LOG_DIR}/upload_github.log" || log_warn "GitHub upload failed"
fi

# ═══ Push lessons ═══
push_lessons_to_git

# ═══ Summary ═══
log_step "Pipeline Summary"
echo "  Model:    ${MODEL_ID}"
echo "  Scheme:   ${SCHEME} / ${METHOD}"
echo "  Status:   ${PIPELINE_STATUS}"
echo "  Duration: ${PIPELINE_DURATION}s"
echo "  Output:   ${RUN_OUTPUT_DIR}"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "  Failed:   ${FAILED_STEPS[*]}"
fi
if [[ -f "${RUN_OUTPUT_DIR}/run_report.md" ]]; then
    echo "  Report:   ${RUN_OUTPUT_DIR}/run_report.md"
fi

exit $([[ "$PIPELINE_STATUS" == "Finished" ]] && echo 0 || echo 1)
