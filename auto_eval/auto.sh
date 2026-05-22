#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    BOLD=''
    NC=''
fi

log_info() { echo -e "${CYAN}[auto.sh]${NC} $*"; }
log_ok() { echo -e "${GREEN}[auto.sh]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[auto.sh]${NC} $*"; }
log_error() { echo -e "${RED}[auto.sh]${NC} $*"; }
log_step() { echo -e "\n${BOLD}${CYAN}========== $* ==========${NC}\n"; }

usage() {
    cat <<'EOF'
Usage:
  bash auto.sh <task_json_file> [options]

Options:
  --skip-upload   Skip all uploads
  --skip-hf       Skip HuggingFace model upload
  --skip-github   Skip GitHub results upload
  --dry-run       Print resolved configuration and exit
  -h, --help      Show this help
EOF
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_error "Required command not found: $cmd"
        exit 1
    fi
}

resolve_first_existing_dir() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_first_existing_file() {
    local candidate
    for candidate in "$@"; do
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_skill_path() {
    local skill_name="$1"
    resolve_first_existing_file \
        "${OPENCLAW_WORKSPACE_DIR}/skills/${skill_name}/SKILL.md" \
        "/root/leaderboard_Agent/tasks/lb_eval/openclaw_config/workspace/skills/${skill_name}/SKILL.md" \
        "/root/leaderboard_Agent/auto_quant/openclaw_home/workspace/skills/${skill_name}/SKILL.md" \
        "/root/leaderboard_Agent/auto_eval/openclaw_home/workspace/skills/${skill_name}/SKILL.md" \
        "/root/.openclaw/workspace/skills/${skill_name}/SKILL.md"
}

resolve_json_file() {
    local input_path="$1"
    if [[ -f "$input_path" ]]; then
        printf '%s\n' "$input_path"
        return 0
    fi
    if [[ -f "$SCRIPT_DIR/$input_path" ]]; then
        printf '%s\n' "$SCRIPT_DIR/$input_path"
        return 0
    fi
    return 1
}

json_status() {
    local path="$1"
    local default_value="${2:-missing}"
    if [[ ! -f "$path" ]]; then
        printf '%s\n' "$default_value"
        return 0
    fi
    python3 - "$path" "$default_value" <<'PY'
import json
import sys

path = sys.argv[1]
default_value = sys.argv[2]

try:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    print(data.get("status", default_value))
except Exception:
    print(default_value)
PY
}

ensure_failure_summary() {
    local target_path="$1"
    local mode="$2"
    local failure_stage="$3"
    local current_status="missing"

    if [[ -f "$target_path" ]]; then
        current_status="$(json_status "$target_path")"
    fi
    if [[ "$current_status" == "success" || "$current_status" == "failed" ]]; then
        return 0
    fi

    run_step \
        "Write fallback $(basename "$target_path")" \
        python3 - "$target_path" "$mode" "$failure_stage" "$MODEL_ID" "$SCHEME" "$METHOD" "$EXPORT_FORMAT" "$DEVICE" "$DEVICE_INDEX" "$NUM_GPUS" "$EVAL_NUM_GPUS" "$RUN_OUTPUT_DIR" "$QUANTIZED_MODEL_DIR" "$LM_EVAL_OUTPUT_DIR" "${FAILED_STEPS[@]}" <<'PY'
import json
import sys
from pathlib import Path

target_path = Path(sys.argv[1])
mode = sys.argv[2]
failure_stage = sys.argv[3]
model_id = sys.argv[4]
scheme = sys.argv[5]
method = sys.argv[6]
export_format = sys.argv[7]
device = sys.argv[8]
device_index = sys.argv[9]
quant_num_gpus = sys.argv[10]
eval_num_gpus = sys.argv[11]
run_output_dir = sys.argv[12]
quantized_model_dir = sys.argv[13]
lm_eval_output_dir = sys.argv[14]
fallback_errors = [err for err in sys.argv[15:] if err]

existing = {}
if target_path.exists():
    try:
        with target_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing = loaded
    except Exception:
        existing = {}

existing_status = existing.get("status")
if existing_status in {"success", "failed"}:
    raise SystemExit(0)

existing_errors = existing.get("errors")
if not isinstance(existing_errors, list):
    existing_errors = []

errors = []
for value in [*existing_errors, *fallback_errors]:
    if value and value not in errors:
        errors.append(value)
if not errors:
    errors = [f"{failure_stage} failed before final summary was written"]

payload = dict(existing)
payload["status"] = "failed"
payload["failure_stage"] = failure_stage
payload["errors"] = errors
payload.setdefault("duration_seconds", None)

if mode == "quant_summary":
    payload.setdefault("model_id", model_id)
    payload.setdefault("scheme", scheme)
    payload.setdefault("method", method)
    payload.setdefault("export_format", export_format)
    payload.setdefault("device", device)
    payload["quant_num_gpus"] = quant_num_gpus
    payload["num_gpus"] = quant_num_gpus
    payload.setdefault("output_dir", run_output_dir)
    payload.setdefault("runtime_output_dir", run_output_dir)
    payload.setdefault("quantized_model_dir", quantized_model_dir)
    payload.setdefault("original_size_mb", None)
    payload.setdefault("quantized_size_mb", None)
    payload.setdefault("compression_ratio", None)
    payload.setdefault("solutions", [])
    payload.setdefault("output_files", [])
elif mode == "accuracy":
    payload.setdefault("model_id", model_id)
    payload.setdefault("model_path", quantized_model_dir)
    payload.setdefault("scheme", scheme)
    payload.setdefault("device", f"{device}:{device_index}")
    payload["eval_num_gpus"] = eval_num_gpus
    payload["num_gpus"] = eval_num_gpus
    payload.setdefault("tasks", {})
    payload.setdefault("eval_framework", None)
    payload.setdefault("lm_eval_output_dir", lm_eval_output_dir)
else:
    raise SystemExit(f"unsupported mode: {mode}")

target_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = target_path.with_name(f"{target_path.name}.tmp")
with tmp_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
tmp_path.replace(target_path)
PY
}

run_step() {
    local title="$1"
    shift

    log_step "$title"

    local rendered_command
    printf -v rendered_command '%q ' "$@"
    if [[ ${#rendered_command} -gt 800 ]]; then
        rendered_command="${rendered_command:0:800} ... [truncated]"
    fi
    log_info "Command: ${rendered_command% }"

    "$@"
    local status=$?
    LAST_EXIT_CODE=$status

    if [[ $status -eq 0 ]]; then
        log_ok "$title succeeded"
    else
        log_warn "$title failed with exit code $status"
        FAILED_STEPS+=("$title (exit=$status)")
    fi

    return 0
}

copy_if_exists() {
    local source_path="$1"
    local target_path="$2"
    local label="$3"

    if [[ -f "$source_path" ]]; then
        run_step "$label" cp "$source_path" "$target_path"
    else
        log_warn "$label skipped: file not found: $source_path"
    fi
}

record_failure() {
    local message="$1"
    log_warn "$message"
    FAILED_STEPS+=("$message")
    LAST_EXIT_CODE=1
}

show_json_if_exists() {
    local title="$1"
    local path="$2"

    if [[ -f "$path" ]]; then
        log_step "$title"
        cat "$path"
    else
        log_warn "$title skipped: file not found: $path"
    fi
}

start_session_monitor() {
    local session_path="$1"
    local label="$2"
    local pid_var="${3:-}"

    if [[ -z "${SESSION_MONITOR:-}" || ! -f "${SESSION_MONITOR:-}" ]]; then
        log_warn "Session monitor unavailable, skipping live session stream for ${label}"
        return 1
    fi

    python3 -u "$SESSION_MONITOR" "$session_path" --label "$label" &
    local monitor_pid="$!"
    if [[ -n "$pid_var" ]]; then
        printf -v "$pid_var" '%s' "$monitor_pid"
    else
        printf '%s\n' "$monitor_pid"
    fi
}

stop_session_monitor() {
    local monitor_pid="${1:-}"
    if [[ -z "$monitor_pid" ]]; then
        return 0
    fi

    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" 2>/dev/null || true
}

start_log_tail() {
    local file_path="$1"
    local label="$2"
    local pid_var="${3:-}"

    bash -lc '
        file_path="$1"
        label="$2"
        while [[ ! -f "$file_path" ]]; do
            sleep 1
        done
        printf "\n========== %s ==========\n\n" "$label"
        exec tail -n +1 -F -- "$file_path"
    ' _ "$file_path" "$label" &

    local tail_pid="$!"
    if [[ -n "$pid_var" ]]; then
        printf -v "$pid_var" '%s' "$tail_pid"
    else
        printf '%s\n' "$tail_pid"
    fi
}

start_artifact_watch() {
    local file_path="$1"
    local title="$2"
    local pid_var="${3:-}"
    local max_lines="${4:-400}"

    bash -lc '
        file_path="$1"
        title="$2"
        max_lines="$3"
        while [[ ! -f "$file_path" ]]; do
            sleep 1
        done
        printf "\n========== %s ==========\n\n" "$title"
        total_lines=$(wc -l < "$file_path" 2>/dev/null || printf "0")
        if [[ "$total_lines" =~ ^[0-9]+$ ]] && (( total_lines > max_lines )); then
            sed -n "1,${max_lines}p" "$file_path"
            printf "[auto.sh] %s truncated: showing first %s/%s lines\n" "$title" "$max_lines" "$total_lines"
        else
            cat "$file_path"
        fi
    ' _ "$file_path" "$title" "$max_lines" &

    local watch_pid="$!"
    if [[ -n "$pid_var" ]]; then
        printf -v "$pid_var" '%s' "$watch_pid"
    else
        printf '%s\n' "$watch_pid"
    fi
}

show_text_if_exists() {
    local title="$1"
    local path="$2"
    local max_lines="${3:-400}"

    if [[ -f "$path" ]]; then
        log_step "$title"
        local total_lines
        total_lines=$(wc -l < "$path" 2>/dev/null || printf '0')
        if [[ "$total_lines" =~ ^[0-9]+$ ]] && (( total_lines > max_lines )); then
            sed -n "1,${max_lines}p" "$path"
            log_warn "$title truncated: showing first ${max_lines}/${total_lines} lines"
        else
            cat "$path"
        fi
    else
        log_warn "$title skipped: file not found: $path"
    fi
}

copy_session_if_exists() {
    local source_path="$1"
    local target_path="$2"
    local copy_label="$3"

    if [[ ! -f "$source_path" ]]; then
        log_warn "$copy_label skipped: file not found: $source_path"
        return 0
    fi

    run_step "$copy_label" cp "$source_path" "$target_path"
}

ensure_runtime_dirs() {
    mkdir -p "$RUN_OUTPUT_DIR" "$LOG_DIR"
}

save_prompt_copy() {
    local file_name="$1"
    local prompt_text="$2"

    ensure_runtime_dirs
    printf '%s\n' "$prompt_text" > "$LOG_DIR/$file_name"
}

ensure_quantize_script_artifact() {
    local quant_script_path="${RUN_OUTPUT_DIR}/quantize.py"
    local legacy_script_path="${RUN_OUTPUT_DIR}/quantize_script.py"

    ensure_runtime_dirs

    if [[ -f "$quant_script_path" ]]; then
        return 0
    fi

    if [[ -f "$legacy_script_path" ]]; then
        run_step "Normalize quantization script artifact" cp "$legacy_script_path" "$quant_script_path"
    fi

    if [[ ! -f "$quant_script_path" ]]; then
        record_failure "Quantization script missing: expected ${quant_script_path}"
    fi
}

ensure_evaluate_script_artifact() {
    local strict_mode="${1:-true}"
    local eval_script_path="${RUN_OUTPUT_DIR}/evaluate.sh"
    local legacy_shell_paths=(
        "${RUN_OUTPUT_DIR}/eval.sh"
        "${RUN_OUTPUT_DIR}/eval_script.sh"
        "${RUN_OUTPUT_DIR}/evaluate_script.sh"
    )
    local legacy_python_paths=(
        "${RUN_OUTPUT_DIR}/evaluate.py"
        "${RUN_OUTPUT_DIR}/eval.py"
        "${RUN_OUTPUT_DIR}/eval_script.py"
        "${RUN_OUTPUT_DIR}/evaluate_script.py"
    )
    local legacy_path=""

    ensure_runtime_dirs

    if [[ -f "$eval_script_path" ]]; then
        return 0
    fi

    for legacy_path in "${legacy_shell_paths[@]}"; do
        if [[ -f "$legacy_path" ]]; then
            run_step "Normalize evaluation script artifact" cp "$legacy_path" "$eval_script_path"
            break
        fi
    done

    if [[ ! -f "$eval_script_path" ]]; then
        for legacy_path in "${legacy_python_paths[@]}"; do
            if [[ -f "$legacy_path" ]]; then
                local message="Legacy Python evaluation script found (${legacy_path}); expected ${eval_script_path} for new runs"
                if [[ "$strict_mode" == "true" ]]; then
                    record_failure "$message"
                else
                    log_warn "$message"
                fi
                return 0
            fi
        done

        local message="Evaluation script missing: expected ${eval_script_path}"
        if [[ "$strict_mode" == "true" ]]; then
            record_failure "$message"
        else
            log_warn "$message"
        fi
    fi
}

normalize_json_gpu_metadata() {
    local target_path="$1"
    local mode="$2"
    local quant_gpus="$3"
    local eval_gpus="$4"

    [[ -f "$target_path" ]] || return 0

    run_step \
        "Normalize $(basename "$target_path") GPU metadata" \
        python3 - "$target_path" "$mode" "$quant_gpus" "$eval_gpus" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
quant_gpus = sys.argv[3]
eval_gpus = sys.argv[4]

with path.open(encoding="utf-8") as handle:
    data = json.load(handle)

if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a JSON object")

if mode == "quant_summary":
    data["quant_num_gpus"] = quant_gpus
    data["num_gpus"] = quant_gpus
elif mode == "accuracy_auto_quant":
    data["eval_num_gpus"] = eval_gpus
    data["num_gpus"] = eval_gpus
elif mode == "accuracy_auto_eval":
    data["num_gpus"] = eval_gpus
else:
    raise SystemExit(f"unsupported mode: {mode}")

with path.open("w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

# ---------------------------------------------------------------------------
# detect_and_patch_method  –  read the real quantization method from the
# model's config.json (quantization_config.quant_method) and patch it into
# quant_summary.json / accuracy.json so the leaderboard shows the correct
# "Method" column.
#
# Usage:  detect_and_patch_method  MODEL_DIR_OR_HF_ID  JSON_FILE [JSON_FILE …]
# ---------------------------------------------------------------------------
detect_and_patch_method() {
    local model_path="$1"; shift
    local json_files=("$@")

    [[ -z "$model_path" ]] && return 0
    [[ ${#json_files[@]} -eq 0 ]] && return 0

    # Filter to existing files only
    local existing=()
    for f in "${json_files[@]}"; do
        [[ -f "$f" ]] && existing+=("$f")
    done
    [[ ${#existing[@]} -eq 0 ]] && return 0

    run_step \
        "Detect and patch quantization method" \
        python3 - "$model_path" "${existing[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

_METHOD_MAP = {
    "compressed-tensors": "CompressedTensors",
    "compressed_tensors": "CompressedTensors",
    "gptq":              "GPTQ",
    "awq":               "AWQ",
    "auto_round":        "AutoRound",
    "auto-round":        "AutoRound",
    "autoround":         "AutoRound",
    "bitsandbytes":      "BitsAndBytes",
    "hqq":               "HQQ",
    "fp8":               "FP8",
    "fbgemm_fp8":        "FBGEMM-FP8",
    "gguf":              "GGUF",
    "quanto":            "Quanto",
    "rtn":               "RTN",
}

def _detect_method(model_path: str) -> str:
    """Try local config.json, then HuggingFace hub."""
    # 1. Local config.json
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.is_file():
        try:
            with cfg_path.open(encoding="utf-8") as f:
                cfg = json.load(f)
            method = (cfg.get("quantization_config") or {}).get("quant_method", "")
            if method:
                return _METHOD_MAP.get(method.lower().strip(), method)
        except Exception:
            pass

    # 2. HuggingFace hub download (for auto_eval with HF model IDs)
    if "/" in model_path and not os.path.isdir(model_path):
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(model_path, "config.json")
            with open(local, encoding="utf-8") as f:
                cfg = json.load(f)
            method = (cfg.get("quantization_config") or {}).get("quant_method", "")
            if method:
                return _METHOD_MAP.get(method.lower().strip(), method)
        except Exception:
            pass

    return ""

model_path = sys.argv[1]
json_files = sys.argv[2:]

detected = _detect_method(model_path)
if not detected:
    print(f"[detect_method] Could not detect method for {model_path}")
    sys.exit(0)

print(f"[detect_method] Detected: {detected}")

for fpath in json_files:
    try:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        current = data.get("method")
        # Patch if missing or was the default "RTN" placeholder
        if not current or current == "RTN":
            data["method"] = detected
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  Patched {os.path.basename(fpath)}: {current!r} -> {detected!r}")
        else:
            print(f"  Kept {os.path.basename(fpath)}: {current!r}")
    except Exception as exc:
        print(f"  Error patching {fpath}: {exc}")
PY
}

write_quant_prompt() {
    cat <<EOF
You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: ${QUANT_SKILL_PATH}

Model: ${MODEL_ID}
Quantization: ${SCHEME} / ${METHOD}
Export format: ${EXPORT_FORMAT}
Quantized Model Output directory: ${QUANTIZED_MODEL_DIR}
Runtime artifact directory: ${RUN_OUTPUT_DIR}
Quantization execution log path: ${QUANT_EXEC_LOG}
Runtime device: ${DEVICE}
Num gpus: ${NUM_GPUS}

Directory responsibilities:
- Write exported model files to: ${QUANTIZED_MODEL_DIR}
- Write runtime artifacts such as quant_summary.json, quantize.py, logs, prompts, copied request/session files, and the venv to: ${RUN_OUTPUT_DIR}

CRITICAL SCRIPT REQUIREMENT:
- Before starting quantization, you MUST first generate the quantization script file:
    ${RUN_OUTPUT_DIR}/quantize.py
- The file name must be exactly: quantize.py
- The script must be a standalone Python program runnable with:
    python3 -u ${RUN_OUTPUT_DIR}/quantize.py
- The generated quantize.py must focus on the core quantization/export logic only.
- Do NOT put venv creation, pip/uv installation, proxy/bootstrap shell setup, summary generation, or unrelated orchestration into quantize.py.
- In this same OpenClaw task, prepare or reuse the Python environment separately before executing quantize.py.
- In this same OpenClaw task, first write quantize.py, then execute that generated script yourself.
- When you execute quantize.py, you MUST stream stdout/stderr into this log file while still printing output:
    python3 -u ${RUN_OUTPUT_DIR}/quantize.py 2>&1 | tee ${QUANT_EXEC_LOG}
- ${QUANT_SUMMARY_JSON} is a final summary artifact, not a progress marker.
- After quantize.py finishes, inspect the exported artifacts and write ${QUANT_SUMMARY_JSON} in a separate finalize step outside quantize.py.
- Do NOT write a success ${QUANT_SUMMARY_JSON} until quantization has finished and the exported artifacts are ready.
- Write ${QUANT_SUMMARY_JSON} atomically via a temporary file and rename/move it into place only at finalize time.
- If quantization fails, still write a minimal failed ${QUANT_SUMMARY_JSON} before exiting non-zero, also atomically.
- Do not use quantize_script.py as the final artifact name

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- This workflow is CUDA-focused. For AutoRound device selection:
    - if Num gpus == 1, prefer device="cuda"
    - if Num gpus > 1, prefer device_map="auto"
  Do NOT default to device_map="0" or device_map="0,1,2,3" unless manual mapping is truly required after auto placement fails.

IMPORTANT - In this same OpenClaw task, after separate quantization and finalize steps, you MUST produce:

${QUANT_SUMMARY_JSON} - structured summary:
{
  "model_id": "${MODEL_ID}",
  "scheme": "${SCHEME}",
  "method": "${METHOD}",
  "export_format": "${EXPORT_FORMAT}",
  "device": "${DEVICE}",
  "quant_num_gpus": "${NUM_GPUS}",
  "num_gpus": "${NUM_GPUS}",
  "output_dir": "${RUN_OUTPUT_DIR}",
  "runtime_output_dir": "${RUN_OUTPUT_DIR}",
  "quantized_model_dir": "${QUANTIZED_MODEL_DIR}",
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "original_size_mb": <float or null>,
  "quantized_size_mb": <float or null>,
  "compression_ratio": <float or null>,
  "errors": [<list of error strings>],
  "solutions": [<list of solution strings>],
  "output_files": [<list of file paths in runtime_output_dir>]
}

Write as valid JSON. If quantization fails, still write quant_summary.json with status=failed.
EOF
}

write_eval_script_prompt() {
    cat <<EOF
You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: ${EVAL_SKILL_PATH}

Quantized model path: ${QUANTIZED_MODEL_DIR}
Runtime artifact directory: ${RUN_OUTPUT_DIR}
Raw lm_eval output directory: ${LM_EVAL_OUTPUT_DIR}
Evaluation script path: ${RUN_OUTPUT_DIR}/evaluate.sh
Evaluation execution log path: ${EVAL_EXEC_LOG}
Evaluation tasks: ${EVAL_TASKS}
Batch size: ${EVAL_BATCH_SIZE}
Num gpus: ${EVAL_NUM_GPUS}

The quantized model was produced by auto_quant with scheme=${SCHEME}, export_format=${EXPORT_FORMAT}.
A venv may already exist at ${RUN_OUTPUT_DIR}/venv (created by auto_quant with --system-site-packages).

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If /root/.venv exists, reuse /root/.venv before creating a new venv.
- If a venv already exists at ${RUN_OUTPUT_DIR}/venv, reuse it - just install lm_eval and vllm into it.
- Use uv pip for dependency installation. Prefer:
    uv pip install --python <venv>/bin/python <packages>
- Do NOT reinstall torch or flash_attn if they already import successfully from the reused environment. Only install them when missing or incompatible.
- Write evaluation outputs, logs, prompts, copied request/session files, and other runtime artifacts to: ${RUN_OUTPUT_DIR}
- Before starting evaluation, you MUST first generate the evaluation script file:
    ${RUN_OUTPUT_DIR}/evaluate.sh
- The file name must be exactly: evaluate.sh
- The script must be a standalone shell program runnable with:
    bash ${RUN_OUTPUT_DIR}/evaluate.sh
- The generated evaluate.sh must focus on Stage A raw lm_eval execution only.
- Do NOT put venv creation, pip/uv installation, package bootstrap, JSON parsing, accuracy.json writing, or destructive cleanup/recreation into evaluate.sh.
- In this same OpenClaw task, prepare or reuse the evaluation environment separately before executing evaluate.sh.
- Prefer direct lm_eval CLI commands and standard shell environment variables over ad-hoc Python wrappers.
- In this same OpenClaw task, first write evaluate.sh, then execute that generated script yourself.
- When you execute evaluate.sh or a direct lm_eval command, you MUST stream stdout/stderr into this log file while still printing output:
    bash ${RUN_OUTPUT_DIR}/evaluate.sh 2>&1 | tee ${EVAL_EXEC_LOG}
- ${ACCURACY_JSON} is a final summary artifact, not a progress marker.
- After Stage A raw lm_eval completes, parse the latest raw results and write ${ACCURACY_JSON} in a separate finalize step outside evaluate.sh.
- Do NOT write a success ${ACCURACY_JSON} until Stage A raw lm_eval output and Stage B parsing have both completed successfully.
- Write ${ACCURACY_JSON} atomically via a temporary file and rename/move it into place only at finalize time.
- If evaluation fails, still write a minimal failed ${ACCURACY_JSON} before exiting non-zero, also atomically.
- The overall evaluation workflow MUST still behave as two idempotent stages:
    Stage A: run lm_eval and persist raw results under ${LM_EVAL_OUTPUT_DIR}
    Stage B: parse the latest raw results into ${ACCURACY_JSON}
- Stage A and Stage B MUST be independently rerunnable.
- If ${LM_EVAL_OUTPUT_DIR} already contains a valid raw results file matching `results_*.json`, do NOT rerun Stage A; only rerun Stage B parsing.
- If Stage B parsing fails after Stage A already succeeded, exit non-zero but preserve the raw lm_eval outputs, logs, and venv so the next retry only reruns parsing.
- Do NOT delete or recreate ${LM_EVAL_OUTPUT_DIR} when raw results already exist.
- Parsing logic should read the latest results file under ${LM_EVAL_OUTPUT_DIR}/**/results_*.json rather than assuming lm_eval must be rerun.
- Do NOT leave the final artifact named eval.sh, eval_script.sh, or evaluate_script.sh.
- When the generated script invokes lm_eval, it MUST pass:
    --output_path ${LM_EVAL_OUTPUT_DIR}
  and MUST set max_gen_toks=2048. The placement depends on the backend:
    - For HF backend (--model hf):   --gen_kwargs max_gen_toks=2048
    - For vLLM backend (--model vllm): append max_gen_toks=2048 inside --model_args
- Do NOT omit --output_path. Keep the raw lm_eval output files under that exact directory for later upload.
- Do NOT omit max_gen_toks=2048 regardless of backend.

IMPORTANT - In this same OpenClaw task, after separate raw-eval and finalize steps, you MUST produce:

${ACCURACY_JSON} - evaluation results:
{
  "model_id": "${MODEL_ID}",
  "model_path": "${QUANTIZED_MODEL_DIR}",
  "scheme": "${SCHEME}",
  "device": "${DEVICE}:${DEVICE_INDEX}",
  "num_gpus": "${EVAL_NUM_GPUS}",
  "tasks": {
    "<task_name>": {
      "accuracy": <float>,
      "accuracy_stderr": <float or null>
    }
  },
  "status": "success" or "failed",
  "duration_seconds": <float>,
  "eval_framework": "lm_eval+vllm" or "lm_eval+hf" or "manual",
  "errors": [<list of error strings if any>]
}

${LM_EVAL_OUTPUT_DIR}/ - raw lm_eval output directory created by:
    lm_eval ... --output_path ${LM_EVAL_OUTPUT_DIR}

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, the script may invoke python only for JSON post-processing, but it must still write accuracy.json with status=failed before exiting non-zero.
EOF
}

JSON_INPUT=""
SKIP_UPLOAD=false
SKIP_HF=false
SKIP_GITHUB=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-upload) SKIP_UPLOAD=true ;;
        --skip-hf) SKIP_HF=true ;;
        --skip-github) SKIP_GITHUB=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$JSON_INPUT" ]]; then
                JSON_INPUT="$arg"
            else
                log_error "Unknown argument: $arg"
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$JSON_INPUT" ]]; then
    usage
    exit 1
fi

JSON_FILE="$(resolve_json_file "$JSON_INPUT")" || {
    log_error "JSON file not found: $JSON_INPUT"
    exit 1
}
JSON_FILENAME="$(basename "$JSON_FILE")"

CONFIG_FILE="${SCRIPT_DIR}/config.env"
if [[ -f "$CONFIG_FILE" ]]; then
    set -a
    source "$CONFIG_FILE"
    set +a
fi

require_command python3

eval "$(python3 - "$JSON_FILE" <<'PY'
import json
import re
import shlex
import sys

json_file = sys.argv[1]
with open(json_file, encoding="utf-8") as handle:
    data = json.load(handle)

job_type = data.get("job_type", "")
script = data.get("script", "")
model = data.get("model", "")
pipeline = "auto_quant" if script == "auto_quant" or job_type == "quantization & evaluation" else "auto_eval"
quant_scheme_full = data.get("quant_scheme", data.get("compute_dtype", "INT4 (W4A16)"))
# Schemes like MXFP4, NVFP4 don't follow the WxAy pattern — use them directly
_known_standalone_schemes = {"MXFP4", "NVFP4"}
if str(quant_scheme_full).upper() in _known_standalone_schemes:
    scheme = str(quant_scheme_full).upper()
else:
    match = re.search(r"W\d+A\d+", str(quant_scheme_full))
    scheme = match.group(0) if match else "W4A16"
model_slug = model.replace("/", "_").replace(" ", "_")
quant_gpu = str(data.get("quant_gpu_nums", data.get("gpu_nums", 1)))
eval_gpu = str(data.get("eval_gpu_nums", data.get("gpu_nums", quant_gpu)))
model_path_override = model if pipeline == "auto_eval" else ""

exports = {
    "JOB_TYPE": job_type,
    "PIPELINE": pipeline,
    "MODEL_ID": model,
    "MODEL_SLUG": model_slug,
    "REVISION": data.get("revision", "main"),
    "SCHEME": scheme,
    "QUANT_SCHEME_FULL": quant_scheme_full,
    "NUM_GPUS": quant_gpu,
    "EVAL_NUM_GPUS": eval_gpu,
    "QUANT_PRECISION": data.get("quant_precision", data.get("precision", "4bit")),
    "QUANT_WEIGHT_DTYPE": data.get("quant_weight_dtype", data.get("weight_dtype", "int4")),
    "MODEL_PATH_OVERRIDE": model_path_override,
}

for key, value in exports.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

OUTPUT_DIR="${OUTPUT_DIR:-/root/.openclaw/workspace/quantized}"
RUNTIME_OUTPUT_BASE_DIR="${RUNTIME_OUTPUT_BASE_DIR:-/root/.openclaw/workspace/quantized/runs}"
METHOD="${METHOD:-RTN}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto_round}"
DEVICE="${DEVICE:-cuda}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
TIMEOUT="${TIMEOUT:-36000}"
EVAL_TASKS="${EVAL_TASKS:-piqa,mmlu,hellaswag}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
OPENCLAW_WORKSPACE_DIR="${OPENCLAW_WORKSPACE_DIR:-}"
OPENCLAW_SESSIONS_DIR="${OPENCLAW_SESSIONS_DIR:-/root/.openclaw/agents/main/sessions}"

MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_SLUG}-${SCHEME}"
RUN_OUTPUT_DIR="${RUNTIME_OUTPUT_BASE_DIR}/${MODEL_SLUG}-${SCHEME}"
QUANTIZED_MODEL_DIR="$MODEL_OUTPUT_DIR"
if [[ "$PIPELINE" == "auto_eval" && -n "${MODEL_PATH_OVERRIDE:-}" ]]; then
    QUANTIZED_MODEL_DIR="$MODEL_PATH_OVERRIDE"
fi

if [[ -z "$OPENCLAW_WORKSPACE_DIR" ]]; then
    OPENCLAW_WORKSPACE_DIR="$(
        resolve_first_existing_dir \
            "/root/leaderboard_Agent/auto_quant/openclaw_home/workspace" \
            "/root/leaderboard_Agent/tasks/lb_eval/openclaw_config/workspace" \
            "/root/.openclaw/workspace"
    )" || {
        log_error "Could not resolve OPENCLAW_WORKSPACE_DIR; set it in config.env"
        exit 1
    }
fi

LOG_DIR="${RUN_OUTPUT_DIR}/logs"
ensure_runtime_dirs
LOG_FILE="${LOG_DIR}/auto.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

FAILED_STEPS=()
LAST_EXIT_CODE=0

QUANT_SKILL_PATH=""
if [[ "$PIPELINE" == "auto_quant" ]]; then
    QUANT_SKILL_PATH="$(resolve_skill_path "auto_quant")" || {
        log_error "Quant skill file not found"
        exit 1
    }
fi
EVAL_SKILL_NAME="auto_eval"
if [[ "$PIPELINE" == "auto_eval" ]]; then
    EVAL_SKILL_NAME="auto_eval_vllm"
fi
EVAL_SKILL_PATH="$(resolve_skill_path "$EVAL_SKILL_NAME")" || {
    log_error "Eval skill file not found for skill: $EVAL_SKILL_NAME"
    exit 1
}
QUANT_SESSION="autoeval_quant_$$"
EVAL_SESSION="autoeval_eval_$$"
QUANT_SUMMARY_JSON="${RUN_OUTPUT_DIR}/quant_summary.json"
ACCURACY_JSON="${RUN_OUTPUT_DIR}/accuracy.json"
LM_EVAL_OUTPUT_DIR="${RUN_OUTPUT_DIR}/lm_eval_results"
QUANT_SCRIPT="${RUN_OUTPUT_DIR}/quantize.py"
EVAL_SCRIPT="${RUN_OUTPUT_DIR}/evaluate.sh"
QUANT_EXEC_LOG="${LOG_DIR}/quant_exec.log"
EVAL_EXEC_LOG="${LOG_DIR}/eval_exec.log"
REQUEST_JSON="${RUN_OUTPUT_DIR}/request.json"
QUANT_SESSION_SRC="${OPENCLAW_SESSIONS_DIR}/${QUANT_SESSION}.jsonl"
EVAL_SESSION_SRC="${OPENCLAW_SESSIONS_DIR}/${EVAL_SESSION}.jsonl"
QUANT_SESSION_DST="${RUN_OUTPUT_DIR}/session_quant_$$.jsonl"
EVAL_SESSION_DST="${RUN_OUTPUT_DIR}/session_eval_$$.jsonl"
FORMATTER="${SCRIPT_DIR}/format_sessions.py"
SESSION_MONITOR="${SCRIPT_DIR}/stream_session.py"
GITHUB_UPLOADER="${SCRIPT_DIR}/upload_results_github.py"

log_step "Resolved configuration"
echo "JSON file           : $JSON_FILENAME"
echo "Job type            : $JOB_TYPE"
echo "Pipeline            : $PIPELINE"
echo "Model               : $MODEL_ID"
echo "Revision            : $REVISION"
echo "Scheme              : $SCHEME ($QUANT_SCHEME_FULL)"
echo "Quant GPUs          : $NUM_GPUS"
echo "Eval GPUs           : $EVAL_NUM_GPUS"
echo "OpenClaw workspace  : $OPENCLAW_WORKSPACE_DIR"
echo "OpenClaw sessions   : $OPENCLAW_SESSIONS_DIR"
echo "Eval skill          : $EVAL_SKILL_NAME"
echo "Quant skill path    : ${QUANT_SKILL_PATH:-'(not used)'}"
echo "Eval skill path     : $EVAL_SKILL_PATH"
echo "Model output dir    : $MODEL_OUTPUT_DIR"
echo "Runtime output dir  : $RUN_OUTPUT_DIR"
echo "Quantized model dir : $QUANTIZED_MODEL_DIR"
echo "Log file            : $LOG_FILE"
echo "Skip upload(all)    : $SKIP_UPLOAD"
echo "Skip HF upload      : $SKIP_HF"
echo "Skip GitHub upload  : $SKIP_GITHUB"

if [[ "$PIPELINE" == "auto_quant" && ! -f "$QUANT_SKILL_PATH" ]]; then
    log_error "Quant skill file not found: $QUANT_SKILL_PATH"
    exit 1
fi
if [[ ! -f "$EVAL_SKILL_PATH" ]]; then
    log_error "Eval skill file not found: $EVAL_SKILL_PATH"
    exit 1
fi

ensure_runtime_dirs
run_step "Copy request JSON" cp "$JSON_FILE" "$REQUEST_JSON"

if [[ "$DRY_RUN" == "true" ]]; then
    log_ok "Dry run complete"
    exit 0
fi

require_command openclaw

QUANT_STATUS="$(json_status "$QUANT_SUMMARY_JSON")"
if [[ "$PIPELINE" == "auto_quant" ]]; then
    if [[ "$QUANT_STATUS" != "success" ]]; then
        QUANT_PROMPT="$(write_quant_prompt)"
        save_prompt_copy "quant_prompt.txt" "$QUANT_PROMPT"
        quant_script_watch_pid=""
        quant_exec_tail_pid=""
        if [[ ! -f "$QUANT_SCRIPT" ]]; then
            start_artifact_watch "$QUANT_SCRIPT" "Generated quantization script" quant_script_watch_pid 400 || true
        fi
        start_log_tail "$QUANT_EXEC_LOG" "Quantization execution log" quant_exec_tail_pid || true
        quant_monitor_pid=""
        start_session_monitor "$QUANT_SESSION_SRC" "quant-live" quant_monitor_pid || true
        run_step \
            "Run auto_quant" \
            env \
                http_proxy="${HTTP_PROXY:-}" \
                https_proxy="${HTTPS_PROXY:-}" \
                HTTP_PROXY="${HTTP_PROXY:-}" \
                HTTPS_PROXY="${HTTPS_PROXY:-}" \
                PYTHONUNBUFFERED=1 \
                openclaw agent --local \
                    --session-id "$QUANT_SESSION" \
                    --message "$QUANT_PROMPT" \
                    --timeout "$TIMEOUT"
        stop_session_monitor "${quant_monitor_pid:-}"
        stop_session_monitor "${quant_script_watch_pid:-}"
        stop_session_monitor "${quant_exec_tail_pid:-}"
        copy_session_if_exists \
            "$QUANT_SESSION_SRC" \
            "$QUANT_SESSION_DST" \
            "Copy quant session log"
        ensure_quantize_script_artifact
        show_text_if_exists "Generated quantization script" "$QUANT_SCRIPT" 400
        QUANT_STATUS="$(json_status "$QUANT_SUMMARY_JSON")"
    else
        log_ok "Quantization already succeeded, skipping auto_quant"
    fi
else
    QUANT_STATUS="success"
fi

if [[ "$PIPELINE" == "auto_quant" ]]; then
    ensure_runtime_dirs
    if [[ ! -f "$QUANT_SESSION_DST" ]]; then
        copy_if_exists "$QUANT_SESSION_SRC" "$QUANT_SESSION_DST" "Copy quant session log"
    fi
    ensure_quantize_script_artifact
    show_text_if_exists "Generated quantization script" "$QUANT_SCRIPT" 400
fi

EVAL_STATUS="$(json_status "$ACCURACY_JSON")"
if [[ "$EVAL_STATUS" != "success" ]]; then
    if [[ "$QUANT_STATUS" == "success" ]]; then
        EVAL_PROMPT="$(write_eval_script_prompt)"
        save_prompt_copy "eval_script_prompt.txt" "$EVAL_PROMPT"
        eval_script_watch_pid=""
        eval_exec_tail_pid=""
        if [[ ! -f "$EVAL_SCRIPT" ]]; then
            start_artifact_watch "$EVAL_SCRIPT" "Generated evaluation script" eval_script_watch_pid 400 || true
        fi
        start_log_tail "$EVAL_EXEC_LOG" "Evaluation execution log" eval_exec_tail_pid || true
        eval_monitor_pid=""
        start_session_monitor "$EVAL_SESSION_SRC" "eval-live" eval_monitor_pid || true
        run_step \
            "Run ${EVAL_SKILL_NAME}" \
            env \
                http_proxy="${HTTP_PROXY:-}" \
                https_proxy="${HTTPS_PROXY:-}" \
                HTTP_PROXY="${HTTP_PROXY:-}" \
                HTTPS_PROXY="${HTTPS_PROXY:-}" \
                PYTHONUNBUFFERED=1 \
                openclaw agent --local \
                    --session-id "$EVAL_SESSION" \
                    --message "$EVAL_PROMPT" \
                    --timeout "$TIMEOUT"
        stop_session_monitor "${eval_monitor_pid:-}"
        stop_session_monitor "${eval_script_watch_pid:-}"
        stop_session_monitor "${eval_exec_tail_pid:-}"
        copy_session_if_exists \
            "$EVAL_SESSION_SRC" \
            "$EVAL_SESSION_DST" \
            "Copy eval session log"
        ensure_evaluate_script_artifact
        show_text_if_exists "Generated evaluation script" "$EVAL_SCRIPT" 400
        EVAL_STATUS="$(json_status "$ACCURACY_JSON")"
    else
        log_warn "Skipping auto_eval because quantization status is $QUANT_STATUS"
    fi
else
    log_ok "Evaluation already succeeded, skipping auto_eval"
    ensure_evaluate_script_artifact false
fi

ensure_runtime_dirs
if [[ ! -f "$EVAL_SESSION_DST" ]]; then
    copy_if_exists "$EVAL_SESSION_SRC" "$EVAL_SESSION_DST" "Copy eval session log"
fi
show_text_if_exists "Generated evaluation script" "$EVAL_SCRIPT" 400

if [[ "$PIPELINE" == "auto_quant" && "$QUANT_STATUS" != "success" ]]; then
    ensure_failure_summary "$QUANT_SUMMARY_JSON" "quant_summary" "quantization"
    QUANT_STATUS="$(json_status "$QUANT_SUMMARY_JSON")"
fi
if [[ "$EVAL_STATUS" != "success" ]]; then
    if [[ "$PIPELINE" == "auto_eval" || "$QUANT_STATUS" == "success" ]]; then
        ensure_failure_summary "$ACCURACY_JSON" "accuracy" "evaluation"
        EVAL_STATUS="$(json_status "$ACCURACY_JSON")"
    fi
fi

# ── Agent-Assisted Failure Diagnosis ─────────────────────────────────────
# When a stage fails, invoke a lightweight OpenClaw diagnosis agent to analyze
# session logs, extract root cause, classify the failure, and merge findings
# back into the summary JSON.  This runs only on failure and with a short timeout.
DIAG_TIMEOUT="${DIAG_TIMEOUT:-3600}"
SKIP_DIAGNOSIS="${SKIP_DIAGNOSIS:-false}"

run_failure_diagnosis() {
    local failed_stage="$1"
    local session_file="$2"
    local exec_log="$3"
    local summary_json="$4"
    local diagnosis_output="${RUN_OUTPUT_DIR}/failure_diagnosis_${failed_stage}.json"

    if [[ "$SKIP_DIAGNOSIS" == "true" ]]; then
        log_warn "Skipping failure diagnosis (SKIP_DIAGNOSIS=true)"
        return 0
    fi
    if ! command -v openclaw &>/dev/null; then
        log_warn "Skipping failure diagnosis: openclaw not available"
        return 0
    fi

    local diag_prompt
    diag_prompt=$(cat <<DIAGEOF
You are a failure diagnosis expert for LLM quantization/evaluation pipelines.

A ${failed_stage} task has FAILED for model "${MODEL_ID}" (scheme: ${SCHEME}, pipeline: ${PIPELINE}).

Available files to examine:
- Session log (JSONL): ${session_file}
- Execution log: ${exec_log}
- Summary JSON: ${summary_json}
- Run output dir: ${RUN_OUTPUT_DIR}

Your task:
1. Read the session log (focus on the LAST 200 lines or error sections) and exec log if they exist
2. Identify the ROOT CAUSE of the failure (not symptoms, the actual cause)
3. Classify the failure into exactly ONE of these categories:
   - oom: GPU/CPU memory insufficient for this model
   - disk_full: Storage space exhausted
   - unsupported_arch: Model architecture not supported by vLLM or transformers
   - gated_repo: HuggingFace access denied (needs token or license agreement)
   - model_incomplete: Missing weights, config, or safetensors files
   - library_incompatible: Version conflicts between torch/vllm/transformers/auto-round
   - driver_old: NVIDIA driver or CUDA toolkit too old
   - timeout: Task exceeded time limit or was unusably slow
   - agent_error: Script was not generated correctly or execution logic was wrong
   - dataset_error: Evaluation dataset/task loading failure
   - network_error: Download failed, connection timeout, HTTP errors
   - unknown: Cannot determine from available logs
4. Extract the key error message (the actual Python traceback or shell error, max 500 chars)
5. Determine if retrying with the same configuration could succeed
6. If retryable, suggest what the retry should do differently

Write your diagnosis ONLY to this file: ${diagnosis_output}
Format as valid JSON:
{
  "failure_category": "<one of the categories above>",
  "root_cause": "<one-line description of what went wrong>",
  "key_error": "<actual error text from logs, max 500 chars>",
  "retryable": true or false,
  "suggested_fix": "<what would fix this, or why retry won't help>",
  "confidence": <float 0.0-1.0>
}

IMPORTANT RULES:
- Only READ files. Do NOT attempt to fix, re-run, or modify anything.
- Write ONLY the single JSON file specified above.
- Be concise and precise. Focus on the root cause, not the full chain of events.
- If logs are missing or empty, set confidence to 0.3 and category to "unknown".
DIAGEOF
)

    log_step "Running failure diagnosis agent (${failed_stage})..."
    local diag_session="diagnosis_${failed_stage}_$$"
    run_step \
        "Failure diagnosis (${failed_stage})" \
        env PYTHONUNBUFFERED=1 \
        openclaw agent --local \
            --session-id "$diag_session" \
            --message "$diag_prompt" \
            --timeout "$DIAG_TIMEOUT"

    if [[ -f "$diagnosis_output" ]]; then
        log_ok "Diagnosis generated: $(basename "$diagnosis_output")"
        # Merge diagnosis findings into the summary JSON
        python3 - "$diagnosis_output" "$summary_json" <<'MERGEPY'
import json
import sys

diag_path = sys.argv[1]
summary_path = sys.argv[2]

try:
    with open(diag_path, encoding="utf-8") as f:
        diag = json.load(f)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    summary["failure_category"] = diag.get("failure_category", "unknown")
    summary["root_cause"] = diag.get("root_cause", "")
    summary["key_error"] = diag.get("key_error", "")
    summary["retryable"] = diag.get("retryable", False)
    summary["suggested_fix"] = diag.get("suggested_fix", "")

    errors = summary.get("errors", [])
    if not isinstance(errors, list):
        errors = []
    root_cause = diag.get("root_cause", "")
    category = diag.get("failure_category", "unknown")
    if root_cause:
        diag_line = f"[{category}] {root_cause}"
        if diag_line not in errors:
            errors.insert(0, diag_line)
    summary["errors"] = errors

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[diagnosis] Merged: category={category}, retryable={diag.get('retryable')}")
except Exception as e:
    print(f"[diagnosis] WARNING: merge failed: {e}", file=sys.stderr)
MERGEPY
    else
        log_warn "Diagnosis agent did not produce output file"
    fi
}

# Run diagnosis on failed stages
if [[ "$PIPELINE" == "auto_quant" && "$QUANT_STATUS" == "failed" ]]; then
    run_failure_diagnosis "quantization" "${QUANT_SESSION_DST:-}" "$QUANT_EXEC_LOG" "$QUANT_SUMMARY_JSON"
fi
if [[ "$EVAL_STATUS" == "failed" ]]; then
    run_failure_diagnosis "evaluation" "${EVAL_SESSION_DST:-}" "$EVAL_EXEC_LOG" "$ACCURACY_JSON"
fi

# ── Smart Retry ──────────────────────────────────────────────────────────
# If diagnosis says retryable, re-run the failed stage with failure context
# injected so the agent can try a different strategy.
MAX_SMART_RETRIES="${MAX_SMART_RETRIES:-1}"
SKIP_SMART_RETRY="${SKIP_SMART_RETRY:-false}"

_read_diag_field() {
    local diag_file="$1" field="$2" default="${3:-}"
    if [[ -f "$diag_file" ]]; then
        python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))" \
            "$diag_file" "$field" "$default" 2>/dev/null || echo "$default"
    else
        echo "$default"
    fi
}

run_smart_retry() {
    local stage="$1"           # "quantization" or "evaluation"
    local original_prompt="$2"
    local summary_json="$3"
    local session_src="$4"
    local session_dst="$5"
    local exec_log="$6"
    local diag_file="${RUN_OUTPUT_DIR}/failure_diagnosis_${stage}.json"

    if [[ "$SKIP_SMART_RETRY" == "true" ]]; then
        log_warn "Smart retry skipped (SKIP_SMART_RETRY=true)"
        return 0
    fi
    if [[ ! -f "$diag_file" ]]; then
        log_warn "Smart retry skipped: no diagnosis file for ${stage}"
        return 0
    fi

    local retryable
    retryable="$(_read_diag_field "$diag_file" "retryable" "False")"
    if [[ "$retryable" != "True" && "$retryable" != "true" ]]; then
        log_warn "Smart retry skipped: diagnosis says not retryable (${stage})"
        return 0
    fi

    local root_cause suggested_fix failure_category
    root_cause="$(_read_diag_field "$diag_file" "root_cause" "unknown")"
    suggested_fix="$(_read_diag_field "$diag_file" "suggested_fix" "")"
    failure_category="$(_read_diag_field "$diag_file" "failure_category" "unknown")"

    log_step "Smart Retry: ${stage} (category: ${failure_category})"
    echo "  Root cause: ${root_cause}"
    echo "  Suggested fix: ${suggested_fix}"

    # Build enhanced prompt with failure context
    local retry_addendum
    retry_addendum=$(cat <<RETRYEOF

⚠️ CRITICAL: PREVIOUS ATTEMPT FAILED — YOU MUST USE A DIFFERENT STRATEGY

The previous attempt at this ${stage} task FAILED. Here is the diagnosis:

Failure category: ${failure_category}
Root cause: ${root_cause}
Suggested fix: ${suggested_fix}

YOU MUST:
1. NOT repeat the same approach that failed — try a different strategy.
2. Apply the suggested fix above if applicable.
3. Key strategies by failure category:
   - oom: reduce batch_size, use device_map="auto" for multi-GPU, offload layers, or use a lighter backend
   - library_incompatible: check version compatibility first, use pip install --upgrade for the conflicting package
   - unsupported_arch: if vLLM fails, fall back to HuggingFace transformers backend with lm_eval --model hf
   - agent_error: carefully verify each generated file is syntactically correct (use python3 -c "import ast; ast.parse(open('file').read())") before running
   - dataset_error: verify task names are correct for lm_eval, use --tasks with validated task IDs
   - timeout: reduce nsamples, use smaller calibration dataset, or skip slow tasks
4. Before writing status=success, verify output files actually exist and contain valid data.
5. If the issue is fundamentally unsolvable (e.g., model too large for available hardware with no workaround), write a clear failed status with explanation instead of trying endlessly.

Previous artifacts may still exist in the workspace. You can inspect them but do NOT assume they are correct.
The previous ${summary_json} has been reset — you must write a new one.
RETRYEOF
)

    local retry_prompt="${original_prompt}${retry_addendum}"

    # Reset the summary JSON so the retry starts fresh
    if [[ -f "$summary_json" ]]; then
        # Backup old summary
        cp "$summary_json" "${summary_json}.pre_retry"
        rm -f "$summary_json"
    fi

    # Clean up exec log for fresh capture
    : > "$exec_log"

    local retry_session="${stage}_retry_$$"
    local retry_session_src="${OPENCLAW_SESSIONS_DIR}/${retry_session}.jsonl"
    local retry_session_dst="${RUN_OUTPUT_DIR}/session_${stage}_retry_$$.jsonl"

    save_prompt_copy "${stage}_retry_prompt.txt" "$retry_prompt"

    local retry_monitor_pid="" retry_exec_tail_pid=""
    start_log_tail "$exec_log" "Retry ${stage} execution log" retry_exec_tail_pid || true
    start_session_monitor "$retry_session_src" "${stage}-retry-live" retry_monitor_pid || true

    run_step \
        "Smart Retry ${stage}" \
        env \
            http_proxy="${HTTP_PROXY:-}" \
            https_proxy="${HTTPS_PROXY:-}" \
            HTTP_PROXY="${HTTP_PROXY:-}" \
            HTTPS_PROXY="${HTTPS_PROXY:-}" \
            PYTHONUNBUFFERED=1 \
            openclaw agent --local \
                --session-id "$retry_session" \
                --message "$retry_prompt" \
                --timeout "$TIMEOUT"

    stop_session_monitor "${retry_monitor_pid:-}"
    stop_session_monitor "${retry_exec_tail_pid:-}"

    # Copy retry session log
    copy_session_if_exists "$retry_session_src" "$retry_session_dst" "Copy ${stage} retry session"

    # Check new status
    local new_status
    new_status="$(json_status "$summary_json")"
    if [[ "$new_status" == "success" ]]; then
        log_ok "Smart Retry succeeded for ${stage}!"
    else
        log_warn "Smart Retry did not resolve ${stage} failure (status: ${new_status})"
        # Write failure summary if still not written
        if [[ "$new_status" != "failed" ]]; then
            if [[ "$stage" == "quantization" ]]; then
                ensure_failure_summary "$summary_json" "quant_summary" "quantization"
            else
                ensure_failure_summary "$summary_json" "accuracy" "evaluation"
            fi
        fi
    fi
    echo "$new_status"
}

# Execute smart retry for failed stages
RETRY_COUNT=0
if [[ "$PIPELINE" == "auto_quant" && "$QUANT_STATUS" == "failed" && $RETRY_COUNT -lt $MAX_SMART_RETRIES ]]; then
    QUANT_PROMPT="$(write_quant_prompt)"
    new_quant_status="$(run_smart_retry "quantization" "$QUANT_PROMPT" "$QUANT_SUMMARY_JSON" "$QUANT_SESSION_SRC" "$QUANT_SESSION_DST" "$QUANT_EXEC_LOG")"
    if [[ "$new_quant_status" == "success" ]]; then
        QUANT_STATUS="success"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        # If quant succeeded on retry, try eval too
        EVAL_STATUS="$(json_status "$ACCURACY_JSON")"
        if [[ "$EVAL_STATUS" != "success" ]]; then
            EVAL_PROMPT="$(write_eval_script_prompt)"
            eval_retry_session="autoeval_eval_retry_$$"
            eval_retry_session_src="${OPENCLAW_SESSIONS_DIR}/${eval_retry_session}.jsonl"
            start_log_tail "$EVAL_EXEC_LOG" "Evaluation execution log (post-retry)" eval_exec_tail_pid || true
            run_step \
                "Run ${EVAL_SKILL_NAME} (after quant retry)" \
                env \
                    http_proxy="${HTTP_PROXY:-}" \
                    https_proxy="${HTTPS_PROXY:-}" \
                    HTTP_PROXY="${HTTP_PROXY:-}" \
                    HTTPS_PROXY="${HTTPS_PROXY:-}" \
                    PYTHONUNBUFFERED=1 \
                    openclaw agent --local \
                        --session-id "$eval_retry_session" \
                        --message "$EVAL_PROMPT" \
                        --timeout "$TIMEOUT"
            stop_session_monitor "${eval_exec_tail_pid:-}"
            copy_session_if_exists "$eval_retry_session_src" "${RUN_OUTPUT_DIR}/session_eval_retry_$$.jsonl" "Copy eval session (post quant retry)"
            EVAL_STATUS="$(json_status "$ACCURACY_JSON")"
        fi
    fi
fi

if [[ "$EVAL_STATUS" == "failed" && $RETRY_COUNT -lt $MAX_SMART_RETRIES ]]; then
    EVAL_PROMPT="$(write_eval_script_prompt)"
    new_eval_status="$(run_smart_retry "evaluation" "$EVAL_PROMPT" "$ACCURACY_JSON" "$EVAL_SESSION_SRC" "$EVAL_SESSION_DST" "$EVAL_EXEC_LOG")"
    if [[ "$new_eval_status" == "success" ]]; then
        EVAL_STATUS="success"
    fi
fi

if [[ "$PIPELINE" == "auto_quant" ]]; then
    normalize_json_gpu_metadata "$QUANT_SUMMARY_JSON" "quant_summary" "$NUM_GPUS" "$EVAL_NUM_GPUS"
    normalize_json_gpu_metadata "$ACCURACY_JSON" "accuracy_auto_quant" "$NUM_GPUS" "$EVAL_NUM_GPUS"
    detect_and_patch_method "$QUANTIZED_MODEL_DIR" "$QUANT_SUMMARY_JSON" "$ACCURACY_JSON"
else
    normalize_json_gpu_metadata "$ACCURACY_JSON" "accuracy_auto_eval" "$NUM_GPUS" "$EVAL_NUM_GPUS"
    detect_and_patch_method "$QUANTIZED_MODEL_DIR" "$ACCURACY_JSON"
fi

SESSION_INPUTS=()
if [[ -f "$QUANT_SESSION_DST" ]]; then
    SESSION_INPUTS+=("$QUANT_SESSION_DST")
fi
if [[ -f "$EVAL_SESSION_DST" ]]; then
    SESSION_INPUTS+=("$EVAL_SESSION_DST")
fi
# Include retry session logs if they exist
for retry_jsonl in "${RUN_OUTPUT_DIR}"/session_*_retry_*.jsonl; do
    [[ -f "$retry_jsonl" ]] && SESSION_INPUTS+=("$retry_jsonl")
done
if [[ ${#SESSION_INPUTS[@]} -gt 0 ]]; then
    run_step "Format session logs" python3 "$FORMATTER" "${SESSION_INPUTS[@]}"
else
    log_warn "Format session logs skipped: no session JSONL files were copied"
fi

show_json_if_exists "Quant summary" "$QUANT_SUMMARY_JSON"
show_json_if_exists "Accuracy summary" "$ACCURACY_JSON"

if [[ "$PIPELINE" == "auto_quant" && "$SKIP_UPLOAD" != "true" && "$SKIP_HF" != "true" ]]; then
    if [[ "$QUANT_STATUS" == "success" ]]; then
        MODEL_SHORT="${MODEL_ID#*/}"
        HF_REPO_NAME="${MODEL_SHORT}-autoround-${SCHEME}"
        run_step \
            "Upload quantized model to HuggingFace" \
            python3 "$SCRIPT_DIR/upload_model_hf.py" \
                "$MODEL_OUTPUT_DIR" \
                "$HF_REPO_NAME" \
                --summary-json "$QUANT_SUMMARY_JSON"
    else
        log_warn "Skipping HuggingFace upload because quantization status is $QUANT_STATUS"
    fi
fi

if [[ "$SKIP_UPLOAD" != "true" && "$SKIP_GITHUB" != "true" ]]; then
    run_step \
        "Upload result artifacts to GitHub" \
        python3 "$GITHUB_UPLOADER" \
            "$RUN_OUTPUT_DIR" \
            "$MODEL_ID" \
            --pipeline "$PIPELINE" \
            --scheme "$SCHEME" \
            --quant-num-gpus "$NUM_GPUS" \
            --eval-num-gpus "$EVAL_NUM_GPUS" \
            --model-output-dir "$QUANTIZED_MODEL_DIR" \
            --request-filename "$JSON_FILENAME"
fi

log_step "Final summary"
echo "Quant status : $QUANT_STATUS"
echo "Eval status  : $EVAL_STATUS"
echo "Model dir    : $QUANTIZED_MODEL_DIR"
echo "Runtime dir  : $RUN_OUTPUT_DIR"
echo "Log file     : $LOG_FILE"

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "Step failures:"
    printf '  - %s\n' "${FAILED_STEPS[@]}"
fi

OVERALL_EXIT=0
if [[ "$PIPELINE" == "auto_quant" && "$QUANT_STATUS" != "success" ]]; then
    OVERALL_EXIT=1
fi
if [[ "$EVAL_STATUS" != "success" ]]; then
    OVERALL_EXIT=1
fi
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    OVERALL_EXIT=1
fi

if [[ $OVERALL_EXIT -eq 0 ]]; then
    log_ok "Pipeline finished successfully"
else
    log_warn "Pipeline finished with failures"
fi

exit "$OVERALL_EXIT"
