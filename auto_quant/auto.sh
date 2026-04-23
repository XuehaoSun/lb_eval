#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# auto.sh — Unified one-click pipeline launcher
#
# Usage:
#   bash auto.sh <task_json_file> [options]
#
# Examples:
#   bash auto.sh Qwen3-8B_quant_request_False_INT4_4bit_int4.json
#   bash auto.sh Qwen3.5-27B_eval_request_False_INT4_4bit_int4.json --no-build
#   bash auto.sh task.json --skip-upload
#
# Options:
#   --no-build      Skip Docker image build (reuse existing image)
#   --skip-upload   Skip HuggingFace and GitHub upload steps
#   --skip-hf       Skip HuggingFace model upload only
#   --skip-github   Skip GitHub results upload only
#   --keep          Keep container running after completion (for debugging)
#   --dry-run       Print configuration and exit without running
#
# Configuration:
#   Place a config.env file in the tasks/ directory (see config.env.template).
#   Values from config.env are used as defaults; JSON fields take priority.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Color output helpers ─────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_info()  { echo -e "${CYAN}[auto.sh]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[auto.sh]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[auto.sh]${NC} $*"; }
log_error() { echo -e "${RED}[auto.sh]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${CYAN}════════ $* ════════${NC}\n"; }

# ── Parse CLI arguments ──────────────────────────────────────────────────────
JSON_INPUT=""
NO_BUILD=false
SKIP_UPLOAD=false
SKIP_HF=false
SKIP_GITHUB=false
KEEP_CONTAINER=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --no-build)     NO_BUILD=true ;;
        --skip-upload)  SKIP_UPLOAD=true ;;
        --skip-hf)      SKIP_HF=true ;;
        --skip-github)  SKIP_GITHUB=true ;;
        --keep)         KEEP_CONTAINER=true ;;
        --dry-run)      DRY_RUN=true ;;
        -h|--help)
            head -30 "$0" | tail -27
            exit 0 ;;
        *)
            if [[ -z "$JSON_INPUT" ]]; then
                JSON_INPUT="$arg"
            else
                log_error "Unknown argument: $arg"
                exit 1
            fi ;;
    esac
done

if [[ -z "$JSON_INPUT" ]]; then
    echo "Usage: bash auto.sh <task_json_file> [options]"
    echo "Run 'bash auto.sh --help' for more information."
    exit 1
fi

# Resolve JSON file path
JSON_FILE="$JSON_INPUT"
if [[ ! -f "$JSON_FILE" ]]; then
    JSON_FILE="$SCRIPT_DIR/$JSON_INPUT"
fi
if [[ ! -f "$JSON_FILE" ]]; then
    log_error "JSON file not found: $JSON_INPUT"
    exit 1
fi
JSON_FILENAME="$(basename "$JSON_FILE")"

# ══════════════════════════════════════════════════════════════════════════════
# Step 0: Load config and prerequisites
# ══════════════════════════════════════════════════════════════════════════════
log_step "Step 0: Loading configuration"

# Load shared config (provides defaults, not overrides)
CONFIG_FILE="${SCRIPT_DIR}/config.env"
if [[ -f "$CONFIG_FILE" ]]; then
    set -a; source "$CONFIG_FILE"; set +a
    log_info "Loaded config from: $CONFIG_FILE"
else
    log_warn "No config.env found. Using defaults. Copy config.env.template to config.env."
fi

# Check prerequisites
for cmd in python3; do
    if ! command -v "$cmd" &>/dev/null; then
        log_error "Required command not found: $cmd"
        exit 1
    fi
done


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Parse JSON and normalize environment variables
# ══════════════════════════════════════════════════════════════════════════════
log_step "Step 1: Parsing task JSON"

# Use python3 to parse JSON and emit shell variable exports
eval "$(python3 << PYEOF
import json, os, re, sys

with open("$JSON_FILE") as f:
    data = json.load(f)

job_type = data.get("job_type", "")
script = data.get("script", "")
model = data.get("model", "")

# Determine pipeline
if script == "auto_quant" or job_type == "quantization & evaluation":
    pipeline = "auto_quant"
else:
    pipeline = "auto_eval"

# Model slug (safe for filenames and container names)
model_slug = model.replace("/", "_").replace(" ", "_")

# Extract scheme code (e.g., "W4A16" from "INT4 (W4A16)")
quant_scheme_full = data.get("quant_scheme", data.get("compute_dtype", "INT4 (W4A16)"))
m = re.search(r"W\d+A\d+", str(quant_scheme_full))
scheme = m.group(0) if m else "W4A16"

# GPU numbers
quant_gpu = data.get("quant_gpu_nums", data.get("gpu_nums", 1))
eval_gpu = data.get("eval_gpu_nums", quant_gpu)

# For eval-only jobs, model is the already-quantized HF model
# For quant jobs, model is the FP model to be quantized
if pipeline == "auto_eval":
    model_path = model  # HF model ID — agent will download it
else:
    model_path = ""  # Will be set after quantization

vars_to_export = {
    "JOB_TYPE": job_type,
    "PIPELINE": pipeline,
    "MODEL_ID": model,
    "MODEL_SLUG": model_slug,
    "REVISION": data.get("revision", "main"),
    "SCHEME": scheme,
    "QUANT_SCHEME_FULL": quant_scheme_full,
    "NUM_GPUS": str(quant_gpu),
    "EVAL_NUM_GPUS": str(eval_gpu),
    "QUANT_PRECISION": data.get("quant_precision", data.get("precision", "4bit")),
    "QUANT_WEIGHT_DTYPE": data.get("quant_weight_dtype", data.get("weight_dtype", "int4")),
    "MODEL_PARAMS": str(data.get("model_params", data.get("params", 0))),
    "HARDWARE": data.get("hardware", "RTX 4090"),
    "INPUT_DTYPE": data.get("input_dtype", "bfloat16"),
    "MODEL_PATH_OVERRIDE": model_path,
    "ARCHITECTURES": data.get("architectures", ""),
    "LICENSE": data.get("license", "unknown"),
}

for k, v in vars_to_export.items():
    # Shell-safe escaping
    v_safe = str(v).replace("'", "'\\''")
    print(f"export {k}='{v_safe}'")
PYEOF
)"

# Derived variables
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONTAINER_NAME="auto-${MODEL_SLUG}-${$}"
JOB_OUTPUT_DIR=$"${OUTPUT_DIR:-/root/.openclaw/workspace/quantized}/${MODEL_SLUG}-${SCHEME}"
QUANTIZED_MODEL_DIR="${OUTPUT_DIR:-/root/.openclaw/workspace/quantized}/${MODEL_SLUG}-${SCHEME}"
PIPELINE_DIR="${REPO_ROOT}/${PIPELINE}"


# Print configuration summary
echo "──────────────────────────────────────────────"
echo "  JSON file      : $JSON_FILENAME"
echo "  Job type       : $JOB_TYPE"
echo "  Pipeline       : $PIPELINE"
echo "  Model          : $MODEL_ID"
echo "  Scheme         : $SCHEME ($QUANT_SCHEME_FULL)"
echo "  GPU (quant)    : $NUM_GPUS"
echo "  GPU (eval)     : $EVAL_NUM_GPUS"
echo "  Host output    : $JOB_OUTPUT_DIR"
echo "  Quantized model out  : $QUANTIZED_MODEL_DIR"
echo "  Pipeline dir   : $PIPELINE_DIR"
echo "──────────────────────────────────────────────"

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Dry run complete. Exiting."
    exit 0
fi
echo $JOB_OUTPUT_DIR

# Create host output directory
mkdir -p "$JOB_OUTPUT_DIR"

# Save the original request JSON to the job output for reference
cp "$JSON_FILE" "$JOB_OUTPUT_DIR/request.json"


METHOD="${METHOD:-RTN}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto_round}"
DEVICE="${DEVICE:-cuda}"
TIMEOUT="${TIMEOUT:-36000}"

EVAL_TASKS="${EVAL_TASKS:-piqa}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

QUANT_SKILL_PATH="/root/.openclaw/workspace/skills/auto_quant/SKILL.md"
EVAL_SKILL_PATH="/root/.openclaw/workspace/skills/auto_eval/SKILL.md"

# Session IDs (use same PID-based suffix so we can find them later)
QUANT_SESSION="autoeval_quant_$$"
EVAL_SESSION="autoeval_eval_$$"

FULL_OUTPUT=$JOB_OUTPUT_DIR

QUANT_STATUS=$(bash -c "
  if [[ -f '${FULL_OUTPUT}/quant_summary.json' ]]; then
    python3 -c \"import json; d=json.load(open('${FULL_OUTPUT}/quant_summary.json')); print(d.get('status','unknown'))\"
  else
    echo 'missing'
  fi
" 2>/dev/null || echo "error")



echo "Quant status: ${QUANT_STATUS}"


if [[ "${QUANT_STATUS}" != "success" ]]; then
    echo ""
    echo ">>> Step 1/2: Running auto_quant first ..."
    echo ""

    QUANT_PROMPT="You are an expert in LLM quantization using the Intel Auto-Round toolkit.
You MUST follow the skill instructions in: ${QUANT_SKILL_PATH}

Model: ${MODEL_ID}
Quantization: ${SCHEME} / ${METHOD}
Export format: ${EXPORT_FORMAT}
Quantized Model Output directory: ${QUANTIZED_MODEL_DIR}
Runtime device: ${DEVICE}
Num gpus: ${NUM_GPUS}

CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.

IMPORTANT — After quantization completes (success or failure), you MUST produce:

${FULL_OUTPUT}/quant_summary.json — structured summary:
{
  \"model_id\": \"${MODEL_ID}\",
  \"scheme\": \"${SCHEME}\",
  \"method\": \"${METHOD}\",
  \"export_format\": \"${EXPORT_FORMAT}\",
  \"device\": \"${DEVICE}\",
  \"num_gpus\": \"${NUM_GPUS}\",
  \"output_dir\": \"${FULL_OUTPUT}\",
  \"quantized_model_dir\": \"${QUANTIZED_MODEL_DIR}\",
  \"status\": \"success\" or \"failed\",
  \"duration_seconds\": <float>,
  \"original_size_mb\": <float or null>,
  \"quantized_size_mb\": <float or null>,
  \"compression_ratio\": <float or null>,
  \"errors\": [<list of error strings>],
  \"solutions\": [<list of solution strings>],
  \"output_files\": [<list of file paths in output_dir>]
}

Write as valid JSON. If quantization fails, still write quant_summary.json with status=failed."

    bash -c "
      export http_proxy=\$HTTP_PROXY && export https_proxy=\$HTTPS_PROXY && \
      openclaw agent --local \
        --session-id '${QUANT_SESSION}' \
        --message '$(echo "$QUANT_PROMPT" | sed "s/'/'\\''/g")' \
        --timeout ${TIMEOUT}
    "

    # Verify quant succeeded
    QUANT_STATUS=$(bash -c "
      if [[ -f '${FULL_OUTPUT}/quant_summary.json' ]]; then
        python3 -c \"import json; d=json.load(open('${FULL_OUTPUT}/quant_summary.json')); print(d.get('status','unknown'))\"
      else
        echo 'missing'
      fi
    " 2>/dev/null || echo "error")

    if [[ "${QUANT_STATUS}" != "success" ]]; then
        echo "ERROR: auto_quant failed or did not produce quant_summary.json (status=${QUANT_STATUS})"
        bash -c "cat '${FULL_OUTPUT}/quant_summary.json' 2>/dev/null || echo '(not found)'"
        exit 1
    fi

    echo ">>> auto_quant completed successfully."
else
    echo ">>> Quantized model already exists, skipping auto_quant."
fi

# ── Copy quant session log to output dir ────────────────────────────────────
echo ""
echo ">>> Copying quant session log ..."

bash -c "
  cp /root/.openclaw/agents/main/sessions/${QUANT_SESSION}.jsonl '${FULL_OUTPUT}/session_quant_$$.jsonl' && \
  echo 'Copied: ${FULL_OUTPUT}/session_quant_$$.jsonl'
" 2>/dev/null || echo "Note: Could not copy quant session log from container"


# ── Step 2: Run auto_eval ────────────────────────────────────────────────────
echo ""
echo ">>> Step 2/2: Running auto_eval ..."
echo ""

EVAL_PROMPT="You are an expert in evaluating quantized LLM models.
You MUST follow the skill instructions in: ${EVAL_SKILL_PATH}

Quantized model path: ${QUANTIZED_MODEL_DIR}
Evaluation tasks: ${EVAL_TASKS}
Batch size: ${EVAL_BATCH_SIZE}
Nuym gpus: ${EVAL_NUM_GPUS}

The quantized model was produced by auto_quant with scheme=${SCHEME}, export_format=${EXPORT_FORMAT}.
A venv may already exist at ${FULL_OUTPUT}/venv (created by auto_quant with --system-site-packages).
CRITICAL ENVIRONMENT NOTE:
- System Python has torch+cuda pre-installed. When creating venvs, ALWAYS use:
    python3 -m venv --system-site-packages <path>
  This ensures the venv inherits torch+cuda. Do NOT pip install torch inside the venv.
- If a venv already exists at ${FULL_OUTPUT}/venv, reuse it — just install lm_eval and vllm into it.

IMPORTANT — After evaluation completes, you MUST produce:

${FULL_OUTPUT}/accuracy.json — evaluation results:
{
  \"model_id\": \"${MODEL_ID}\",
  \"model_path\": \"${FULL_OUTPUT}\",
  \"scheme\": \"${SCHEME}\",
  \"device\": \"${DEVICE}:${DEVICE_INDEX}\",
  \"tasks\": {
    \"<task_name>\": {
      \"accuracy\": <float>,
      \"accuracy_stderr\": <float or null>
    }
  },
  \"status\": \"success\" or \"failed\",
  \"duration_seconds\": <float>,
  \"eval_framework\": \"lm_eval+vllm\" or \"lm_eval+hf\" or \"manual\",
  \"errors\": [<list of error strings if any>]
}

The accuracy values MUST be real numbers from actual evaluation runs.
Write as valid JSON. If evaluation fails, still write accuracy.json with status=failed."

bash -c "
  export http_proxy=\$HTTP_PROXY https_proxy=\$HTTPS_PROXY && \
  openclaw agent --local \
    --session-id '${EVAL_SESSION}' \
    --message '$(echo "$EVAL_PROMPT" | sed "s/'/'\\''/g")' \
    --timeout ${TIMEOUT}
"

EVAL_EXIT=$?

echo "=========================="
echo "Eval exit code: ${EVAL_EXIT}"
echo ""


# ── Copy eval session log to output dir ─────────────────────────────────────
echo ">>> Copying eval session log ..."
bash -c "
  cp /root/.openclaw/agents/main/sessions/${EVAL_SESSION}.jsonl '${FULL_OUTPUT}/session_eval_$$.jsonl' && \
  echo 'Copied: ${FULL_OUTPUT}/session_eval_$$.jsonl'
" 2>/dev/null || echo "Note: Could not copy eval session log from container"

# Print results
bash -c "
  echo '--- summary.json (quant) ---'
  cat '${FULL_OUTPUT}/summary.json' 2>/dev/null || echo '(not found)'
  echo ''
  echo '--- accuracy.json (eval) ---'
  cat '${FULL_OUTPUT}/accuracy.json' 2>/dev/null || echo '(not found)'
"

# ── Format session logs to markdown ──────────────────────────────────────────
echo ""
echo ">>> Formatting session logs to markdown ..."

bash -c "
  if command -v python3 &>/dev/null; then
    python3 -c \"
import json, sys
from pathlib import Path

def format_content_block(block):
    btype = block.get('type', '')
    if btype == 'text':
        return block.get('text', '')
    elif btype == 'thinking':
        return '**Thinking:** ' + block.get('thinking', '')
    elif btype == 'toolCall':
        name = block.get('name', 'unknown')
        args = block.get('arguments', {})
        args_str = json.dumps(args, indent=2, ensure_ascii=False) if isinstance(args, dict) else str(args)
        return f'**Tool:** \`{name}\`\n\`\`\`json\n{args_str}\n\`\`\`'
    elif btype == 'toolResult':
        content = block.get('content', [])
        if isinstance(content, str):
            return f'**Result:**\n\`\`\`\n{content}\n\`\`\`'
        parts = []
        for c in content if isinstance(content, list) else []:
            if isinstance(c, dict):
                parts.append(c.get('text', ''))
        return '\n'.join(parts)
    return f'[{btype}]'

def format_message(msg):
    message = msg.get('message', {})
    role = message.get('role', 'unknown')
    content = message.get('content', [])
    ts = msg.get('timestamp', '')
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        ts_str = dt.strftime('%H:%M:%S')
    except:
        ts_str = ts
    header = f'### [{ts_str}] {role.upper()}'
    if isinstance(content, str):
        body = content
    elif isinstance(content, list):
        parts = [format_content_block(b) for b in content if isinstance(b, dict)]
        body = '\n\n'.join(parts)
    else:
        body = str(content)
    return f'{header}\n\n{body}\n'

def format_session(jsonl_path, out_path):
    with open(jsonl_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    msgs = [r for r in records if r.get('type') == 'message']
    meta = next((r for r in records if r.get('type') == 'session'), {})
    sid = meta.get('id', jsonl_path.stem)
    lines = [f'# Session: {sid}', '', f'- **Session ID:** \`{sid}\`', f'- **Timestamp:** {meta.get(\"timestamp\",\"\")}', '']
    step = 'Step 1: Quantization' if 'quant' in sid.lower() else 'Step 2: Evaluation' if 'eval' in sid.lower() else 'Session'
    lines += [f'## {step}', '']
    for m in msgs:
        lines.append(format_message(m))
        lines.append('')
    out_path.write_text('\n'.join(lines))
    print(f'Written: {out_path}')

# Format both sessions
quant_jsonl = Path('${FULL_OUTPUT}/session_quant_$$.jsonl')
eval_jsonl = Path('${FULL_OUTPUT}/session_eval_$$.jsonl')
if quant_jsonl.exists():
    format_session(quant_jsonl, quant_jsonl.with_suffix('.md'))
if eval_jsonl.exists():
    format_session(eval_jsonl, eval_jsonl.with_suffix('.md'))
  \"
  fi
" 2>/dev/null || echo "Note: Could not format session logs (python3 not available in container)"



echo "=== Auto-Eval Pipeline Done ==="

# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Upload quantized model to HuggingFace (auto_quant only)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PIPELINE" == "auto_quant" && "$SKIP_UPLOAD" != "true" && "$SKIP_HF" != "true" ]]; then
    log_step "Step 5: Uploading model to HuggingFace"

    # Check if HF tokens are configured
    if [[ -z "${HF_TOKENS:-}" ]]; then
        log_warn "HF_TOKENS not set. Skipping HuggingFace upload."
        log_warn "Set HF_TOKENS in config.env to enable model upload."
    else
        # Determine repo name: {ModelName}-autoround-{Scheme}
        # e.g., Qwen3-8B-autoround-W4A16
        MODEL_SHORT="${MODEL_ID#*/}"  # Remove org prefix
        HF_REPO_NAME="${MODEL_SHORT}-autoround-${SCHEME}"

        log_info "Uploading model to HuggingFace as: $HF_REPO_NAME"
        python3 "$SCRIPT_DIR/upload_model_hf.py" \
            "$QUANT_MODEL_HOST_DIR" \
            "$HF_REPO_NAME" \
            --summary-json "$JOB_OUTPUT_DIR/quant_summary.json" \
            || log_warn "HuggingFace upload failed. Results are saved locally."
    fi
else
    if [[ "$PIPELINE" == "auto_quant" ]]; then
        log_info "Skipping HuggingFace upload (--skip-upload or --skip-hf)"
    fi
fi

