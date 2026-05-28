#!/bin/bash
# Phase 1: Environment Setup
# Deterministic script — installs auto-round and dependencies with version control.
#
# Environment variables (set by parent auto.sh):
#   AUTO_ROUND_REF     — "latest" | version | branch | commit SHA
#   TRANSFORMERS_REF   — "auto" | version | branch | commit SHA
#   EXPORT_FORMAT      — "auto_round" | "llm_compressor"
#   EVAL_BACKEND       — "hf" | "vllm"
#   LM_EVAL_VERSION    — minimum lm_eval version (default: 0.4.10)
#   VLLM_VERSION       — vllm version (empty = latest)

set -euo pipefail

AUTO_ROUND_REF="${AUTO_ROUND_REF:-latest}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-auto}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto_round}"
EVAL_BACKEND="${EVAL_BACKEND:-hf}"
LM_EVAL_VERSION="${LM_EVAL_VERSION:-0.4.10}"
VLLM_VERSION="${VLLM_VERSION:-}"

echo "=== Phase 1: Environment Setup ==="
echo "  AUTO_ROUND_REF=${AUTO_ROUND_REF}"
echo "  TRANSFORMERS_REF=${TRANSFORMERS_REF}"
echo "  EXPORT_FORMAT=${EXPORT_FORMAT}"
echo "  EVAL_BACKEND=${EVAL_BACKEND}"

# ═══ Helper: resolve ref to pip install spec ═══
resolve_install_spec() {
    local pkg_name="$1"
    local git_url="$2"
    local ref="$3"

    case "$ref" in
        latest|"")
            echo "${pkg_name}" ;;
        [0-9]*)
            echo "${pkg_name}==${ref}" ;;
        *)
            echo "${pkg_name} @ git+${git_url}@${ref}" ;;
    esac
}

# ═══ Step 1: Install auto-round ═══
AR_SPEC=$(resolve_install_spec "auto-round" \
    "https://github.com/intel/auto-round.git" "${AUTO_ROUND_REF}")
echo "[setup_env] Installing: ${AR_SPEC}"
pip install ${AR_SPEC} 2>&1 | tail -5

# ═══ Step 2: Transformers version override ═══
if [ "$TRANSFORMERS_REF" != "auto" ]; then
    TF_SPEC=$(resolve_install_spec "transformers" \
        "https://github.com/huggingface/transformers.git" "${TRANSFORMERS_REF}")
    echo "[setup_env] Overriding transformers: ${TF_SPEC}"
    pip install ${TF_SPEC} 2>&1 | tail -3
fi

# ═══ Step 3: Install lm_eval ═══
pip install "lm-eval>=${LM_EVAL_VERSION}" 2>&1 | tail -3

# ═══ Step 4: llm_compressor export deps ═══
if [ "$EXPORT_FORMAT" == "llm_compressor" ]; then
    echo "[setup_env] Installing llm_compressor + compressed-tensors..."
    pip install "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@main" 2>&1 | tail -3
    pip install "compressed-tensors @ git+https://github.com/vllm-project/compressed-tensors.git@main" 2>&1 | tail -3
fi

# ═══ Step 5: vllm eval backend ═══
if [ "$EVAL_BACKEND" == "vllm" ]; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    echo "torch==${TORCH_VER}" > /tmp/torch_constraint.txt

    VLLM_SPEC="${VLLM_VERSION:+vllm==${VLLM_VERSION}}"
    VLLM_SPEC="${VLLM_SPEC:-vllm}"

    echo "[setup_env] Installing ${VLLM_SPEC} (torch constraint: ${TORCH_VER})..."
    pip install "${VLLM_SPEC}" -c /tmp/torch_constraint.txt 2>&1 | tail -5 || {
        echo "[WARN] Constraint install failed, trying --no-deps"
        pip install "${VLLM_SPEC}" --no-deps 2>&1 | tail -3
        pip install ray outlines msgspec partial-json compressed-tensors 2>&1 | tail -3
    }
    pip install "lm-eval[api]" 2>&1 | tail -3
fi

# ═══ Step 6: Auxiliary deps ═══
pip install loguru hf_transfer sentencepiece protobuf accelerate datasets 2>&1 | tail -3 || true

# ═══ Step 7: Model-specific dependency pre-flight ═══
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MODEL_ID:-}" ]; then
    echo "[setup_env] Running pre-flight dependency check for ${MODEL_ID}..."
    python3 "${SCRIPT_DIR}/preflight_deps.py" --model "${MODEL_ID}" --install || {
        echo "[WARN] Pre-flight dep install had issues (non-fatal, agent can retry)"
    }
fi

# ═══ Step 8: Verify ═══
echo ""
echo "[setup_env] Verification:"
python3 -c "
import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')
import auto_round; print(f'  auto_round={auto_round.__version__}')
import transformers; print(f'  transformers={transformers.__version__}')
import lm_eval; print(f'  lm_eval={lm_eval.__version__}')
" || { echo "FATAL: Environment verification failed"; exit 1; }

echo ""
echo "=== Phase 1: DONE ==="
