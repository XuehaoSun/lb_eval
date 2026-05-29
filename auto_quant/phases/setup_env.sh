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
uv pip install ${AR_SPEC} 2>&1 | tail -5

# ═══ Step 2: Transformers version override ═══
if [ "$TRANSFORMERS_REF" != "auto" ]; then
    TF_SPEC=$(resolve_install_spec "transformers" \
        "https://github.com/huggingface/transformers.git" "${TRANSFORMERS_REF}")
    echo "[setup_env] Overriding transformers: ${TF_SPEC}"
    uv pip install ${TF_SPEC} 2>&1 | tail -3
fi

# ═══ Step 3: Install lm_eval ═══
uv pip install "lm-eval>=${LM_EVAL_VERSION}" 2>&1 | tail -3

# ═══ Step 4: llm_compressor export deps ═══
if [ "$EXPORT_FORMAT" == "llm_compressor" ]; then
    echo "[setup_env] Installing llm_compressor + compressed-tensors..."
    uv pip install "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@main" 2>&1 | tail -3
    uv pip install "compressed-tensors @ git+https://github.com/vllm-project/compressed-tensors.git@main" 2>&1 | tail -3
fi

# ═══ Step 5: vllm eval backend ═══
if [ "$EVAL_BACKEND" == "vllm" ]; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    echo "torch==${TORCH_VER}" > /tmp/torch_constraint.txt

    VLLM_SPEC="${VLLM_VERSION:+vllm==${VLLM_VERSION}}"
    VLLM_SPEC="${VLLM_SPEC:-vllm}"

    echo "[setup_env] Installing ${VLLM_SPEC} (torch constraint: ${TORCH_VER})..."
    uv pip install "${VLLM_SPEC}" -c /tmp/torch_constraint.txt 2>&1 | tail -5 || {
        echo "[WARN] Constraint install failed, trying --no-deps"
        uv pip install "${VLLM_SPEC}" --no-deps 2>&1 | tail -3
        uv pip install ray outlines msgspec partial-json compressed-tensors 2>&1 | tail -3
    }
    uv pip install "lm-eval[api]" 2>&1 | tail -3
fi

# ═══ Step 6: Auxiliary deps ═══
uv pip install loguru hf_transfer sentencepiece protobuf accelerate datasets 2>&1 | tail -3 || true

# ═══ Step 6.5: Verify torch+CUDA driver compatibility ═══
# If torch was pre-installed with a CUDA version newer than the driver supports,
# reinstall a compatible version automatically.
echo "[setup_env] Checking torch/CUDA driver compatibility..."
python3 - <<'PYEOF'
import subprocess, sys, re

try:
    import torch
except ImportError:
    print("[setup_env] torch not installed, installing default...")
    subprocess.run(["uv", "pip", "install", "torch"], check=True)
    import torch

torch_version = torch.__version__
cuda_available = torch.cuda.is_available()

if cuda_available:
    print(f"[setup_env] torch={torch_version}, CUDA available — OK")
    sys.exit(0)

# CUDA not available — check if it's a driver mismatch
# Try to get the driver-supported CUDA version via nvidia-smi
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print("[setup_env] No NVIDIA GPU detected, CPU-only mode")
        sys.exit(0)
    driver_version = result.stdout.strip().split('\n')[0]
    print(f"[setup_env] NVIDIA driver: {driver_version}")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("[setup_env] nvidia-smi not found, assuming no GPU")
    sys.exit(0)

# Get driver-supported CUDA version
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    if not cuda_match:
        print("[setup_env] Could not determine driver CUDA version, skipping")
        sys.exit(0)
    driver_cuda = cuda_match.group(1)  # e.g. "12.1"
    print(f"[setup_env] Driver supports CUDA: {driver_cuda}")
except Exception:
    sys.exit(0)

# Determine which torch CUDA build to install
# Map driver CUDA major.minor to PyTorch index URL suffix
major_minor = driver_cuda.split('.')
cuda_major = int(major_minor[0])
cuda_minor = int(major_minor[1]) if len(major_minor) > 1 else 0

# PyTorch available CUDA builds: cu118, cu121, cu124, cu126
if cuda_major < 11 or (cuda_major == 11 and cuda_minor < 8):
    cu_tag = "cu118"
elif cuda_major == 11:
    cu_tag = "cu118"
elif cuda_major == 12 and cuda_minor < 1:
    cu_tag = "cu118"
elif cuda_major == 12 and cuda_minor < 4:
    cu_tag = "cu121"
elif cuda_major == 12 and cuda_minor < 6:
    cu_tag = "cu124"
else:
    cu_tag = "cu126"

index_url = f"https://download.pytorch.org/whl/{cu_tag}"
print(f"[setup_env] torch {torch_version} CUDA mismatch with driver (CUDA {driver_cuda})")
print(f"[setup_env] Reinstalling torch with {cu_tag} from {index_url}...")

# Reinstall torch + torchaudio + torchvision with correct CUDA
subprocess.run(
    ["uv", "pip", "install", "--reinstall", "torch", "torchaudio", "torchvision",
     "--index-url", index_url],
    check=False
)

# Verify
import importlib
importlib.invalidate_caches()
result = subprocess.run(
    [sys.executable, "-c", "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"],
    capture_output=True, text=True
)
print(f"[setup_env] After reinstall: {result.stdout.strip()}")
if "cuda=True" not in result.stdout:
    print("[WARN] CUDA still not available after torch reinstall")
PYEOF

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
