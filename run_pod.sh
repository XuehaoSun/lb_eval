#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env (if present) ────────────────────────────────────────────────────
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "[run.sh] Loading env from $ENV_FILE"
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
fi

DOCKER_FILE="docker/Dockerfile"
IMAGE_NAME="${IMAGE_NAME:-cuda-openclaw:local}"
CONTAINER_NAME="${CONTAINER_NAME:-cuda-openclaw-new}"
BUILD_IMAGE=true

for arg in "$@"; do
    case "$arg" in
        --no-build) BUILD_IMAGE=false ;;
        *) echo "Usage: ./run_pod.sh [--no-build]"; exit 1 ;;
    esac
done

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "[run_pod.sh] Building image: $IMAGE_NAME ..."
    docker build \
        --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
        --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
        -t "$IMAGE_NAME" -f "$DOCKER_FILE" "$SCRIPT_DIR"
    echo "[run_pod.sh] Build done."
fi


# ── Remove old container with same name ───────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[run.sh] Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[run.sh] Starting container: $CONTAINER_NAME"

# Optional host output directory — mount into container for persistence
VOLUME_ARGS=""
HOST_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-./}"
if [[ -n "${HOST_OUTPUT_DIR:-}" ]]; then
    mkdir -p "$HOST_OUTPUT_DIR"
    CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-/root/.openclaw/workspace/quantized}"
    VOLUME_ARGS="-v ${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}"
    echo "[run.sh] Mounting volume: ${HOST_OUTPUT_DIR} -> ${CONTAINER_OUTPUT_DIR}"
fi


# Allow caller to inject extra docker args (e.g., GPU constraints)
DOCKER_EXTRA_ARGS="${DOCKER_EXTRA_ARGS:-}"


# MINIMAX_API_KEY
# HF_TOKENS

docker run -d \
    --name "$CONTAINER_NAME" \
    --network host \
    --gpus all \
    -e MINIMAX_API_KEY="${MINIMAX_API_KEY:-}" \
    -e HTTP_PROXY="${HTTP_PROXY:-}" \
    -e HTTPS_PROXY="${HTTPS_PROXY:-}" \
    -e http_proxy="${HTTP_PROXY:-}" \
    -e https_proxy="${HTTPS_PROXY:-}" \
    $VOLUME_ARGS \
    $DOCKER_EXTRA_ARGS \
    "$IMAGE_NAME" \
    sleep infinity

echo "[run.sh] Container $CONTAINER_NAME is running."
echo "[run.sh] Use: docker exec -it $CONTAINER_NAME bash"
