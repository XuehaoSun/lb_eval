#!/bin/bash
set -x

# ── Configuration ───────────────────────────────────────────────────
MAX_GIT_RETRIES=5        # more retries for parallel-push resilience
RETRY_BASE_SLEEP=3       # seconds; doubles each attempt
HF_REPO_NAME="Intel/ld_results"

# ── Helpers ─────────────────────────────────────────────────────────
function prepare_repo() {
    workspace=$(pwd)
    git clone --depth 1 --filter=blob:none --sparse \
        "https://github.com/XuehaoSun/lb_eval.git" lb_eval_backup
    cd lb_eval_backup
    git sparse-checkout set status results
}

function get_status() {
    if [ "${status}" ]; then
        cd "$workspace/lb_eval_backup/status"
        sed -i "s/\"status\":.*/\"status\": \"${status}\",/g" "${requestJson}"
    else
        # ── Push results ────────────────────────────────────────────
        model_name=$(echo "${requestJson}" | awk -F '_eval' '{print $1}')
        mkdir -p "$workspace/lb_eval_backup/results/${model_name}"
        find "${BUILD_SOURCESDIRECTORY}/evaluation" -name "results_*.json" \
            -exec cp {} "results/${model_name}" \; 2>/dev/null || true
        push_hf_dataset
        # NOTE: pending_requests/ files are intentionally kept — the dispatcher
        # uses status/ (not pending_requests/) for scheduling decisions.
    fi
}

function push_results() {
    cd "$workspace/lb_eval_backup"
    git config --global user.email "xuehao.sun@intel.com"
    git config --global user.name "Sun,Xuehao"
    git remote set-url origin "https://XuehaoSun:${TOKEN}@github.com/XuehaoSun/lb_eval.git"
    git add . && git commit -m "${commitMessage}" || echo "[git_update] Nothing to commit"
    run_git_push
    rm -rf "${workspace}/lb_eval_backup"
}

function run_git_push() {
    # Retry with exponential backoff for parallel-push conflicts
    local n=0
    local sleep_sec=${RETRY_BASE_SLEEP}
    until [ "$n" -ge ${MAX_GIT_RETRIES} ]; do
        git pull --rebase origin main && git push && return 0
        n=$((n + 1))
        echo "[git_update] Push failed (attempt ${n}/${MAX_GIT_RETRIES}), retrying in ${sleep_sec}s ..."
        sleep ${sleep_sec}
        sleep_sec=$((sleep_sec * 2))
    done
    echo "[git_update] ERROR: push failed after ${MAX_GIT_RETRIES} attempts"
    return 1
}

function push_hf_dataset() {
    local local_data_path="$workspace/lb_eval_backup/results/${model_name}"
    local target_data_path="${model_name}"
    if [ ! -d "${local_data_path}" ]; then
        echo "[git_update] Skipping HF upload: ${local_data_path} does not exist"
        return 0
    fi
    hf auth login --token ${HUGGINGFACE_TOKEN}
    hf upload ${HF_REPO_NAME} "${local_data_path}" "${target_data_path}" --repo-type=dataset
}

function main() {
    prepare_repo
    get_status
    push_results
}

main
