#!/bin/bash
set -x

# ── Configuration ───────────────────────────────────────────────────
MAX_GIT_RETRIES=5        # more retries for parallel-push resilience
RETRY_BASE_SLEEP=3       # seconds; doubles each attempt

# ── Helpers ─────────────────────────────────────────────────────────
function prepare_repo() {
    workspace=$(pwd)
    git clone https://github.com/XuehaoSun/lb_eval.git lb_eval_backup
    git clone https://huggingface.co/datasets/Intel/ld_results
    cd lb_eval_backup
    git checkout main
}

function get_status() {
    if [ "${status}" ]; then
        cd "$workspace/lb_eval_backup/status"
        sed -i "s/\"status\":.*/\"status\": \"${status}\",/g" "${requestJson}"
    else
        # ── Push results ────────────────────────────────────────────
        model_name=$(echo "${requestJson}" | awk -F '_eval' '{print $1}')
        mkdir -p "$workspace/lb_eval_backup/results/${model_name}" \
                 "$workspace/ld_results/${model_name}"
        find "${BUILD_SOURCESDIRECTORY}/evaluation" -name "results_*.json" \
            -exec cp {} "results/${model_name}" \; 2>/dev/null || true

        rsync -avP "$workspace/lb_eval_backup/results/${model_name}/" \
                   "$workspace/ld_results/${model_name}/" 2>/dev/null || true

        # ── Remove processed request from requests/ ─────────────────
        # This ensures the request is not re-dispatched in the next
        # pipeline run.  The corresponding status/ file is kept
        # (already set to Finished by git-status-template).
        local req_file="$workspace/lb_eval_backup/requests/${requestJson}"
        if [ -f "${req_file}" ]; then
            echo "[git_update] Removing processed request: requests/${requestJson}"
            rm -f "${req_file}"
            # Clean up empty parent directory
            local req_dir
            req_dir=$(dirname "${req_file}")
            rmdir "${req_dir}" 2>/dev/null || true
        fi
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

    cd "$workspace/ld_results"
    git add . && git commit -m "${commitMessage}" || echo "[git_update] Nothing to commit"
    git remote set-url origin "https://lvkaokao:${HUGGINGFACE_TOKEN}@huggingface.co/datasets/Intel/ld_results"
    run_git_push
    rm -rf "${workspace}/ld_results"
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

function main() {
    prepare_repo
    get_status
    push_results
}

main
