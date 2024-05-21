#!/bin/bash
set -x

function prepare_repo() {
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login --token $HUGGINGFACE_TOKEN

    workspace=$(pwd)
    git clone https://github.com/XuehaoSun/lb_eval.git lb_eval_backup
    git clone https://huggingface.co/datasets/Intel/ld_results
    cd lb_eval_backup
    git checkout main
}

function get_status() {
    if [ ${status} ]; then
        cd $workspace/lb_eval_backup/status
        sed -i "s/\"status\":.*/\"status\": \"${status}\",/g" ${requestJson}
    else
        # push results
        model_name=$(echo ${requestJson} | awk -F '_eval' '{print $1}')
        mkdir -p $workspace/lb_eval_backup/results/${model_name} $workspace/ld_results/${model_name}
        find ${BUILD_SOURCESDIRECTORY}/evaluation -name "results_*.json" -exec cp {} results/${model_name} \;

        rsync -avP $workspace/lb_eval_backup/results/${model_name}/* $workspace/ld_results/${model_name}
    fi
}

function push_results() {
    cd $workspace/lb_eval_backup
    git config --global user.email "xuehao.sun@intel.com"
    git config --global user.name "Sun,Xuehao"
    git remote set-url origin https://XuehaoSun:"${TOKEN}"@github.com/XuehaoSun/lb_eval.git
    git add . && git commit -m "${commitMessage}"
    run_git_push
    rm -rf ${workspace}/lb_eval_backup

    cd $workspace/ld_results
    git add . && git commit -m "${commitMessage}"
    run_git_push
    rm -rf ${workspace}/ld_results
}

function run_git_push() {
    # retry 3 times in case of parallel push conflict
    n=0
    until [ "$n" -ge 3 ]; do
        git pull --rebase origin main && git push && break
        n=$((n + 1))
        sleep 3
    done
}

function main() {
    prepare_repo
    get_status
    push_results
}

main
