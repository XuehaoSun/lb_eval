#!/bin/bash
set -x

workspace=$(pwd)
git clone https://github.com/XuehaoSun/lb_eval.git lb_eval_backup
cd lb_eval_backup
git checkout main

if [ ${status} ]; then
    cd status
    sed -i "s/\"status\":.*/\"status\": \"${status}\",/g" ${requestJson}
else
    # push results
    username=$(echo ${requestJson} | cut -d'/' -f2 )
    mkdir -p results/${username}
    find ${BUILD_SOURCESDIRECTORY}/evaluation -name "results_*.json" -exec cp {} results/${username} \;
fi

git config --global user.email "xuehao.sun@intel.com"
git config --global user.name "Sun,Xuehao"
git remote set-url origin https://XuehaoSun:"${TOKEN}"@github.com/XuehaoSun/lb_eval.git
git add . && git commit -m "${commitMessage}"

# retry 3 times in case of parallel push conflict
n=0
until [ "$n" -ge 3 ]
do
  git pull --rebase origin main && git push && break
  n=$((n+1))
  sleep 3
done

rm -rf ${workspace}/lb_eval_backup