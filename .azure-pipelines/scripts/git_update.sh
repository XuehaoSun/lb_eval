#!/bin/bash
set -x

git clone https://github.com/XuehaoSun/lb_eval.git lb_eval_backup
cd lb_eval_backup
git checkout main
cd lb_eval_backup/status
sed -i "s/\"status\":.*/\"status\": \"${status}\"/g" ${requestJson}

git config --global user.email "xuehao.sun@intel.com"
git config --global user.name "Sun,Xuehao"
git remote set-url origin https://XuehaoSun:"${TOKEN}"@github.com/XuehaoSun/lb_eval.git
git add . && git commit -m "${commitMessage}"

# retry 3 times in case of parallel push conflict
n=0
until [ "$n" -ge 3 ]
do
  git pull && git push && break
  n=$((n+1))
  sleep 3
done

rm -rf lb_eval_backup