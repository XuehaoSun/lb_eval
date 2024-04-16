#!/bin/bash
set -xe

PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
    --config_name=*)
        config_name=$(echo $i | sed "s/${PATTERN}//")
        ;;
    *)
        echo "Parameter $i not recognized."
        exit 1
        ;;
    esac
done

git clone https://github.com/intel/intel-extension-for-transformers.git /intel-extension-for-transformers
shell_path=/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/quantization/run_generation.py

git clone https://github.com/XuehaoSun/lb_eval.git /lb_eval_backup
git checkout ${GITHUB_REF##*/}
cd /lb_eval_backup
git config --global user.email "xuehao.sun@intel.com"
git config --global user.name "Sun, Xuehao"
echo "Running" > status.txt
git remote set-url origin https://XuehaoSun:$(TOKEN)@github.com/XuehaoSun/lb_eval.git
git add . && git commit -m "Update status" && git push

cd /lb_eval
params=$(python parse_config.py $config_name)

cmd="python -u $shell_path --trust_remote_code $params"
${cmd}

cd /lb_eval_backup
echo "Finished" > status.txt
git add . && git commit -m "Update status" && git push
