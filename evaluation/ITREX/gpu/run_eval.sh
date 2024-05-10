#!/bin/bash
set -xe

export TOKENIZERS_PARALLELISM=false

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

cd /lb_eval/evaluation/ITREX/gpu

export http_proxy=http://child-jf.intel.com:912
export https_proxy=http://child-jf.intel.com:912
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/dataset/hf_cache/

cmd="python -m pip install optimum==1.19.1"
eval ${cmd}
cmd="python -m pip install --no-cache-dir auto-gptq==0.7.1 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/"
eval ${cmd}
cmd="python -m pip install --no-cache-dir einops==0.8.0"
eval ${cmd}

cmd="python run_generation.py --request-file /lb_eval/requests/$config_name"
eval ${cmd}
