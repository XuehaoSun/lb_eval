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

export http_proxy=http://child-ir.intel.com:912
export https_proxy=http://child-ir.intel.com:912

cmd="python run_generation.py --request-file /lb_eval/requests/$config_name"
eval ${cmd}
