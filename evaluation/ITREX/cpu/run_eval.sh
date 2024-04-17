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

cd /lb_eval
params=$(python parse_config.py $config_name)

cmd="python -u $shell_path --trust_remote_code $params"
${cmd}
