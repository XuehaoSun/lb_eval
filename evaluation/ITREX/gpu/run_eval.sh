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

cd /lb_eval/evaluation/ITREX/cpu
# params=$(python /lb_eval/evaluation/parse_config.py /lb_eval/requests/$config_name)
# cmd="python -u run_generation.py --trust_remote_code $params"
cmd="python run_generation.py --request-file /lb_eval/requests/$config_name"
eval ${cmd}

# update results
# todo
