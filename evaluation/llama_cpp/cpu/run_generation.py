import argparse
import os
import re
import time
import json
import torch
import logging
from transformers import AutoConfig, AutoTokenizer
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
    AutoModel,
)
from transformers.utils import check_min_version
from intel_extension_for_transformers.transformers.utils import str2bool
from optimum.intel.generation.modeling import TSModelForCausalLM
from intel_extension_for_transformers.transformers import (
    MixedPrecisionConfig,
    SmoothQuantConfig,
    BitsAndBytesConfig,
    RtnConfig,
    AwqConfig,
    TeqConfig,
    GPTQConfig,
    AutoRoundConfig,
)

from config import eval_batch_size, tasks_shots_map, rename_tasks_map, results_template
import copy
import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "--request-file", type=str, default=None, help="model_name_or_path of peft model"
)
parser.add_argument("--use_neural_speed", action="store_true")
parser.add_argument("--batch_size", default=56, type=int, help="batch size num.")

args = parser.parse_args()

with open(args.request_file) as f:
    request_json = json.load(f)

print(request_json)

config = AutoConfig.from_pretrained(
    request_json["model"],
    use_cache=True,  # to use kv cache.
    trust_remote_code=True,
)

# chatglm
if config.model_type == "chatglm":
    AutoModelForCausalLM = AutoModel
# tokenizer
if config.model_type == "llama":
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(request_json["model"])
else:
    tokenizer = AutoTokenizer.from_pretrained(
        request_json["model"], trust_remote_code=True
    )

if request_json["quant_type"] == "GPTQ" and request_json["hardware"] == "cpu":
    config.quantization_config["disable_exllama"] = True

user_model = AutoModelForCausalLM.from_pretrained(
        request_json["model"],
        config=config,
        trust_remote_code=True,
        use_neural_speed=args.use_neural_speed
        )

from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate
pretrained = ',pretrained=' + request_json["model"]
commit_hash = request_json["revision"]
eval_args = "tokenizer=" + request_json["model"] + ",dtype=" + request_json["compute_dtype"] +",_commit_hash=" + \
                commit_hash + ",trust_remote_code=" + str(True)
if user_model is None:
    eval_args += pretrained
"""
if args.use_neural_speed:
    eval_args += pretrained
    q_conf = user_model.config.quantization_config
    if isinstance(q_conf, dict):
        q_algo = q_conf.get("quant_method", None)
    else:
        q_algo = q_conf.quant_method.value
    if q_algo.upper() in ["AWQ", "GPTQ", "AUTOROUND"]:
        eval_args += ",use_gptq=True"
"""
eval_tasks = []
eval_shots = []
for each in tasks_shots_map:
    eval_tasks.append(each)
    eval_shots.append(tasks_shots_map[each])

import fnmatch
from lm_eval import tasks
# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

task_names = pattern_match(eval_tasks, tasks.ALL_TASKS)

print(f"Selected Tasks: {task_names}")

results = evaluate(
        model="hf-causal",
        model_args=eval_args,
        user_model=user_model,
        batch_size=args.batch_size,
        tasks=task_names,
        model_format="neural_speed" if args.use_neural_speed else "torch",
    )

end_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')

final_results = copy.deepcopy(results_template)
final_results["config_general"]["job_id"] = request_json["job_id"]
final_results["config_general"]["start_time"] = request_json["job_start_time"]
final_results["config_general"]["end_time"] = end_time
final_results["config_general"]["model_name"] = request_json["model"]
final_results["config_general"]["model_dtype"] = request_json["precision"]
final_results["config_general"]["model_size"] = request_json["params"]

final_results["task_info"] = request_json
final_results["quantization_config"] = config.quantization_config

rename_results = {}
rename_versions = {}
for task in results["results"]:
    result = results["results"][task]
    version = results["versions"][task]
    if task in rename_tasks_map:
        name = f"harness|{rename_tasks_map[task]}|{tasks_shots_map[task]}"
    else:
        if task.startswith("hendrycksTest"):
            name = f"harness|{task}|{tasks_shots_map['hendrycksTest-*']}"
        else:
            name = f"harness|{task}|{tasks_shots_map[task]}"

    rename_results[name] = result
    rename_versions[name] = version
final_results.update({"results": rename_results, "versions": rename_versions})

# dumped = json.dumps(final_results, indent=2)


user_name = ""
model_path = request_json["model"]
if "/" in request_json["model"]:
    user_name = request_json["model"].split("/")[0]
    model_path = request_json["model"].split("/")[1]

result_path = f"results_{end_time}.json"

print("Creating results file")

with open(result_path, "w") as f:
    f.write(json.dumps(final_results, indent=4))
