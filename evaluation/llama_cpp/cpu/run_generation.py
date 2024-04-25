import argparse
import os
import re
import time
import json
import torch
import logging

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

from llama_cpp_lm import WrapperGGUFLM

pretrained = 'gguf_model=' + request_json["model"] + ",ftype=" + request_json["gguf_ftype"]
commit_hash = request_json["revision"]
model_args = pretrained + ",dtype=" + request_json["compute_dtype"] +",_commit_hash=" + commit_hash

print(f"Initializing {request_json['model']} model, with arguments: {model_args}")
gguf_lm = WrapperGGUFLM.create_from_arg_string(
        model_args,
        {
            "batch_size": 1, # llama.cpp only support bs=1
            "max_batch_size": 1,
            "device": request_json["hardware"],
            })

"""
data = {'prompt': 'Sarah was a much better surgeon than Maria so Sarah always got the easier cases.', 'suffix': None, 'max_tokens': 1, 'temperature': 0.0, 'top_p': 0.95, 'min_p': 0.05, 'echo': True, 'stop': None, 'stream': False, 'logprobs': 10, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'logit_bias': None, 'seed': None, 'model': None, 'top_k': 40, 'repeat_penalty': 1.1, 'mirostat_mode': 0, 'mirostat_tau': 5.0, 'mirostat_eta': 0.1, 'grammar': None}

print(gguf_lm.model(**data))
"""

from lm_eval.evaluator import simple_evaluate


eval_tasks = []
eval_shots = []
for each in tasks_shots_map:
    eval_tasks.append(each)
    eval_shots.append(tasks_shots_map[each])

import fnmatch
from lm_eval import tasks
from lm_eval.tasks import TaskManager

task_manager = TaskManager("INFO")

# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

task_names = pattern_match(eval_tasks, task_manager.all_tasks)

print(f"Selected Tasks: {task_names}")

results = simple_evaluate(
    gguf_lm,
    model_args=model_args,
    tasks=task_names,
    batch_size=1,
    device=request_json["hardware"],
    write_out=True,
)

results = {}
results["results"] = eval_results["results"]
results["versions"] = eval_results["versions"]
results["n-shot"] = eval_results["n-shot"]
results["date"] = eval_results["date"]
results["config"] = eval_results["config"]
print(results)


end_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')

final_results = copy.deepcopy(results_template)
final_results["config_general"]["job_id"] = request_json["job_id"]
final_results["config_general"]["start_time"] = request_json["job_start_time"]
final_results["config_general"]["end_time"] = end_time
final_results["config_general"]["model_name"] = request_json["model"]
final_results["config_general"]["model_dtype"] = request_json["precision"]
final_results["config_general"]["model_size"] = request_json["params"]

final_results["task_info"] = request_json
quantization_config = {"quant_method": request_json["quant_type"],
        "ftype": request_json["gguf_ftype"],}
final_results["quantization_config"] = quantization_config

rename_results = {}
rename_versions = {}
for task in results["results"]:
    result = results["results"][task]
    version = results["versions"].get(task, None)
    if task in rename_tasks_map:
        name = f"harness|{rename_tasks_map[task]}|{tasks_shots_map[task]}"
    else:
        if task.startswith("mmlu"):
            name = f"harness|{task}|{tasks_shots_map['mmlu']}"
        else:
            name = f"harness|{task}|{tasks_shots_map[task]}"

    rename_results[name] = result
    rename_versions[name] = version

final_results.update({"results": rename_results, "versions": rename_versions,
    "n-shot": results["n-shot"], "date": results["date"], "config": results["config"]})


user_name = ""
model_path = request_json["model"]
if "/" in request_json["model"]:
    user_name = request_json["model"].split("/")[0]
    model_path = request_json["model"].split("/")[1]

result_path = f"results_{end_time}.json"

print("Creating results file")

with open(result_path, "w") as f:
    f.write(json.dumps(final_results, indent=4))
