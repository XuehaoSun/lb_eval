import json
import re
import os
from collections import defaultdict


def findAllFile(base):
    all_files = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root, f)
                all_files.append(fullname)
    return all_files

def already_submitted_models(requested_models_dir: str) -> set[str]:
    depth = 1
    file_names = []
    users_to_submission_dates = defaultdict(list)

    for root, _, files in os.walk(requested_models_dir):
        current_depth = root.count(os.sep) - requested_models_dir.count(os.sep)
        if current_depth == depth:
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(root, file), "r") as f:
                    info = json.load(f)
                    # {quant_type}_{precision}_{weight_dtype}_{compute_dtype}.json
                    quant_type = info.get("quant_type", "None")
                    weight_dtype = info.get("weight_dtype", "None")
                    compute_dtype = info.get("compute_dtype", "None")
                    file_names.append(f"{info['model']}_{info['revision']}_{quant_type}_{info['precision']}_{weight_dtype}_{compute_dtype}")

                    # Select organisation
                    if info["model"].count("/") == 0 or "submitted_time" not in info:
                        continue

                    try:
                        organisation, _ = info["model"].split("/")
                    except:
                        print(info["model"])
                        organisation = "local" # temporary "local"
                    users_to_submission_dates[organisation].append(info["submitted_time"])

    return file_names, users_to_submission_dates


def main():
    results_path = '../../results'
    status_path = '../../status'

    result_files = findAllFile(results_path)
    status_files = findAllFile(status_path)
    print(len(result_files))
    print(len(status_files))

    results_dup = []
    for result_file in result_files:
        with open(result_file) as f:
            r_data = json.load(f)
        # We manage the legacy config format
        config = r_data.get("config_general")
        # Precision
        model = config.get('model_name')

        precision = config.get("precision")
        quant_type = config.get("quant_type")
        weight_dtype = config.get("weight_dtype", "int4")
        compute_dtype = r_data["task_info"].get("compute_dtype")
        double_quant = r_data["quantization_config"].get("bnb_4bit_use_double_quant", False)
        group_size = r_data["quantization_config"].get("group_size", -1)
        revision = r_data["task_info"].get("revision")

        request_file = f"{model}_{revision}_{quant_type}_{precision}_{weight_dtype}_{compute_dtype}"
        if request_file not in results_dup:
            results_dup.append(request_file)
    print(results_dup[0])


    status_dup = []
    requested_models, users_to_submission_dates = already_submitted_models(status_path)
    print(requested_models)
    print(len(requested_models))
    print(requested_models[0])
    print(users_to_submission_dates)

    for each in requested_models:
        if each not in results_dup:
            print(each)




main()
