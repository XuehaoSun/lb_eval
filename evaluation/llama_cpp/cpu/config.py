EVAL_SCRIPT = "evaluation/eval_normal.sh"

"""
tasks_shots_map = {"winogrande": 0,
        "arc_easy": 0,
        "arc_challenge": 0,
        "truthfulqa_mc":0,
        "hellaswag": 0,
        "hendrycksTest-*": 0,
        "lambada_openai": 0,
        "piqa": 0,
        "openbookqa": 0,
        "boolq": 0}
"""

tasks_shots_map = {"winogrande": 0,
        "arc_challenge": 0,
        "truthfulqa_mc2":0,
        "piqa": 0,}

rename_tasks_map = {"arc_easy": "arc:easy",
        "arc_challenge": "arc:challenge",
        "truthfulqa_mc": "truthfulqa:mc",
        "truthfulqa_mc2": "truthfulqa:mc2",
        "truthfulqa_mc1": "truthfulqa:mc1",
        "lambada_openai": "lambada:openai"}

eval_batch_size=56
result_path="llm_evaluation_results.json"


results_template = {
  "config_general": {
    "lighteval_sha": "no", # not depend itrex version
    "num_few_shot_default": None,
    "num_fewshot_seeds": None,
    "override_batch_size": None,
    "max_samples": None,
    "job_id": "",
    "start_time": 145237.069553293,
    "end_time": 155718.702369908,
    "total_evaluation_time_secondes": "",
    "model_name": "Intel/neural-chat-7b-v3-1",
    "model_sha": "",
    "model_dtype": "4bit",
    "model_size": "4.24 GB"
  },
  "results": {}
  }
