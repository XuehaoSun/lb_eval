import argparse
import base64
import json
import sys

import requests

# === Pipeline configuration ===
ORG_NAME = "lpot-inc"
PROJECT_NAME = "lb_eval"
PIPELINE_ID = 78  # https://dev.azure.com/lpot-inc/lb_eval/_build?definitionId=78
BRANCH_NAME = "main"


def build_headers(pat: str) -> dict:
    # Azure DevOps requires a Base64-encoded PAT for Basic auth
    auth_base64 = base64.b64encode(f":{pat}".encode()).decode()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Basic {auth_base64}",
    }


def build_payload(variables: dict, branch: str) -> dict:
    # Pipeline Run API expects variables as {"name": {"value": "..."}}
    return {
        "variables": {key: {"value": val} for key, val in variables.items()},
        "resources": {"repositories": {"self": {"refName": f"refs/heads/{branch}"}}},
    }


def trigger_pipeline(pat: str, scripts_path: str, gpu_id: str, request_json: str) -> None:
    url = f"https://dev.azure.com/{ORG_NAME}/{PROJECT_NAME}" f"/_apis/pipelines/{PIPELINE_ID}/runs?api-version=7.1"
    headers = build_headers(pat)
    variables = {
        "SCRIPTS_PATH": scripts_path,
        "GPU_ID": gpu_id,
        "REQUEST_JSON": request_json,
    }
    payload = build_payload(variables, BRANCH_NAME)

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code in (200, 201):
        run_info = response.json()
        print("Pipeline triggered successfully.")
        print(f"Run ID:  {run_info['id']}")
        print(f"Run URL: {run_info['_links']['web']['href']}")
    else:
        print(f"Failed to trigger pipeline (HTTP {response.status_code}).")
        print(f"Error details: {response.text}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger an Azure DevOps pipeline run via the REST API.")
    parser.add_argument(
        "--pat",
        required=True,
        help="Personal Access Token (PAT) for Azure DevOps authentication.",
    )
    parser.add_argument(
        "--scripts-path",
        default="auto_quant",
        help="Value for the SCRIPTS_PATH pipeline variable.",
    )
    parser.add_argument(
        "--gpu-id",
        default="NVIDIA GeForce RTX 4090",
        help="Value for the GPU_ID pipeline variable.",
    )
    parser.add_argument(
        "--request-json",
        default="request.json",
        help="Value for the REQUEST_JSON pipeline variable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trigger_pipeline(args.pat, args.scripts_path, args.gpu_id, args.request_json)
