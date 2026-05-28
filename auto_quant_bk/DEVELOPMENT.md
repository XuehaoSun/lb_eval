# auto_quant Developer Guide

This document describes the `tasks/lb_eval/auto_quant` workflow: required environment variables, script responsibilities, runtime flow, and artifact layout.

## 1. Scope

The `auto_quant` workflow is the container-side entrypoint used to:

1. Read a quantization/evaluation request JSON
2. Run quantization with OpenClaw
3. Run evaluation with OpenClaw
4. Upload the quantized model to Hugging Face
5. Write result artifacts back to the `lb_eval` GitHub repository

The current implementation separates:

- **Model output dir**: only quantized model files
- **Runtime output dir**: logs, prompts, request copies, sessions, summaries, evaluation output, venv

This avoids runtime artifacts being removed when quantization/export rewrites the model directory.

## 2. Main scripts

| Script | Purpose |
| --- | --- |
| `run_pod.sh` | Build the image and start the container |
| `auto.sh` | Main entrypoint inside the container |
| `format_sessions.py` | Convert OpenClaw session JSONL files to Markdown |
| `upload_model_hf.py` | Upload quantized model directory to Hugging Face |
| `hf_shared_ledger.py` | Shared git-backed ledger for multi-machine HF account coordination |
| `upload_results_github.py` | Copy runtime artifacts into the `lb_eval` GitHub results repo and push |

## 3. Typical run sequence

### 3.1 Start the container

Run on the host:

```bash
cd /root/leaderboard_Agent/tasks/lb_eval
bash run_pod.sh
```

If needed, configure the container name, image name, or output mount via the environment before running `run_pod.sh`.

### 3.2 Run auto_quant inside the container

```bash
cd /root/leaderboard_Agent/tasks/lb_eval/auto_quant
bash auto.sh Qwen3-0.6B_quant_request_False_W4A16_4bit_int4.json
```

Optional flags:

```bash
bash auto.sh <task.json> --dry-run
bash auto.sh <task.json> --skip-upload
bash auto.sh <task.json> --skip-hf
bash auto.sh <task.json> --skip-github
```

## 4. Runtime flow

`auto.sh` performs the following steps:

1. Load `config.env`
2. Parse the request JSON and derive:
   - pipeline type
   - model id
   - quantization scheme
   - quant/eval GPU counts
3. Resolve OpenClaw workspace and session directories
4. Resolve:
   - **model output dir**
   - **runtime output dir**
5. Start unified logging to `logs/auto.log`
6. Copy the input request JSON into the runtime output dir
7. If quantization has not already succeeded:
   - generate quant prompt
   - run `openclaw agent --local`
   - expect `quant_summary.json`
8. Copy quant session JSONL from the OpenClaw sessions directory
9. If evaluation has not already succeeded and quantization succeeded:
   - generate an eval-script prompt
   - use `auto_eval` for the quantization+evaluation flow, or `auto_eval_vllm` for evaluation-only requests
   - run `openclaw agent --local` to write `evaluate.sh`
   - let OpenClaw execute the generated `evaluate.sh` within the same autonomous task
   - expect `accuracy.json`
   - expect raw `lm_eval_results/` written via `lm_eval --output_path`
10. Copy eval session JSONL
11. Format copied session JSONL files into Markdown
12. If enabled and quantization succeeded:
   - upload model directory to Hugging Face
13. If enabled:
   - upload runtime artifacts to the GitHub results repo
14. Exit non-zero if any required stage failed

### Important behavior

- All shell output is written to one log: `logs/auto.log`
- Step failures are recorded and the script continues when possible
- The final exit code still reflects pipeline failure
- Prompt files are saved as runtime artifacts, but prompt text is generated in memory first
- New evaluation runs generate a first-class runtime artifact: `evaluate.sh`
- GitHub artifact upload now includes both `quantize.py` and `evaluate.sh` when present
- `auto.log` now follows the OpenClaw session JSONL during execution and prints incremental user / assistant / tool summaries
- `auto.log` also tails fixed execution logs (`logs/quant_exec.log`, `logs/eval_exec.log`) when OpenClaw streams script stdout/stderr via `tee`
- `auto.log` prints the generated `quantize.py` / `evaluate.sh` artifacts (truncated when very long)
- Quantization and evaluation remain OpenClaw-owned; `auto.sh` only adds live session visibility, script display, and session artifact export

## 5. Directory layout

Assume:

- `OUTPUT_DIR=/root/.openclaw/workspace/quantized`
- model slug = `Qwen_Qwen3-0.6B`
- scheme = `W4A16`

Then:

```text
Model output dir:
/root/.openclaw/workspace/quantized/Qwen_Qwen3-0.6B-W4A16

Runtime output dir:
/root/.openclaw/workspace/quantized/runs/Qwen_Qwen3-0.6B-W4A16
```

### Model output dir

Expected contents:

- quantized weights
- tokenizer/config files
- model metadata required for inference/upload

This directory is the source for Hugging Face model upload.

### Runtime output dir

Expected contents:

- `logs/auto.log`
- `logs/quant_prompt.txt`
- `logs/eval_prompt.txt`
- `request.json`
- `quant_summary.json`
- `accuracy.json`
- `session_quant_<pid>.jsonl`
- `session_eval_<pid>.jsonl`
- `session_quant_<pid>.md`
- `session_eval_<pid>.md`
- `venv/` created and reused by the OpenClaw tasks when needed

This directory is the source for GitHub artifact upload.

## 6. Environment variables

The workflow reads `tasks/lb_eval/auto_quant/config.env`.

Do **not** put real secrets in documentation or commit them to git. Use local-only values or secret injection in your runtime.

### 6.1 Hugging Face upload

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_TOKENS` | Yes, if HF upload is enabled | Comma-separated Hugging Face tokens |
| `HF_UPLOAD_ORGS` | Recommended | Comma-separated org/user names matching the tokens |
| `HF_ACCOUNT_IDS` | Optional | Stable logical account ids for ledger accounting |
| `HF_ACCOUNT_CAPACITY_GB` | Optional | Per-account capacity assumption used by the ledger |
| `HF_USAGE_FILE` | Optional | Local JSON ledger path for non-shared mode |

### 6.2 Shared Hugging Face ledger

Used when multiple workers or machines share the same HF accounts.

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_SHARED_LEDGER_ENABLED` | Optional | Enable shared ledger mode |
| `HF_SHARED_LEDGER_REPO` | Yes in shared mode | Git-backed ledger repo, usually a HF dataset repo or other git remote |
| `HF_SHARED_LEDGER_TOKEN` | Usually yes | Token used to clone/pull/push the ledger repo |
| `HF_SHARED_LEDGER_BRANCH` | Optional | Branch used by the ledger repo |
| `HF_SHARED_LEDGER_CLONE_DIR` | Optional | Local cache/clone path for the shared ledger repo |
| `HF_SHARED_LEDGER_RESERVATION_TTL_SECONDS` | Optional | Reservation expiry time |
| `HF_SHARED_LEDGER_GIT_USER_NAME` | Optional | Git author name for ledger commits |
| `HF_SHARED_LEDGER_GIT_USER_EMAIL` | Optional | Git author email for ledger commits |

### 6.3 GitHub results upload

| Variable | Required | Purpose |
| --- | --- | --- |
| `GIT_TOKEN` | Yes, if GitHub upload is enabled | Token used for authenticated push |
| `GIT_REPO` | Yes when repo must be cloned | Remote `lb_eval` repo URL |
| `GIT_BRANCH` | Optional | Target branch |
| `GIT_RESULTS_REPO_DIR` | Optional | Existing local clone of the results repo |
| `GIT_RESULTS_CLONE_DIR` | Optional | Clone destination when no local repo is provided |
| `GIT_USER_NAME` | Optional | Git author name for result commits |
| `GIT_USER_EMAIL` | Optional | Git author email for result commits |

### 6.4 OpenClaw / runtime

| Variable | Required | Purpose |
| --- | --- | --- |
| `MINIMAX_API_KEY` | Usually yes | API key used by OpenClaw agent |
| `HTTP_PROXY` | Optional | HTTP proxy |
| `HTTPS_PROXY` | Optional | HTTPS proxy |
| `OPENCLAW_WORKSPACE_DIR` | Recommended | OpenClaw portable workspace root; used to resolve skills |
| `OPENCLAW_SESSIONS_DIR` | Recommended | Directory containing OpenClaw session JSONL files |

If `OPENCLAW_WORKSPACE_DIR` is empty, `auto.sh` auto-detects it in this order:

1. `/root/leaderboard_Agent/auto_quant/openclaw_home/workspace`
2. `/root/leaderboard_Agent/tasks/lb_eval/openclaw_config/workspace`
3. `/root/.openclaw/workspace`

### 6.5 Pipeline defaults

| Variable | Required | Purpose |
| --- | --- | --- |
| `METHOD` | Optional | Quantization method label passed into the prompt |
| `EXPORT_FORMAT` | Optional | Export format label passed into the prompt |
| `DEVICE` | Optional | Runtime device, usually `cuda` |
| `DEVICE_INDEX` | Optional | Device index used in the eval summary |
| `TIMEOUT` | Optional | OpenClaw agent timeout |
| `EVAL_TASKS` | Optional | Evaluation tasks, e.g. `piqa,mmlu,hellaswag,gsm8k` |
| `EVAL_BATCH_SIZE` | Optional | Evaluation batch size |

### 6.6 Output directories

| Variable | Required | Purpose |
| --- | --- | --- |
| `OUTPUT_DIR` | Optional | Base directory for quantized model export |
| `RUNTIME_OUTPUT_BASE_DIR` | Optional | Base directory for runtime artifacts |

Defaults:

- `OUTPUT_DIR=/root/.openclaw/workspace/quantized`
- `RUNTIME_OUTPUT_BASE_DIR=/root/.openclaw/workspace/quantized/runs`

### 6.7 Container startup

Used mainly by `run_pod.sh`.

| Variable | Required | Purpose |
| --- | --- | --- |
| `IMAGE_NAME` | Optional | Docker image name |
| `CONTAINER_NAME` | Optional | Docker container name |
| `CONTAINER_OUTPUT_DIR` | Optional | Container-side path used for output volume mount |
| `DOCKER_EXTRA_ARGS` | Optional | Extra arguments passed to `docker run` |

## 7. Request JSON expectations

`auto.sh` expects a JSON file containing fields such as:

| Field | Purpose |
| --- | --- |
| `script` | Determines whether the pipeline is `auto_quant` or `auto_eval` |
| `job_type` | Used as an additional signal for pipeline selection |
| `model` | Model id, e.g. `Qwen/Qwen3-0.6B` |
| `revision` | Model revision |
| `quant_scheme` or `compute_dtype` | Used to derive `W4A16`-style scheme label |
| `quant_gpu_nums` | Quantization GPU count |
| `eval_gpu_nums` | Evaluation GPU count |
| `quant_precision` | Precision label |
| `quant_weight_dtype` | Weight dtype label |

## 8. Hugging Face upload behavior

`upload_model_hf.py` uploads the **model output dir**, not the runtime dir.

Ignored content includes:

- `logs/**`
- `session_*.jsonl`
- `session_*.md`
- `quant_summary.json`
- `summary.json`
- `accuracy.json`
- `request.json`
- `lm_eval_results/**`
- `quantize_script.py`
- local ledger files

### Local ledger mode

When shared mode is disabled:

1. Load or initialize `HF_USAGE_FILE`
2. Estimate upload size from non-ignored files
3. Rank candidate accounts by remaining capacity
4. Try upload account by account
5. Update the local ledger after success

### Shared ledger mode

When shared mode is enabled:

1. Clone or update the shared ledger repo
2. Compute current account state from append-only events
3. Reserve space on one account
4. Upload model
5. Write a `commit` event on success
6. Write a `release` event on failure

Event types:

- `reserve`
- `commit`
- `release`

The shared ledger implementation exists and has been validated locally, but it should still be tested in the real shared environment before relying on it for production coordination.

## 9. GitHub artifact upload behavior

`upload_results_github.py` uploads the **runtime output dir** into the `lb_eval` results repository.

Copied artifacts include:

- `quant_summary.json`
- `summary.json` if present
- `accuracy.json`
- `quantize.py`
- `evaluate.sh`
- `lm_eval_results/`
- `logs/`
- `session_*.jsonl`
- `session_*.md`

Output layout in the results repo:

```text
results/<org>/<artifact_name>/run_<timestamp>/...
results/<org>/<artifact_name>/results_<timestamp>.json
```

If `GIT_RESULTS_REPO_DIR` is empty, the script can clone the repo automatically.

## 10. Session formatting

`format_sessions.py` accepts:

- one or more JSONL files
- one or more directories containing session JSONL files
- an optional `--output-dir`

Example:

```bash
python3 format_sessions.py session_quant_123.jsonl session_eval_123.jsonl
python3 format_sessions.py /path/to/runtime-output-dir
```

## 11. Common development notes

### Re-run behavior

- If `quant_summary.json` already has `status=success`, quantization is skipped
- If `accuracy.json` already has `status=success`, evaluation is skipped
- If a reused runtime dir already contains `evaluate.sh`, `auto.sh` reuses that script for the next evaluation attempt
- Older runtime dirs with legacy `evaluate.py` still work as historical artifacts, but new runs are expected to generate `evaluate.sh`
- New evaluation scripts are expected to split raw `lm_eval` execution from result parsing so a parsing failure does not force `lm_eval` to run again on the next retry
- Successful `quant_summary.json` and `accuracy.json` are treated as final summary artifacts and should be written atomically at the end; if OpenClaw fails before that, `auto.sh` writes a minimal failed fallback summary so upload/status write-back still works
- Generated `quantize.py` should contain only core quantization/export logic, and generated `evaluate.sh` should contain only raw `lm_eval` execution; environment preparation and parsing/finalization stay as separate steps in the same OpenClaw task

### Failure handling

- A failing step is recorded in `FAILED_STEPS`
- Later steps still run when possible
- The final exit code is non-zero if:
  - quantization failed for an `auto_quant` pipeline
  - evaluation failed
  - any wrapped step failed

### Logging

- All shell output goes to one file: `logs/auto.log`
- Prompt copies are written to the runtime log directory
- OpenClaw session JSONL files are copied from `OPENCLAW_SESSIONS_DIR`

## 12. Recommended development checklist

1. Confirm `config.env` points to the correct OpenClaw workspace and sessions path
2. Run `bash auto.sh <task.json> --dry-run`
3. Run the real task
4. Inspect:
   - runtime `logs/auto.log`
   - `quant_summary.json`
   - `accuracy.json`
   - `evaluate.sh`
   - copied session JSONL/Markdown
5. Confirm the model dir contains only upload-worthy model files
6. Confirm GitHub upload contains runtime artifacts only
