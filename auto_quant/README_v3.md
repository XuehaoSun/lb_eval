# auto_v3 Pipeline — 使用指南

> Phases-based 量化 + 评估 + 上传一体化管道，支持 agent 自动修复

---

## 目录

1. [快速开始](#快速开始)
2. [运行模式](#运行模式)
3. [Task JSON 格式](#task-json-格式)
4. [config.env 配置](#configenv-配置)
5. [环境变量覆盖](#环境变量覆盖)
6. [版本控制](#版本控制auto-roundtransformers)
7. [Pipeline 阶段详解](#pipeline-阶段详解)
8. [BitLesson 系统](#bitlesson-系统)
9. [输出目录结构](#输出目录结构)
10. [常见用法示例](#常见用法示例)
11. [Pre-flight 依赖检测](#pre-flight-依赖检测)
12. [多 GPU 支持](#多-gpu-支持)
13. [容错与重试机制](#容错与重试机制)
14. [输出格式兼容性](#输出格式兼容性)
15. [故障排查](#故障排查)
16. [配置参考表](#配置参考表)

---

## 快速开始

```bash
# 1. 配置 config.env（填入 HF token 和 GitHub token）
cp config.env config.env.bak
vim config.env

# 2. 运行 dry-run 检查配置是否正确
bash auto_v3.sh /path/to/task.json --dry-run

# 3. 正式运行
bash auto_v3.sh /path/to/task.json
```

---

## 运行模式

```bash
bash auto_v3.sh <task_json_file> [options]
```

| 选项 | 说明 |
|------|------|
| `--dry-run` | 仅打印解析后的配置，不执行任何操作 |
| `--skip-upload` | 跳过所有上传步骤（本地调试用） |
| `--skip-agent` | 跳过 agent 修复循环，失败立即退出 |
| `-h, --help` | 显示帮助 |

**典型使用场景：**

```bash
# 本地调试：不上传、不用 agent
bash auto_v3.sh task.json --skip-upload --skip-agent

# CI 生产：完整执行
bash auto_v3.sh task.json

# 只看配置是否正确
bash auto_v3.sh task.json --dry-run
```

---

## Task JSON 格式

Pipeline 通过读取 task JSON 获取任务参数。兼容 leaderboard 提交的标准格式：

### 最小 JSON（必需字段）

```json
{
  "model": "Qwen/Qwen3-0.6B"
}
```

仅 `model` 为必需字段，其余使用默认值。

### 完整 JSON（所有可选字段）

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN",
  "iters": 0,
  "export_format": "auto_round",
  "auto_round_ref": "latest",
  "transformers_ref": "auto",
  "request_filename": "Qwen_Qwen3-0.6B_quant_request_False_W4A16_4bit_int4.json"
}
```

### 字段说明

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `model` | **必填** | HuggingFace 模型 ID，如 `Qwen/Qwen3-0.6B` |
| `scheme` | `W4A16` | 量化方案：`W4A16` / `MXFP4` / `NVFP4` / `MXFP8` |
| `method` | `RTN` | 量化方法：`RTN`(iters=0) / `TUNING`(iters=200) |
| `iters` | *auto* | 若指定，覆盖 method：`0` → RTN, `>0` → TUNING |
| `export_format` | `auto_round` | 导出格式：`auto_round`(hf backend) / `llm_compressor`(vllm backend) |
| `auto_round_ref` | `latest` | auto-round 版本控制（详见[版本控制](#版本控制auto-roundtransformers)） |
| `transformers_ref` | `auto` | transformers 版本控制 |
| `request_filename` | `""` | 关联的 request 文件名（用于状态写回） |

### 兼容 Leaderboard Status JSON

Pipeline 也可直接使用 leaderboard 的 status JSON 文件作为输入：

```bash
# 直接用 lb_eval/status 下的文件
bash auto_v3.sh ../status/Qwen/Qwen3-0.6B_quant_request_False_W4A16_4bit_int4.json
```

此时，`scheme` 会从 `quant_type` 字段回退读取。

---

## config.env 配置

`config.env` 位于 `auto_quant/` 目录，包含所有密钥和默认配置。

### 必须配置的项

```bash
# ═══ HuggingFace 上传 ═══
HF_TOKENS=hf_xxx,hf_yyy              # 逗号分隔，多 token 自动 failover
HF_UPLOAD_ORGS=LeaderboardModel1,LeaderboardModel2  # 对应每个 token 的 org

# ═══ GitHub 结果上传 ═══
GIT_TOKEN=ghp_xxx                     # GitHub PAT，有 repo 写入权限
GIT_REPO=https://github.com/user/lb_eval.git
GIT_USER_NAME=your-name
GIT_USER_EMAIL=your@email.com

# ═══ Agent API Key（如需 agent 修复） ═══
MINIMAX_API_KEY=your_key_here
```

### 可选配置项

```bash
# ═══ Pipeline 默认行为 ═══
METHOD=RTN                    # 默认量化方法 (RTN|TUNING)
EXPORT_FORMAT=auto_round      # 默认导出格式
DEVICE=cuda                   # 设备
DEVICE_INDEX=0                # GPU 索引
EVAL_TASKS=piqa,mmlu,hellaswag  # 评估任务
EVAL_BATCH_SIZE=8             # 评估 batch size
TIMEOUT=36000                 # 超时（秒）

# ═══ 输出路径 ═══
RUNTIME_OUTPUT_BASE_DIR=      # 留空使用默认 output/runs/
OUTPUT_DIR=                   # 留空使用 auto_quant/output/

# ═══ 代理 ═══
HTTP_PROXY=http://proxy:port
HTTPS_PROXY=http://proxy:port
```

---

## 环境变量覆盖

所有参数均可通过环境变量覆盖（优先级：环境变量 > task JSON > config.env）：

```bash
# 指定 GPU
DEVICE_INDEX=1 bash auto_v3.sh task.json

# 多卡评估
NUM_GPUS=2 bash auto_v3.sh task.json

# 自定义评估任务
EVAL_TASKS="arc_easy,winogrande,piqa" bash auto_v3.sh task.json

# 增加 agent 修复尝试次数（默认 3）
MAX_FIX_ATTEMPTS=5 bash auto_v3.sh task.json

# 指定输出目录
OUTPUT_DIR=/data/results bash auto_v3.sh task.json
```

### 完整环境变量列表

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL_ID` | *from JSON* | 模型 ID |
| `SCHEME` | `W4A16` | 量化方案 |
| `METHOD` | `RTN` | 量化方法 |
| `ITERS` | *auto* | 量化迭代次数 |
| `EXPORT_FORMAT` | `auto_round` | 导出格式 |
| `EVAL_BACKEND` | *auto* | 评估后端（由 export_format 推导） |
| `AUTO_ROUND_REF` | `latest` | auto-round 版本引用 |
| `TRANSFORMERS_REF` | `auto` | transformers 版本引用 |
| `DEVICE` | `cuda` | 设备类型 |
| `DEVICE_INDEX` | `0` | GPU 索引 |
| `DEVICE_MAP` | `auto` | 模型加载 device_map |
| `EVAL_TASKS` | `piqa,mmlu,hellaswag` | 评估任务 |
| `EVAL_BATCH_SIZE` | `8` | 评估批大小 |
| `NUM_GPUS` | `1` | GPU 数量 |
| `MAX_FIX_ATTEMPTS` | `3` | Agent 最大修复次数 |
| `OUTPUT_DIR` | `./output` | 输出根目录 |
| `LM_EVAL_VERSION` | `0.4.10` | lm_eval 最低版本 |
| `VLLM_VERSION` | *latest* | vLLM 版本 |

---

## 版本控制（auto-round/transformers）

`AUTO_ROUND_REF` 和 `TRANSFORMERS_REF` 支持多种版本引用格式：

| 值 | 含义 | 实际安装 |
|----|------|---------|
| `latest` | 最新 PyPI 版本 | `pip install auto-round` |
| `0.13.0` | 指定版本号 | `pip install auto-round==0.13.0` |
| `main` | Git 分支 | `pip install git+https://github.com/intel/auto-round@main` |
| `abc123` | Git commit SHA | `pip install git+https://github.com/intel/auto-round@abc123` |
| `feature/xxx` | Git 分支名 | `pip install git+https://github.com/intel/auto-round@feature/xxx` |

**`TRANSFORMERS_REF` 额外支持：**

| 值 | 含义 |
|----|------|
| `auto` | 不覆盖，使用 auto-round 安装时带入的版本 |

### 使用示例

```bash
# 使用 auto-round 开发分支 + 最新 transformers
AUTO_ROUND_REF=main TRANSFORMERS_REF=latest bash auto_v3.sh task.json

# 锁定特定版本（可重复性）
AUTO_ROUND_REF=0.13.0 TRANSFORMERS_REF=4.46.0 bash auto_v3.sh task.json

# 测试某个 PR fix
AUTO_ROUND_REF=fix/nvfp4-export bash auto_v3.sh task.json
```

---

## Pipeline 阶段详解

```
┌─────────────────────────────────────────────────────────────────┐
│                    auto_v3.sh (orchestrator)                     │
├─────────┬──────────────┬─────────────────┬────────────────────── │
│ Phase 1 │    Phase 2   │     Phase 3     │        Phase 4       │
│setup_env│  quantize.py │   evaluate.sh   │  upload (HF+GitHub)  │
│         │  (wrapper.sh)│                 │                       │
└─────────┴──────────────┴─────────────────┴──────────────────────┘
     ↓ failure        ↓ failure        ↓ failure
  agent_fix_loop   agent_fix_loop   agent_fix_loop
  (max 3 retries)  (max 3 retries)  (max 3 retries)
```

### Phase 1: setup_env.sh

**功能：** 安装 auto-round 及依赖包

- 根据 `AUTO_ROUND_REF` 安装 auto-round
- 根据 `TRANSFORMERS_REF` 覆盖 transformers（若非 `auto`）
- 安装 `lm-eval >= 0.4.10`
- 如果 `EVAL_BACKEND=vllm`，安装 vllm + llm_compressor

### Phase 2: quantize.py (via quantize_wrapper.sh)

**功能：** 对模型进行量化

- 使用 Recipe 字典定义每种 scheme 的量化参数
- 支持 W4A16 / MXFP4 / NVFP4 / MXFP8
- 方法：RTN（iters=0）或 TUNING（iters=200）
- 开启 `trust_remote_code=True`
- 输出到 `quantized_model/` 子目录
- 生成 `quant_summary.json` 记录量化元信息

### Phase 3: evaluate.sh

**功能：** 对量化模型进行评估

- 支持 `hf` 和 `vllm` 两种 backend
- 多卡支持（hf: parallelize, vllm: tensor_parallel_size）
- 自动解析 lm_eval 输出为 `accuracy.json`
- **零精度检测：** 任何 task 的 acc=0 视为失败

### Phase 4: Upload

**功能：** 上传结果

- **模型 → HuggingFace Hub：** 多 token/多 org failover，共享 ledger 管理配额
- **结果 → GitHub lb_eval：** 自动过滤 secret，写回 status

---

## BitLesson 系统

pipeline 在修复过程中自动积累经验（`lessons/` 目录）。

### 文件位置

```
lb_eval/lessons/
  ├── setup_env.jsonl      # 环境安装错误经验
  ├── quantize.jsonl       # 量化错误经验
  ├── evaluate.jsonl       # 评估错误经验
  └── README.md
```

### 单条 Lesson 格式

```json
{
  "timestamp": "2026-05-28T14:00:00",
  "model": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN",
  "error_signature": "ImportError: cannot import name ...",
  "error_traceback": "... (full traceback, for human review) ...",
  "status": "fixed",
  "solution": "Agent fixed on attempt 1"
}
```

### 手动管理

```bash
# 压缩 lessons（合并相似条目）
python3 lessons/compact_lessons.py lessons/quantize.jsonl

# 查看统计
wc -l lessons/*.jsonl

# Lessons 会在每次 pipeline 结束后自动 git push
```

---

## 输出目录结构

每次运行在 `output/runs/<ModelName>-AutoRound-<Scheme>-<Method>/` 下生成：

```
output/runs/Qwen3-0.6B-AutoRound-W4A16-RTN/
├── request.json              # 原始 task JSON 副本
├── quant_summary.json        # 量化结果摘要
├── accuracy.json             # 评估精度汇总
├── run_report.md             # 运行报告（自动生成）
├── session_*.jsonl           # OpenClaw agent session 原始日志
├── session_*.md              # Agent session 可读 markdown
├── quantized_model/          # 量化后模型文件
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── lm_eval_results/          # 评估原始输出
│   ├── results_*.json
│   └── ...
└── logs/                     # 运行日志
    ├── auto.log              # 整体 pipeline stdout+stderr
    ├── setup_env.log
    ├── quantize.log
    ├── evaluate.log
    ├── upload_hf.log
    ├── upload_github.log
    └── agent_fixes/          # agent 修复记录
        ├── setup_env/
        │   ├── prompt_1.txt  # 发给 agent 的修复指令
        │   ├── attempt_1.log # agent 输出
        │   └── retry_1.log   # 修复后重跑日志
        └── quantize/
            └── ...
```

### 回传到 GitHub 的完整内容

`upload_results_github.py` 将以下全部推送到 `lb_eval/results/{org}/{artifact}/run_{timestamp}/`：

| 文件 | 说明 |
|------|------|
| `request.json` | 原始任务输入（可追溯是谁提交的什么任务） |
| `quant_summary.json` | 量化元信息 |
| `accuracy.json` | 评估精度 |
| `run_report.md` | 自动生成的运行报告 |
| `quantize.py` | 实际使用的量化脚本 |
| `evaluate.sh` | 实际使用的评估脚本 |
| `logs/` | 全部日志（含 `auto.log`、各 phase 日志、agent_fixes/） |
| `lm_eval_results/` | lm_eval 原始输出 |
| `session_*.jsonl` | Agent session 原始数据 |
| `session_*.md` | Agent session 可读 markdown |
| `failure_diagnosis_*.json` | 失败诊断（失败时生成） |
| `results_{timestamp}.json` | 聚合文件（Leaderboard 直接读取） |

---

## 常见用法示例

### 1. 基础量化 — W4A16/RTN

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "scheme": "W4A16",
  "method": "RTN"
}
```

```bash
bash auto_v3.sh task_qwen3_w4.json
```

### 2. 高精度量化 — W4A16/TUNING

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "scheme": "W4A16",
  "method": "TUNING"
}
```

```bash
bash auto_v3.sh task_llama_tuning.json
```

### 3. MXFP4 量化 — 使用 vLLM 评估

```json
{
  "model": "Qwen/Qwen3-4B",
  "scheme": "MXFP4",
  "export_format": "llm_compressor"
}
```

```bash
bash auto_v3.sh task_mxfp4.json
```

### 4. 指定版本测试兼容性

```json
{
  "model": "google/gemma-4-E2B-it",
  "scheme": "W4A16",
  "auto_round_ref": "0.13.0",
  "transformers_ref": "4.46.0"
}
```

```bash
bash auto_v3.sh task_version_lock.json --skip-upload
```

### 5. 大模型多卡量化评估

```bash
NUM_GPUS=4 DEVICE_MAP=auto bash auto_v3.sh task_70b.json
```

### 6. 自定义评估任务

```bash
EVAL_TASKS="arc_easy,arc_challenge,hellaswag,piqa,winogrande,mmlu" \
  bash auto_v3.sh task.json
```

### 7. 本地调试（不上传、不 agent）

```bash
bash auto_v3.sh task.json --skip-upload --skip-agent
```

### 8. 批量运行（结合 CI）

```bash
for json in ../status/Qwen/*.json; do
  echo "Processing: $json"
  bash auto_v3.sh "$json" 2>&1 | tee "logs/$(basename $json .json).log"
done
```

---

## 故障排查

### Q: dry-run 报 unbound variable

检查 `config.env` 中变量定义顺序。如有变量互相引用，确保被引用的在前面定义，或使用 `${VAR:-}` 默认值语法。

### Q: Phase 1 (setup_env) 失败

常见原因：
- 网络问题导致 pip install 失败 → 检查代理配置
- `AUTO_ROUND_REF` 指向不存在的分支/tag → 确认 ref 有效
- 依赖冲突 → 设 `TRANSFORMERS_REF` 为具体版本

### Q: Phase 2 (quantize) OOM

- 减小模型规模或调整 `DEVICE_MAP=auto` 使用多卡
- RTN 模式内存消耗远小于 TUNING 模式
- 检查 `DEVICE_INDEX` 指向的 GPU 是否有足够显存

### Q: Phase 3 (evaluate) 零精度

- 量化可能损坏了模型权重（极端配置下）
- 尝试不同的 `EVAL_BATCH_SIZE`
- 确认 `QUANTIZED_MODEL_DIR` 下有完整的模型文件

### Q: Agent 修复循环无效

- 检查 `MINIMAX_API_KEY` 是否配置
- 查看 `logs/agent_fixes/` 下的 prompt 和 attempt 日志
- 如果反复 drift（同一错误），说明需要人工干预
- 增加 `MAX_FIX_ATTEMPTS` 数值或 `--skip-agent` 手动修复

### Q: 上传失败

- **HF:** 检查 HF_TOKENS 是否有写权限，org 名称是否对应
- **GitHub:** 检查 GIT_TOKEN 的 repo 权限，GIT_REPO 地址是否正确
- 网络问题：设置 `HTTP_PROXY` / `HTTPS_PROXY`

---

## 配置参考表

| 量化方案 | bits | group_size | 导出格式 | 评估后端 | 适用场景 |
|----------|------|-----------|---------|---------|---------|
| W4A16 | 4 | 128 | auto_round | hf | 通用，主推 |
| MXFP4 | 4 | 32 | llm_compressor | vllm | Ampere+ GPU |
| NVFP4 | 4 | -1 | llm_compressor | vllm | Ada Lovelace+ |
| MXFP8 | 8 | 32 | llm_compressor | vllm | 高精度需求 |

| 量化方法 | iters | 说明 |
|----------|-------|------|
| RTN | 0 | Round-to-Nearest，快速，无需校准数据 |
| TUNING | 200 | 有校准数据优化，精度更高，耗时更长 |

---

## Pre-flight 依赖检测

Pipeline 在量化前自动运行 `preflight_deps.py`，检测并安装模型所需的额外依赖：

| 检查项 | 机制 | 示例 |
|--------|------|------|
| transformers 版本 | 比较 config.json 中 `transformers_version` 与已安装版本 | 模型需 4.45，装了 4.43 → 升级 |
| requirements.txt | 下载模型仓库的 requirements.txt | 自定义模型的额外包 |
| 已知架构映射 | KNOWN_DEPS 字典（mamba→mamba-ssm, phi→einops 等） | Jamba 模型自动装 mamba-ssm |
| custom code 导入 | 试运行 `AutoConfig.from_pretrained(trust_remote_code=True)` 捕获 ImportError | 自定义代码依赖 tiktoken |

如果 preflight 失败，不会阻断 pipeline——仍进入量化阶段，交由 agent fix loop 处理。

---

## 多 GPU 支持

### 量化（Phase 2）

```bash
# 使用 device_map="auto" 自动分配多卡（accelerate 处理）
NUM_GPUS=4 DEVICE_MAP=auto bash auto_v3.sh task.json
```

quantize.py 使用 `AutoModelForCausalLM.from_pretrained(device_map="auto")`，accelerate 自动将模型分片到多卡。

### 评估（Phase 3）

| Backend | 多卡机制 | 触发条件 |
|---------|----------|----------|
| hf | `--model_args parallelize=True` | `NUM_GPUS > 1` |
| vllm | `--model_args tensor_parallel_size=N` | `NUM_GPUS > 1` |

```bash
# 4卡评估（自动传递给 lm_eval）
NUM_GPUS=4 bash auto_v3.sh task.json
```

---

## 容错与重试机制

### Agent Fix Loop

```
Phase 执行 → 失败 → 提取错误上下文 → 检查 BitLesson → 构建 prompt → 调用 Agent → 重跑验证
                                    ↑                                              ↓
                                    └──── 重复最多 MAX_FIX_ATTEMPTS 次 ←──── 仍失败 ─┘
```

**Drift 检测：** 如果连续两次的错误签名完全相同，提前终止循环（避免浪费）。

### GitHub Push 重试

`upload_results_github.py` 的 push 有 5 层防护：

1. 正常 push → 成功退出
2. 非 fast-forward 冲突 → `pull --rebase` 后重推
3. rebase 冲突 → `rebase --abort` + `reset --hard FETCH_HEAD` + 重新 commit
4. 指数退避等待（2s, 4s, 8s, 16s, 32s）
5. 最终失败返回 exit 1（日志中有详细错误）

### Git Repo 初始化容错

如果本地 `lb_eval/` 目录存在但不是 git repo（前次 clone 中断等），自动删除重新 clone，不再报错退出。

### Pipeline 状态始终回传

即使量化或评估失败，GitHub upload 仍然执行——确保失败日志和诊断信息也能回传。仅 HF 模型上传要求 `PIPELINE_STATUS == "Finished"`。

---

## 输出格式兼容性

新版输出与旧版/Leaderboard 完全兼容：

### Leaderboard 读取路径

```
Leaderboard App
     ↓ 读取
results_*.json（聚合文件）
     ↓ 内嵌
├── quant_summary（量化信息）
├── accuracy（评估精度）
└── 元数据（model_id, artifact_name, generated_at, copied_files ...）
```

Leaderboard **不直接读** `quant_summary.json` / `accuracy.json`，它读 `upload_results_github.py` 生成的 `results_{timestamp}.json` 聚合文件。

### 关键兼容字段

| 字段 | 来源 | Leaderboard 用途 |
|------|------|-----------------|
| `quant_summary.scheme` | quantize.py | 显示 Scheme 列 |
| `quant_summary.method` | quantize.py | 显示 Method 列（RTN/TUNING） |
| `quant_summary.hf_repo` | upload_model_hf.py 补写 | 生成 Artifact 下载链接 |
| `quant_summary.output_files` | quantize.py | artifact 完整性检测 |
| `accuracy.tasks.{name}.accuracy` | evaluate.sh | 计算各 task 得分 |

### 新增字段（Leaderboard 安全忽略）

`ar_scheme`, `is_moe`, `ignore_layers`, `architecture`, `lm_eval_output_dir`, `eval_num_gpus`

---

## 故障排查（补充）

### Q: GitHub push 提示 "Username for 'https://github.com':"

**原因：** `GIT_TOKEN` 为空或未生效。

检查项：
1. 确认 `config.env` 中 `GIT_TOKEN=ghp_xxx` 非空
2. 注意：pipeline 读取的是 **`auto_quant/config.env`**，不是外层文件
3. Token 需要有 `repo` 权限（Contents: read/write）
4. 如用 fine-grained token，确认对目标仓库有推送权限

### Q: "target repo dir exists but is not a git repo"

此错误已修复。当前版本会自动清理非 git 目录后重新 clone。如果仍遇到，手动删除：

```bash
rm -rf auto_quant/lb_eval && bash auto_v3.sh task.json
```

### Q: 在 Azure 环境运行正常但本地失败

Azure 通过 `update_config_env.py --set` 注入 secrets：
```yaml
- script: python update_config_env.py --set HF_TOKENS=$(HF_TOKEN) --set GIT_TOKEN=$(GIT_TOKEN)
```

本地需手动填写 `config.env`，确保所有 token 字段非空。

### Q: run_report.md 显示 "N/A" 或 phase 全部 skipped

旧版 pipeline 产出的目录不含 `logs/` 子目录，report 退化到仅展示 quant_summary + accuracy 数据。这是正常行为。
