# Auto-Quant Agent LLM Benchmark 设计方案

## 1. 背景与动机

### 1.1 系统架构

`auto_quant` pipeline（`auto.sh`）负责自动化 LLM 量化与评测，包含 4 个阶段：

```
Phase 1: setup_env     → 安装 AutoRound/transformers/lm_eval，预检依赖
Phase 2: quantize      → 执行量化（RTN/Tuning），生成 quant_summary.json
Phase 3: evaluate      → lm_eval 评测，生成 accuracy.json
Phase 4: upload        → 上传模型到 HF Hub + 写回 results/status 到 lb_eval
```

当 Phase 1-3 任一阶段失败时，`agent_fix_loop`（定义在 `phases/agent_fix_loop.sh`）会：
1. 提取最后 100 行错误日志
2. 加载历史 lessons（来自 `lessons/*.jsonl`）
3. 构建 prompt 调用 LLM Agent（通过 `openclaw agent --local`）
4. Agent 尝试修复（安装依赖、修改配置等）
5. 重新运行失败的 phase 脚本验证修复
6. 最多重试 `MAX_FIX_ATTEMPTS`（默认 10）次
7. 记录 lesson（fixed / still_failing / drift）

### 1.2 当前数据（MiniMax-M2.7）

从 `lessons/*.jsonl` 统计（72 条记录）：

| 状态 | 数量 | 占比 | 说明 |
|------|------|------|------|
| still_failing | 37 | 51.4% | Agent 尝试后仍未修复 |
| drift | 27 | 37.5% | 同一错误反复出现，触发 drift 检测退出 |
| fixed | 4 | 5.6% | Agent 成功修复 |
| verified | 4 | 5.6% | 已验证的 seed/固定方案 |

**修复率仅 5.6%**，说明当前 Agent LLM 在此 workload 上效果有限。切换更强的 LLM 可能显著提升修复率，但也伴随更高成本。因此需要一套指标体系来系统评估。

### 1.3 目标

定义一套可量化、可重复的 benchmark 指标，用于：
- 对比不同 LLM（GPT-4o、Claude Sonnet、DeepSeek-V3 等）在此 workload 上的表现
- 指导 LLM 选型决策（效果 vs 成本的最优点）
- 发现 pipeline 本身的可改进点（某些错误类别是否不应交给 Agent）

---

## 2. 指标体系（5 大维度 17 个指标）

### 2.1 效果指标（Effectiveness）— 能不能修好

| 编号 | 指标 | 定义 | 计算方式 | 重要性 |
|------|------|------|---------|--------|
| E1 | **Fix Rate（修复率）** | Agent 介入后最终修复成功的比例 | `fixed / (fixed + drift + still_failing)` | ⭐⭐⭐ |
| E2 | **First-Attempt Fix Rate（首次修复率）** | 第一次 agent 调用就修复的比例 | `fixed_on_attempt_1 / total_agent_invocations` | ⭐⭐⭐ |
| E3 | **Phase-wise Fix Rate（分阶段修复率）** | 按 setup_env / quantize / evaluate 分别统计修复率 | 各 phase 的 `fixed / total` | ⭐⭐ |
| E4 | **Error Category Coverage（错误类别覆盖率）** | 能修复多少种不同类别的错误 | `fixed_categories / total_categories` | ⭐⭐ |
| E5 | **Drift Rate（重复率）** | Agent 反复尝试同一方案导致 drift 退出的比例 | `drift / total_agent_invocations` | ⭐⭐ |

**解读**：
- E1 是最核心指标，直接衡量 Agent 价值
- E2 区分"运气好一次修好"vs"需要多次摸索"
- E5 高说明 Agent 缺乏多样化解题思路，容易陷入死循环

### 2.2 效率指标（Efficiency）— 修好的代价

| 编号 | 指标 | 定义 | 计算方式 | 重要性 |
|------|------|------|---------|--------|
| C1 | **Tokens per Fix（每次修复 token 消耗）** | 成功修复一个问题的平均 token 用量 | `sum(tokens_for_fixed_runs) / fixed_count` | ⭐⭐⭐ |
| C2 | **Tokens per Attempt（每次尝试 token 消耗）** | 每次 agent 调用的平均 token 数 | `total_tokens / total_attempts` | ⭐⭐ |
| C3 | **Cost per Fix（每次修复成本, USD）** | 美元计价的修复成本 | `tokens_per_fix × price_per_token` | ⭐⭐⭐ |
| C4 | **Attempts to Fix（修复所需尝试次数）** | 成功修复平均需要几次尝试 | `sum(attempt_number_for_fixed) / fixed_count` | ⭐⭐ |
| C5 | **Agent Wall Time（Agent 执行时间）** | Agent 从接收 prompt 到返回结果的时间 | `agent_end_time - agent_start_time` | ⭐ |

**解读**：
- C1 + C3 结合看，某些 LLM 修复率高但 token 贵，可能不划算
- C4 直接影响 pipeline 总时长（每多一次 attempt = 重跑一次 phase 脚本）
- C5 高意味着 pipeline 被 Agent 阻塞时间长，影响吞吐量

### 2.3 质量指标（Quality）— 修好的好不好

| 编号 | 指标 | 定义 | 计算方式 | 重要性 |
|------|------|------|---------|--------|
| Q1 | **Fix Durability（修复持久性）** | 修复后同类错误在后续 run 是否重现 | `durable_fixes / total_fixes` | ⭐⭐ |
| Q2 | **Side Effect Rate（副作用率）** | 修复一个 phase 是否导致后续 phase 新增失败 | `runs_with_new_failure_after_fix / fixed_count` | ⭐⭐ |
| Q3 | **Accuracy Delta（精度影响）** | Agent 修复后模型评测精度 vs 正常流程精度 | `avg_accuracy_fixed - avg_accuracy_normal` | ⭐ |

**解读**：
- Q1 衡量修复是"治本"还是"打补丁"（如 Agent 可能硬编码绕过而非真正修复）
- Q2 重要：例如 Agent 升级 transformers 修好了 quantize，但破坏了 evaluate
- Q3 确保 Agent 修复不会偷偷降低模型质量

### 2.4 鲁棒性指标（Robustness）— 稳不稳定

| 编号 | 指标 | 定义 | 计算方式 | 重要性 |
|------|------|------|---------|--------|
| R1 | **Timeout Rate（超时率）** | Agent 执行超时（超过 AGENT_TIMEOUT）的比例 | `timeouts / total_attempts` | ⭐⭐ |
| R2 | **API Error Rate（API 错误率）** | rate_limit / auth / network 错误的比例 | `api_errors / total_attempts` | ⭐⭐ |
| R3 | **Forbidden Action Rate（违规操作率）** | Agent 执行了禁止操作（降级 torch、修改 eval tasks）的比例 | `harmful_actions / total_attempts` | ⭐⭐⭐ |

**解读**：
- R1 高说明 LLM 推理太慢或生成太长
- R2 是 provider 稳定性问题，非 LLM 能力本身（但影响实际可用性）
- R3 是安全底线，某些 LLM 更倾向于"大刀阔斧"修改，可能造成破坏

### 2.5 经济性指标（Economics）— 值不值

| 编号 | 指标 | 定义 | 计算方式 | 重要性 |
|------|------|------|---------|--------|
| $1 | **ROI（投入产出比）** | Agent 修复省下的人工时间价值 / token 成本 | `(fixed_count × human_fix_time_hours × hourly_rate) / total_cost` | ⭐⭐⭐ |
| $2 | **Marginal Value of Retry（边际重试价值）** | 每多一次重试的增量修复概率 | `P(fix\|attempt=n) - P(fix\|attempt=n-1)` | ⭐⭐ |

**解读**：
- $1 > 1 说明用 Agent 比人工便宜，ROI 越高越好
- $2 帮助决定 MAX_FIX_ATTEMPTS：如果第 3 次之后边际价值趋近于 0，就不必重试 10 次

---

## 3. 额外建议指标

除核心 17 个指标外，以下指标在特定场景下也有价值：

| 指标 | 说明 | 适用场景 |
|------|------|---------|
| **Lesson Generation Quality** | Agent 修复后自动生成的 lesson 是否可复用于未来相同错误 | 评估 Agent 的知识沉淀能力 |
| **Context Window Utilization** | Lessons 上下文占总 prompt 的比例，及其对修复质量的影响 | 优化 prompt 工程，决定加载多少 lessons |
| **Error Escalation Report Quality** | Agent 无法修复时，生成的错误报告是否足够帮助人工定位 | 即使修不好，也要有价值 |
| **Multi-phase Coherence** | 跨 phase 修复时，Agent 是否理解上下游依赖（如 setup_env 影响 quantize） | 评估 Agent 的系统思维 |
| **Idempotency（幂等性）** | 同一 case 跑两次，Agent 是否给出一致的修复方案 | 结果可预测性 |
| **Forbidden Action Compliance** | Agent 是否严格遵守约束（不降级 torch、不修改 eval tasks） | 安全性评估 |
| **Model Size Scaling** | 量化目标模型大小（0.5B / 7B / 70B）对 Agent 修复成功率的影响 | 发现 Agent 在大模型场景的短板 |
| **Latency to First Token** | API 首 token 响应速度 | 影响 pipeline 吞吐量 |
| **Fix Diversity Score** | 面对同一错误，Agent 在多次重试中是否尝试了不同方案 | Drift Rate 的细化版 |
| **Lesson Retrieval Precision** | 加载的 lessons 中有多少实际被 Agent 采纳使用 | 优化 lesson 系统 |

---

## 4. 数据采集方案

### 4.1 现有数据点

Pipeline 已经采集的数据：

| 数据 | 位置 | 内容 |
|------|------|------|
| 错误日志 | `{run}/logs/{phase}.log` | Phase 脚本的完整 stdout+stderr |
| Agent 日志 | `{run}/logs/agent_fixes/{phase}/attempt_*.log` | Agent 的输出和操作 |
| Agent Prompt | `{run}/logs/agent_fixes/{phase}/prompt_*.txt` | 发送给 Agent 的完整 prompt |
| 重试日志 | `{run}/logs/agent_fixes/{phase}/retry_*.log` | Agent 修复后重跑的日志 |
| Lessons | `lessons/{phase}.jsonl` | 结构化的错误+修复记录 |
| Pipeline 计时 | `auto.sh` 中的 `PIPELINE_START/END` | 总耗时 |
| 量化摘要 | `{run}/quant_summary.json` | 量化参数和结果 |
| 评测结果 | `{run}/accuracy.json` | 各 task 精度 |
| 运行报告 | `{run}/run_report.md` | 人类可读的汇总报告 |

### 4.2 需要新增的数据采集

在 `agent_fix_loop.sh` 的 `agent_fix_loop()` 函数中新增：

```bash
# ─── 新增：Agent 执行计时 ───
local agent_start_ts=$(date +%s)
run_openclaw_fix "${fix_prompt}" "${agent_log}" || true
local agent_end_ts=$(date +%s)
local agent_wall_time=$((agent_end_ts - agent_start_ts))

# ─── 新增：解析 Agent 日志提取 metrics ───
local agent_model=""
local agent_provider=""
local tokens_input=0
local tokens_output=0
local exit_reason="normal"

if [ -f "${agent_log}" ]; then
    # 从 openclaw 日志解析模型信息
    agent_model=$(grep -oP 'model=\K[^ ]+' "${agent_log}" | head -1)
    agent_provider=$(grep -oP 'provider=\K[^ ]+' "${agent_log}" | head -1)

    # 解析 token 用量（如果 openclaw 输出了 usage 信息）
    tokens_input=$(grep -oP 'input_tokens=\K\d+' "${agent_log}" | tail -1)
    tokens_output=$(grep -oP 'output_tokens=\K\d+' "${agent_log}" | tail -1)

    # 检测退出原因
    if grep -q "TIMEOUT" "${agent_log}"; then
        exit_reason="timeout"
    elif grep -q "rate_limit" "${agent_log}"; then
        exit_reason="rate_limit"
    elif grep -q "auth.*fail\|unauthorized" "${agent_log}"; then
        exit_reason="auth_error"
    fi
fi
```

### 4.3 指标输出格式

每次 agent 调用输出一条 JSONL 记录到 `{run}/agent_metrics.jsonl`：

```json
{
  "run_id": "run_2026-06-16-01-03-48",
  "model_id": "Quazim0t0/Escarda-86M",
  "scheme": "W4A16",
  "method": "TUNING",
  "phase": "quantize",
  "attempt": 1,
  "agent_model": "MiniMax-M2.7",
  "agent_provider": "minimax",
  "tokens_input": 3200,
  "tokens_output": 1800,
  "tokens_total": 5000,
  "agent_wall_time_seconds": 45.3,
  "exit_reason": "normal",
  "error_signature": "expected `,` or `}` at line 1 column 9",
  "error_category": "tokenizer_corrupt",
  "outcome": "still_failing",
  "lesson_status": "still_failing",
  "actions_detected": ["pip install transformers>=4.55"],
  "prompt_tokens_estimate": 4500,
  "lessons_loaded_count": 8,
  "timestamp": "2026-06-16T01:04:05Z"
}
```

### 4.4 汇总输出 `benchmark_results.json`

每次 benchmark 跑完后，汇总所有 run 的 agent_metrics，输出：

```json
{
  "benchmark_id": "bench_2026-06-17",
  "agent_model": "GPT-4o",
  "agent_provider": "openai",
  "test_cases_total": 20,
  "metrics": {
    "effectiveness": {
      "fix_rate": 0.45,
      "first_attempt_fix_rate": 0.30,
      "drift_rate": 0.10,
      "error_category_coverage": 0.67,
      "phase_fix_rates": {
        "setup_env": 0.60,
        "quantize": 0.35,
        "evaluate": 0.50
      }
    },
    "efficiency": {
      "tokens_per_fix_avg": 8500,
      "tokens_per_attempt_avg": 4200,
      "cost_per_fix_usd": 0.085,
      "attempts_to_fix_avg": 1.8,
      "agent_wall_time_avg_seconds": 32.5
    },
    "quality": {
      "fix_durability": 0.90,
      "side_effect_rate": 0.05,
      "accuracy_delta": -0.002
    },
    "robustness": {
      "timeout_rate": 0.03,
      "api_error_rate": 0.02,
      "forbidden_action_rate": 0.00
    },
    "economics": {
      "total_cost_usd": 1.70,
      "total_tokens": 84000,
      "roi": 12.5,
      "marginal_retry_value": [0.30, 0.12, 0.03, 0.01]
    }
  },
  "composite_score": 0.72
}
```

---

## 5. 对比实验设计

### 5.1 实验矩阵

```
┌─────────────────────┬───────────────┬────────────────────┬──────────────┐
│ Agent LLM           │ Provider      │ 价格 ($/1M tokens) │ 预期定位     │
├─────────────────────┼───────────────┼────────────────────┼──────────────┤
│ MiniMax-M2.7        │ minimax       │ ~$0.5              │ 基线（当前） │
│ GPT-4o              │ openai        │ ~$5.0              │ 高能力参照   │
│ GPT-4o-mini         │ openai        │ ~$0.3              │ 低成本对比   │
│ Claude 3.5 Sonnet   │ anthropic     │ ~$3.0              │ 强代码能力   │
│ Claude 3.5 Haiku    │ anthropic     │ ~$0.25             │ 低成本对比   │
│ DeepSeek-V3         │ deepseek      │ ~$0.5              │ 高性价比     │
│ Qwen2.5-72B         │ dashscope     │ ~$0.8              │ 中文+代码    │
│ Llama-3.1-70B       │ together/local│ ~$0.9              │ 开源参照     │
└─────────────────────┴───────────────┴────────────────────┴──────────────┘
```

### 5.2 控制变量

所有实验共享：
- **同一套测试 case**（从历史 failures 提取，见 5.3）
- **同一份 lessons 上下文**（冻结当前 `lessons/*.jsonl` 快照）
- **同一个 prompt template**（`build_fix_prompt()` 不变）
- **同一个 MAX_FIX_ATTEMPTS = 3**（Benchmark 模式下降低到 3 以节省成本）
- **同一个 AGENT_TIMEOUT = 600**
- **同一个运行环境**（Python 3.12 + torch 2.6.0 + CUDA 12.4）

唯一变量：Agent LLM（通过 openclaw 配置切换）

### 5.3 可重放测试集构建

从现有 72 条 lessons + 49 条失败分析中，提取标准测试集：

```
benchmark/test_cases/
├── tc_001_tokenizer_corrupt.json          # tokenizer JSON 损坏
├── tc_002_missing_dep_ouro.json           # 模型需要特殊依赖
├── tc_003_multimodal_unsupported.json     # 多模态模型（预期 unsolvable）
├── tc_004_cuda_driver_mismatch.json       # CUDA 驱动不匹配
├── tc_005_transformers_version.json       # transformers 版本过低
├── tc_006_config_json_corrupt.json        # config.json 格式错误
├── tc_007_meta_device_params.json         # 模型过大，参数在 meta device
├── tc_008_monkey_patch_conflict.json      # auto_round monkey_patch 冲突
├── tc_009_trust_remote_code.json          # 需要 trust_remote_code
├── tc_010_eval_task_not_found.json        # 评测 task 未找到
├── tc_011_oom_during_quantize.json        # 量化过程 OOM
├── tc_012_pip_install_conflict.json       # pip 依赖冲突
├── ...
└── manifest.json                          # 测试集元数据
```

每个 test case 结构：

```json
{
  "id": "tc_001",
  "category": "tokenizer_corrupt",
  "phase": "quantize",
  "difficulty": "medium",
  "model": "Quazim0t0/Escarda-86M",
  "scheme": "W4A16",
  "method": "TUNING",
  "error_log_file": "tc_001_error.log",
  "environment_snapshot": {
    "python": "3.12.13",
    "torch": "2.6.0+cu124",
    "transformers": "4.52.0",
    "auto_round": "0.5.1"
  },
  "expected_fix_type": "reinstall_or_skip",
  "is_solvable": true,
  "human_solution": "pip install tokenizers>=0.21 或更新 transformers",
  "notes": "tokenizer.json 文件在 HF Hub 上损坏，不是本地问题"
}
```

**难度定义**：
- `easy`：pip install 一个包即可，明确的依赖缺失
- `medium`：需要分析错误链条，可能需要版本调整
- `hard`：需要理解模型架构或修改代码逻辑
- `unsolvable`：结构性问题（多模态模型、模型过大），Agent 应识别并报告而非盲目重试

### 5.4 实验执行流程

```
1. 冻结环境快照（pip freeze > benchmark/env_snapshot.txt）
2. 冻结 lessons 快照（cp lessons/ benchmark/lessons_snapshot/）
3. 对每个 LLM:
   a. 配置 openclaw 使用该 LLM
   b. 对每个 test case:
      i.   恢复环境到快照状态
      ii.  注入 error condition（replay error 或实际跑 model）
      iii. 运行 agent_fix_loop
      iv.  采集 agent_metrics.jsonl
   c. 汇总该 LLM 的所有指标
4. 生成对比报告（benchmark_report.md）
```

---

## 6. 综合评分公式

### 6.1 AgentScore（0-100 分）

```
AgentScore = w1 × E1_norm + w2 × (1 - E5_norm) + w3 × C_norm
           + w4 × (1 - R1_norm) + w5 × E2_norm

其中:
  w1 = 0.35  Fix Rate（最重要：能不能修）
  w2 = 0.15  Anti-Drift（不要白费功夫）
  w3 = 0.20  Cost Efficiency（性价比）
  w4 = 0.10  Reliability（稳定性）
  w5 = 0.20  First-Shot（一次修好效率最高）

归一化:
  E1_norm = fix_rate              （已是 0-1）
  E5_norm = drift_rate            （已是 0-1）
  C_norm  = 1 - min(cost_per_fix / budget_cap, 1)   # budget_cap = $1.00
  R1_norm = timeout_rate          （已是 0-1）
  E2_norm = first_attempt_fix_rate（已是 0-1）
```

### 6.2 评分示例（假设数据）

```
              MiniMax-M2.7   GPT-4o    Claude Sonnet  DeepSeek-V3
Fix Rate      0.056          0.450     0.500          0.400
1st Fix Rate  0.028          0.300     0.350          0.300
Drift Rate    0.375          0.100     0.080          0.120
Cost/Fix($)   $2.50          $0.85     $0.60          $0.25
Timeout Rate  0.050          0.030     0.020          0.040
─────────────────────────────────────────────────────────────
AgentScore    8.2            62.5      71.3           68.0
```

### 6.3 可视化建议

使用雷达图展示 5 维度对比：

```
         Effectiveness
              │
    Quality ──┼── Efficiency
              │
   Robustness ┼ Economics
```

每个 LLM 一条折线，直观对比各维度强弱。

---

## 7. 实现路线图

### Phase 1: 数据采集增强（1-2 天）

- [ ] 修改 `agent_fix_loop.sh`：新增 agent 计时、token 解析、exit_reason 检测
- [ ] 新增 `save_agent_metrics()` 函数，输出 `agent_metrics.jsonl`
- [ ] 修改 `generate_report.py`：在报告中包含 agent metrics 摘要
- [ ] 验证 openclaw 日志中是否包含 token usage（若无，需要在 openclaw 侧添加）

### Phase 2: 测试集构建（1-2 天）

- [ ] 从 lessons + failure analysis 提取 15-20 个代表性 test case
- [ ] 为每个 case 创建 error_log + environment_snapshot + 元数据
- [ ] 标注 difficulty 和 is_solvable
- [ ] 编写 `benchmark/prepare_test_env.sh`：环境恢复脚本

### Phase 3: Benchmark Runner（2-3 天）

- [ ] 编写 `benchmark/run_benchmark.sh`：自动遍历 LLM × test_case 矩阵
- [ ] 编写 `benchmark/analyze_results.py`：计算所有指标 + 综合评分
- [ ] 编写 `benchmark/generate_report.py`：生成对比报告（Markdown + JSON）
- [ ] 支持 `--llm` 参数指定单个 LLM 跑（增量实验）

### Phase 4: 报告与可视化（1 天）

- [ ] 生成对比表格 + 雷达图
- [ ] 输出最优 LLM 推荐 + 成本估算
- [ ] 输出 MAX_FIX_ATTEMPTS 优化建议（基于 marginal retry value）

---

## 8. 附录

### A. 错误类别分类体系

基于 `analyze_failures.py` 的分析结果：

| 类别 | 占比 | 可修复性 | Agent 应对策略 |
|------|------|---------|---------------|
| multimodal_unsupported | 27% | 不可修复 | 识别并报告，不重试 |
| missing_dependency | 10% | 可修复 | `pip install <pkg>` |
| config_json_corrupt | 12% | 部分可修复 | 尝试 fallback config |
| needs_more_gpus | 10% | 不可修复（环境限制） | 识别并报告 |
| transformers_version | 8% | 可修复 | `pip install -U transformers` |
| tokenizer_corrupt | 5% | 部分可修复 | `pip install -U tokenizers` |
| cuda_driver_mismatch | 5% | 可修复 | 重装匹配 CUDA 的 torch |
| oom_killed | 8% | 部分可修复 | 减小 batch_size 或 gc |
| eval_task_error | 5% | 可修复 | 更新 lm_eval |
| other | 10% | 不确定 | 通用推理 |

### B. Token 价格参考（2026-06）

| Model | Input ($/1M) | Output ($/1M) | Context Window |
|-------|-------------|---------------|----------------|
| GPT-4o | $2.50 | $10.00 | 128K |
| GPT-4o-mini | $0.15 | $0.60 | 128K |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 200K |
| Claude 3.5 Haiku | $0.25 | $1.25 | 200K |
| DeepSeek-V3 | $0.27 | $1.10 | 64K |
| Qwen2.5-72B | $0.40 | $1.20 | 128K |
| MiniMax-M2.7 | ~$0.50 | ~$2.00 | 128K |

### C. 关键文件位置

```
auto_quant/
├── auto.sh                         # Pipeline 主入口
├── config.env                      # 配置（含 Agent provider）
├── phases/
│   ├── agent_fix_loop.sh           # Agent 修复循环（核心采集点）
│   ├── setup_env.sh                # Phase 1
│   ├── quantize_wrapper.sh         # Phase 2 wrapper
│   ├── quantize.py                 # Phase 2 核心
│   ├── evaluate.sh                 # Phase 3
│   └── generate_report.py          # 报告生成
├── lessons/
│   ├── setup_env.jsonl             # 7 条 lessons
│   ├── quantize.jsonl              # 64 条 lessons
│   └── evaluate.jsonl              # 1 条 lesson
├── tools/
│   └── analyze_failures.py         # 失败分析工具
├── docs/
│   └── agent_benchmark_design.md   # 本文档
└── benchmark/                      # [待创建] Benchmark 相关
    ├── test_cases/                 # 标准测试集
    ├── run_benchmark.sh            # Benchmark runner
    ├── analyze_results.py          # 指标计算
    └── reports/                    # 对比报告输出
```
