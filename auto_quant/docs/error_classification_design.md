# 错误分类与 Drift 检测设计

> 目标:让 OpenClaw fix-loop「分析问题、判断是否重复、判断是否值得继续」更准。
> 核心原则:**确定性分类负责精度(precision),Agent 负责召回(recall)与长尾,自学习负责收敛。**

---

## 1. 背景:两个真实 bug

### 1.1 Drift 误判(fix-loop 提前退出)

`test.log` 里,第 1 次失败是 **OOM**,Agent 修复后重跑,第 2 次失败是 **shape mismatch**
——两个完全不同的错误,却被判成「相同错误 / drift」,导致 loop 在第 2 次就中止
(`MAX_FIX_ATTEMPTS=10` 实际只跑了 2 次)。

**根因(off-by-one)**:每轮结尾把 `phase_log` 重新指向 `retry_{N-1}.log`,
下一轮的 drift 检查就变成 `retry_{N-1}.log` 与它自己比 → 永远相等 → 假 drift。

**修复**:不再「文件 vs 文件」,而是把「分类结果」跨轮持久化到变量,
当前轮 vs 上一轮比较(见第 3 节)。

### 1.2 `classify_error` 的三个缺陷(实测)

拿真实错误样本喂旧版 `taxonomy.classify_error`,证明:

| 缺陷 | 现象 |
|------|------|
| 覆盖缺口 | SHAPE / meta-device / CPU-fallback 全部落 `unknown` |
| 危险误判 | 良性可选文件 404(如 `generation_config.json` Not Found)→ 命中 `model_unavailable`(标记为不可修复)→ 直接放弃一个本可修复的任务 |
| 顺序脆弱 | dict「第一个命中即返回」,宽签名(如 `RuntimeError.*CUDA`)会盖住更具体的真实故障 |

---

## 2. 分层分类设计

```
┌─ L1  确定性 taxonomy (classify_error)   ── 精度/快/可复现,命中即准
│      先剥噪声行 → 按 CLASSIFY_PRIORITY 取「最具体」类别(不再第一个命中即返回)
│
├─ L1.5 去噪文本相似度 (logs_are_similar)  ── 对任意文本可用;两边都 unknown 时的 drift 兜底
│      difflib 比较去噪后日志,忽略 404/进度条等噪声
│
├─ L2  Agent 语义分析 (ERROR_CLASS)        ── 召回/语义/长尾;taxonomy 结果作为「先验」注入 prompt
│      Agent 可复用 taxonomy 类名,也可产出新的 snake_case 类名
│
└─ L3  自学习                              ── Agent 的 category 写入 lesson,供未来晋升进 taxonomy
```

**关键约束**:只有 Agent 的 `VERDICT: UNFIXABLE` 能提前终止 loop,正则永远不能。
正则只提供「先验」和「drift 信号」,不做最终裁决。

---

## 3. Drift 判定(三层择优)

每轮计算 `eff_class`(有效分类):

```
eff_class = Agent 的 ERROR_CLASS(存在且非 unknown)
          else taxonomy 类别
```

判定「是否相同错误」:

1. 若当前与上一轮 `eff_class` 都已知且非 unknown → 直接比较是否相等
2. 否则回退到 `logs_are_similar`(去噪相似度)比较两轮 errtail
3. **进度覆盖**:若量化跑到更深的 `layers.N`(比历史最深更深)→ 说明在推进 →
   重置 `drift_count`(即使错误看似相同,也不算卡死)
4. `drift_count >= DRIFT_THRESHOLD`(默认 2)才中止
5. **Fail-safe**:相同性无法判定时,既不中止也不重置(宁可多跑,不要错杀)

> 这样 OOM→shape 不再算 drift;真正卡在同一个 shape 错误上、且没有 layer 进展时,
> 才会在第 2 次累积后中止。

---

## 4. 为什么 taxonomy 覆盖不全「不是问题」— precision vs recall

用户质疑:「你优化的 taxonomy 如何覆盖不同种类的错误?」

在真实 124 条 lesson 语料上实测:**优化后仍有 65% 落 `unknown`**。
但拆解这 81 条 unknown,发现**绝大多数不是覆盖问题,而是数据质量问题**:

| 类别 | 数量 | 性质 |
|------|------|------|
| 空签名 | 19 | `save_lesson` 签名提取 bug(见第 5 节),错误文本丢失 |
| 噪声行当签名 | 39 | 把进度条/404/时间戳行当成了错误签名 |
| 真正的新错误 | 23 | 例:`invalid group reference 1 at position 22`、`Unrecognized configuration class`、`Expected attn_mask dtype` |

**结论:72%(58/81)的 unknown 是数据质量 bug,不是 taxonomy 覆盖不够。**

由此确立分工原则:

- **taxonomy 用 precision 衡量,不用 recall 衡量。**
  它只需要「命中的都对」(命中即准),不需要「覆盖所有错误」。
  盲目往里加正则去追长尾,只会引入误判(如 1.2 的 404 误判),得不偿失。
- **长尾覆盖(recall)是 Agent 的职责。** Agent 有完整日志 + 语义理解,
  天然能处理 taxonomy 没见过的新错误。
- **收敛靠 L3 自学习。** 反复出现的 unknown 错误,由 Agent 打上稳定 category,
  积累后再「晋升」为 taxonomy 的一条正式签名——让覆盖面**数据驱动地增长**,
  而不是靠人拍脑袋写正则。

一句话:**taxonomy 做「准」,Agent 做「全」,自学习做「收敛」。**

---

## 5. `save_lesson` 签名提取修复(最高杠杆)

第 4 节数据显示:58/81 的 unknown 源自 `save_lesson` 写坏了签名。旧逻辑:

- 取**第一行**含 "error/exception/failed" 的行 → 常抓到包装摘要行而非真正的异常
- 保留 `15:51:56 [ERROR]` 时间戳前缀 → 同一错误不同时刻 → 签名不同 → **去重失效**
- 包装行消息为空(`Quantization failed:`)时,真错误在更深处却被忽略

**新逻辑(复用同一套 taxonomy 去噪 + 分类)**:

1. 用 `_strip_noise` 先去掉 404/进度条等噪声行
2. 剥掉时间戳 / 日志级别前缀(`HH:MM:SS [ERROR]`、ISO 时间戳)→ 签名跨轮稳定
3. 优先取**最深(最后一条)真实 Python 异常**(traceback 底部才是真正抛出的错误),
   而非第一行提到 "error" 的;再退回带真实消息的包装行;最后退回去噪后最后一行
4. 写入时**持久化 `error_category`**(调用 `classify_error`)→ 为覆盖度量与 L3 自学习提供原料

实测效果:

| 输入 | 旧签名 | 新签名 / 类别 |
|------|--------|--------------|
| `15:51:56 [ERROR] Quantization failed: ...meta device` | 带时间戳前缀 | `Quantization failed: ...meta device` / `meta_device_error` |
| traceback(含 404 噪声 + 底部 ValueError) | 抓到 404 或首个 error 行 | `ValueError: invalid group reference...` / `unknown` |
| `Quantization failed:`(空)+ 深层 RuntimeError | 空签名 | `RuntimeError: shape mismatch...` / `shape_mismatch` |

### 5.1 lesson 现在存「log + 分析 + 解决方案」三件套

旧 lesson 只有报错 log 和一句 grep 出来的 fix,**agent 的分析被丢掉了**。现在
`save_lesson` 接收 agent 的完整结构化诊断(`extract_agent_analysis` 把
`COMPONENT / ERROR_CLASS / ROOT_CAUSE_HYPOTHESIS / EVIDENCE_RESULT / FIX_TIER / FIX_PLAN`
解析成 JSON),落成结构化字段:

| 维度 | 字段 | 来源 |
|------|------|------|
| 报错 log | `error_signature`(去噪+取最深异常)、`error_traceback`(去噪末 50 行) | 日志 |
| 报错**分析** | `agent_root_cause`、`agent_component`、`agent_evidence`、`agent_category`(agent 语义类)、`error_category`(确定性类) | Agent + taxonomy |
| 解决方案 | `solution`(FIX_PLAN,note 太薄时自动改用 agent 的 FIX_PLAN)、`fix_tier` | Agent |

这样每条 lesson 自洽:**为什么错(分析)+ 怎么修(方案)+ 原始证据(log)**——
既能喂回未来的 prompt,也是 L3 晋升的原料。

---

## 6. L3 自学习:learned overlay 晋升

**目标**:taxonomy 反复判 `unknown`、但 agent 反复给同一个 `ERROR_CLASS` 的错误,
自动学成一条保守正则,补进覆盖面——**数据驱动,不靠人工写正则**。

### 6.1 存储与读取(不污染人工 taxonomy)

- 学到的签名写进**独立文件** `error_analysis/learned_signatures.json`,与人工
  `TAXONOMY` 完全隔离,可审计、可手动裁剪。
- `taxonomy.classify_error` **只在人工 taxonomy 全部未命中后**才查 learned overlay
  → 人工分类的 precision 永不受影响,learned 只负责缩小 `unknown` 尾巴。
- overlay 文件缺失/损坏 → 返回空,fail-safe 不影响任何现有分类。

### 6.2 晋升工具 `promote_lessons.py`

扫描 `lessons/*.jsonl`,按 agent 语义类分组,**所有门槛都为 precision 优先**:

1. `error_category == unknown` —— taxonomy 确实分不出(真缺口)
2. agent 给了稳定、非通用的 snake_case 类 —— 真语义标签
3. 该类**复现 ≥ 阈值**(默认 3)次 —— 不是一次性
4. 能从复现文本派生出**足够具体**的签名(≥16 字符、≥2 个字母词);
   数字/路径/hex 归一为 `\d+`,类名等变化部分用最长公共子串消掉
5. 派生签名在**当前人工 taxonomy 下仍是 unknown** —— 不与人工类冲突/重复

默认 dry-run 只打印提案;`--apply` 才写 overlay(幂等合并,按 (category, signature) 去重)。

### 6.3 实测

- `invalid group reference 1 at position 22`(数字变化 x4)→ 学成
  `invalid group reference \d+ at position \d+`,能命中**没见过的**新数字组合
- `Unrecognized configuration class <XConfig> ...`(类名变化 x3)→ 最长公共子串派生
  `Unrecognized configuration class`,泛化到全新配置类
- 仅出现 2 次的错误(低于阈值)→ **不晋升**
- 人工类优先:`CUDA out of memory` 仍走确定性 `out_of_memory`,overlay 不抢

> 运维闭环:量化跑一批 → agent 修 + 写 lesson(带分析) →
> `python3 promote_lessons.py --lessons-dir lessons`(先看提案)→ `--apply` →
> 下一批同类错误直接被确定性命中,不再劳烦 agent。

---

## 7. 对 GitHub 回传的影响

回传链路 `error_analysis/analyze_failures.py` 通过 `quick_classify → classify_error`
使用同一套 taxonomy,产出 3 个回传产物都带 `category`:

- GitHub `lb_eval` 仓库的 `failure_analysis.md`
- HuggingFace 社区 discussion
- lessons JSONL(经 `push_lessons` 单独推送)

taxonomy 优化对回传是**净正向**且**无破坏**:

- 16 个类别字段齐全 → `quick_classify` 不会 KeyError 打断回传
- `classify_error(text)` 签名与返回值不变 → 向后兼容;learned overlay 也会被回传复用
- **Agent 超时兜底路径**(prompt 写死 ~90s,超时很常见):此时回传报告直接用
  确定性分类。补齐了 `_CATEGORY_ATTRIBUTION` 映射的 3 个新类别
  (`shape_mismatch` / `meta_device_error` / `device_mismatch`),
  否则超时 + 这些错误(恰是 W4A16 / CPU-fallback 场景)会退化成 `unknown` 归因
- 诊断同时保留 `pattern_category`(确定性)与 `category`(Agent),
  两者不一致时在报告里并列展示,互相印证

---

## 8. 待办(高价值,按杠杆排序)

- [x] 修复 drift off-by-one
- [x] 优化 taxonomy(新增 3 类、修 404 误判、去噪 + 优先级匹配)
- [x] 集成 Agent(taxonomy 先验注入 prompt + `ERROR_CLASS` 输出 + 三层 drift)
- [x] 修复 `save_lesson` 签名提取 + 持久化 `error_category` + agent 分析字段
- [x] 补齐回传兜底的 `_CATEGORY_ATTRIBUTION` + 并列展示两种 category
- [x] **L3 自动晋升**:`promote_lessons.py` + learned overlay + taxonomy 读端集成
- [ ] 让 `analyze_failures.py` 事后诊断也回写 lesson(root_cause 等已入 lesson,可进一步统一)
- [ ] 把高频确定性修复(依赖缺失等)前置进 `setup_env.sh`,减少 Agent 介入
- [ ] 定期(或 CI)跑 `promote_lessons.py --apply`,让覆盖面持续自增长
