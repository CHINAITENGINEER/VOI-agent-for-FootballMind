# FootballMind 正文证据投影与全文状态 `g` 设计 v1

日期：2026-10-24  
状态：设计完成，尚未接入正式 runtime、尚未替换当前在线 `g`

## 目标

当前界面已经能显示正文全文和语义抽取结果，但读者需要自己从全文中寻找真正支持某条 claim 的句子。第一阶段要把正文中**可追溯、可核验、与当前比赛有关的原句**单独投影出来，增强证据链的可读性；第二阶段再建立一个允许 `g` 读取全文的独立研究入口。

两项能力必须分开：

1. 证据投影只改善展示和审计，不改变 `f`、`g`、概率、收益或选择结果。
2. 全文 `g` 是新的 post-read 研究接口，不冒充当前在线的 pre-query `g`，也不自动上线。

## 当前实现的事实边界

当前在线链路位于 `demo/live_g.py`：

```text
市场先验 + 已采用 claims + 标题/年龄/状态
    -> 在线 g 的 pre-query 特征
    -> 选择是否读取正文
    -> 取得正文
    -> DeepSeek typed-v3 语义抽取
    -> quote/source assertion 校验
    -> 当前 f 重新计算 prediction_after
```

`demo/live_g.py::_decision` 在正文尚未取得时调用 `football_research.acquisition.state_candidate_gain` 的查询特征。`football_research.acquisition.historical_gain_head::text_vector` 和相关预查询代码使用的是标题的词/二元词 TF-IDF（训练时最多 32 个词项）以及公开状态关系；正文尚未可见，所以此处不可能使用正文 embedding。

已有的离线 post-read 入口位于 `football_research/acquisition/postread_g_use.py`。它的 `build_features(..., mode='body_state')` 接受调用者提供的 384 维正文向量，拼接当前状态、与已采用证据的相似度、候选向量、已采用向量均值、差值和乘积。它只允许 `g_seen_ids` 中的候选进入评分，并把 `g_seen_ids` 与 `f_adopted_ids` 分开保存。它不负责生成 embedding，也没有绑定到在线 runtime。

正文向量有两类已存在的研究产物，不能混称：

* `tools/vectorize_e_drive_football_corpus.py` 生成本地 `sentence-transformers/all-MiniLM-L6-v2` 的 384 维 masked-mean、L2 归一化向量，长正文按 token 分块后聚合；正文不静默截断。
* `tools/build_batch_training_readiness.py` 所交接的历史 bridge 使用 384 维字符 TF-IDF（3–5 gram）正文向量，并生成 393 维 body+state 向量。

因此，“文章先 embedding 再进入 g”对当前在线运行是不成立的；对离线 post-read 研究入口是成立的，但必须明确 encoder、版本、正文 SHA 和可见性。

## A. 可审计的正文重要片段投影

### A1. 输入

每个已读取正文步骤保留：

* `body`、`raw_sha256`、`input_id`、`url`、`source_published_at`、请求/返回/保存/交接时钟；
* typed-v3 `claims`，包括 `team`、`entity`、`kind`、`value`、`scope`、`statement`、`quotes`；
* 当前比赛身份和 `prediction_before/after`。

### A2. 投影规则

投影器只从 claim 的逐字 `quotes` 在原文中定位片段，不生成摘要，不改写正文，不把标题当正文。每个片段都输出：

```json
{
  "claim_index": 0,
  "quote": "正文中的连续原句",
  "quote_verified": true,
  "body_start": 123,
  "body_end": 245,
  "body_context": "引句前后有限上下文",
  "entity": "球员或球队",
  "team": "home|away|unknown",
  "kind": "availability|selection|tactical_plan|...",
  "value": "unavailable|doubtful|...",
  "scope": "this_match|team_context|historical|unresolved",
  "statement": "语义抽取的结构化表述"
}
```

### A3. 展示排序，不改变模型决策

UI 可以按以下固定、可解释规则排序显示前 3 条，并保留“查看全部”：

1. `quote_verified=true` 优先；
2. `scope=this_match` 优先于 `team_context`、`historical`、`unresolved`；
3. `kind` 属于 `availability`、`selection`、`tactical_plan`、`schedule`、`stakes` 优先；
4. 有明确 `entity`、`team` 和 `value` 优先；
5. 同一 `body_start/body_end` 或同一 claim 去重；
6. 冲突、条件、不确定性保持原样并打标签，不能因为“重要”而删掉反例。

排序结果只叫 `display_priority` 或 `evidence_display_rank`，不能叫信息价值、收益或可信度。它不进入 `g` 特征，不改变 f 输入，不生成 gain。

### A4. 失败和缺失

* 引句不在正文中：保留 claim，`quote_verified=false`，不把它放入“已核验原句”区。
* claim 没有引句：保留在“未提供可核验引句”区，不补造原句。
* 只有 `other/unmodeled`：显示为未建模上下文，不移植为伤停、首发或本场事实。
* 来源时间未知：显示 `UNKNOWN`，不把抓取时间改成发布时间。
* 同一正文多版本：保留 body SHA、版本和来源时间，不能合并成一个独立样本。

当前 `demo/live_g.py::_evidence_highlights` 已实现精确 quote、offset、context；`demo/static/live.js::evidenceHighlightHtml` 已能展示这些字段。下一步只需增加展示排序字段和冲突/不确定性标签，不需要改变 f/g 计算。

## B. 全文加状态的 post-read `g`

### B1. 决策位置

```text
来源发现 -> 正文获取 -> g 读取全文/正文向量
        -> g 选择 adopt 或 skip
        -> 被 adopt 的正文才进入 claims/f
        -> f 输出 prediction_after
```

全文已经被读取，因此 g 的总系统成本必须包含来源和正文读取成本。post-read g 只能声称减少后续 f 处理成本，不能声称节省已经发生的正文获取成本。

### B2. 明确的状态和候选输入

`s`：

* `prediction_before` 和其来源模型版本；
* 当前已采用 evidence 的 IDs、claim 摘要、时间和来源；
* 候选菜单总数、已读数、剩余预算；
* 来源侧、文章年龄、重复/冲突/不确定性标记；
* 当前比赛、球队和赛季关联。

`d`：

* 正文或正文向量；
* `body_sha256`、encoder 名称/版本、向量维度、chunk 规则；
* 正文可见性和读取时钟；
* 可选的 claim-backed evidence projection，但必须标记为 `g_read`，不等于 `f_adopted`。

明确禁止：赛果、`prediction_after`、`realized_gain`、赛后文章、未来信息和由结果反推的正文标签。

### B3. 特征布局

第一版保持现有低复杂度接口，不直接训练大模型：

```text
[public_numeric(11 + host_count), max_similarity(1), body_embedding(384),
 adopted_state_mean(384), body_minus_state(384), body_times_state(384)]
```

因此当前 `postread_g_use.py` 的 `body_state` 维度是 `1548 + host_count`，不是一个固定的 1546 维接口；`host_count` 来自本次数据集的来源主机字典。另一个历史 bridge 中的 393 维 body+state 向量是不同的研究产物，不能与这里的布局混用。同时保留 `body_only`、`title_state`、`title_local` 作为对照。embedding 生成器、版本、SHA 和可见性写入 dataset manifest；训练和验证按 `physical_match_id` 分组。

### B4. 标签和防泄漏

固定 f 后，赛果可用时才计算：

```text
loss_before = -log(p_before[y])
loss_after  = -log(p_after[y])
realized_gain = loss_before - loss_after
```

正、零、负收益全部保留。`g` 的输入构建函数必须不能访问 `prediction_after`、`outcome` 或 `realized_gain`；标签生成与输入构建分离，避免把标签反向写入正文表示。

`g_seen_ids` 与 `f_adopted_ids` 分开记录：g 可以读过正文但不把它交给 f；只有 `f_adopted_ids` 才进入 f 的状态。

## C. 交付和验收顺序

1. **证据投影 v1**：在现有 run schema 上增加 `display_priority`、`priority_reasons`、`uncertainty_flags`、`conflict_flags`；生成 UI 只读投影和单元测试。
2. **全文状态 dataset v1**：复用已有正文和向量，补齐 `body_sha256`、encoder、版本、时钟、`g_seen/f_adopted` 字段；不调用网络。
3. **离线对照**：固定 f，比较常数、title-only、body-only、body+state；按比赛分组输出 MSE、Spearman、top-k、最终 log loss 和完整成本。
4. **错误分析**：检查正文表示、状态表示、f 能力、g 选择、重复/冲突、时间版本和标签噪声。
5. **独立比赛验证**：只有独立前瞻轨迹和完整 before/after/outcome 链路通过后，才讨论是否新增正式 runtime 接缝。

## 当前结论

* 重要片段可以先做，而且应当是 claim-backed 原句和 offset，不应先做自由摘要。
* 当前在线 g 是标题/状态 pre-query g，不是正文 embedding g。
* 当前离线 post-read g 已支持 384 维正文向量加状态，但仍是研究接口，`production_ready=false`，不能自动获得在线选择权。
* 下一步最高收益是先完成证据投影排序和 dataset schema，再在固定 f 上做同条件 body+state 对照，而不是直接把离线模型接入在线运行。
