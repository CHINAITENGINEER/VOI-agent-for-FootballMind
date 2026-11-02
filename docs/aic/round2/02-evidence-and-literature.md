# FootballMind 最终事实与证据清单 + 文献对照矩阵

版本：round2-v1　日期：2026-10-25
配套文件：[01-audit-report.md](01-audit-report.md)（第二轮审计报告）
本文件中的每一项证据均可通过所列文件路径与脚本直接复核

---

## 一、两个数据集必须分开（强制口径）

后续所有材料必须使用下列标签，禁止跨集累加或跨集比较绝对数值。

| 标签 | 名称 | 规模 | 切分 | 用途 | 定义文件 |
|---|---|---|---|---|---|
| **A 集** | postread 语料（干净扩展版） | 51 场 / 92 篇 / **289 行** | 比赛级 39/6/6；行级 241/24/24 | 文本表示、信息价值估计、策略诊断 | `E:\FootballMind\experiments\postread-clean-expansion-20261025-v1\dataset-summary.json` |
| **B 集** | Understat EPL 历史重建集 | 380 场 → 7 折 → **290 场**测试 | 7 个时间折 | 真实历史策略比较（含正文证据更新） | `docs\parallel\status\round177.md` L71–83 |

A 集补充统计：唯一标题 79；正文词数中位 313（最长 1713）；来源域名 9 个，`leedsunited.com` 166 行（57%）；行级 `realized_gain` 分布 = 正 60 / 零 114 / 负 115。

---

## 二、f 与 g 的接口表（技术级核验结果）

| 组件 | 输入 | 输出 | 数学定义 | 实际代码位置 | 当前证据 |
|---|---|---|---|---|---|
| **f** | 有序证据事件集合（含 `source_published_at`、`source_host`、`document_url`、`entity`、`atoms{concept,predicate,scope}`）+ 初态 | 1X2 概率 `p_t` | 见下 §2.1 | `football_research\acquisition\entity_evidence_updates.py`；冻结模型 `quoted-entity-count-semantic-match-residual-f-v1`，SHA `34ce858f02165c20779a05b6bed20f1c121008ea7fbe389b64022508f0d7b583` | 15 场回放：7 场概率变化、8 场不变；gain 见 `fm-f-v1-legacy-prior-replay-20261025-v1` |
| **f 的证据准入子模块** | 证据事件 | 保留/超越/冲突列表 + 审计记录 | 见下 §2.2 | 同上 `resolve()` | 代码内强制 `published > cutoff` 抛 ValueError；冲突保留为 `unresolved_source_disagreement` |
| **g** | `public_query`（数值特征 + 标题文本）+ `snapshot`（`physical_match_id`、`used_input_ids`、`claims`） | 标量 `ŝ ∈ ℝ`（估计的有符号收益） | 见下 §2.3 | `football_research\acquisition\state_candidate_gain.py`（`VERSION='historical-state-candidate-gain-v1'`） | A 集 289 行；35 切分诊断；Inter 39 条真实菜单评分 |
| **两步前瞻策略** | 状态 `s_t`、预算 `B`、候选动作 `a` | 动作选择或停止 | 见下 §2.4 | `football_research\acquisition\policy_return_continuation_g.py`；`football_agent\text_voi_policy.py` | Round179 合成机制（960 训练 / 240 留出） |
| **方向性门控** | 正文 + 方向标签 + 主体 + 方向依据 + 逐字引文 | `supports`/`refutes`/`no_answer`（+ `unknown + availability_only`） | 规则契约（非学习） | `football_agent\directional_evidence.py` | `mechanism-03/audit.json`：5 条观察 → 1 supports + 4 no_answer。**默认不进入数值求解器** |
| **运行时预算** | — | 硬上限 | — | `demo\live_g.py` L37–41 | `MAX_SOURCE_REQUESTS=10`、`MAX_BODY_READS=5`、`MAX_PROVIDER_CALLS=5`、`RUN_TIMEOUT_MINUTES=15`、`MAX_DYNAMIC_HOSTS=2` |
| **LLM 用在哪** | — | — | — | `demo\live_g.py`（DeepSeek Chat Completions 用于语义抽取/分类） | **数值策略与确定性工具负责选择与停止**；LLM 不参与候选估值 |

### 2.1 f 的数学定义（忠实于代码）

`resolve(row)` 对每条证据事件 `e` 计算 `status(e) → (facet, value, pending, scope)`：

- `concept ∈ PENDING_SELECTION = {possible_bench, club_eligibility_pending}` → `('selection', concept, pending=True, scope)`
- `concept ∈ CONFIRMED_SELECTION` → `('selection', concept, pending=False, scope)`
- `predicate` 以 `availability:` 开头 → `('availability', value, value == 'doubtful', scope)`
- 其他 → `('context:<concept|predicate>', …, False, scope)`（**训练态与医疗态是独立 facet，不互斥**）

超越规则（保留为数值输入的替代关系）：

> 若旧事件 `old` 为 `pending`，且存在同组 `new` 满足
> `(facet='selection' ∧ value ∈ CONFIRMED_SELECTION ∧ scope='this_match')`
> `∨ (facet='availability' ∧ value ∈ {available, unavailable} ∧ scope = old.scope = 'this_match')`，
> 且 `new.source_host = old.source_host`、`new.document_url ≠ old.document_url`、`new.published > old.published`，
> 且所有候选 `new` 的 `value` 一致 → `old` 被超越；
> 否则保留。多个不兼容的已解析断言**不静默裁决**，记入 `conflicts`。

### 2.2 证据准入的三条硬约束（代码级）

1. `published > cutoff` → 抛 `ValueError('Future evidence cannot enter this state')`
2. `e['physical_match_id'] != row['physical_match_id']` → 抛错（比赛绑定校验）
3. 重复 `event_id` → 抛错

### 2.3 g 的数学定义（忠实于代码）

**特征向量**（15 个关系量 + 数值公共特征 + 标题词向量）：

```
x = [ base.numeric(public_query) , base.text_vector(encoder, title) , relations ]
relations =
  known_entities, pending_entities, conflicting_entity_facets, known_body_claims,
  entity_title_overlap, pending_entity_title_overlap, known_body_title_jaccard,
  title_injury, title_lineup, title_team_news,
  pending_x_injury, pending_x_lineup, pending_x_team_news,
  known_availability_x_injury, known_selection_x_lineup
```

各分量均归一化到 [0,1]；`pending_count = min(|pending|,5)/5`；`availability/selection` 同理；`*_title_overlap` 使用**精确整短语匹配**（不做姓氏/别名猜测）；`known_body_title_jaccard` 取正文与标题 token 的 Jaccard 最大值。

**训练目标**：`signed_gain`。**拟合方式**：`Ridge(alpha, solver='svd')`，样本权重 `w_i = 1/(n_matches · n_i)`（比赛等权）。

**输出**：`estimate = x·w + b`（标量）。

> **准确表述**：g 估计的是「在当前可见状态 H 下，把候选 a 的正文并入后、**同一固定 f** 的对数损失有符号改善」的条件期望。它是**经验有符号风险改善估计**，不是精确 Bayes EVSI，不是长期 Bellman Q，也不是经济收益。

### 2.4 两步前瞻公式（忠实于 `submission-brief.md`）

```
V₁(a | s_t)        = R(p_t) − E_z[ R(p_{t+1}) ] − c(a)
V₂(a | s_t, B)     = V₁(a | s_t)
                   + E_{z,M'}[ max( 0, max_{b ∈ K(s', B−c(a))} V₁(b | s') ) ]
R(p)               = −Σ_y p(y) log p(y)          （log loss 下的模型条件期望风险）
c(a)               = Σ_u λ_u C_u(a)              （声明成本 → 损失单位）
```

**适用边界（必须与公式同时出现）**：
- `R(p)` 是 log loss 对应的模型条件期望风险，**不能替代任意实际行动的效用函数**
- 未知未来 URL 以冻结训练请求簇近似规划；**规划枚举被禁止用于真实正文更新**
- `V₂` 评价的是**有限两步子策略**；滚动重规划的最终长轨迹价值 ≠ 某一步的 `V₂`
- 受限子策略价值非正而仍有未知项时 → 暂缓（不是"已证明信息无用"）

### 2.5 方向性门控与 no_answer 的准确语义

| 状态 | 触发条件 | 对后验的作用 |
|---|---|---|
| `supports` / `refutes` | 方向标签、主体、方向依据、逐字引文**四项同时成立** | 允许进入联合模型更新 |
| `no_answer` | 正文存在但不足以支撑该场次的具体断言（如仅说明"处于康复/训练状态"） | **不改变后验** |
| `unknown + availability_only` | 只有状态没有影响方向 | **不改变后验**，且该扩展**默认不接入数值求解器** |

---

## 三、四类证据分级表

### 3.1 已实现（有代码 + 运行记录证明功能存在；可在材料中描述，但须说明范围）

| 能力 | 证据位置 |
|---|---|
| 单场比赛闭环：身份核验 → 候选生成 → 估值 → 获取 → 正文准入 → 更新 → 再决策 → 报告 | `demo\live_g.py`；`docs\agent\backend-release.md` |
| 硬预算约束（10/5/5 请求、15 分钟超时、每轮最多 2 个新域名） | `demo\live_g.py` L37–41 |
| 零网络精确回放（真实原始字节重建完整决策） | Round180 三场比赛八份返回；Round181 独立目录验收 |
| 证据超越与冲突保留（不静默裁决） | `entity_evidence_updates.py::resolve` |
| 重复引文去重（`_digest`） | `state_candidate_gain.py` L58 |
| 方向性证据契约（四条件联合） | `football_agent\directional_evidence.py` |
| 未知价值不被写成零（三分状态：有支持正值 / 有支持非正值 / 未知） | `submission-brief.md`；runtime LIMITATIONS |
| 论文级可复现性：SHA 绑定、封版证据表、主张机器索引 | `var\parallel\agent\round181\evidence\`；`var\releases\footballmind-r181.zip`（SHA-256 `67d540529d3a7bbf3293be7b0068b1cd3ac5d6ad667cafe90b50b8a90769f949`） |
| 扩展语料已落盘待编码：2179 篇 / 1218 场 / 1,544,005 tokens | `postread-semantic-training-ready-20261025-v1\READ_FIRST.txt` |
| 收缩（shrinkage）实现与验证选权 | `postread-reversal-corrected-20261025-v2\run-script.py` |

### 3.2 已验证（有适当实验与对照支持某项性能结论；引用时必须给出指标、样本与条件）

| 结论 | 指标 | 样本 | 条件 | 文件 |
|---|---|---|---|---|
| 在统一读出（alpha=10）下，**没有任何文本配置在 35 切分均值上优于恒零基线** | 比赛等权 macro MSE；零 0.00089320 / train_mean 0.00089625 / source_mean 0.00088991；最低文本配置 0.00094081 | 8 编码器 × 7 臂 × 35 切分 | A 集；alpha 固定 10；仅训练折拟合的标准化与 TF-IDF/SVD | `fm-embedding-bench-20261025-v1\results\exp1_summary.json` |
| **GTE-small 正文臂**是唯一 90% CI 完全低于两条基线的配置 | 平均差 −5.85e−5（vs source_mean）/ −6.05e−5（vs zero），CI 上界均 < 0 | 33 切分（rand30 + loco3）配对 bootstrap | 验证集选 alpha | `exp1_paired_bootstrap.json` |
| **Qwen32 MLP 在 6/6 组均未低于恒零基线**；6 组比赛配对 bootstrap 区间全部跨 0 | 测试 gain MSE | 6 组 × 3 种子（18 次 MLP 重训） | A 集；214 输入宽、128×64 头 | `postread-qwen-stability-20261025-v1\readable-report.txt` |
| 标题通道在该切分上**有害**：原标题 test MSE 0.0021539；替换为训练均值 0.0014257；替换为全零 0.0014138 | MSE | P0 原切分测试 6 场 / 24 行 | 同上 | `postread-shortcut-ablation-20261025-v1\readable-report.txt` L10–15 |
| 验证集最优模型在 P0 测试上**落后恒零 0.000935** | MSE 超额 | P0 | — | 同上 L23 |
| g 在真实菜单上输出近常数：`gain ∈ [0.019519, 0.023954]`，跨度 **0.004435**，std **0.000812** | — | Inter–Fiorentina 当轮 g 实际打分的 39 条 | 单场、单次真实运行 | `exp2_ranking.json` → `g_baseline_inter` |
| 合成机制：预算 0.095 下两步 VoI 的 `loss+cost = 0.947120`，固定/随机 = 0.980864；**但纯 log loss 固定/随机更优（0.900448 vs 0.908912）** | log loss 与 loss+cost | 960 训练 / 240 独立留出；3600 轨迹 | 合成环境 | `submission-brief.md`；Round179 |
| B 集主预算（0.05/2）下条件 VoI **未优于不查询**；同表内固定顺序全查亦更优 | log loss：不查询 1.080229；context residual VoI 1.086871；**固定顺序全查 1.086803** | 290 场、7 折 | 该轮冻结预测器与成本权重 | `round177.md` L75–83 |
| B 集低预算（0.03/1）下纯 log loss 微幅改善（1.079009 < 1.080229），**成本加权后变差（1.085854）** | log loss / loss+cost | 290 场 | 同上 | `round181.md` L53 |
| 冻结后新取得 50 场 B 集：修正策略**全部不查询**，与基线相同 | 1.090590 | 50 场 | 冻结模型 | `round181.md` L54 |
| 收缩修复：**6 组中 3 组验证选权为 0**（chrono → train_mean；loco_chelsea → source_mean；global_time_51 → zero） | 权重 | 6 个命名切分 | 权重候选 {0,0.05,0.1,0.25,0.5,0.75,1}，仅验证集选择 | `postread-reversal-corrected-20261025-v2\acceptance-notes.txt` L7–12 |
| P0 原切分 6 场测试：全文方案相对 `no_information` 平均差 −0.011849，95% CI [−0.020011, −0.003814]（改善 3 / 变差 0 / 相同 3） | 逐场配对 log loss 差 | 6 场 | 每场最多采用 2 篇；**探索性 bootstrap** | 同上 L17 |
| 30 次重排在**实际有效模型**上分散到 15 种不同模型 | 选中计数 | 30 组 | 同一批比赛 | 同上 L14 |
| 遮盖标题的**独立测试**来源识别准确率 33.33%；样本内为 81.3% | 比赛等权准确率 | 独立 6 场 / 12 篇 vs 全部 289 行 | 两个统计量不可互换 | 同上 L22 |
| 来源集中度：`leedsunited.com` 占 166/289 行（57%） | — | A 集 | — | `readable-report.txt` §2.1 |
| 语料外球队查询 OOV 率 30.8%–58.3% | — | 5 个独立菜单 | 词表由 39 场训练集拟合 | `exp2_ranking.json` → `token_diagnostics` |
| 5 场独立菜单共 202 条候选 / 22 条相关 / 14 条 borderline | 单标注者判读（规则与边界项全部公开） | 5 场 | 非双盲 | `exp2_ranking.json`；`bench\menu_labels.json` |

### 3.3 待验证（有设计或初步证据，不足以证明结论；只能作为研究假设或下一步目标）

| 能力 | 现有证据 | 缺什么 |
|---|---|---|
| 方向性门控降低错误更新率 | `mechanism-03` 路径真实运行 | **完全缺少消融**：从未测过去掉任一项后的错误更新率变化 |
| no_answer 优于无约束更新 | 5 条观察中 4 条 no_answer；后验仅在 1 条 supports 时变化 ≤2.8pp | 缺对照：允许 no_answer 更新时的错误率 |
| 两步前瞻优于固定策略 | 合成 loss+cost 优；**纯 log loss 劣** | 缺多发现动作场景（现仅 1 个初始发现动作 + 2 个相同转载，导致"固定 = 随机"） |
| 收缩改善策略 | P0 上 CI 不含 0，但 3/6 组选权为 0 | 需在**未被查看**的冻结集上验证；当前为已查看小样本的探索性 bootstrap |
| 信息价值估计可行 | Qwen256 Ridge 在 3/6 组优于零基线；30 重划 16/30 | 6 组配对 bootstrap 全跨 0；且改变的是向量类型与维度双重因素 |
| 市场赔率同条件对照 | f-v1 回放曾用 `Avg.no_vig_probabilities` 作先验 | `quote_observed_at` **全部 UNKNOWN**、`point_in_time_verified` **全部 false**；无同条件三臂比较 |
| 跨语言能力 | 无 | **零实验**。现有 embedding 对语料外球队 OOV 31%–58%；日语 J.League 官网为 JS 渲染、服务端无文章链接 |
| 跨领域迁移 | 无 | 零实验。需重新定义损失、工具集、身份契约与时间契约 |
| 系统在独立比赛上改善预测 | 无 | 无一场新的前瞻确认比赛 |

### 3.4 不应宣称（现有证据不支持，或与结果矛盾）

| 禁止表述 | 原因 |
|---|---|
| 「首次提出信息价值驱动的 Agent」「首次将 VoI 引入 AI」 | Howard 1966 起已定型 |
| 「已证明 g 能提高预测准确率」 | B 集主预算为负结果；A 集无配置优于恒零 |
| 「实现全局最优停止」 | 无理论保证；仅在声明的有限菜单内停止 |
| 「概率已校准」 | 项目自述未成立；`model_conditioned` 概率未做 ECE |
| 「稳定盈利」 | 无投资组合、无交易成本模型；成本为 log-loss 换算额度 |
| 「多智能体协作形成算法创新」 | 自述为开发效率收益，未建立产品收益；MAST 亦不支持默认增益假设 |
| 「跨语言 / 跨领域能力已具备」 | 零实验 |
| 「g 的预测精度为 0.544」 | 目标不等价（见审计报告 §二） |
| 「所有同类方法都失败」 | 只测了 8 个编码器、Ridge/MLP 两种读出、单一目标函数 |
| 「佐证材料已具备」 | 无专利、软著、论文、合作意向、用户反馈、试点证明 |

---

## 四、文献对照矩阵（原始来源已逐条核验）

### 4.1 构成直接先例的文献（必须主动承认）

| 文献 | 原始来源核验 | 实际研究内容 | 对应的项目主张 | 是否构成直接先例 | 对创新性判断的影响 |
|---|---|---|---|---|---|
| **WC2026-Agents**<br>*FIFA World Cup 2026 as a Contamination-Free Benchmark for LLM Forecasting Agents: Four Models, a Bookmaker, and 104 Matches*<br>Ding, Guo, Xu（2026-07-20） | ✅ 逐字核验：[arXiv:2607.17765](https://arxiv.org/abs/2607.17765) | 四模型（Claude Opus 4.8 / GPT-5.5 / Gemini 3.1 Pro / Grok）执行 search–act–reflect 循环，输出 1X2 分布 + 虚拟 100 美元下注；博彩市场为第五方基线 | 「自主检索 → 概率预测 → 赛后评估」 | **是（最强）** | ⚠️ 摘要原句 *"the four agents issue an identical top pick in 92% of matches and none beats the market's Brier score"* —— **"Agent 自主检索新闻能否超越市场"在 2026 年已被答为"不能"**。项目若以准确率为卖点，主张已被实质覆盖 |
| **WorldCupArena**<br>Wang, Gui, Rao, Di, Tang, Liang（v1 2026-07-20 / v2 2026-08-31） | ✅ 逐字核验：[arXiv:2607.18084](https://arxiv.org/abs/2607.18084) | 赛前 24 小时：S1 给统一证据包（禁工具）或 S2 只给赛程自行联网；提交比分/赛果/球员预测，赛后打分；市场与人类球迷为基线 | 同上 | **是** | 摘要只说相对基线"小幅提升"。**修正**：正文报告无检索版本得分更高，但作者明确声明 *"the comparison is not a causal retrieval ablation"*——**不能引用为"检索无增益"的因果结论** |
| **OptionWorld**<br>Zhang, Liu, Tian, Feng, Liu（2026-10-20） | ✅ 逐字核验：[arXiv:2610.28491](https://arxiv.org/abs/2610.28491) | 追踪长期个人助理在模拟日常任务中的信息获取、判断、行动与结果 | 「信息获取是独立失败维度」 | 是（但不涉足球） | 摘要确含 35.5% 与 "needed information is not obtained or combined"。**任务域为长期个人助理，与足球无关**；可作问题动机外部支持，不可作方法对照 |
| **Self-RAG / FLARE / CRAG** | ✅ [2310.11511](https://arxiv.org/abs/2310.11511) / [2305.06983](https://arxiv.org/abs/2305.06983) / [2401.15884](https://arxiv.org/abs/2401.15884) | 按需触发检索；反思 token 判定相关性/支持性；低置信多轮获取 | 「自主生成信息需求」「正文级证据验证」 | **是** | 三件套均已存在 |
| **SIM-RAG**<br>*Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing*<br>Diji Yang, Linda Zeng, Jinmeng Rao, Yi Zhang，SIGIR '25, pp.1305–1315 | ✅ DOI 与著录核验：[10.1145/3726302.3730018](https://doi.org/10.1145/3726302.3730018) | 训练 Critic 判断信息是否充分、是否继续检索 | 「何时停止获取」 | **是** | 项目增加了成本项与物理预算，但不构成方法首创 |

### 4.2 理论基础的先行文献（不可主张首创）

| 文献 | 核验 | 与项目的关系 | 是否削弱主张 |
|---|---|---|---|
| **Howard 1966, Information Value Theory** | ✅ DOI 与著录完全正确：[10.1109/TSSC.1966.300074](https://doi.org/10.1109/TSSC.1966.300074)；IEEE Trans. SSC 2(1):22–26 | VoI/EVPI 的定义 | 削弱「信息价值」的原创性主张 |
| **Chaloner & Verdinelli 1995, Bayesian Experimental Design: A Review** | ✅ [10.1214/ss/1177009939](https://doi.org/10.1214/ss/1177009939) | 「该获取什么信息」的形式化 | 削弱 |
| **Krause & Guestrin, Near-optimal Nonmyopic Value of Information in Graphical Models** | ✅ [arXiv:1207.1394](https://arxiv.org/abs/1207.1394)（UAI 2005） | 预算约束下的非短视信息价值与次模近似 | **最强削弱**：预算化信息选择已有可证明近似 |
| **Strong 2015 / Heath 2019（EVSI 回归估计）** | ✅ [10.1177/0272989X15575286](https://doi.org/10.1177/0272989X15575286) / [10.1177/0272989X19837983](https://doi.org/10.1177/0272989X19837983) | 用回归/机器学习估计新增信息的期望价值 —— **与 g 的角色同构** | **最强削弱**：g 不是新概念。**注：Hey 等只在大样本或模拟中验证，未报告小样本真实文本的失效边界 → 这是项目唯一可守的空间** |
| **Rainforth et al. 2024, Modern Bayesian Experimental Design** | ✅ [10.1214/23-STS915](https://doi.org/10.1214/23-STS915) | 用深度学习摊销估计 EIG | 削弱 |
| **Russell & Wefald 1991, Principles of Metareasoning** | ✅（Springer/AIJ；作者主页可核） | 把推理步骤当作带效用的动作（VOC），决定是否继续推理 | 削弱「何时停止」的原创性 |
| **DynaKRAG**<br>Zhou 等（v1 2026-07-07 / v2 2026-08-22） | ✅ [arXiv:2607.06507](https://arxiv.org/abs/2607.06507) | 把检索/改写/充分性判断/停止建模为状态条件动作选择；含 `sufficiency_check`、`stop_answer` 等原子操作 | **是**：项目「基于信息价值与成本预算的停止」已被可学习控制器覆盖。原文含 *"additional retrieval is not uniformly beneficial"* |
| **BALAR: A Bayesian Agentic Loop for Active Reasoning**<br>Echarghaoui, Wu, Fox（2026-05） | ✅ [arXiv:2605.05386](https://arxiv.org/abs/2605.05386) | 贝叶斯主动追问 + 信息增益驱动停止 | 是 |
| **Evidential Deep Learning**（Sensoy 等 2018） | ✅ [arXiv:1806.01768](https://arxiv.org/abs/1806.01768) | `evidence = 0 → 高不确定/可拒绝` | 削弱 `no_answer` 的原创性；项目的差异在**语义适用性**而非证据量 |

### 4.3 领域基线（回答"为什么不直接用市场赔率"）

| 文献 | 核验 | 关键结论 |
|---|---|---|
| **Wunderlich & Memmert 2018, The Betting Odds Rating System** | ✅ [10.1371/journal.pone.0198668](https://doi.org/10.1371/journal.pone.0198668) | 「赛前赔率比比赛结果本身含更多信息」；近 15,000 场验证 |
| **Clegg, Song, Cartlidge 2026, market-calibrated AFT for in-play football** | ✅ [arXiv:2605.16066](https://arxiv.org/abs/2605.16066) | 「market calibration is the dominant driver of predictive accuracy」 |
| **Brown et al. 2018, Forecasting With Social Media: Evidence From Tweets On Soccer Matches** | ✅ [10.1111/ecin.12506](https://doi.org/10.1111/ecin.12506) | 1380 万推文 + Betfair 价格；文本含赔率外增量，**但仅在进球/红牌后瞬间** |
| **Kaunitz et al. 2017, Beating the bookies with their own numbers** | ✅ [arXiv:1710.02824](https://arxiv.org/abs/1710.02824) | 盈利来自赔率结构错价，且随即被限注 |
| **Dixon & Coles 1997** | ✅ [10.1111/1467-9876.00065](https://doi.org/10.1111/1467-9876.00065) | Poisson + 低比分修正；「模型 vs 赔率」框架自 1997 年成熟 |

### 4.4 失败模式与时代背景（用于问题动机）

| 文献 | 核验 | 关键结论 |
|---|---|---|
| **MAST：Why Do Multi-Agent LLM Systems Fail?** Cemri 等 | ✅ [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)；[NeurIPS 2025 D&B Track](https://proceedings.neurips.cc/paper_files/paper/2025/file/b1041e52d3be19f0a9bc491657488e4a-Paper-Datasets_and_Benchmarks_Track.pdf) | 7 个 MAS 框架、200+ 任务、**1600+ 标注轨迹**（人工精析 150 条），14 种失败模式；三大类为 system design / inter-agent misalignment / task verification |
| **Loud Failures, Quiet Failures** Kraishan（2026-10-07） | ✅ [arXiv:2610.10062](https://arxiv.org/abs/2610.10062)。⚠️ **arXiv 页面无 "IEEE CogMI 2026" 字样，该录用信息无法证实，已删除** | 工具静默错误时 agent 仅 **58.8%** 察觉；显式报错 91.3% |
| **Calibration, Not Answer Selection** Xi 等（2026-09-27） | ✅ [arXiv:2609.33290](https://arxiv.org/abs/2609.33290) | 口头置信度追踪的是"是否愿下决心"而非"是否正确"；隐藏态探针 ECE 低 5–38 倍 |
| **Lost in the Middle**（2023）/ **Chroma Context Rot**（2025） | ✅ [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) / [research.trychroma.com](https://research.trychroma.com/context-rot) | 中段信息坍塌；相似干扰项危害大于无关项；长度增长本身降低可靠性 |
| **GRACE：Scoped Verification for Long-Horizon Agentic Context Evolution** Hsu & Lu | ✅ [arXiv:2607.09175](https://arxiv.org/html/2607.09175)。**修正：pass^3 终值为 0.673 ± 0.136（此前误写 0.661）** | 长程上下文演化的结构化验证 |
| **Every pooling rule has its world: matching probability combination rules to situations and stakes** Tammet, Järv, Draheim（2026-08-11） | ✅ [arXiv:2608.11275](https://arxiv.org/abs/2608.11275) | 独立证据假设违背/重复证据的合并问题 |
| **Provenance-Aware Belief Updating in Human-AI Feedback Loops** Huang 等（2026-10-09） | ✅ [arXiv:2610.13008](https://arxiv.org/abs/2610.13008) | 来源血缘感知的信念更新（LQET） |
| **Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG**（v1 标题为 TARG）Wang, Wei, Ling | ✅ [arXiv:2511.09803](https://arxiv.org/abs/2511.09803)。**修正：不是综述；FLARE/Self-RAG/CRAG 出现在其相关工作中** | 训练无关的自适应检索门控 |
| **Halawi et al. 2024, Approaching Human-Level Forecasting with Language Models** | ✅ [arXiv:2402.18563](https://arxiv.org/abs/2402.18563) | 检索增强 LM 预测系统接近竞争性人类预测者 |

---

## 五、当前可主张的增量（严格限缩后）

1. **小样本真实文本下条件信息价值估计的失效边界**——文献中的 EVSI 回归类工作（Strong 2015、Heath 2019）均在模拟或大样本中验证；**未检索到在「51 场真实比赛 / 92 篇真实新闻 / 289 行 / 来源集中 57% / 标签零值占 39.4% / 特征列半数恒零」这一条件下报告失效边界的公开工作**。
   - 表述限制：这是**「在本人检索范围内未找到」**，不是「全世界没有人做过」。
   - 可复核性：35 切分 × 8 编码器 × 7 臂基准、配对 bootstrap、700 点零误差跨基准一致性。
2. **可审计的保守证据准入契约**（方向 + 主体 + 方向依据 + 逐字引文四项联合，状态描述降级为 `availability_only`）——在 Self-RAG/FLARE 的反思 token 与 EDL 的证据量弃答中不显式存在。
   - 表述限制：**待验证**。缺消融，缺校准。
3. **一个可复用的诊断协议**——把「验证集选模型 → 测试落后基线」的代价量化到单点（P0：验证最优 test MSE 0.002154 vs 恒零 0.001219；超额 +0.000935），并给出 30 次重排分布（选出 15 种不同模型）。

**不能主张的**：任何形式的算法原创性、最优性、跨域/跨语言能力、经济收益、市场优势。
