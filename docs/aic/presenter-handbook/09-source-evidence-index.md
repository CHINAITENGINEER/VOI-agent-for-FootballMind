# 09 原始证据索引（现场核查用）

**使用方式**：当评委要求"拿证据出来"，或你自己想确认某个数字时，从这里查。

**标记说明**：
- 🟢 **有原始产物**：可直接打开文件核对
- 🟡 **仅有报告摘要**：数值来自报告文本，未找到更底层的逐条产物
- 🔴 **未找到原始证据**：不得引用

---

## 一、数据与语料

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| 主注册表规模 | 26,271 条唯一 ID | `E:\FootballMind\experiments\full-history-sequential-fg-20261025-v1\final-analysis\data-coverage.json` → `master_registry_rows` | 🟢 |
| 结构化监督比赛数 | 24,911 | 同上 → `structured_supervised_matches` | 🟢 |
| 正文序列比赛数 | 1,520 | 同上 → `eligible_sequence_matches`；`data\dataset-summary.json` → `sequence_matches` | 🟢 |
| 去重正文数 | 3,001 | 同上 → `eligible_document_unique_shas`；`body_routes_mutually_exclusive.eligible_sequence_body` | 🟢 |
| 源顺序前缀数 | 4,521 | `research-delivery-v1\readable-report.txt` §2「全部源顺序前缀共 4521 条，含每场 s0」 | 🟡 |
| 六段划分 | 294 / 74 / 479 / 160 / 193 / 320 | `data\dataset-summary.json` → `sequence_matches_by_role` | 🟢 |
| 六段交集为 0 | 15 个交集字段全为 0 | `final-analysis\split-integrity.json` → `role_intersections` | 🟢 |
| 13 项数据自检通过 | `all_passed: true` | `data\data-validation.json` | 🟢 |
| 字段定义与元数据名 | 22 个元数据名 | `data\schema.json` | 🟢 |
| 时间对齐（日级） | 3,001 / 3,001 严格早于比赛日 | `final-analysis\data-coverage.json` → `eligible_publication_day_alignment` | 🟢 |
| 时间戳缺口 | `source_available_at` 3,001 全缺；`kickoff` 缺 2,882 | 同上 → `eligible_fields_missing` | 🟢 |
| 测试集开发暴露 | test 55 / 320；validation 39 / 193 | `data\data-validation.json` → `previously_exposed_main_test_matches`；`readable-report.txt` §2 | 🟢 |
| 来源域名分布 | 38 个域名（`www.chelseafc.com` 295 篇最多） | `final-analysis\data-coverage.json` → `eligible_source_domain` | 🟢 |
| 来源扫描范围声明 | 「已识别本地来源族的覆盖，不证明穷尽」 | 同上 → `source_discovery_scope.coverage_claim` | 🟢 |

---

## 二、模型配置

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| f 架构 | `set_residual_mlp`，参数 26,823 / 可训练 26,819 | `models\f\model.json` | 🟢 |
| f 的元数据名 | 22 个（生效 16 个） | 同上 → `config.metadata_names` / `effective_metadata_names` | 🟢 |
| f 的训练规模 | f_train 294 场 / f_dev 74 场 | 同上 → `research_metadata.f_train_matches` / `f_dev_matches` | 🟢 |
| f 的 best_epoch | **1** | 同上 → `research_metadata.best_epoch` | 🟢 |
| f 的优化器配置 | AdamW lr 0.001 / wd 0.01 / betas [0.9,0.999] / eps 1e-8 / grad clip 5.0 / batch 32 / max 200 / patience 25 | 同上 | 🟢 |
| f 的损失 | 交叉熵，先对前缀平均（含 s0），再按比赛等权 | 同上 → `research_metadata.loss` | 🟢 |
| s0 的含义 | `"calibrated market prior; no text residual"` | 同上 → `s0` | 🟢 |
| f 参数在推理期不变 | `parameter_updates_during_predict: false` | 同上 | 🟢 |
| f 的 SHA | `b36b2cf87f379034224c314794efc6ec59266dd4cf4752383965c41fb63661e0` | 同上 → `f_model_sha256` | 🟢 |
| g 的七个头与参数 | zero 0 / train_mean 1 / state_ridge 350 / fulltext_ridge 1141 / local_metadata_refit_ridge 52 / fulltext_mlp 各 154,369 | `models\g\model.json` → `configurations` | 🟢 |
| g 的输入维度 | 1140；`state_width` 349 | 同上 → `feature_dimension` / `state_width` | 🟢 |
| g 的 13 个特征块与起止 | 完整分块表 | `models\g\feature-schema.json` → `blocks` | 🟢 |
| g 的禁止输入 | `outcome` / `realized_gain` / `candidate_after_probability` / `candidate_after_hidden_state` | 同上 → `forbidden_inputs` | 🟢 |
| 开发集选定策略 | `state_ridge`，阈值 0.0，预算 3 | `models\g\model.json` → `selected` | 🟢 |
| g 的 SHA | `5af947711279765838adb02e7dc09c2a9525a89e90db3e23e4f68a865ed88eff` | 同上 → `g_model_sha256` | 🟢 |
| g 的训练规模 | 479 场 / 7,600 行拟合；160 场 / 2,846 行开发 | 同上 → `rows: 10446` / `matches: 639`；`readable-report.txt` §4 | 🟢 |
| 标签分布（集合 f 侧） | 正 5,904 / 零 0 / 负 4,542 | 同上 → `positive` / `zero` / `negative` | 🟢 |
| failure_rows | 4,540（`input_preparation / sequence_not_eligible`） | 同上 → `failure_rows`；`readable-report.txt` §4 | 🟢 |
| 编码器配置 | Qwen3-Embedding-0.6B，revision `97b0c614…`，1024 原生 / 256 前缀 | `f-fit-vectors\encoder-provenance.json`；`models\f\model.json` → `encoder_provenance` | 🟢 |
| 编码器权重 SHA | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` | 同上 | 🟢 |
| GRU 参数 | 67,079 / 可训练 67,075，隐状态 64 | `research-delivery-v1\readable-report.txt` §3 | 🟡 |

---

## 三、实验结果

### 3.1 f 的 heldout 指标

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| 原始市场先验 | validation 0.965245 / test 1.020413 | `research-delivery-v1\f-heldout-metrics.csv` | 🟢 |
| 校准先验（s0） | validation 0.963980 / test 1.022476 | 同上 | 🟢 |
| 集合 f | validation 0.962041 / test 1.022482 | 同上 | 🟢 |
| GRU f | validation 0.958916 / test 1.026598 | 同上 | 🟢 |
| Brier / accuracy | 随行给出 | 同上；`f-models\evaluation\selected-model-metrics.json` | 🟢 |
| 独立数值复算 | 各命名切分误差 < 1e-12 | `f-models\evaluation-verification-v1\metric-reproduction.csv` | 🟢 |

### 3.2 完整策略比较

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| 五策略 × 四预算全表 | budget ∈ {1,2,3,6} | `policy-evaluation\set\strategy-comparison.json` → `summary`（88 行） | 🟢 |
| 预算 3 的五策略值 | 0.963980 / 0.962062 / 0.962063 / 0.963420 / 0.965245（验证）；1.022476 / 1.022446 / 1.022450 / 1.021563 / 1.020413（测试） | 同上 | 🟢 |
| 配对 bootstrap | validation 差 +0.001357 CI [−0.000466, +0.003016]；test 差 −0.000886 CI [−0.002299, +0.000438] | `research-delivery-v1\readable-report.txt` §5 | 🟡 |
| bootstrap 参数 | 比赛为抽样单元、2,000 次、seed 20261025 | 同上 | 🟡 |
| 实验条件 | `threshold = 0.0`；`candidate_pool = all eligible documents from frozen sequence; no gain-based truncation`；`random_repeats = 10` | `strategy-comparison.json` | 🟢 |
| state_ridge 同状态同分 | 「334 个多候选决策状态中全部候选同分」 | `var\releases\footballmind-defense-20261025-rc1\bundle-config.json` → `limitations[2]` | 🟢 |
| GRU 侧极端停止 | test 320 场只采用 10 次；115 场英超全部停止 | `research-delivery-v1\readable-report.txt` §4 | 🟡 |

### 3.3 g 的回归与排序指标

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| 恒零基线 macro MSE | g_fit 0.00020807 / g_dev 0.00018271 | `provenance\g-recorded-metrics.json` | 🟢 |
| state_ridge macro MSE | g_fit 0.00020628 / g_dev 0.00018614 | 同上 | 🟢 |
| state_ridge Spearman | g_fit 0.2213 / g_dev 0.0096 | 同上 | 🟢 |
| state_ridge 正类符号准确率 | g_fit 0.6266 / g_dev 0.5130 | 同上 | 🟢 |
| 全文 MLP macro MSE | g_fit 0.0000067 / g_dev 0.0002199 / test 0.0002524 | `research-delivery-v1\readable-report.txt` §4 | 🟡 |
| 恒零 test macro MSE | 0.00021162 | 同上 | 🟡 |

### 3.4 顺序敏感性

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| 集合 f 置换不变 | 最终 L1 = 0 | `research-delivery-v1\readable-report.txt` §6 | 🟡 |
| GRU 顺序效应 | 多文 324 场最大 L1 均值 ≈ 0.001374，最大 0.0122995，无类别变化 | 同上 | 🟡 |
| 完整池规模 | 513 场 / 1,041 篇；两 f 共 6,156 行 | 同上 | 🟡 |

---

## 四、冻结演示包

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| release_id | `footballmind-defense-20261025-rc1` | `var\releases\footballmind-defense-20261025-rc1\bundle-config.json` | 🟢 |
| manifest SHA | `dcb05e9766310843390c3c7de11a43608a23ae3160547fb6ab9006753b55bcf8` | `validation\acceptance.json` | 🟢 |
| 核心文件 SHA 校验 | 364 个通过 | 同上 → `core_files: 364` | 🟢 |
| 案例规模 | 160 场 / 342 篇 | `bundle-config.json` → `cases` / `unique_bodies` | 🟢 |
| 默认案例 | `fd:E0:2425:2025-01-01:Brentford:Arsenal` | 同上 → `default_case_id` | 🟢 |
| 默认策略 | `fulltext_mlp_seed20261027`，阈值 0.01，预算 3 | 同上 → `presentation_default` | 🟢 |
| 默认 MLP 配置 | 154,369 参数，hidden [128,64]，selected_epoch 42，epochs_run 67 | 同上 | 🟢 |
| 开发集选定策略 | `state_ridge`，阈值 0.0，参数 350，alpha 10.0，输入 349 维 | 同上 → `development_selected_policy` | 🟢 |
| 案例选择规则 | 全部 160 场 g_dev，按比赛日期排序，**不按赛果筛选** | 同上 → `case_selection` | 🟢 |
| 160/160 逐条一致 | `selected_ids / stop / p_after` 一致，**最大概率差 = 0** | `validation\acceptance.txt` L11 | 🟢 |
| 逐场完整记录 | 160 行 | `validation\all-cases-default-run.jsonl` | 🟢 |
| 赛果读取被拦截 | **触发 0 次** | `validation\acceptance.txt` L13 | 🟢 |
| 依赖锁定 | numpy 2.3.5 / torch 2.11.0 / scikit-learn 1.8.0 / scipy 1.16.3 / joblib 1.5.3 / Python 3.13.9 | `bundle-config.json` | 🟢 |
| 六条自述限制 | 含"非实时预测""未证实优于随机""state_ridge 非排序模型""成本已发生" | 同上 → `limitations` | 🟢 |

### 4.1 Brentford–Arsenal 案例逐项

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| 三组概率 | 市场 14.24/19.79/65.97；f 12.95/19.39/67.65；采用后 13.41/20.01/66.58 | `validation\default-run.json` → `p_before` / `p_after`；`data\cases.json` → `prior` | 🟢 |
| 5 条候选得分 | 0.006819 / **0.014555** / −0.000924 / 0.011135 / 0.007537 | `validation\default-run.json` → `steps[0].candidate_scores` | 🟢 |
| 采用编号 | `sha256:84a19abe57e23a022be9db4a35655a43efdf48b385506d93a930451b97d007e8` | 同上 → `selected_ids` | 🟢 |
| 采用文章 | `arsenal.com/news/pick-your-mens-player-of-the-month-for-december` | `data\cases.json` → `documents[1].source_url` | 🟢 |
| 停止原因 | `predicted_gain_below_frozen_threshold` | `validation\default-run.json` → `stop_reason` | 🟢 |
| 实算 Δ_f | **−0.015919** | **本项目自算**（`log p_after(Y) − log p_before(Y)`，Y = away），非原始文件字段 | 🟡 |
| 赛果 | `outcome_class: 2`（客胜） | `settlement\outcomes.json` | 🟢 |
| 市场先验来源 | football-data.co.uk `E0.csv`，SHA `d0c8ce4a…` | `data\cases.json` → `prior_provenance` | 🟢 |
| 市场先验时点 | **`prior_time_verified: false`** | 同上 | 🟢 |

> ⚠️ **Δ_f = −0.015919 是我们用公开记录自算的**，不是冻结包里的现成字段。若评委要求，可以现场用 `p_before`、`p_after` 与 `outcome_class = 2` 手算验证。

---

## 五、早期实验（C 集与 D 集）

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| C 集规模 | 51 场 / 92 篇 / 289 行；比赛级 39/6/6；行级 241/24/24 | `E:\FootballMind\experiments\postread-clean-expansion-20261025-v1\dataset-summary.json` | 🟢 |
| C 集标签分布 | 正 60 / 零 114 / 负 115 | `postread-semantic-training-ready-20261025-v1\READ_FIRST.txt` | 🟢 |
| C 集特征退化 | 57 状态列中 35 恒零；29 候选列中 11 恒零 | `postread-qwen-stability-20261025-v1\readable-report.txt` | 🟢 |
| 35 切分 × 8 编码器基准 | 56 个文本配置无一优于恒零 | `var\work\fm-embedding-bench-20261025-v1\results\exp1_summary.json`；`readable-report.txt` | 🟢 |
| 基线值 | 恒零 0.00089320 / 训练均值 0.00089625 / 来源均值 0.00088991 | 同上 → `baselines` | 🟢 |
| 最低文本配置 | 0.00094081（project_tfidf_sparse 正文臂） | 同上 → `per_encoder.project_tfidf_sparse.body.test_mse_mean` | 🟢 |
| 700 对照点零误差 | `n_points: 700`，`max_abs_diff: 0.0` | 同上 → `audit_crosscheck` | 🟢 |
| GTE-small 正文臂 CI | vs 来源均值 −5.85e-5 [−9.36e-5, −2.60e-5]；vs 恒零 −6.05e-5 [−9.55e-5, −2.45e-5] | `results\exp1_paired_bootstrap.json` | 🟢 |
| 标题消融 | 原标题 test 0.0021539 → 训练均值 0.0014257 → 全零 0.0014138；恒零 0.0012191 | `postread-shortcut-ablation-20261025-v1\readable-report.txt` | 🟢 |
| 30 重排选出 15 种模型 | 逐项计数 | `postread-reversal-corrected-20261025-v2\acceptance-notes.txt` L14 | 🟢 |
| 收缩选权 | P0 0.75 / chrono **0** / arsenal 0.1 / chelsea **0** / liverpool 0.5 / global_time **0** | 同上 L7–12 | 🟢 |
| P0 配对比较 | 全文方案 vs no_information 平均差 −0.011849，CI [−0.020011, −0.003814] | 同上 L17 | 🟢 |
| 遮盖标题独立识别率 | 33.33%（独立 6 场 / 12 篇）；**81.3% 是全部 289 行的样本内计算** | 同上 L22 | 🟢 |
| D 集规模 | 380 场 → 7 折 → 290 场测试 | `docs\parallel\status\round177.md` L71 | 🟢 |
| D 集主预算结果 | 不查询 1.080229 / context residual 1.086871 / **固定顺序全查 1.086803** | 同上 L79–83 | 🟢 |
| D 集低预算结果 | 1.079009（纯）/ 1.085854（加成本） | `docs\parallel\status\round181.md` L53 | 🟢 |
| 冻结后 50 场 | 1.090590，全部不查询 | 同上 L54 | 🟢 |
| 早期在线运行 | 3 场 / 8 次真实请求；概率未变 | `docs\parallel\status\round180.md` | 🟡 |

---

## 六、旧版代码与机制

| 机制 | 代码路径 | 标记 |
|---|---|---|
| 旧版在线运行时（含硬预算常量） | `demo\live_g.py` L37–41（`MAX_SOURCE_REQUESTS=10` / `MAX_BODY_READS=5` / `MAX_PROVIDER_CALLS=5` / `RUN_TIMEOUT_MINUTES=15` / `MAX_DYNAMIC_HOSTS=2`） | 🟢 |
| 旧版 g（15 个关系特征 + 标题 TF-IDF） | `football_research\acquisition\state_candidate_gain.py` | 🟢 |
| 旧版证据超越与冲突保留 | `football_research\acquisition\entity_evidence_updates.py` | 🟢 |
| 新版 f（集合 / GRU） | `football_research\acquisition\sequential_evidence_f.py` | 🟢 |
| 新版元数据构造 | `football_research\acquisition\full_history_sequence_metadata.py` | 🟢 |
| 新版 g | `football_research\acquisition\sequential_postread_g.py` | 🟢 |
| 四项证据准入 | `football_agent\directional_evidence.py` | 🟢 |
| 冻结包入口 | `var\releases\footballmind-defense-20261025-rc1\footballmind_defense.py` | 🟢 |
| 答辩演示服务 | `demo\defense.py` + `demo\static\defense.html/js/css` | 🟢 |

---

## 七、旧版 g 在真实菜单上的评分（**注意解读限制**）

| 结论 | 数值 | 原始路径 | 标记 |
|---|---|---|---|
| g 在 Inter 39 条子集上的分数跨度 | gain ∈ [0.019519, 0.023954]，跨度 0.004435，std 0.000812 | `var\work\fm-embedding-bench-20261025-v1\results\exp2_ranking.json` → `g_baseline_inter` | 🟢 |
| 排序 AUC | 0.544（**标签是单标注者判读的"是否相关"**） | 同上 | 🟢 |
| Qwen3 同一子集 AUC | 0.918 | 同上 → `per_query.inter_fiorentina.encoders.qwen3_06b.auc_scored_subset` | 🟢 |
| 5 场菜单规模 | 202 条候选 / 22 条相关 / 14 条 borderline；Inter 单场 141 条中相关 5 条、文章 36 条 | 同上 | 🟢 |
| OOV 率 | 0.308 / 0.333 / 0.500 / 0.583 | 同上 → `token_diagnostics` | 🟢 |

> 🔴 **重要解读红线**：0.544 的标签是**人工判读的相关性**，而 g 的输出是 **Δ_f 分数**。**两者目标不同，不得把 0.544 表述为"g 的精度"。** 可辩护的部分只有**输出跨度 0.004435 与 std 0.000812**（近常数性），这与标签定义无关。

---

## 八、文献（均已核验原始来源）

| 文献 | 标识 | 核验 |
|---|---|---|
| Howard 1966, Information Value Theory | DOI `10.1109/TSSC.1966.300074`；IEEE Trans. SSC 2(1):22–26 | ✅ 逐字核验 |
| Chaloner & Verdinelli 1995 | DOI `10.1214/ss/1177009939` | ✅ |
| Strong et al. 2015（EVSI 回归） | DOI `10.1177/0272989X15575286` | ✅ |
| Heath et al. 2019 | DOI `10.1177/0272989X19837983` | ✅ |
| Rainforth et al. 2024 | DOI `10.1214/23-STS915` | ✅ |
| Krause & Guestrin（非短视 VoI） | arXiv `1207.1394`（UAI 2005） | ✅ |
| SIM-RAG（*…through Self-Practicing*）Diji Yang 等，SIGIR '25 pp.1305–1315 | DOI `10.1145/3726302.3730018` | ✅ 著录核验 |
| DynaKRAG | arXiv `2607.06507`（v1 2026-07-07 / v2 2026-08-22） | ✅ |
| Self-RAG | arXiv `2310.11511` | ✅ |
| FLARE | arXiv `2305.06983` | ✅ |
| CRAG | arXiv `2401.15884` | ✅ |
| Retrieval as a Decision（v1 名 TARG） | arXiv `2511.09803`（v1 2025-11-12 / v2 2026-04-14） | ✅ 标题已变更 |
| ReAct | arXiv `2210.03629` | ✅ |
| Reflexion | arXiv `2303.11366` | ✅ |
| MAST *Why Do Multi-Agent LLM Systems Fail?* | arXiv `2503.13657`；NeurIPS 2025 **D&B Track**；**1600+ 标注轨迹** | ✅ 含修正 |
| Evidential Deep Learning | arXiv `1806.01768` | ✅ |
| Zhao et al. 2012 Latent Truth Model | arXiv `1203.0058` | ✅ |
| OptionWorld | arXiv `2610.28491`（2026-10-20） | ✅ 含 35.5% 原句 |
| Calibration, Not Answer Selection | arXiv `2609.33290`（2026-09-27） | ✅ 含标题补全 |
| Loud Failures, Quiet Failures | arXiv `2610.10062`（2026-10-07） | ✅；**"IEEE CogMI 2026 已录用"无法证实，已删除** |
| SUPPORT / Provenance-Aware Belief Updating | arXiv `2610.13008` | ✅ 全名补全 |
| Every pooling rule has its world | arXiv `2608.11275` | ✅ 副标题补全 |
| GRACE | arXiv `2607.09175` | ✅；**pass^3 终值为 0.673 ± 0.136** |
| **WC2026-Agents** | arXiv `2607.17765`（2026-07-20） | ✅ 摘要原句核验 |
| **WorldCupArena** | arXiv `2607.18084`（v1 2026-07-20 / v2 2026-08-31） | ✅；**"非因果消融"原句核验** |
| Dixon & Coles 1997 | DOI `10.1111/1467-9876.00065` | ✅ |
| Wunderlich & Memmert 2018 | DOI `10.1371/journal.pone.0198668` | ✅ |
| Kaunitz et al. 2017 | arXiv `1710.02824` | ✅ |
| Brown et al. 2018 | DOI `10.1111/ecin.12506` | ✅ |
| Clegg, Song, Cartlidge 2026 | arXiv `2605.16066` | ✅ |
| Halawi et al. 2024 | arXiv `2402.18563` | ✅ |
| Lost in the Middle | arXiv `2307.03172` | ✅ |
| Chroma Context Rot | `research.trychroma.com/context-rot` | ✅ |
| tau-bench | arXiv `2406.12045` | ✅ |

---

## 九、竞赛规则来源

| 项 | 来源 | 日期 |
|---|---|---|
| 算法创新赛道通知（〔2026〕11 号） | `https://www.aicomp.cn/notice/notice-1/3674.html` | 2026-04-29 |
| 算法模型创新赛题页（含评分细则） | `https://www.aicomp.cn/tracks/tracks-2/3788.html` | 2026-05-06 |
| 算法创新赛统一材料规范 | `https://www.aicomp.cn/tracks/tracks-2/3775.html` | 2026-05-08 |
| 赛区划分通知（〔2026〕49 号） | `https://www.aicomp.cn/notice/notice-1/5503.html` | 2026-10-17 |
| 国际赛区通知 | `https://www.aicomp.cn/notice/notice-1/5335.html` | 2026-09-29 |
| 大赛通知（〔2026〕9 号） | `https://www.aicomp.cn/notice/notice-1/3618.html` | 2026-04-29 |

---

## 十、未找到原始证据的项（**不得引用**）

| 项 | 状态 |
|---|---|
| 本团队实际报名的赛区 / 赛题 / 当前阶段 | 🔴 代码库内无记录，**必须由团队自行确认** |
| 26,271 条注册记录中"未跨源确认"的分项条数 | 🔴 只有布尔标记 `master_not_all_confirmed_independent_settled_matches: true` |
| 4,540 条 failure_rows 的逐项原因分布 | 🔴 只有归类说明 |
| GRU f 的完整元数据掩码配置 | 🔴 报告只说"存在差异"，未给清单 |
| 市场先验报价的真实时点 | 🔴 `prior_time_verified = false` |
| 是否使用过 Argilla 等标注平台 | 🔴 未找到证据；**已核实的人工介入只有单标注者菜单判读** |
| 门控消融实验 | 🔴 **从未做过**（面板仅 5 条观察） |
| 概率校准（ECE / 可靠性图） | 🔴 **未做** |
| 跨语言 / 跨领域迁移实验 | 🔴 **未做** |
| 投资组合与交易成本模型 | 🔴 **未定义** |
