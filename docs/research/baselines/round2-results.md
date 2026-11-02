# 第二轮真实运行与复现（2026-09-22）

## 交付状态

已完成不可变预测账本、版本赛果追加、幂等结算/评分、精确批量特征连接、七模型CLI装配、模型元数据，以及真实Understat历史重建评估。目标始终为常规90分钟加补时胜平负，排除加时和点球。

第一轮冻结产物保留；本轮产物全部在 `var/parallel/baselines/round2/`。没有修改采集器、迁移、Agent、展示或共享依赖。公共类型直接从 `football_spider.models.contracts` 导入，私有importlib bridge已移除。PredictionService签名保持兼容。

本轮先复用已落地的Understat规范化解析器与FeatureStore纯compute函数，在研究目录生成有完成标记的**研究消费快照**，manifest的producer为research、production_platform_export=false。随后平台正式COMPLETE导出就绪，已另接官方版本进行完整评估；两个冻结版本分开保存。来源字段解析与Elo算法没有研究侧副本。

## 正式平台导出接入（同日追加）

平台完成目录 `var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda/`，feature_version=`prematch-season-v2.0`，manifest SHA256 `a3065657f0a6e7e8385e783a9169e23d35e419469c54061e1da91ee4806493ad`。380比赛/760特征/402映射/0特征排除，规范比赛及所有特征使用平台canonical UUID。研究消费者验证完成标记、全文件哈希、时点及独立xg_inputs血缘；760快照全部接受。

正式结果产物 `round2/understat/platform-evaluation-v1/`，实验设计保持此前冻结版本。`understat/platform-feature-comparison.json` 用规范source_id进行380场显式身份对照，760快照的6个模型字段均一致；平台完整最近5场窗口要求与研究“最近5个已可见值”规则在这份数据上未产生数值差异，规则本身仍分别版本化。

正式版本已完成7窗口、290共同测试场、3480测试预测。下文六位小数指标表同时适用于正式版本与早期研究快照；完整精度分别保存在各自backtest/report.json，ID排序/优化引起的极小差异保留原值，不宣称跨命名空间逐字节相同。正式版本应优先用于平台/Agent消费，早期研究版本用于独立对照。

正式账本也已实际完成3480 replay登记→290真实赛果追加→3480 scored；重复结算290 duplicate、指标相同、预测hash不变。预测集合SHA256 `eeec34599c48905525c13767b12d7c8478e85448e16b9d91ca56a8c6da179dc5`，证据 `platform-evaluation-v1/summary.json` / `settlement-score.json`。正式复核 `understat/platform-verification.json` 同样通过6756概率检查、98模型加载及84次模型重算精确一致。第一轮和前瞻集合保持原hash，见round2/final-integrity.json。

正式平台全链路复跑命令（新输出目录）：

```powershell
python -m football_research.evaluation.reconstruction --platform-export var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --design var/parallel/baselines/round2/understat/experiment-design.json --feature-version prematch-season-v2.0 --output var/parallel/baselines/round2/understat/platform-evaluation-replay
python -m football_research.evaluation.verify_reconstruction --run-dir var/parallel/baselines/round2/understat/platform-evaluation-v1 --output var/parallel/baselines/round2/understat/platform-verification-repeat.json
```

正式operational导出也已导入并运行：`round2/understat/platform-operational/backtest/report.json`，380最终标签存在，但760历史赛前快照全部因fixture_not_known_at_requested_cutoff排除，NO_ELIGIBLE_TEST_PREDICTIONS、评分0。真实入库晚于这些历史比赛，未回填operational可见时间。

平台新增真实刷新观测后已再次消费全档案，证据 `round2/settlement-platform-refresh/`。本次刷新目标不在71条前瞻登记清单中，相关18场仍无合格新结果，因此68 no_label/3 pending保持不变。该刷新由平台任务执行，研究侧没有直接联网。

## 两类真实证据分别报告

### 实际前瞻登记

- 2026-09-20T12:20:29.888300Z冻结输入，随后在本地实际predict并登记：18场、71条，频率/Poisson/DC各18条、市场17条。issued_at由执行时系统记录，且全部早于档案开球。
- 账本 `round2/ledger-prospective/`；模型及运行报告 `round2/prospective/20260920T122029-87dcaf99-cb69-4760-a00c-9d7fff17ab80/`。
- 2026-09-22重新读取已有原始档案，714比赛、193完整赔率快照。选中的18场仍只有scheduled记录，合格新增结果0。账本现在为 **68条/17场no_label、3条/1场pending、0 scored**。pending是 `spftll:match:3013698`，档案开球2026-10-21T18:00Z；保留来源赛程，不猜测新日期。
- 列表、逐条issued_at、应到标签时间与状态见 `round2/settlement-20260922/score.json`；缺标签原因见同目录 `results.report.json`。重复执行相同空结果输入没有新增记录。
- 前瞻预测集合SHA256仍为 `5d1c376778a06d47081530deb420155f0ce8e70500d6cfc00cef5823368f79b3`。本地时钟及文件哈希是本地执行证据，不是第三方时间戳服务。
- 第一轮54条预测导入 `round2/ledger-replay/`，全部replay、issued_at=null。两次登记分别54 registered、54 duplicate；没有倒填前瞻身份。第一轮预测文件SHA256仍为 `744b3db10da1782f799afb4be8a490f30bf60da6f3bc79dd3fd092f60b563178`。

### 有条件的历史重建

真实原文是collector研究档案，不是tests夹具。Understat EPL/2025：380场、20队、每队38场、380不同主客队配对，全部有常规赛果；平台结构审计通过。赛季时间2025-08-15至2026-05-24，观测时间2026-09-20T11:59:41.602689Z，原文SHA256 `d401b18a3386d8168f03727b43b1ab5dc9092e0469e77a1561136baf25151f42`。

消费入口 `football_research.data_adapters.reconstruction_bundle` 调用平台 `normalize_understat`，接收MatchRecord/TeamStatistic；Elo直接调用 `football_spider.features.store.compute`，参数初始1500、K=20、主场优势65。先按结果可用时间筛选history，再交给compute。研究适配只计算最近5场**已可见**规范化xG/xGA均值，保留逐场输入版本和可见时间。目标单场xG/赛果、来源forecast、赛季最终汇总都不进入特征。

重建假设：赛果在开球后3小时可用，xG在开球后24小时可用，赛程提前168小时可知；时间按平台解释为UTC，依据其已归档的官方前端时区转换证据，未作独立逐场开球核验。2026年观测到的修订后比分/xG、最终赛程和延期变动可能带来事后偏差。所有输出都是historical_reconstruction，登记身份均为replay。

冻结研究包 `round2/understat/research-bundle-v1/`，manifest SHA256 `e23ac5d75f313881aea7efbf52d8a1a5e64f11e8390569b8f48d7213b0a1d66e`；380 matches、760 features，feature_version=`research-normalized-history-v1`。所有760快照的hash、四键、字段、历史输入时间检查通过，拒绝0、冲突0。

实验设计在2026-09-20先行冻结，SHA256 `e6e09702acc18b895619d518dcac9e18009eaed57e7ba4d33ec58dbb7cbaba9d`。本次未因测试结果改模型、调参、融合或拟合校准器。7个扩展训练窗口、30天验证与30天测试；同场两提前量在同一分区，每个提前量分别训练；训练标签严格早于相应评估预测截止。

| 窗口 | 训练比赛（频率/Poisson/DC/logistic/tree） | Elo训练比赛 | 验证比赛 | 测试比赛 |
|---|---:|---:|---:|---:|
| 0 | 60 | 50 | 30 | 35 |
| 1 | 90 | 80 | 30 | 55 |
| 2 | 120 | 110 | 55 | 50 |
| 3 | 180 | 170 | 50 | 42 |
| 4 | 230 | 220 | 41 | 37 |
| 5 | 271 | 261 | 37 | 30 |
| 6 | 309 | 299 | 30 | 41 |

Elo剔除首轮10场没有既往评级的训练比赛；logistic/tree的显式null只由训练窗口拟合的插补器处理。它们使用双方Elo及双方rolling_xg_for/against共6字段。Poisson/DC使用既往双方进球和球队ID；频率使用训练窗口联赛H/D/A频数；市场要求同预测时点完整1X2赔率。本来源无合格历史报价，市场测试覆盖0，未使用forecast冒充市场。

**6个有覆盖模型在同一290场上的测试结果：**

| 模型 | 24h log loss | 1h log loss | Brier（两提前量相同） | 准确率 | 每提前量测试场 |
|---|---:|---:|---:|---:|---:|
| 频率 | 1.108492 | 1.108492 | 0.673332 | 40.34% | 290 |
| Poisson | 1.064950 | 1.064950 | 0.643207 | 43.79% | 290 |
| Dixon–Coles | 1.066393 | 1.066393 | 0.643808 | 43.79% | 290 |
| Elo三分类映射 | 1.073555 | 1.073555 | 0.647450 | 44.48% | 290 |
| 多项逻辑回归 | 1.153696 | 1.153696 | 0.676154 | 42.76% | 290 |
| 决策树 | 1.229388 | 1.229388 | 0.737765 | 35.86% | 290 |
| 市场去水 | — | — | — | — | 0 |

Brier为三类误差平方和，范围[0,2]。两提前量分别报告；其模型输入数值在本赛程及延迟假设下相同，因而结果相同，未视为580个独立比赛。测试总计3480条=290×6×2；连同验证共6756条。7×7×2=98模型文件中14市场文件无可评分报价；其余84份均有预测。

逐场配对以UTC自然周为块，2000次固定种子bootstrap，290场/27周，95%描述性区间。频率减Poisson的log loss差0.043542，区间[0.006076,0.083836]；频率减DC为0.042099，区间[0.004250,0.083376]。完整30组按模型/提前量配对损失、覆盖及区间见backtest/report.json。区间未覆盖跨周依赖、多模型比较或重建假设误差；单赛季条件实验不构成稳定预测优势证据。

12张校准图及各箱计数JSON已生成，Poisson 24h图已人工渲染检查。图是诊断，没有用测试集拟合校准器。

## 登记与重复结算实测

历史test预测3480条已全部登记replay，追加290个真实赛果版本后scored=3480。相同输入再次结算，290 duplicate；两次有效指标与逐预测状态相等。冻结预测集合SHA256保持 `599c1d052afaeaceb6c6a7d8c2942aff79502ed0ffc3f4182fe1a0084c8b2484`。

证据在 `round2/understat/evaluation-v1/{summary,settlement-score}.json`、`results.jsonl`、`ledger-reconstruction/`。评分逐条标明label_versions，赛果版本含source、source_reference、available_at、observed_at、recorded_at、revision_id、supersedes及重建假设。矛盾有效版本保持conflict；明确修订追加新版本，旧版保留；取消、延期、争议不进入常规比分评分。

真实数据没有人为制造修订。修订、重试中断、并发锁、hash损坏、取消/延期/争议等机制由合成独占测试验证。

## 可直接执行的命令

在 `D:\a1-足彩` 执行。研究模型使用现有全局Python3.13；离线消费平台compute使用现有平台Python3.11，因为该模块导入其已安装的psycopg。两个环境未安装/升级依赖。

```powershell
# 1. 从真实档案消费平台解析/Elo，输出新研究版本（目录须尚不存在）
football_spider/.venv/Scripts/python.exe -m football_research.data_adapters.reconstruction_bundle --source-manifest var/parallel/collectors/understat-research/data-manifest.json --output var/parallel/baselines/round2/understat/research-bundle-replay

# 2. 批量特征 → 固定设计滚动训练 → 登记test预测 → 追加结果 → 评分及重复核验
python -m football_research.evaluation.reconstruction --platform-export var/parallel/baselines/round2/understat/research-bundle-v1 --design var/parallel/baselines/round2/understat/experiment-design.json --feature-version research-normalized-history-v1 --output var/parallel/baselines/round2/understat/evaluation-replay

# 3. 独立批量训练和预测（已经实跑：训练60场，640预测，120过早快照明确拒绝）
python -m football_research.baselines train --dataset var/parallel/baselines/round2/understat/evaluation-v1/dataset.jsonl --model logistic --as-of 2025-10-01T00:00:00Z --competition understat:competition:EPL --feature-file var/parallel/baselines/round2/understat/research-bundle-v1 --feature-version research-normalized-history-v1 --lead-hours 24 --model-parameters var/parallel/baselines/round2/understat/numeric-parameters.json --output var/parallel/baselines/round2/understat/cli-logistic-replay.json
python -m football_research.baselines batch-predict --dataset var/parallel/baselines/round2/understat/evaluation-v1/dataset.jsonl --artifact var/parallel/baselines/round2/understat/cli-logistic-replay.json --lead-hours 24 1 --feature-file var/parallel/baselines/round2/understat/research-bundle-v1 --feature-version research-normalized-history-v1 --output var/parallel/baselines/round2/understat/cli-batch-replay

# 4. 分开的登记 → 追加结果 → 评分命令；既有历史账本可幂等复跑
python -m football_research.baselines register-forecasts --predictions var/parallel/baselines/round2/understat/evaluation-v1/test-predictions.jsonl --ledger var/parallel/baselines/round2/understat/evaluation-v1/ledger-reconstruction --output var/parallel/baselines/round2/understat/register-repeat.json
python -m football_research.baselines settle-forecasts --results var/parallel/baselines/round2/understat/evaluation-v1/results.jsonl --ledger var/parallel/baselines/round2/understat/evaluation-v1/ledger-reconstruction --output var/parallel/baselines/round2/understat/settle-repeat.json
python -m football_research.baselines score-forecasts --ledger var/parallel/baselines/round2/understat/evaluation-v1/ledger-reconstruction --output var/parallel/baselines/round2/understat/score-repeat.json

# 5. 平台增加新观测之后，对真实前瞻账本只追加赛果/评分
python -m football_research.data_adapters export --output var/parallel/baselines/round2/next-settlement/dataset.jsonl
python -m football_research.baselines export-results --dataset var/parallel/baselines/round2/next-settlement/dataset.jsonl --ledger var/parallel/baselines/round2/ledger-prospective --output var/parallel/baselines/round2/next-settlement/results.jsonl
python -m football_research.baselines settle-forecasts --results var/parallel/baselines/round2/next-settlement/results.jsonl --ledger var/parallel/baselines/round2/ledger-prospective --output var/parallel/baselines/round2/next-settlement/settle.json
python -m football_research.baselines score-forecasts --ledger var/parallel/baselines/round2/ledger-prospective --output var/parallel/baselines/round2/next-settlement/score.json

# 6. 模型消费说明与冻结产物复核
python -m football_research.baselines list-models --artifact var/parallel/baselines/round2/understat/cli-logistic.json
python -m football_research.evaluation.verify_reconstruction --run-dir var/parallel/baselines/round2/understat/evaluation-v1 --output var/parallel/baselines/round2/understat/verification-repeat.json
```

`reconstruction`的platform-export也接受平台以后发布的正式COMPLETE目录；按其manifest指定feature_version。feature/model字段变化须选用实际有覆盖的模型并新建实验设计版本，不覆盖此次冻结实验。

## 验证与文件清单

```powershell
python -m pytest tests/research/baselines tests/research/evaluation -q --junitxml=var/parallel/baselines/round2/tests-final.xml
football_spider/.venv/Scripts/python.exe -m pytest tests/research/baselines/test_reconstruction_bundle.py -q --junitxml=var/parallel/baselines/round2/platform-consumer-tests.xml
```

全局环境261 passed、1 skipped（24.76s）；含正式平台独立xg_inputs的目标/未来血缘防护。跳过的是需要平台psycopg的消费测试模块，随后在平台环境4 passed。不是未验证跳过。真实产物复核 `understat/verification.json`：6756条概率、训练标签截止及目标排除通过；760特征血缘通过；98模型加载、84次模型重算概率逐值完全一致。该复核是已训练模型加载一致性，并非所有模型独立重训的声明。正式版本另写platform-verification.json。

- `baselines/api.py、__main__.py、metadata.py、prospective.py`：兼容服务、CLI、模型要求、实际前瞻发行。
- `evaluation/ledger.py、ledger_cli.py、runner.py、reconstruction.py、verify_reconstruction.py`：冻结/版本/评分、批量滚动与实测复核。
- `data_adapters/_contracts.py、batch_features.py、platform_history.py、reconstruction_bundle.py、settlement.py`：公共导入、精确快照、完成导出/离线平台消费、结果投影。
- `tests/research/baselines/test_{batch_features,metadata,platform_history,reconstruction_bundle}.py` 与 `tests/research/evaluation/test_{ledger,settlement_adapter,feature_runner,feature_integration_review}.py`：新增机制与回归。
- 文档：本文件、round2-ledger.md、round2-history-input.md、README与本角色status/requests/dependencies。

交付前再次读取platform-interfaces、平台/Agent状态及新acceptance.json：公共根导入已消费；平台独立验证报告已为OK/acceptance_complete，正式历史及operational完成导出均已接入。生产DATABASE_URL配置与前瞻18场合格新标签仍是外部接点。Agent已接既有无赔率三模型、模型目录与精确feature消费，研究接口说明见requests/baselines/004；研究侧没有将平台代办保留为整段等待。
