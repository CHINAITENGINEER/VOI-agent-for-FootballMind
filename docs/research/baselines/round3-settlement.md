# 第三轮真实前瞻结算与五时钟契约

2026-09-22，执行截止约 09:50 UTC。研究侧消费平台已冻结的精确18场结果导出，没有新增网络请求。

## 修复与兼容

`ResearchMatch` 及 JSONL 投影新增可空的 `source_published_at`、`observed_at`、`ingested_at`、`operational_available_at`。来源发布时间保持来源证据值；未知时为 null。逐场 observed_at 优先于导出 manifest 的整个批次观测上界，真实接收时间与入库时间分别保留。

- operational: `operational_available_at >= max(observed_at, ingested_at)`；结果的 `available_at >= operational_available_at`。因此14:00收到、14:05入库、14:05可用合法。
- historical_reconstruction: `available_at` 继续使用明确假设，和今天真实观察/入库时间分开；必须携带 `availability_assumption`。
- ledger: `recorded_at` 由实际入账时钟生成。所有实时时钟及可用时间不得晚于当前时钟。评分额外要求 recorded/observed/ingested/operational_available/available 均不晚于评分 as_of。
- `time_contract_version=separate-clocks-v2` 显式标识新载荷。旧账本的无新字段载荷保持原字段、校验和版本哈希；未重写任何旧预测、赛果文件。旧格式缺入库证据时继续遵守原有严格时间限制。
- 平台已隔离记录的 `settlement_status/pending_reason/blocked_observation_id/version_id` 经研究 audit 与导出跳过报告保留。缺标签与矛盾原文均不作为赛果评分。

## 真实执行结果

输入目录：`var/parallel/collectors/round2/results/results-export-v1-31797adea32d326cd39e76f55be063b56d601d9519e6afff14a9e1b75a1e0711/`。
平台 manifest SHA256：`72f553a26443ecbab5940162cb0a81a6bd659e5babf5a5b42276c09ce7d5647d`。

原 `round2/ledger-prospective` 的71条／18场冻结预测追加12个真实 final标签，实际48条／12场被评分。其余20条属于5场 `HALF_EXCEEDS_FULL` 质量隔离，账本状态 `no_label`；3条属于未来／空源赛程 `spftll:match:3013698`，当前 `pending`。平台的6场pending与账本20 no_label + 3 pending语义不同，报告同时保留。

| 模型族 | 独立比赛 | log loss | Brier（三类和） | 准确率 |
|---|---:|---:|---:|---:|
| 频率 | 12 | 1.153158 | 0.695701 | 0.333333 |
| Poisson | 12 | 1.021888 | 0.598131 | 0.500000 |
| Dixon–Coles | 12 | 1.026129 | 0.603138 | 0.500000 |
| 市场去水 | 12 | 1.040067 | 0.627960 | 0.416667 |

12场仅作真实前瞻结算功能验证；样本不足以建立稳定模型排名。模型族汇总用于上述共同12场描述；详细score报告仍按真实model_version分组，标签版本逐条列出。

5场隔离ID：2993795、2993799、2993800、3018813、3018818（均为 `spftll:match:` 命名空间）。质量记录及其原始观测引用保存于 `verification.json.quality_rows`，没有从矛盾比分挑选标签。

重复相同结算输入得到12 duplicate，指标相等。71个预测文件逐文件SHA256均相等，集合SHA256仍为 `5d1c376778a06d47081530deb420155f0ce8e70500d6cfc00cef5823368f79b3`。

产物均在 `var/parallel/baselines/round3/settlement/`：`before.json`、`dataset.jsonl`、`results.jsonl`、`results.report.json`、`settle.json`、`settle-repeat.json`、`score.json`、`score-repeat.json`、`verification.json`。`before.json`记录实际追加前状态和每个冻结文件hash，保留为证据。

## 复跑命令

在 `D:\a1-足彩` 执行。下面重建消费输入，并再次幂等结算已有12版本；重新生成报告应选择不同报告名，保留本轮首次运行证据。

```powershell
python -X utf8 -m football_research.data_adapters.platform_history --export var/parallel/collectors/round2/results/results-export-v1-31797adea32d326cd39e76f55be063b56d601d9519e6afff14a9e1b75a1e0711 --output var/parallel/baselines/round3/settlement/dataset-rerun.jsonl
python -X utf8 -m football_research.baselines export-results --dataset var/parallel/baselines/round3/settlement/dataset-rerun.jsonl --ledger var/parallel/baselines/round2/ledger-prospective --output var/parallel/baselines/round3/settlement/results-rerun.jsonl
python -X utf8 -m football_research.baselines settle-forecasts --results var/parallel/baselines/round3/settlement/results-rerun.jsonl --ledger var/parallel/baselines/round2/ledger-prospective --output var/parallel/baselines/round3/settlement/settle-rerun.json
python -X utf8 -m football_research.baselines score-forecasts --ledger var/parallel/baselines/round2/ledger-prospective --output var/parallel/baselines/round3/settlement/score-rerun.json
python -m pytest tests/research/evaluation/test_settlement_clocks.py tests/research/evaluation/test_ledger.py tests/research/evaluation/test_settlement_adapter.py tests/research/baselines/test_platform_history.py -q
```

上述独占回归最终98项通过，包含14:00/14:05分离、未来四类时钟、operational回填拒绝、重建假设分离、旧载荷逐字节幂等、逐场投影与隔离原因保留。构造用例位于tests，与真实结果目录分开。
