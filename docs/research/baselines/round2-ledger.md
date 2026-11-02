# 第二轮预测账本、追加赛果与评分

实现：`football_research/evaluation/ledger.py`、`ledger_cli.py`。独占机制测试：`tests/research/evaluation/test_ledger.py`。原有 PredictionService 和公共概率输出保持兼容。

## 实际运行

2026-09-20 12:19:58 UTC，将第一轮真实预测导入 `var/parallel/baselines/round2/ledger-replay/`：54 条、18 场；重复导入报告 54 条 duplicate。它们均为 replay，issued_at 为 null，真实本次登记时间保存在 registered_at。第一轮文件中的 prediction_cutoff 没有被冒充为生成时间。

首次评分：54 条 pending、0 条赛果版本、0 条 scored。pending 包括尚未开始以及比赛开始后仍在结果等待窗口的比赛；等待窗口默认 kickoff+3 小时，超过后仍缺标签报告 no_label。每条列出具体 match_id、kickoff、label_due_at 和 reason。三小时只决定状态分类，不生成标签或估计比分。

实际报告：`ledger-replay-registration.json`、`ledger-replay-registration-repeat.json`、`ledger-replay-score.json`，均位于 `var/parallel/baselines/round2/`。首次 forecast_snapshot_sha256：`7fd3d98b9dd66072c26faa42e436b466175aebb18913e8fc19b40a93b520e101`。该快照哈希只依赖冻结预测文件，与后续评分时钟或追加标签无关。

这批档案只验证登记和状态链路。真实新发预测由根任务另行调用 issue_forecast 生成并保存；测试里的比分和结果修订属于构造机制，独立于正式账本。

根任务随后在 `round2/ledger-prospective/` 登记 71 条、18 场新发预测。本子任务于 2026-09-20 12:23:59 UTC 进行只读独立核验：71 条全部通过 cutoff≤generation_started_at≤issued_at≤registered_at<kickoff、登记延迟上限、实际内部调用路径、输入哈希及文件内容哈希检查；检查前后所有文件 SHA256 一致。当时 71 pending、0 条结果版本。完整逐条证据为 `round2/prospective-ledger-verification.json`，冻结快照哈希 `5d1c376778a06d47081530deb420155f0ce8e70500d6cfc00cef5823368f79b3`。

## 可直接运行的 CLI

在 `D:\a1-足彩` 执行：

```powershell
# 导入第一轮记录；任意文件导入都保留 replay 身份
python -X utf8 -m football_research.evaluation.ledger_cli register-forecasts --ledger var/parallel/baselines/round2/ledger-replay --predictions var/parallel/baselines/real-asof-final/predictions.jsonl --output var/parallel/baselines/round2/ledger-replay-registration-repeat.json

# 只读评分，输出逐条 pending/无标签/冲突及各模型分数
python -X utf8 -m football_research.evaluation.ledger_cli score-forecasts --ledger var/parallel/baselines/round2/ledger-replay --output var/parallel/baselines/round2/ledger-replay-score.json

# 有经过核实的新结果后追加；重复执行相同输入为 duplicate
python -X utf8 -m football_research.evaluation.ledger_cli settle-forecasts --ledger var/parallel/baselines/round2/ledger-prospective --results var/parallel/baselines/round2/results.jsonl --output var/parallel/baselines/round2/settlement-report.json
python -X utf8 -m football_research.evaluation.ledger_cli score-forecasts --ledger var/parallel/baselines/round2/ledger-prospective --output var/parallel/baselines/round2/prospective-score.json

# 独占机制测试
python -X utf8 -m pytest tests/research/evaluation/test_ledger.py -q
```

`results.jsonl` 是后续真实标签文件路径，不是随交付附赠的虚构赛果。当前未有标签时，score 仍完整报告 pending/no_label。已有预测 JSONL 可以自带 kickoff；缺失时传 `--fixtures` 指向研究 dataset JSONL 或纯 fixture JSONL。

实时发预测命令结构：

```powershell
python -X utf8 -m football_research.evaluation.ledger_cli issue-forecasts --ledger var/parallel/baselines/round2/ledger-prospective --model MODEL.json --fixtures DATASET.jsonl --match-id SOURCE:match:MATCH_ID --output var/parallel/baselines/round2/issuance-report.json
```

每次调用以执行瞬间作为 cutoff，内部执行 model.predict，完成时间成为 issued_at，并在开赛前落盘。可用 `--prediction-cutoff` 指定刚冻结的时点，默认最多滞后 300 秒；`--max-latency-seconds` 显式调整容许延迟且被记录。所有输入仍需通过 PredictionService 的赛程可见时间、模型训练时间、特征时间和目标隔离检查。市场模型需要匹配 cutoff 的真实报价特征，用 `--features` 提供每场一条规范特征 JSONL。未来 cutoff、过期 cutoff、开赛后生成、时钟倒退、生成/登记超时均拒绝登记。

## Python / Agent 接口

```python
from football_research.baselines import PredictionService
from football_research.evaluation.ledger import ForecastLedger

ledger = ForecastLedger("var/parallel/baselines/round2/ledger-prospective")
service = PredictionService.load("MODEL.json")
registered = ledger.issue_forecast(service, fixture, features=None)
# registered: status=registered|duplicate、forecast_id、record
# record.prediction 是原公共 Prediction 的 JSON 结构

appended = ledger.settle_result(result_row)
# appended: status=appended|duplicate、label_version、record
report = ledger.score()
```

导入 API：`ledger.import_forecast(prediction, kickoff=..., source_reference=...)`。它始终登记 replay；外部声称的 issued_at 仅保留为 `claimed_issued_at_unverified`，未来的声称时间校验报错。

规范化真实档案的结果导出：

```python
from football_research.data_adapters.settlement import export_results

rows, coverage = export_results(dataset, match_ids=selected_match_ids)
operations = ledger.settle_results(rows)
```

`dataset` 必须是 ResearchDataset；适配器读取现有规范化字段和血缘，不读取来源原始响应。`match_ids=None` 导出全部合格标签；传空列表导出零条。finished 转为 final；cancelled/postponed/disputed 保留；suspended 转为 disputed 并注明来源状态；scheduled/live/unknown 只报告未到结算状态。输出保留数据版本、source_reference、完整 input_manifest、稳定 revision_id。相同数据重复导出和结算均幂等；适配器不猜测 supersedes。

`settle_results(rows)` 为批量入口，返回每行操作结果列表。整批取得一次文件锁并校验读取一次预测/结果快照，随后用内存索引追加每条结果；同批较后的结果可 supersedes 同批先前已追加的版本。`settle_result(row)` 保持兼容，内部使用同一实现。每条原子提交，中间失败时已提交前缀保留，修正后重新提交整批即可幂等恢复；这一行为不声称整批回滚。CLI settle-forecasts 同样使用批量入口，避免数千条预测与数百个标签逐行重复解析。

观察时间证据来自每场规范化 manifest 的 `|at:`、显式 `audit.match_observed_at` 或平台整份快照的 `audit.source_observed_at`。最终比分还要求 result_available_at 和常规时间语义；延期取消等使用状态实际被观测时间，避免把首次赛程可见时间误作较晚状态更新时间。缺观察时间、缺血缘、未来观察、结果时间关系矛盾、未知来源、缺重建假设等逐场跳过，coverage 给出具体原因与遗漏 ID。批量结算增强后，21 项适配测试与 37 项账本测试合计 58 项已通过。

## 结果 JSONL 单行字段

以下只说明结构，示例中的 TEST ID 与比分不用于真实评估：

```json
{"match_id":"TEST:match:1","information_mode":"operational","status":"final","score_period":"regulation","home_goals":2,"away_goals":1,"source":"SOURCE","source_reference":"sha256:RAW_HASH|observation:OBSERVATION_ID","revision_id":"source-result-v1","kickoff":"2026-09-20T14:00:00Z","available_at":"2026-09-20T16:10:00Z","observed_at":"2026-09-20T16:10:00Z","supersedes":[]}
```

- `match_id` 与预测的完整命名空间一致；不进行数字 ID 猜测或跨来源映射。
- `available_at` 是本结果版本首次可用时间；`observed_at` 是来源实际观测时间，要求 available_at≤observed_at≤当前系统时间。程序另写真实 recorded_at。final 结果的 available_at 不得早于 kickoff。first_available_at 记录该比赛状态/结果流最早出现时间；first_final_label_available_at 单独记录最终标签最早出现时间，避免把较早的 scheduled/postponed 状态时间当作比分可用时间。
- `kickoff` 可省略，仅在账本对此 match_id+information_mode 存在唯一赛程时推断；否则明确提供。
- `status` 支持 final、scheduled、postponed、cancelled、disputed。final 必须有非负整数常规时间比分，其余状态不带比分。
- 每个来源的 revision_id 在同一比赛和模式内标识固定内容。修改已使用 revision_id 的内容报错。新修订用新 revision_id，并在 supersedes 中列出被取代的 label_version 哈希。
- 修订必须引用已存在、同比赛、同模式的版本，时间不得早于被修订版本。旧版本始终保留。独立来源相同结果可作为互证；不同结果且没有明确 supersedes 解决时报告 conflict，暂停评分。
- historical_reconstruction 必须额外提供 `availability_assumption`，说明结果延迟假设；其结果仅连接相同 information_mode 的预测。
- 延期后的 final 如 kickoff 与原冻结赛程不同，原预测报告 schedule_changed，不按同一 ID 静默评分。

## 存储与评估行为

`forecasts/<forecast_id>.json` 与 `results/<label_version>.json` 分目录保存。预测 ID 由比赛、cutoff、model_version、dataset_version、information_mode 与 prospective/replay 身份确定；同槽位概率、manifest 或 kickoff 改动报冲突。每个文件包含 SHA256 完整性字段，读取时校验。结果的 label_version 是规范化结果内容 SHA256，记录首次可用时间与写入时间。

写入使用跨进程 OS 锁、同目录临时文件、flush/fsync 和原子替换；已提交文件从不覆盖。锁随进程退出释放，未提交临时文件不参加读取。幂等重试复用第一次登记/追加的时间与完整原文件。评分只读，追加赛果及修订不改变预测文件哈希。存储完整性用于发现损坏和未同步更新哈希的改动；拥有整个目录修改权限的管理员不受第三方审计约束。prospective 是本地时钟和内部 predict 调用证据，不宣称外部时间公证。

评分按 registration_kind、information_mode、model_version 分组，绝不把 replay/reconstruction 与 operational prospective 合并成一个总分。每条 scored 记录给出实际使用的 label_version、互证版本、来源、比分和逐场 loss；报告含 log loss、三类平方和 Brier、准确率、按 cutoff 提前量分组以及校准分箱。另列实际 issued_at 的提前量，避免把 cutoff 与生成时间混淆。无样本时指标为 null。

`score --as-of` 同时过滤预测登记时间、标签 recorded_at、observed_at 与 available_at，用于重现旧评分视图；未来 as_of 报错。重复结算后的指标、标签版本与预测快照哈希相同。默认 score 的 as_of 是本次真实执行时间，因此报告时间本身会变化。

测试覆盖并发重复登记/结算、实际发预测时钟、未来/陈旧 cutoff、慢生成、开赛边界、时钟倒退、冻结概率冲突、结果修订与历史视图、独立来源冲突和显式解决、取消延期争议、延期后赛程变化、模式隔离、无标签、非法标签、文件篡改及 CLI 全链路。所有构造数据仅存在于 tests 与 pytest 临时目录。
