# FootballMind Agent 工具执行与事件回放

本模块负责真实工具执行、可审计运行记录和本地回放。入口位于 `football_agent/` 与 `demo/`，无需修改共享 CLI、采集器、数据库、公共契约或基线实现。

2026-09-22：主动信息获取的真实公开状态→价值估计→获取→部分预测→停止闭环见 [round3.md](round3.md)，当前真实入口为 http://127.0.0.1:8769/ 。模型选择、平台冻结特征、真实刷新接线及第二轮命令见 [round2.md](round2.md)。下方表格保留第一轮背景；当前状态以round3及 `docs/parallel/status/agent.md` 为准。

## 启动和验收

所有命令在 `D:\a1-足彩` 执行；推荐 `-X utf8` 避免 Windows 控制台中文编码差异。核心与 HTTP 展示使用 Python 标准库；真实工具复用现有环境的 pydantic 和 F 基线依赖。

```powershell
Set-Location -LiteralPath 'D:\a1-足彩'
python -X utf8 -m demo --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。选择真实档案回放或测试模式，选择比赛和北京时间截止点，初始化后可逐工具执行、一次运行至停止、指定主题查询、等待、恢复、停止或暂缓。事件滑块/自动播放只呈现选中事件的快照；首事件不会显示最终证据、概率或费用。完整运行记录可下载 JSON。

真实档案示例（档案标注赫塔费—马拉加；来源 ID 不是已映射的 canonical ID）：

```powershell
python -X utf8 -m football_agent run --mode real --archive-replay --match-id spftll:3013707 --cutoff 2026-09-20T11:10:00Z --output var/parallel/agent/real-run.json
```

截止 UTC 11:10 等于北京时间 19:10。可先列比赛或查看实际接口配置：

```powershell
python -X utf8 -m football_agent matches --mode real --cutoff 2026-09-20T11:10:00Z
python -X utf8 -m football_agent capabilities --mode real
python -X utf8 -m football_agent run --mode test --scenario normal --output var/parallel/agent/test-run.json
python -X utf8 -m football_agent.validate
python -m pytest tests/agent tests/demo -q
```

`football_agent.validate` 不联网，执行 9 个合成场景和 2 个真实工具场景，产物放在 `var/parallel/agent/acceptance/`。`validation.json` 索引完整结果，每个 `runs/<run_id>/run.json` 都有逐调用 blob。运行审计：

```powershell
python -X utf8 -m football_agent --store-dir var/parallel/agent/acceptance/runs verify RUN_ID
python -X utf8 -m football_agent --store-dir var/parallel/agent/acceptance/runs show RUN_ID
```

把 RUN_ID 替换为该验证清单内的实际 UUID。正常停止 CLI 返回 0；暂缓/等待返回 2，与工具真实状态保持一致。缺参数/无效输入同样返回 2，不把失败写成成功。

## 当前接通范围

| 工具 | 实现与验证 | 边界 |
|---|---|---|
| query_matches | 读取既有原始档案，复用 Observation 与 normalize，按时点投影比赛 | 来源命名空间 ID；显式比赛不在赛前则暂缓 |
| read_evidence | 原始正文校验、公布/观测/已知入库时间过滤、来源与版本引用 | 原始证据可见性，不代表当时平台已生成相同派生特征 |
| get_features | agent-market-input-v1，选完整同快照、常规时间、不让球 1x2 报价 | 尚未接完整 PostgreSQL FeatureStore；Elo/伤停标缺失 |
| predict | 实际调用 football_research.baselines.api.PredictionService / market | stateless 市场去水基线；结果不是训练效果或策略增益结论 |
| refresh_information：ArchiveReplayTools | 披露截止点以内的后续已有档案，零外部请求 | 明确 archive_replay；不声称进行了线上刷新 |
| refresh_information：LocalArchiveTools | 注入既有采集服务；缺服务返回 NEEDS_CONFIGURATION | 真实联网刷新尚未验收；无另写爬虫 |
| get_run / RunStore | 原子运行存档、内容寻址调用输入输出、hash 链校验、恢复 | 单写者执行；跨进程同一 run_id 的并发调度交 E |
| Agent planner | ToolPlanningPolicy 以 plan_next_action 工具接入，实测相同预算与审计 | 编排接口及测试替身已验证；未配置外部 LLM 服务 |

公共领域契约复用 `football_spider.models.contracts` 的 schema 1.0 和 F 的 Prediction。`ToolResult/Budget/Decision` 是本模块私有执行封装。真实特征是市场输入投影，没有另写模型计算胜率。

真实样本的 1x2 报价为 1.82 / 2.83 / 4.41，F 工具返回主胜 0.4864267590647423、平局 0.31282568957520535、客胜 0.2007475513600524，model_version=`market-proportional-1.0`。这些数值证明接口和血缘通路；未宣称数据已跨源校准或预测质量得到验证。后续披露的准确总进球报价未改变 1x2 输入，模型输出保持不变，重复观测保留审计而不制造变盘。

## 状态、动作与预算

```text
INITIALIZED -> query_matches -> MATCH_READY
            -> read_evidence -> EVIDENCE_READY
            -> get_features  -> FEATURES_READY
            -> predict       -> PREDICTED
            -> policy -> refresh_information
                         -> 有效新证据 -> EVIDENCE_READY -> 再取特征/预测
                         -> 无新增     -> STOPPED
                         -> 429        -> WAITING
                         -> 缺配置/故障/冲突 -> DEFERRED
```

显式 `wait` 保存原状态；`resume` 在 Retry-After 到期后恢复，普通下一步和 run-to-completion 不忙轮询。STOPPED/DEFERRED 是终态，重复点击不增加调用或费用；修改截止点或配置需创建新运行。

默认预算：20 次总工具调用、3 次信息查询、3 credits、60 秒累计工具执行耗时、单工具 10 秒。cost 单位是适配器 credits；本地 I/O 为 0，不等同人民币。界面等待时间不消耗工具延迟预算。每次调用前检查估计费用和剩余额度；成功、失败、超时、429 均保留实际耗时与已报告费用。超时/崩溃且实际费用未知时按预估保守记账，明确 `actual_cost_known=false` 或 `unreconciled_calls`，不把估计宣称实付金额。服务若回报超额费用，日志保留实值并暂缓，不截断账面支出。

工具超时后晚到的返回值被舍弃；线程边界不撤销服务已经发生的外部操作，因此该运行暂缓，服务方应提供自身超时/幂等与核账。不会自动重试来源请求。

固定策略依次查询 odds、lineups、injuries；主题去重，无新增有效证据即停止。费用、主题和总调用限制同样约束手动查询和后续 Agent planner。规则是工程基线，没有研究创新或增益声明。

## 证据与时点语义

- 原始 observed_at 与 available_at 均不得晚于截止；存在 ingested_at 时取两者最大值。可信公布时间和供应商报价时间也要通过截止门槛。
- 不采用 SQLite 行创建时间、文件修改时间或比赛日期冒充字段首次可见时间。
- 当前 `operational` 指以真实已收到的原始证据为边界、使用当前解析器和基线重放；不是恢复过去某次未保存的模型执行，也没有实现可信公开历史档案的 historical_reconstruction 模式。
- ArchiveReplayTools 使用真实档案的原始时间，按固定披露流程演示后续信息查询；这不是对历史操作员真实查询顺序的断言。
- 同源、同种类、相同内容哈希消息去重；实际新观测仍记录返回引用和重复事件。版本修订保留旧事件，明确替换关系。
- 过期/未来/其他比赛/错误模式证据隔离并记录原因。相同事实跨源冲突、相同时点互异版本、显式冲突标记进入暂缓。真实赔率只取完整 1x2 快照，避免混用盘口或半场。
- 缺模型时 prediction 为空或保留此前已得到的结果并标记非当前判断；从不临时生成“真实概率”。新有效证据到达后立即撤下当前概率，等待新的模型调用完成。

## 审计与恢复

每个事件包含递增序号、执行时间、类型、数据、当时完整快照、previous_hash 和 event_hash。每次工具调用包含调用 ID、input_ref、output_ref、status/error、cost、latency、预算。报告同时提供证据版本、模型/特征版本、更新前后概率及停止/暂缓原因。

文件路径：

```text
var/parallel/agent/.../runs/<run_id>/run.json
var/parallel/agent/.../runs/<run_id>/blobs/<sha256>.json
```

写入使用临时文件 + fsync + 原子替换；读取运行时先验证整条事件链、输入输出 blob 和最终快照。认证字段在持久化前脱敏；校验针对实际已保存数据，不在校验时再次脱敏。hash 链用于检出损坏/不一致，并未接入外部签名或防篡改日志服务。

工具开始前与整个状态转换均有持久标记。进程若在工具执行、工具返回已存盘、或证据已写但状态未提交时退出，重启进入 DEFERRED，保留待核账/待复核详情，不重复执行来源请求。验收使用实际子进程强制退出验证这些边界。

## 替换策略与接入服务

```python
from football_agent.engine import Workflow
from football_agent.policy import FixedOrderPolicy, CallbackPolicy, ToolPlanningPolicy
from football_agent.types import Decision

# G 的纯计算策略；设置自己的版本，全部动作继续受执行器约束。
policy = CallbackPolicy(lambda run: Decision("stop", reason="POLICY_DECISION"),
                        version="acquisition-policy-example-v1")
workflow = Workflow(tools, policy=policy)

# 外部 Agent 的推理本身也是有费用/超时的工具。
# tools 须实现 plan_next_action 和 estimate_cost，返回
# ToolResult(data={"action": "query", "topic": "odds", "reason": "..."}, cost=...)
workflow = Workflow(tools, policy=ToolPlanningPolicy())
```

纯回调用于本地策略计算；远程推理应走 ToolPlanningPolicy，让推理调用同样计入预算和日志。工具层通过 `execute(tool, inputs)`、`estimate_cost(tool, inputs)`、`capabilities()` 对接，每项独立可用。在线采集注入 `LocalArchiveTools(refresh_service=...)`，对象提供 `execute(inputs)->ToolResult` 与 `estimate_cost(inputs)`；成功后也只读取符合历史截止的档案，杜绝刷新倒灌。

接口请求、共享导入兼容项与后续依赖见 `docs/parallel/requests/agent/001-interfaces.md`。最终实测状态见 `docs/parallel/status/agent.md`。
