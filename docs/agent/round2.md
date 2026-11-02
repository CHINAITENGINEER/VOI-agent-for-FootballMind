# Agent 第二轮运行与验收

工作目录为 `D:\a1-足彩`。第一轮记录保留；新运行默认写入 `var/parallel/agent/round2/runs/`。本轮增加服务端模型目录、原模型恢复、真实训练模型的输入门槛、多报价选择、跨进程互斥，以及 live 刷新关联运行。

## 本地启动

```powershell
Set-Location -LiteralPath 'D:\a1-足彩'
python -X utf8 -m demo --host 127.0.0.1 --port 8765
python -X utf8 -m football_agent models
python -X utf8 -m football_agent run --mode real --archive-replay --model-id frequency-england --match-id spftll:3003895 --cutoff 2026-09-20T11:25:00Z --initial-only
python -X utf8 -m football_agent run --mode real --archive-replay --model-id market --match-id spftll:3003895 --cutoff 2026-09-20T11:25:00Z --initial-only
python -X utf8 -m football_agent show RUN_ID
python -X utf8 -m football_agent verify RUN_ID
python -X utf8 -m football_agent resume RUN_ID --action next
```

页面为 `http://127.0.0.1:8765/`。模型下拉框列出真实模型、版本、哈希、训练截止点与输入需求。已保存 run 的恢复沿用原模型和运行配置；模型文件、注册哈希或配置变化时显式返回配置变更。客户端只接收模型 ID；模型文件与白名单路径由服务端启动参数配置。

默认目录包含市场模型和西甲/英超/德甲的 frequency、Poisson、Dixon–Coles，共 10 项。训练模型使用第一轮真实冻结工件，训练截止为 `2026-09-20T11:22:02+00:00`。信息模式与比赛联赛必须匹配；选择另一联赛模型不会按名称猜测 ID。频率/Poisson/DC 的推理依赖已训练状态及合法赛程，赔率缺失不再阻止调用。市场模型要求一份完整、同版本、同口径的常规时间 1x2 报价。

## 时间、刷新与审计

`archive_replay` 保持固定 prediction_cutoff，新的网络观测按实际到达时间过滤。原始返回与观测引用仍写入 `report.refresh_results`，不会把今天获取的信息倒填成历史可见。

`live` 保留父运行原截止点、原证据及原预测；成功归档刷新后创建新截止点的关联子运行。子运行继承全部预算上限、累计工具次数、查询数、费用、请求数和延迟。刷新失败、429、缺配置或请求上限到达时保留相应状态。内容未变化不作为人为调整概率的理由。

父子交接持久化子运行初始快照及哈希，恢复沿用同一子 ID。工具调用中断保守记录未知费用/请求预留，不隐式重复调用。父子引用、预算转移、每个工具输入输出哈希和证据版本都可在 JSON 中检查。

同一 run 由操作系统文件锁保护，Windows 使用 msvcrt，POSIX 使用 flock。CLI 和 HTTP 的并发执行返回 RUN_BUSY/HTTP 409，不提前恢复仍执行中的调用。崩溃后系统释放锁；锁文件保持稳定路径。细节与构造证据见 [locking.md](locking.md)。

## 平台消费

平台公共包导入已接通，Agent 原有 sys.path 兼容桥已删除。`PlatformTools` 接受服务端配置的已完成导出目录，消费研究侧既有 `verify_completed_export`、`load_completed_history_export` 与 `FeatureIndex`。完成标记、manifest 和各文件 SHA256 全部验证。查询与特征连接采用精确 match_id/cutoff/information_mode/feature_version；保留平台 ID，不按球队名称拼接。

`historical_reconstruction` 单独显示延迟假设和事后修订风险。历史最终赛果、当前比赛 xG 和来源 forecast 不复制进赛前 fixture；对 Elo/数值模型使用平台逐场特征，Agent 不计算评级。

平台刷新与读取新接口逐项发布于 `docs/parallel/round2/platform-interfaces.md`。刷新领域服务由平台维护，Agent 仅转换工具结果和展示状态。具体真实验证记录随服务完成后写入本轮验收产物；缺服务保留 NEEDS_CONFIGURATION。

## 现有实测记录

- 无赔率真实模型与 direct PredictionService 完全一致：`models/validation.json`；频率 run `7783e5b0-6001-4903-b965-ebf49c7635c0`，Poisson `676c7387-af55-480a-bbe8-f9fc9bc89788`，DC `0a8b3b97-2f16-4055-9fe3-d7a988b83569`。
- 真实档案浏览器模型选择与恢复：`demo/`；市场 run `8578cab0-2524-4417-b877-cd6452e9f1a2`，frequency run `30b35993-b0cb-42cc-95ed-657d9e64e030`。
- Windows 实际子进程锁与崩溃验证：`lock-validation/validation.json`。这是构造执行验证，费用数值为测试记账。

以上相对产物路径均位于 `var/parallel/agent/round2/`。测试模式使用独立合成证据，并在页面及报告明确标记。功能联通与重复运行一致性不代表预测效果或查询策略收益已得到验证。

## 2026-09-22 继续实施结果

新增 `collector_refresh.py`，通过平台解释器子进程调用刚发布的 `football_platform.api.RefreshService`，领域接口保持独立。HTTP请求、限流、重试、归档、入库均由平台执行。子进程受工具剩余时长约束；request_id 持久记录避免同请求重复执行。原有 collect() 的适配留作兼容，CLI/HTTP 启用刷新时优先正式领域服务。

```powershell
# 主展示入口：默认10个真实模型；启用平台刷新，每次最多2个来源请求
python -X utf8 -m demo --port 8765 --enable-platform-refresh

# 独立平台特征展示：真实Parquet、固定Elo映射、精确历史时点
python -X utf8 -m demo --port 8766 --model-catalog var/parallel/agent/round2/frozen-feature/catalog.json --platform-export var/parallel/agent/round2/frozen-feature/binding.json

# 零联网重跑真实冻结特征闭环；新run ID写入validation.json
python -X utf8 -m football_agent.validate_frozen

# 精确恢复已有领域刷新run，只读；不重发请求
python -X utf8 -m football_agent --enable-platform-refresh resume 98c62242-37ec-4c2f-9b19-ce42c2bbf844 --action next

python -m pytest tests/agent tests/demo -q --junitxml=var/parallel/agent/round2/regression-final.xml
```

`frozen_features.py` 消费已有真实 PostgreSQL 验收报告与 FeatureStore Parquet，保留文件hash、manifest和单来源ID对应证据。没有自行计算Elo或写入数据库。`binding.json` 是Agent自己的只读配置，不是新公共契约/平台导出。`spftll:3003895` 对应 canonical `69bb2f11-d1c7-598e-b3ac-0ad84744b65d` 来自平台实际验收target；跨源确认没有额外推断。

真实冻结特征run：`c5a4e1b1-69fa-4f72-a2b3-9ffa6f48145e`，cutoff `2026-09-20T11:44:15.289432Z`，特征 `prematch-v1.1`，模型 `elo-davidson-mapping-1.0`，**固定参数未拟合**。输出与direct PredictionService完全一致；模型SHA `eb9f113134285332e5ba43730ff7a2c26cbcb3abac6f2c305c34122436eb8b55`，Parquet SHA `fbe0ff59cf2d01df6228a20165d68876ebb68c8dbb02de2767801698f7794185`，manifest `be18eb99c529fecbdaa4fbde80230dcbb268bfa1e807771214703c297899f60c`。验收位于 `frozen-feature/validation.json`、`report.json`、`browser-final/`。

真实联网领域服务run：`98c62242-37ec-4c2f-9b19-ce42c2bbf844`，实际1个请求/0重试，原始观测 `bb4fe1f0-b1bd-4d2f-aee7-ca1ec5a64a00`，返回HTTP200但来源 `PROVIDER_ERROR`/“数据为空”；平台 `PARTIAL`、network `FAILED`、数据库 `NEEDS_DATABASE`。Agent最终 `DEFERRED/FAILED`，保留原训练模型预测和原始引用，无新增live子run。1 credit为本地预算预留，实际货币费/供应商credit未知；界面和JSON分开说明。详见 `real-refresh/platform-domain-report.json`、`real-refresh/browser/`。

在正式服务发布前还通过既有collect执行了2个有界请求，响应均为空。最初仅按HTTP200判断的验收已被纠正：`real-refresh/correction.json` 明确否定原 `validation.json` 中成功推进live的结论，旧父子run `b16b123c-e349-4f02-9b04-e169117dce3c` / `50cb62ec-209c-49e2-be61-79e630bcb2d0` 只保留作为问题审计。修复后同一原始响应零联网重新处理为 `NO_DATA`、原概率不变；不将这两条旧run作为成功验收。当前适配复用平台normalize验证业务正文，空data和无合格事实均停止。本次共3个真实外部请求，均已归档。

当前合并回归 **219 passed in 66.57s**。新真实冻结特征与正式刷新失败路径各通过4项浏览器检查，均0 JavaScript错误，覆盖恢复、概率与服务输出一致、raw引用、早期快照隔离、移动截图；原10项浏览器和构造live/锁测试证据保留。

剩余外部依赖：生产数据库未配置；完整历史赛季冻结导出尚待完成标记；本轮来源空返回，因此“真实新合格证据进入live并更新预测”仍未获得成功实例。代码层父子run/预算/崩溃/不变内容已通过构造验收，真实运行准确保留失败。没有将失败、构造或旧档案伪装为成功的实时数据更新。
