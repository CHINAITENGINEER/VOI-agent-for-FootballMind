# FootballMind 本地事件回放

该入口直接调用 `football_agent.engine.Workflow`，界面不计算胜率、不生成事实。HTTP 服务和页面只需项目已有 Python 环境；不增加共享依赖。工具、特征、模型与策略的接通情况在页面“预算上限与工具状态”中读取实际适配器状态。

## 启动与使用

在 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'D:\a1-足彩'
python -m demo --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765>。`Ctrl+C` 停止服务。第二轮默认运行档案位于 `D:\a1-足彩\var\parallel\agent\round2\runs`；第一轮档案保留，读取时显式传 `--store-dir var/parallel/agent/runs`。服务只接受 IPv4 回环地址或 localhost，并检查 Host、Origin、静态路径、运行 ID 和 JSON 输入。

服务端可用 `--model-catalog PATH` 配置额外模型白名单，用 `--platform-export PATH` 接通完成的平台冻结导出。默认目录提供市场基线及已有西甲、英超、德甲的频率、Poisson、Dixon–Coles 真实训练模型。客户端只提交 `model_id`，模型实例、文件路径、注册表路径与数据路径字段均拒绝；所有概率继续由 `PredictionService` 返回。

1. 选择真实档案或明确标记的测试模式。截止时点输入采用北京时间；请求转换为带时区的 UTC 时间。比赛列表按此时点读取，并筛选当时仍未开赛的比赛。
2. 选择模型、执行时间模式和信息模式。模型说明随目录展示真实版本、训练截止、SHA256 与必需输入；市场模型是截止点报价转换，没有训练拟合。`archive_replay` 固定历史截止，刷新产生的未来观测不进入旧预测；`live` 通过平台刷新服务生成关联的新决策运行并保留父记录，父子共享原预算。`operational` 与 `historical_reconstruction` 分开显示，适配器或模型不支持所选模式时保留真实错误。
3. 没有平台导出配置时，默认勾选“逐批披露本地档案”：用 `ArchiveReplayTools` 展示已留存的真实观测，外部请求为零。关闭勾选后使用真实刷新接点；未配置时保留 `NEEDS_CONFIGURATION`。平台配置存在时使用其 reader 与刷新接口，勾选框不会将平台数据替换成其他来源。
4. 初始化后，点击“执行下一步”逐次调用工具，或“运行至停止”执行固定策略。模型结果就绪后可指定赔率、阵容或伤停查询；等待状态通过“恢复”显式继续，限流的 Retry-After 到期前保持等待。
5. 用时间线、滑块或自动播放浏览事件。概率、证据、预算和状态全部来自所选事件的快照；回到初始化时不展示后续结果。选中工具调用或返回，点击“查看实际工具输入”或“查看实际工具返回 / 原始引用”，读取经过哈希校验的完整工具内容。切换事件后收起上一条工具内容，原始观测与模型/特征/证据版本保留在详情中。
6. 下载完整运行记录 JSON，或复制页面带 `?run=<run_id>` 的地址重新打开。也可展开“打开已有运行记录”并粘贴 ID。重启先校验审计，再恢复该运行保存的模型 ID、哈希、数据 reader、模式与档案路径。版本或哈希变化返回 `RUN_CONFIGURATION_CHANGED`，不会静默改用市场模型。

当前已验证的真实样本：`spftll:3013707`，预测截止 `2026-09-20T11:10:00Z`（北京时间 `2026-09-20 19:10:00`），比赛为赫塔费 vs 马拉加。它依赖本地已有档案，档案缺失时页面如实显示缺数据。真实概率来自研究模块的市场基线；界面不宣称预测有效性或策略增益。

第二轮真实训练模型样本：`spftll:3003895`，预测截止 `2026-09-20T11:25:00Z`，模型 `frequency-england` / `league-frequency-1.0`，训练截止 `2026-09-20T11:22:02Z`。实际工具返回主/平/客 `0.4782608695652174 / 0.2608695652173913 / 0.2608695652173913`。这类模型的预测输入不要求赔率；比赛、联赛身份与训练模型必须匹配。模型训练时间晚于预测截止时仍会正确拦截。

```powershell
python -X utf8 -m football_agent models
python -X utf8 -m football_agent run --archive-replay --model-id frequency-england --match-id spftll:3003895 --cutoff 2026-09-20T11:25:00Z --initial-only
python -X utf8 -m football_agent resume RUN_ID --action next
```

测试模式有常规更新、空结果、超时、限流、重复、过期、冲突、预算耗尽、模型故障 9 个场景。合成概率仅验证执行与交互，不作为实际比赛胜率。默认测试比赛和截止由测试工具返回。

## HTTP 接口

| 方法与路径 | 输入与行为 |
|---|---|
| `GET /api/models` | 返回服务端模型白名单、输入要求、训练截止、版本与状态 |
| `GET /api/config` | 可带 `mode=real/test`、`scenario`、`archive_replay=true/false`、`model_id`、`time_mode`、`information_mode`，返回实际能力状态 |
| `GET /api/matches` | 同上，可带时区明确的 `cutoff`；原样返回比赛工具状态、数据、错误 |
| `POST /api/runs` | `mode`、`scenario`、`archive_replay`、`model_id`、`time_mode`、`information_mode`、`match_id`、`prediction_cutoff`、可选 `budget` |
| `GET /api/runs/{run_id}` | 返回持久化运行记录 |
| `POST /api/runs/{run_id}/action` | `action=next/run/query/wait/resume/stop/defer`；query 另带 `topic` |
| `GET /api/runs/{run_id}/report` | 下载完整运行记录 JSON，包含事件快照及工具引用 |
| `GET /api/runs/{run_id}/blobs/{sha256}` | 读取该运行事件已引用的工具输入/输出；审计链和内容哈希均校验后返回 |

预算字段沿用 `football_agent.types.Budget`：`max_calls`、`max_queries`、`max_requests`、`max_cost`、`max_elapsed_seconds`、`max_tool_seconds`。费用以工具声明的 `credits` 计数；本地档案读取计费为零，合成场景使用明确的测试费用。HTTP 请求在本进程内序列化，执行器另持有同一运行的跨进程单写者锁；其他 CLI/HTTP 进程占用时响应 HTTP 409 / `RUN_BUSY`。

## 验证

```powershell
python -m pytest tests/demo -q
node --check demo/static/app.js
```

测试覆盖 HTTP 边界、缺数据/缺模型原样展示、真实引擎的合成执行与 JSON 报告、非法预算/截止时点、等待与恢复动作，以及应用重启后的工具模式与测试场景恢复。

已安装 Playwright 和浏览器时，可在另一个终端对运行中的服务执行可重复的浏览器检查：

```powershell
python tests/demo/browser_smoke.py
```

该脚本依赖现有 Playwright 环境，不是运行展示入口的依赖。也可使用 `--browser-path` 指向兼容版本的 Chromium。它验证真实市场模型、真实频率模型与测试模式、模型选择和保存配置恢复、事件回放防未来数据泄漏、播放/暂停、JSON 下载、390px 布局及 JavaScript 错误，并将第二轮报告和截图保存到 `var/parallel/agent/round2/demo`。若真实样本档案不存在，输出中明确记录相应真实验证未执行。

## 范围与当前限制

- 本模块只负责本地 HTTP 与展示；没有新增爬虫、数据库类型或模型。
- 工具执行由确定性工作流和可替换策略完成；这里不接外部大模型编排服务。
- 真实档案逐批披露是回放交互，不意味着现场采集已接通，也不意味着每次查询产生新信息。
- 模型暂不可用、缺信息、证据质量问题和预算耗尽均由工作流状态表达，不填充替代概率。
- 页面显示按既有证据得到的概率及输入血缘，不将概率变动解释成单条证据的已验证因果影响。

