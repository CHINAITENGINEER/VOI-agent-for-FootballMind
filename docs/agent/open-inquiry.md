**开放信息自主调查入口。** 本入口让真实 LLM 根据当前知识状态提出待核实问题，选择搜索或读取动作，取得证据后重新决定查询、等待、暂缓或停止。首次真实运行记录在 [第十六轮报告](../parallel/round16/README.md)。

第十七轮新增可选的 [假设调查协议](hypothesis-inquiry.md)：在配置中指定 `inquiry_protocol: hypothesis-inquiry-v1`，记录假设的适用范围、查询前答案分支与查询后实际证据反馈。未指定时继续使用原协议。真实续查见 [第十七轮报告](../parallel/round17/README.md)。

第十八轮新增 [适用关系与前提协议](applicability-inquiry.md)：`applicability-inquiry-v1` 支持已核验正文继承、带引文的跨范围关系以及 `reflect` 后读取计算缺口再选择查询。旧协议继续按原语义运行。[真实运行记录](../parallel/round18/README.md)

复用已有 `RunStore`、跨进程锁、`RawArchive` 和 `RequestTransport`。历史数值规划仍通过原 `AcquisitionWorkflow` 使用；新增的 `InquiryWorkflow` 处理参数化问题和开放文本，保留 `UNSUPPORTED_OPEN_TEXT`，当前选取依据为 `llm_bounded_exploration`。两种入口的研究结论分开报告。

运行使用现有 Codex CLI 登录，每步以临时、只读进程生成结构化结果，禁用直接执行、浏览、应用及并行代理能力。工具请求由项目执行层统一完成并计费。具体调用方式依据 [OpenAI 官方非交互模式文档](https://learn.chatgpt.com/docs/non-interactive-mode) 和本机 `codex exec --help`。模型服务的格式 Schema 不接受 `uniqueItems`；该限制保留在本地完整校验中，发送给服务的子集只移除此项。

```powershell
# 对一个提前填写并核验过比赛身份、实际截止时点和来源的任务启动真实调查。
python -m football_agent.inquiry_cli --config PATH_TO_TASK.json --root PATH_TO_NEW_RUN

# 每次只推进一步；再次使用 --resume 继续同一 READY 运行。
python -m football_agent.inquiry_cli --config PATH_TO_TASK.json --root PATH_TO_NEW_RUN --step
python -m football_agent.inquiry_cli --config PATH_TO_TASK.json --root PATH_TO_NEW_RUN --resume RUN_ID --step

# 对本轮已保存的真实运行，导出报告或离线核验，不执行任何外部获取。
python -m football_agent.inquiry_cli --config docs/agent/open-inquiry-task.json --root var/parallel/agent/round16/open-inquiry-v4 --resume ffc5e4eb-2b2e-44e6-bc19-5523e4ad98d1 --export-only
python -m football_agent.inquiry_verify --root var/parallel/agent/round16/open-inquiry-v4 --run-id ffc5e4eb-2b2e-44e6-bc19-5523e4ad98d1 --output var/parallel/agent/round16/open-inquiry-v4/verification.json
```

[本轮任务文件](open-inquiry-task.json) 保存的是 2026-09-26 的实际案例及其截止时间。新任务使用独立文件，填写未来截止时间与真实来源依据；超过截止时间的任务会停止，日期不会自动平移。当前 `WAITING/DEFERRED` 是持久终态，等待原因与时间可查，尚未自动建立未来唤醒任务。中断的未知结果保留预算预留，重启不会自动重试。

当前依赖在 [requirements-inquiry.txt](../../football_agent/requirements-inquiry.txt) 中单独声明，不改动既有平台环境。Codex CLI 需由调用机器提供并已经登录，库不会读取或复制凭据文件。`configured=true` 只说明发现命令行程序，实际可用性由调用结果确认。

| 文件 | 作用 |
|---|---|
| `football_agent/inquiry_contract.py` | 问题、预期答案作用、候选动作、来源表述/假设/冲突/未知、停止协议；原文锚点检查 |
| `football_agent/inquiry_provider.py` | JSON 结构化 CLI 调用、公开输入、临时工作目录、实际 usage 和错误记录 |
| `football_agent/inquiry_tools.py` | 有界搜索与页面读取、原始归档、来源域和已发现链接限制、确定性正文提取 |
| `football_agent/inquiry.py` | 请求前持久化、预算、实际返回后的重规划、状态更新和可读报告 |
| `football_agent/inquiry_verify.py` | 零网络重放来源、时间、输入可见性、查询身份、原文投影及成本 |
| `tests/agent/test_open_inquiry.py` | 内容驱动的后续查询、失败/中断、引用真实性、未知价值、截止点等行为检查 |

证据边界：

- 每条 `source_claim` 都要求引用已经取得的成功页面，并且引文必须出现在公开正文中。这是原文锚点检查，语义映射仍标记 `llm_extraction_quote_checked_meaning_unverified`。
- 搜索结果为发现线索，搜索摘要不进入来源事实；网页中的指令只作为页面数据。
- 每次 HTTP 尝试均归档，失败消耗请求预算；不自动重定向或重试。来源费用与 CLI 货币费用未报告时保持为空。
- 页面实际观察、提取可用时间与任务截止时间分开。发布时间缺失保持未知，赛后返回不成为赛前可用证据。
- 当前文本状态会改变后续调查，但尚未接通文本到数值预测的可验证映射，`prediction=null`。
- 当前 VoI 数值保持为空；停止/暂缓是预算约束下的语义判断，尚未证明所有剩余信息的价值非正。
- CLI 返回实际 usage，但此路径未报告具体响应模型标识，保存为未知。CLI 调用数与模型服务内部 HTTP 次数分开。

首次真实回放发现新闻正文链接被前 80 条导航链接挤出。读取器 v2 优先正文区域链接，再保留导航，并显示链接截断信息。v1 原始轨迹仍按 v1 提取规则重放；v2 对同一原文的离线投影另存，不回写到旧 Agent 状态。当前静态 HTML 读取也仍可能缺少浏览器动态内容，这类空缺需作为获取能力限制保留。
