# Round193：比赛输入、持久任务与可见状态

2026-10-07，北京时间。执行既有剩余计划的 M1，并并行建立 W3 的输入接口。本项没有重置原计划，也没有宣布整个 M1 已通过真实端到端验收。

## 交付行为

新增 `football_agent.live_match_workflow`，由原 `football_agent.footballmind` CLI 提供入口。用户只输入比赛 ID，或有序双方名称及可选日期；部署设置、研究阶段、预算、来源能力由后台提供。名称有多个未来匹配时返回 ambiguous，不猜比赛。

准备器使用已有 SPFTLL 市场接口、精确实体映射和官方身份网关，自动生成初始化与身份文件。准备成功后登记 f_train，再以 news4 / evidence5 / candidate7 和来源配置 v3 运行既有 Agent。真实模型未就绪时明确为有限探索，numeric VoI 为 UNKNOWN。

任务持久化请求、输入哈希、学习登记与状态迁移。完整采集恢复会重放原始响应，逐项核对 cycle、receipt、候选选择、成本和最终状态，再进入 awaiting_result；不完整请求进入 reconciliation_required。不同任务目录中相同 canonical event 不会重新获得 Actor 预算。准备阶段也先落请求意图，再执行 HTTP；无原始返回的未明请求不自动重发。

结果任务到期后使用已核验的原比赛身份做独立核验。默认开球后130分钟，最多两次结果请求，间隔至少60分钟；每次一个请求，并复核 receipt 与单条原始归档的 ID、URL、哈希、时间和预算。结果不能进入此前 Actor 状态。当前自动结果适配仅支持既有 UEFA 范围。

批次先固定输入顺序和配置，再逐个执行，最多20个输入。失败和不支持保留在原批次，不自动补成功案例。`task-tick` 是一次有界扫描，执行到期工作；本项没有安装或启动后台常驻服务，也不声称任务正在等待某个离线执行器。

## 命令

在项目根目录运行，输出目录必须是新目录：

```powershell
python -m football_agent.footballmind start --match "spftll:match:比赛ID" --output var/live-tasks/my-match
python -m football_agent.footballmind start --match "主队 vs 客队" --date 2026-10-08 --output var/live-tasks/my-match-2
python -m football_agent.footballmind task-status --task var/live-tasks/my-match
python -m football_agent.footballmind task-resume --task var/live-tasks/my-match --settle
python -m football_agent.footballmind task-tick --task-root var/live-tasks
python -m football_agent.footballmind task-replay --task var/live-tasks/my-match --output var/replay/my-match
python -m football_agent.footballmind start-batch --match "比赛ID1" --match "比赛ID2" --output var/live-batches/batch-1
python -m football_agent.footballmind batch-resume --batch var/live-batches/batch-1
```

这些是命令格式，不代表示例比赛已接入。默认部署设置在 `config/live-workflow.json`；准备、Actor、赛果预算分开。每场 Actor 上限4次。旧 `config/footballmind.json` 与旧比赛入口保留原样。长期执行器仍需显式调用 tick；本项尚未完成长期无人值守供给验收。Windows 默认长路径限制下，应使用较短的任务根目录。

## 与自主信息选择直接相关的接口

新增 `encode_visible_state(public, actions)`，输出 `before`、`candidates`、`audit`：

- before 包含已经取得的正文、完整公开行动顺序、实际来源与时钟、当前概率和明确提供的初始先验。
- 已读父页上所有标题和链接都进入 before，不依赖候选子集。点击文章后不得再把已知标题计成新信息收益。
- candidate 只含工具参数、成本、状态和已知标题的引用，不包含未读正文。
- 原文精确去重但保留多次来源引用；FAILED、NO_DATA 与 unknown 区分；未来文本和无效绑定从输入中排除。

这只是确定性结构输入，尚未连接或训练新 f / g。它检查输入内的绑定，原始真实性仍由既有 raw gateway 保证。公开历史缺少的失败回执时钟不做补造。

供给试运行协议已冻结20场新物理比赛的目标、每场4次Actor上限、失败保留、A/B/C分期与四层覆盖统计。20是运行目标，不代表训练充分性。原有7场不计入这20场目标；当前并未启动20场已准入采集。

## 真实结果及验收边界

市场盘点实际取得13场未来比赛，其中12场有完整1X2报价。当前这些候选没有新接通的官方身份网关；并非不存在未来比赛。

按开球时间、provider ID 排序，对前两场完整报价比赛实际执行新 CLI：3085210（芬超）和2910909（巴甲）。各一个 GetInfo 请求，均返回 `official_identity_provider_not_registered`，没有正文 Actor 请求、学习登记或合格新样本。恢复同一批次和一次 tick 均零新增HTTP，原始输入保持一致。

本项共10次联网尝试：市场盘点2次、两个新赛事官方来源探查6次、CLI验收准备2次。六次官方访问均未取得HTTP响应，其中直连芬兰来源为证书验证失败，直连巴西来源为DNS解析失败。它们证明本环境未取得对应证据，不证明网站无赛程，也不能证明尝试路径有效。原始空响应、异常和时间均留存。

准备→登记→Actor→完整回放→到期结果→training_eligible 已由禁止联网的构造集成验证；不完整回执、篡改cycle、重复比赛预算、过期报价、失败批次和结果来源绑定分别覆盖。真实外网只通过不支持状态与恢复验收，不能用构造测试冒充真实新比赛成功。

旧 Round192 零网验收再次通过：11条真实轨迹精确回放、32条实际响应、6场/26条合格监督输入、原始文件和旧验收不变。当前真实 f 仍为 not_ready / model=null；没有新 critic 或正VoI结论。

最终测试数字与全部源码/产物哈希见 `var/parallel/agent/round193/acceptance.json`。复核：

```powershell
python var/parallel/agent/round193/verify_round193.py
python var/parallel/agent/round192/verify_round192.py
```

## 下一步的主阻塞

M1 还缺一场新的、真实来源支持的完整采集及到期赛果；W2 还缺可持续新比赛供给。下一项优先取得一种近期赛事的真实官方格式，接通该赛事范围的身份与结算适配，再由本入口批量登记。不能把现有 UEFA 规则直接套到芬超/巴甲，不能通过放宽身份、时间或训练门槛把当前候选改成合格样本。

共享输入已经准备好，随后可并行实现先验约束的小型 f 和固定 f 下的有符号收益回归 g。样本未到位时只验机制，真实训练与确认仍遵守原分期。新增信息源的目的始终是让 Agent 学会选择信息和停止，不增加人工题材菜单或新前端。
