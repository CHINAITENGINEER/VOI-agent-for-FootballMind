# 单场开放信息选择：已运行的研究版本

更新：2026年10月9日，北京时间。提交截止为10月25日；10月20—22日预留人工体验，23—24日修复和封版。唯一总计划仍为 [footballmind-delivery-plan.md](footballmind-delivery-plan.md)。本说明对应真实数据和实际执行，不是未来功能清单。

## 当前入口：语义情景与数学 VoI

### 最新完成案例：新正文合同的端到端运行

`var/submission/deepseek-evidence-only-h2-20261009/report.md` 对应重新执行的完整链路，
采用DeepSeek Pro、开启思考／low、候选发现v2、方向合同、二步规划、4次来源请求预算及8次语义
调用上限。实际6次API调用均通过协议，合计284.07秒、38,316个报告token，无语义失败或修正调用。
来源是历史归档，来源网络请求0；4次来源预算单位包含启动目录，不能把它们称为真实联网HTTP。

| 实际步骤 | Agent／求解器行为 | 实际返回和处理 |
| --- | --- | --- |
| 启动与发现 | 从已见元数据提名3条查询、2篇文章，再生成情景与观察律 | 没有提前读取候选正文 |
| 第一次获取 | 选伤停正文，两步净值下包络约0.00257926 nats | 只有可用性信息；方向未知，门控将`refutes`降为`no_answer` |
| 再发现与第二次获取 | 提出3条新查询，数值选择确认首发的查询 | 归档搜索没有可用正文，保持`no_answer`；此步有数值并列候选 |
| 第三次获取 | 选剩余可执行的赛前预览，净值下包络约0.000508211 nats | 没有明确本场方向效应，保持`no_answer` |
| 停止 | 来源请求预算耗尽 | 不是所有潜在信息都没有价值，也不是开放世界充分性证明 |

三次无答案都保持各声明模型的预测与后验不变（1e-12容差内），两次无答案后继续选择其他动作。
名义初终概率均约53.4884%／30.2326%／16.2791%。新LLM提案不同于旧运行，不将运行间差异归因于
正文隔离单一因素。这次记录覆盖了拒绝无依据更新的完整路径，尚未覆盖真实有信息方向更新；模型
未重新训练，经验g保持deferred。历史重建不能验证原赛前网页版本或新的前瞻预测效果。

现在可对任意采用新合同的保存运行执行闭环验收（输出必须是新文件）：

```powershell
python tools/check_semantic_evidence_loop.py `
  --run var/submission/deepseek-evidence-only-h2-20261009/run.json `
  --output var/submission/my-evidence-loop-acceptance.json
```

它检查实际分类输入与本次取得正文的绑定、VoI动作与观察的顺序、方向门控、无答案前后所有声明
模型的不变性、终态概率和停止原因，同时重算原数学轨迹。不调用来源或模型、不改原run、不作
事实或语义真值证明。`coverage.unexercised` 显式列出未触达的分支；`passed`不等于所有能力均被
这个案例验证。当前真实产物为同前缀的 `-acceptance.json`：2条正文分类、3次无答案转换、2次继续
选取、0次有信息方向更新，4周期数学回放通过。

另对该轨迹完成同状态h1/h2对照，见同前缀的 `-horizon.txt`：4个状态的动作均相同，没有h1停止而
h2获取的状态；两步规划值变大不自动证明策略更好。这些对照新增网络和语义调用均为0。

### DeepSeek 直接 API 通道（2026-10-09 新增）

`tools/run_footballmind_semantic_case.py` 与 `tools/run_historical_semantic_case.py`
均支持 `--provider deepseek`；原 `--provider codex-cli` 保留兼容。Provider 通过同一
`configuration/decide` 接口注入，数值 VoI、正文准入、方向合同与停止规则没有迁移到 LLM。
DeepSeek 默认选择 `deepseek-v4-pro` 与开启思考；也可通过 `--model deepseek-flash`
或 `--model deepseek-v4-pro` 显式选择。
`--thinking enabled` 使用固定且记录在配置中的 `reasoning_effort=low`，用于限制推理时延；
`--thinking disabled` 关闭思考。不同配置必须使用不同输出目录，运行中不自动切模型。

现在也可显式指定 `--observation-thinking enabled|disabled`，只覆盖正文分类阶段；省略时继承
全局 `--thinking`，发现、概率提案与候选扩展不受影响，各次回执保存实际模式。默认继续开启思考：
本轮两份归档正文的无思考对照虽然较快，但其中一次引句被改写，未通过逐字校验，不能直接用于更新。

安装可选 transport 依赖：`python -m pip install -r football_agent/requirements-deepseek.txt`。
密钥只从当前进程的 `DEEPSEEK_API_KEY` 读取，或由调用方以参数注入；不写项目文件。
独立 PowerShell 终端可以这样配置（输入内容不回显）：

```powershell
$deepseekSecret = Read-Host "DeepSeek API key" -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', $deepseekSecret).Password
python tools/run_historical_semantic_case.py `
  --provider deepseek --model deepseek-v4-pro --thinking enabled `
  --candidate-discovery-version 2 --directional-evidence `
  --planning-horizon 2 --requests 4 --semantic-calls 8 `
  --provider-timeout 120 --max-output-tokens 8192 `
  --rediscoveries 1 --output var/submission/my-deepseek-historical-run
```

该命令使用已归档的 Leeds–Fulham 材料执行新的真实语义调用，属于**历史重建**：
文章后来采集，不能证明当时正文版本，不是新的前瞻预测或预测效果实验。
`historical-panel-manifest.json` 分别统计来源网络请求（零）和真实语义 API 尝试数。
每次试验需新输出目录；真实赛前运行仍须通过原 `case.json` 的截点和身份校验，
不能修改旧截点来复用已经过期的赛前案例。

DeepSeek 的 JSON Output 仅保证 JSON 模式，完整 schema 放在 system 消息中，
本地继续核验候选数、实际可见 ID、概率归一、方向与逐字证据。传输层拒绝空返回、
截断、工具调用、多 choice、重复 JSON key 和非有限数。服务器返回的内部思考内容不归档。
连接、响应头和完整正文共用总截止时间；无 HTTP 隐式重试、无重定向和静默 Provider 切换。
runtime 原有的一次语义修正仍计入调用预算，不能篡改失败提案后接受。

每个调用保存输入、schema、输出与回执；回执区分请求模型和服务端报告模型，保存 token
及缓存计数、耗时、失败代码。不推算真实扣款；超时的远端用量未知。失败调用同样保留
runtime 中的耗时和已返回用量。`--replay` 无需 API key，不调用任何模型。

已执行的完整记录为 `var/submission/deepseek-historical-h2-20261009-low/`：Pro开启思考、low强度，
7次语义调用约297.76秒，4周期数学回放通过。实际动作是伤停正文 → 再发现后的战术查询 →
赛前预览 → 来源预算耗尽停止；第一条观察校验失败，第二条查询在归档环境没有正文，均未更新。
第三条观察通过方向合同及逐字引用校验，名义概率由53.49%／30.23%／16.28%变为
57.15%／28.47%／14.38%。这是未校准语义模型下的更新，不能据此声称预测更准确。
完整历史记录保留两次错误的 `unknown/home/availability_only` 返回；随后提示词补齐
`unknown → subject=unknown` 的映射并拆分报错，未覆盖、修改或重新分类该轨迹。
修正后的同一输入单次真实复测存于 `var/submission/deepseek-directional-contract-check-20261009/`，
37.39秒通过，返回 `unknown/unknown/availability_only`，保持不可用于后验更新。该复测不拼接进
原运行，也不能外推为协议零失败率。

默认配置的选择依据是本次接线诊断：Flash无思考两次超出候选数量上限，Pro默认高思考在提案时
超时，无思考返回空提名，Pro低思考完成上述链路。样本不足以作模型优劣排名或时延保证。

官方接口依据：[模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、
[Chat Completions](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/)、
[JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)。本地 API Provider
接通不等于经验 `g` 训练完成；语义情景概率继续标为未经校准的假设。

### 正文分类隔离与真实时延对照

新运行标记 `observation_input_contract=semantic-observation-evidence-only-v2`。分类器仅接收比赛
身份、当前动作的精确问题与方向合同、本次已取得且准入的正文；基线概率、情景先验、条件胜平负、
观察似然、VoI及其他候选不进入分类输入。这样避免把模型的数值预期当作正文证据；仍不能排除语义
模型本身或问题措辞的偏差。原有正文投影长度、精确引用验证和方向更新门控保持不变。

对旧DeepSeek运行的 `record-002` 和 `record-006` 用相同正文进行独立复测：

| 分类模式 | 第一篇：伤停正文 | 第二篇：赛前预览 | 两次调用合计 |
| --- | --- | --- | --- |
| Pro，开启思考、low | 47.54秒，通过；只有可用性，不允许方向更新 | 33.19秒，通过；无答案 | 80.74秒 |
| Pro，关闭思考 | 6.33秒，引句非逐字正文，拒绝 | 4.20秒，通过；无答案 | 10.53秒 |

对应归档为 `var/submission/deepseek-evidence-only-20261009/result.json` 与
`var/submission/deepseek-evidence-only-nonthinking-20261009/result.json`，合计4次模型API调用、
0次来源网络请求。两组输入正文完全相同；数值隔离后输入JSON字节分别减少42.29%和18.20%，
不是全文被截短，也不是总prompt token或时延同比减少。后执行的关闭思考调用有缓存命中，两份
样本没有随机化或重复，不能把时间差归因于思考模式单一因素，不能外推准确率或时延保证。

旧 `record-006` 曾返回 `home_positive`；新的两次独立分类均未支持方向更新。旧运行仍保留原值，
这些复测不是新端到端轨迹，也不将其拼接回旧run。当前没有证据支持把旧上调称为稳定语义结论。

### 信息选择的假设敏感性

新增离线诊断入口：

```powershell
python tools/analyze_semantic_acquisition.py `
  --run var/submission/deepseek-historical-h2-20261009-low/run.json `
  --output var/submission/my-acquisition-sensitivity.json
```

输出JSON和同名TXT，要求新路径。先核验原数学回放和比赛／截点，再固定已发生证据、候选身份、
请求预算与原规划器，逐状态单因素重算；没有HTTP或LLM调用。五类预设格点的定义是：

- 先验：一项情景权重乘0.5或2后归一，并重新对齐同一外部基线。
- 结果差异：已对齐的条件胜平负分布与基线混合，保留系数0.5或0。
- 来源区分能力：一个可执行动作的观察似然与其原先验预测边际混合，保留系数0.75、0.5或0。
- 可回答率：一个可执行动作的支持／反驳概率乘0.75或0.5，其余质量移入跨情景相同的无答案桶。
- 获取成本：每次请求的损失等价成本乘0.5、2或10，整数请求预算不变。

来源与可回答率只改变未尝试、仍可执行的动作，已观察动作的似然不改；这些变更及成本变更必须
保持当前预测不变。先验或结果分布变更可能改变有信息历史下的后验，报告显式保存变化量。每个
格点由现有联合模型验证和求解，不直接修改动作评分。所有格点各自重规划；它们不共同构成一个
针对全体格点的稳健策略。格点没有概率权重，翻转计数不是统计置信度或风险发生率。

已运行产物 `var/submission/acquisition-sensitivity-20261009/deepseek-low-v2.{json,txt}`：
3个可执行状态共104个格点，另有1个预算终止状态不计稳定性。第一步伤停与战术的两步净值差约
1.11e-16 nats，远小于1e-12容差；第二步也存在并列。不能用确定的日志动作声称它明显优于其他
信息。把伤停来源区分能力保留为75%时会转向战术查询；较高成本或零区分能力也可使策略停止。
这是当前VoI依赖来源模型假设的直接诊断，不是经验 `g` 已训练的证明。

诊断同时分解已发生调用开销：旧完整运行约297.76秒中，两次无效分类共90.02秒。该值帮助定位
优化对象，不等于这些时间已经全部节省。默认不采用本轮证据校验失败的快速配置。

### 核心运行机制

当前执行版本由 Agent 根据本场公开状态自主提出信息需求和查询，生成有限情景与观察概率假设；数学模块计算期望 log-loss 风险下降及成本，执行选中的动作并按实际正文重新决策。用户不提交研究问题，也不指定伤停、天气或战术的优先级。预先给定的是一场比赛、来源权限、基线、预算和截止时间。

新 CLI 默认 `--projection-version 8 --candidate-discovery-version 2`：入口索引从实际已归档 HTML
保留最多320条可见且符合来源权限的链接；Agent 查看有界标题目录，提出搜索词与待建模文章 ID，
两类合计最多8项。数学模块随后逐候选估值，语义提名不直接触发读取。原先“查询优先再加目录前几条”
的隐含限制已移除；旧版本仍可显式使用 `--projection-version 7 --candidate-discovery-version 1`。
首次真实观察后允许一次基于历史的再发现，v2 可提出新的搜索或未建模文章，均需先建模再估值。
没有语义提名时不会用 DOM 前几条自动补齐，也不会将空菜单解释为信息已经充分。

`candidate-discovery.json` 与报告分别保存目录展示数、上游截断、提名数与未建模数。
这仍是有界的语义候选生成：未提名和超出目录上限的条目没有被数值估值，不能称它们没有价值，
也不能把当前菜单的停止解释为整个开放环境的最优停止。

规划器默认使用一步视野；需要展示“首步即时价值低、但可能打开第二条高价值信息”时，
可显式加入 `--planning-horizon 2`。该模式只在冻结的可见候选菜单内规划一项后续
动作。2026-10-09 起，新版本按每个可能返回选择所有污染模型共享的一项稳健续动作，
再逐模型积分同一策略；它是可实施的逐分支策略，不声称全局最优策略树。
已尝试动作、同事实组约束与请求预算在第二步继续生效；即使损失成本为零，也不能超请求数。
运行把即时 `net` 与前视 `planning_net` 分开记录；它不是训练好的 `g`、
全局开放世界停止证书，也不会为反事实分支预取正文。

推荐直接对一份已完成（或已有合法决策状态）的冻结运行做同状态 h1/h2 对照：

```powershell
python tools/compare_semantic_horizons.py `
  --run var/submission/p6-semantic-scenario/live-01/run.json `
  --output var/submission/my-frozen-horizon-analysis.json
```

该命令先核验原数学回放，再固定全部情景、似然、证据、候选、成本和预算，只改变规划深度；
不联网、不调用 LLM，也不拼造未执行的后续轨迹。后期状态属于原策略的实际轨迹，不能假定
另一策略也会到达它们。已有结果见 `var/submission/frozen-horizon-20261009/`：正常成本主运行
两种规划都停止；历史零成本机制的 6 个状态中 1 个动作选择不同，但两种动作顺序的二步价值
差约1.11e-16，远低于1e-12容差，只能视为并列；没有一步停止而二步获取的状态。
带并列标注的输出为该目录中的 `live-01-reviewed` 和 `mechanism-03-reviewed`。

若比较两份独立运行，比赛相同还不够；首个冻结状态的概率模型、来源动作绑定、证据和预算
也必须一致，否则报告 `not_comparable`。可用：

```powershell
python tools/compare_semantic_horizons.py `
  --h1 <h1>/run.json `
  --h2 <h2>/run.json `
  --output <comparison>.json
```

比较器会拒绝不同比赛或截止时间，并精确回放原记录。两份独立运行中任一未完成，或两次
LLM 提案不同，都不会被当作可控对照。只有在同一冻结状态中 h1 停止、h2 选择的共同策略
净值下包络严格为正，才记录为二步启用获取的机制证据；名义净值为正不够。空列表也是
有效结果，不能用两个独立模型的动作差异声称二步改善。

新运行默认使用 `semantic-directional-observation-v1`。只有逐字正文确实回答本场问题并明确说明主队／客队影响方向，`supports/refutes` 才能保留为概率更新；“球员受伤、康复、训练或入选”这类没有影响方向的状态会被降为 `no_answer`。这道门控解决了“信息存在”被误当成“球队变强或变弱”的语义偷换。方向合同现在还拒绝同一支持／反驳桶内同时放入两个相反的球队方向，或把 `mixed` 与单队方向混放，避免用过宽合同让方向门控失去意义。旧冻结运行如需原协议兼容，可显式加入 `--legacy-observation`。

这仍然不是中文语义蕴含证明：本地校验能证明方向标签、主体、原文逐字引用和动作合同彼此一致，但不能证明引用的句子在现实语义上真的蕴含该方向，也不能校准影响幅度。该部分继续作为可审计的 Provider 语义判断，并在报告中标记为未校准假设；未来若有时间戳信息流，可再训练方向分类器和 `g`。

已保存两份真实运行，必须按各自实验角色阅读：

- `var/submission/p6-semantic-scenario/live-01/report.md`：正常成本0.001 nats/HTTP，8候选净价值下包络均非正。1次入口HTTP、2次LLM、没有后续读取，初终概率62.031458% / 25.392844% / 12.575698%。
- `var/submission/p6-semantic-scenario/mechanism-01/report.md`：零损失成本的执行机制诊断，6HTTP、6LLM、6轮、1次扩展。2次搜索失败，3篇真实正文均不足以回答本场问题，5个no_answer使所有模型的后验不变，预算耗尽。这不是主策略有效性的实证。

冻结的是实际语义提案、协议、观察律与决策参数。真实调用使用CLI默认模型，其回执没有远端模型身份，不宣称已固定模型权重。先验、条件胜平负和观察似然属于未经校准的语义模型假设；污染模型集合的上下界不是置信区间。HTTP折算成本与实际LLM用量分开记录，货币费用未知，LLM开销尚未进入VoI目标。

在项目根目录开启新的赛前运行（输出目录必须尚不存在）：

```powershell
python tools/run_footballmind_semantic_case.py --output var/submission/manual-semantic-case --requests 6 --semantic-calls 12 --rediscoveries 1
```

每次语义 Provider 调用的默认硬截止时间为180秒。若只做有限时长的演示，可以显式降低它，例如：

```powershell
python tools/run_footballmind_semantic_case.py `
  --output var/submission/manual-semantic-case-fast `
  --requests 4 --semantic-calls 8 --rediscoveries 1 --provider-timeout 45
```

超时会保存诊断并使本次运行失败闭合；系统不会把截断输出当作提案，也不会用默认概率替代 Provider 返回。由于运行时最多允许一次有界修复，较短截止时间可以避免一次坏提案把演示拖成长时间，但会增加 `runtime_incomplete` 的概率。修复现在按错误类型分级：正文引用或 schema 形状错误允许一次修复；概率和动作能力契约错误直接闭合并记录 `failure_code`，不再重复消耗一个长 Provider 调用，也不做静默归一化。

新增 `--resume-from <原失败目录>/run.json` 只支持初始化阶段的继续：原运行必须保存了当前版本的
案例／协议绑定、恰好一次已完成且可核对的入口 HTTP、失败的语义调用，并且尚无概率提案、
数学周期或正文观察。所有案例、来源投影、发现版本、Provider、规划深度和预算设置必须相同，
实际时间仍须在原预测截点前，输出必须使用另一个新目录。已接受的发现结果可以复用；入口
不重新读取，失败调用仍占用累计预算。恢复来源与文件 SHA-256 会保存到 `resume-provenance.json`。
这是本地记录一致性核验，不是外部真实性认证，也不是新增独立实验。缺少绑定的旧运行会被拒绝，
不能把别次成功的 Provider 诊断响应移入本次运行。不会自动无限重试。

2026-10-09 的通道复测曾成功：原 Python Provider 的极小 JSON 调用用时13.29秒；原先超时的
同一发现输入与协议用时22.73秒返回3条合法自主查询。此前失败发生在发现阶段，不能确定
归因于情景协议复杂度；有一次旧诊断包含网络重试信号，根因仍未确定。保持 Codex CLI 默认
模型；当时尚未接入独立 API（现已新增上文 DeepSeek 通道）。随后 `prospective-shared-h2-20261009` 完成1次入口HTTP后，首次发现调用
再次90秒超时，已保存 `runtime_incomplete`，没有候选或概率更新。因此当前只能确认间歇成功，
不能称完整在线运行已恢复。部分事件、旧案例或默认概率都不能拼成语义提案；单次调用成功
也不等于有效正文获取或策略效果成立。

### 单场输入契约与最小演示

人工体验时只提交比赛身份；不要先写“请查询伤停／天气／战术”等研究问题。先用无网络的准备命令生成可审计请求和初始未知目录：

```powershell
python tools/prepare_footballmind_match.py `
  --match "Santos vs Flamengo" `
  --competition Brasileirao `
  --kickoff 2026-10-10T20:00:00+08:00 `
  --output var/submission/manual-match
```

该命令只规范化比赛身份、时间和候选信息能力，不调用来源、不训练模型，也不替用户决定先查哪类信息。它会写入 `request.json`、`initial-state.json` 和 `bundle.json`。没有可靠开球时间时可以省略 `--kickoff`；状态会明确标为 `unresolved`，后续先解决比赛身份，系统不会猜日期。输出目录必须是新的，避免把一次运行的证据覆盖掉。

要把准备态接入完整语义 VoI 运行器，必须显式提供同一场比赛的来源 profile 和结构化基线。物化器不会生成均匀先验，也不会从旧案例复制来源：

```powershell
python tools/materialize_footballmind_case.py `
  --bundle var/submission/manual-match/bundle.json `
  --source-profile var/submission/manual-match/source-profile.json `
  --baseline var/submission/manual-match/baseline.json `
  --output var/submission/manual-match-case
```

`source-profile.json` 至少要绑定主客队、`allowed_hosts`、`seed_urls` 和 `bootstrap_url`；`baseline.json` 必须给出明确的 `[主胜, 平, 客胜]` 三分类概率，以及同场 `match_id`、同一 `prediction_cutoff` 和 `source` 或 `model_version`。物化成功后，`case.json` 才能作为语义运行器的 `--case-dir`。缺少其中任一项会返回 `case_not_ready`，这表示资料尚未具备运行条件，不代表系统可以用旧比赛继续猜测。

如果要运行数学语义 VoI，`--match` 只允许绑定已经注册且主客顺序一致的语义案例：

```powershell
python tools/run_footballmind_semantic_case.py `
  --match "Arsenal vs Leeds" `
  --case-dir var/submission/p1-single-match/initial-state `
  --output var/submission/manual-arsenal-leeds `
  --requests 6 --semantic-calls 12 --rediscoveries 1
```

若要运行有限两步互补信息模式，在同一注册案例上使用新的输出目录：

```powershell
python tools/run_footballmind_semantic_case.py `
  --match "Arsenal vs Leeds" `
  --case-dir var/submission/p1-single-match/initial-state `
  --output var/submission/manual-arsenal-leeds-h2 `
  --requests 6 --semantic-calls 12 --rediscoveries 1 --planning-horizon 2
```

运行器会核对案例中的规范主队、客队、来源能力和启动入口；`case.json` 必须在 `source_profile.bootstrap_url` 中显式登记一个同时属于 `seed_urls`、且主机属于 `allowed_hosts` 的 URL。上例使用已经登记 Arsenal–Leeds 身份的 `p1-single-match/initial-state/case.json`。`p6-semantic-scenario` 目录只保存已完成的运行回执，不含可作为 `--case-dir` 的 `case.json`，不能把它写成输入目录。换成未注册的比赛、交换主客或把旧案例当作新比赛使用都会拒绝。这样做是为了防止把一场比赛的基线、来源证据或历史状态静默套到另一场比赛。案例可以同时登记规范名和用户可见别名，例如规范名为 `Leeds United` 时允许输入 `Arsenal vs Leeds`；`vs`、`vs.`、`v` 和多空格写法会统一解析，但别名冲突会直接报错，不会猜测。对尚未注册的比赛，`prepare_footballmind_match.py` 只生成身份和未知目录，不能直接伪造完整语义案例；要继续运行，必须另行完成来源绑定、启动 URL 和结构化基线，并写入案例，或显式传入 `--bootstrap-url`。不要为了让演示跑通而复用旧案例。

人工审阅 `initial-state.json` 和最终 `report.md` 时，依次检查：系统是否从比赛状态生成多个候选未知；是否依据当前状态选择下一动作；返回后是否更新候选和概率；是否在剩余动作价值低于成本或预算／截止时间约束时停止。`empirical_g=deferred` 是当前明确状态，表示经验 g 尚未用于动作选择；报告中的概率和 VoI 是冻结语义模型假设下的机制结果，不是预测准确率证明。

在实际体验前先运行一次底层机制验收，确认演示环境没有把评估信息、固定优先级或赛果泄漏给 Actor：

```powershell
python tools/check_footballmind_autonomous_mechanism.py `
  --output var/submission/mechanism-acceptance/report.json
```

当前内置 8 个离线场景、67 项检查应全部通过。该验收覆盖“已出线不等于轮换／弱阵”“平局即可出线仍需单独查询战术”“伤停与天气的条件差异”以及来源冲突、过期、缺失、重复时的保守处理；它验证机制边界，不把场景假设当作赛果准确率或经验 `g` 校准。

本场截点是2026-10-09 11:30 UTC，即北京时间19:30。超过截点会拒绝发起这场比赛的新赛前采集；以后使用已存回放，不能回填抓取时间。这个CLI使用现有一场比赛配置，没有新增通用赛事解析或商业覆盖。

无需网络的数学重放：

```powershell
python tools/run_footballmind_semantic_case.py --replay var/submission/p6-semantic-scenario/live-01/run.json --output var/submission/manual-semantic-replay
python tools/run_footballmind_semantic_case.py --replay var/submission/p6-semantic-scenario/mechanism-01/run.json --output var/submission/manual-mechanism-replay
```

回放逐周期从冻结参数重建数学模型并比较决策，不再次调用LLM、不发HTTP，不声称重新验证语义分类真伪。已通过1轮/6轮精确回放。正文引文、工具返回、候选、每轮净值与停止原因均在对应run.json/report.md；原响应另存source-archive。

运行后的人工审阅重点是：Agent提出的未知是否适合本场、有限情景有没有漏掉关键机制、概率假设是否可辩护、引用是否确实回答本场问题，以及停止理由是否被错误解释成“所有信息都足够”。不要求看到人为制造的概率变化。

追加的 `mechanism-03` 在相同历史面板上把零成本预算扩展到6次请求，出现了一条真正可检查的条件更新：搜索无答案后，Agent选择读取第一篇正文；分类器引用 Daniel James 康复及 Bornauw 回归训练的连续原文，将该观察判为 `supports + match_specific`。第二篇正文虽提到可选择球员，但不能证明战术改变，保持 `no_answer`。5个观察中有1个支持、4个无答案，名义概率从53.4884% / 30.2326% / 16.2791%变为50.7306% / 31.0413% / 18.2281%。这只是预训练语义假设和历史出版物重建下的条件更新，不是独立真值验证；[机制03报告](../var/submission/p7-historical-semantic/mechanism-03/report.md)和[同面板对照](../var/submission/p7-historical-semantic/mechanism-03/controls.md)保留完整证据。

审阅时不要把该 `supports` 解读成“确认利兹关键缺阵”：正文证明的是健康状态信息存在，尚未证明方向、关键性或影响幅度。方向性语义压缩的风险和建议的 `home_negative/home_positive/.../unknown` 原子标签见 [机制03审计](../var/submission/p7-historical-semantic/mechanism-03/audit.json)。

当前代码把这些方向放在 `directional_evidence` 扩展中：新运行用它作为实际观察的资格门控，旧协议和冻结运行保持不变。人工测试可以检查标签是否有依据；门控只阻止无方向的状态信息进入更新，不提供影响大小校准，也不应把它解释成概率变化或 VoI 增益本身。

## 历史重建机制案例

按元数据规则从已有 C 段资料选择的 Leeds–Fulham 面板见 `var/submission/p7-historical-semantic/case-selection/`。选择只看时间、官方来源元数据和 raw hash；面板排除了 outcome、label、目标比分、先验和 VoI。正常成本本地重放：

```powershell
python tools/run_historical_semantic_case.py --output var/submission/p7-historical-semantic/live-new --requests 4 --semantic-calls 10 --request-loss-cost 0.0001
```

此命令默认使用归档面板，不产生网络请求；它把 source publication 保留为历史事实，把实际 capture 时间保留为后来归档元数据，不能宣称是当时的完整站点快照。已完成的 `live-03` 在目录发现后停止。若需要仅验证正文准入、分类和重估链，可使用零成本机制参数：

```powershell
python tools/run_historical_semantic_case.py --output var/submission/p7-historical-semantic/mechanism-new --requests 4 --semantic-calls 10 --request-loss-cost 0
```

零成本运行不是策略成功证据。已完成 `mechanism-02` 实际读取两篇正文、保存3个分类观察，均未达到支持／反驳条件，后验不变。`controls.json` / `controls.md` 在同一个已返回面板上比较自适应 VoI、固定顺序、冻结后验消融和零采集；缺少返回会标 `incomplete`，不补造未来分支或实测增益。

## 下方为保留的经验路线对照

下文P2–P5训练与live-03用于复现既有负结果，不是当前语义情景入口。当前主线已经实现；旧“待选择”状态由上方及唯一计划取代。

最新A-only预测器选模在 `var/submission/p5-core-algorithm/A-selected-f-g/`：纯基率胜出，重建B正文增益全0。它是当前经验路线的负结果，不替换下文live-03为完成版。Agent不将这种模型零响应当成外部信息无价值，而保留UNKNOWN。可用下列命令复现（新输出目录、无需联网）：

```powershell
python tools/train_footballmind_single_case.py --cases var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json --dataset-version reproduce-A-selected-f-g --f-fit-scheme a_group_loo_v1 --g-fit-scheme b_group_loo_v1 --text-projection observed-news-titles-v1 --output var/submission/reproduce-A-selected-f-g
```

经验路线的限制促成了上方已实现的语义情景与数学VoI版本；两条路线的概率依据和研究主张分别记录。

## 当前可复现的案例

Arsenal–Leeds，2026年10月10日11:30 UTC开球。来源与身份探测在 `var/submission/p1-single-match/source-probe/`；初态和独立结构化对照在 `var/submission/p1-single-match/initial-state/`。

主运行是 `var/submission/p3-single-match/live-03/`。用户侧只需确定比赛；研究部署预先配置来源入口、模型、成本与预算，不输入调查问题、新闻重要性或查询顺序。

**当前核心验收未通过。** 工程循环可重放，真实导语可入模，但B内选模得到常数收益基线，尚未学出有效的状态条件信息价值。先前live-02的“正文进入f”说法已纠正：其4周期实际正文入模数均为0；导航与链接状态引起的概率变化不能等同正文处理。旧原档原样保留，纠正报告在 `var/submission/p4-review/live-02-corrected-report/`。

本次模型与场景是开发案例，不是独立确认试验。英文字符/主客角色特征模型支持机制研究，不宣称已经掌握战术、指代与复杂因果推理。

## 实际模型和数据

当前冻结模型：`var/submission/p2-historical-context/real-open-menu-v4/model.json`。

版本：`historical-role-context-adapter-v1:fe9f4ecd7ed3c7db61bcc2d9`。

数据：`var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json`，26场独立英超比赛、62篇实际赛前正文、112条真实目录标题。历史页面是在本次实际获取的；发布时间用于重建赛前可用性，无法排除后来修订。真实抓取时间和原始哈希单独保存，没有回填成历史抓取。

| 环节 | 独立比赛 | 计算记录 |
| --- | ---: | ---: |
| A：拟合角色文本预测器 f | 15 | 255个可见状态 |
| B：同一个冻结f生成增量并拟合g | 5 | 46个状态—动作对，20正、26负 |
| C：保留的回顾性机制检查 | 6 | 120个增量，严格规则80数值、40未知 |

每场获得相同总训练权重，并归一化使所有样本权重合计1。状态枚举不会把同场变成更多独立比赛。f复用角色分块字符TF-IDF与正则化softmax；g候选包括状态/候选交互ridge与常数模型，仅按B内整场留一误差选择。每折词表与IDF也仅由训练比赛建立。

预先声明的岭强度为20、10、1、0.1、0.01、0.001、0.0001，另含常数基线。常数验证MSE为1.018935e-6，岭20为1.021115e-6，其他更弱正则化也没有胜出。因此v4真实保存常数均值−0.0002657493；不能把它称为学会了条件排序。C与现场结果不参与选模。每折结果在model.json的 `metadata.g_selection` 中。

    delta = log f(y | H + 实际正文) - log f(y | H)
    估计净收益 = g(H, a) - 0.001 × 请求数

初始分布来自A的比赛结果频率及每类0.5平滑。已有结构化模型独立保留作比较，没有作为未经该文本训练支持的先验偷偷注入。

历史正文实际能改变f的词法表示，但不等于已学会其足球作用。旧v3中24组条件估值小幅变化，与常数基线的查询/停止判断完全相同；真正可比较的4对C排序没有反转。B只有5场（2主胜、3平、0客胜），其中4场提供同状态候选比较，只有6个不同比赛×文章对。C按比赛等权的实际增量均值为−0.000305443；这是枚举返回诊断，不是策略收益，没有效果优势结论。

## 严格模式和研究泛化模式

严格模式保留全部原检查。对主案例首页24个文章候选，它全部返回UNKNOWN：完整已见状态的角色组合没有足够匹配历史。

研究模式明确启用 `state_generalization=True`：保持工具/来源、先验域、候选角色、信息存在标志、余弦至少0.2及至少两个独立比赛的要求，仅将整个状态角色组合完全一致改为外推诊断。模型权重、成本及原严格模式不变。

这类数值标为 `prototype_state_generalization`，只描述支持规则。v4的21个数值候选仍使用同一常数均值，不能把状态支持门说成条件收益学习。有限邻域不是可靠校准或精确Bayes EVSI。未知来源、候选角色或不足的独立邻域仍返回UNKNOWN。

## 实际发生的过程

预算在查询前固定：总请求最多6次，其中未知信息探索最多3次；请求的损失折算成本为0.001。这是研究目标函数参数，不是API价格。

1. 读取Leeds官网入口：取得真实目录，结构投影排除导航、类别和条款，保留新闻标题，动态生成文章候选。
2. 当前21个候选能通过研究支持门，使用常数均值−0.0002657493；扣除成本后为−0.0012657493，不能进入正收益读取。其他值未知。
3. 未知探索实际读取 `from-west-yorkshire-to-the-world-stage-leeds-stars-on-international-duty`。v7绑定当前Gallery的实际日期、标题和导语，1段导语进入同一f；4条图片图注不入模，也不推断本场首发/健康/轮换。
4. 重新估值后，继续有限探索搜索本场预览，实际失败，保留失败记录。
5. 未知探索额度用完，终态defer；总请求预算未用完，没有信息充分性证书。实际次序由该运行的探索种子和配置生成，未手写。

初始文本模型分布为主胜57.5758%、平局21.2121%、客胜21.2121%；最终为57.7377%、21.1176%、21.1447%。变化不代表准确度提高。本次实际3次HTTP、2个成功页面、1次失败，全部动作仍为有限探索。终态有1段导语、72个不同标题输入，19个链接标签被投影排除；数值估值是常数基线，未发生学得正收益驱动的读取。

不能声称节省21次请求，也不能把支持门改变探索集合当作条件收益模型有用的证据。

## 同一状态下的动作对照

`live-03/support-rule-ablation.json` 固定同一公共状态、f/g权重、成本、预算及随机种子，只比较严格规则和研究泛化规则。旧 `live-02/support-rule-ablation-v2.json` 保留。

支持门对照仅回答候选可估范围如何影响探索，不回答学习内容是否有决策增益。对照分支只复算下一动作，没有获取或捏造其后续返回、概率或收益。条件选择要在同一个已知状态、共同候选菜单上另与零排序/常数基线比较，不能以候选集合不同代替此项。

同一真实轨迹的原始响应重投影和完整决策回放已通过，3份返回、4个周期完全一致，回放网络请求为0。

## 运行命令

在项目根目录运行。每次使用新的输出目录，以保留已经完成的研究记录。

训练相同方案：

```powershell
python tools/train_footballmind_single_case.py --cases var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json --dataset-version leeds-real-open-menu-v4 --g-fit-scheme b_group_loo_v1 --text-projection observed-news-titles-v1 --output var/submission/reproduce-training
```

实际联网执行：

```powershell
python tools/run_footballmind_single_case.py --model var/submission/p2-historical-context/real-open-menu-v4/model.json --output var/submission/reproduce-live --state-generalization --projection-version 7
```

本主案例的联网信息截止为2026年10月9日11:30 UTC，即北京时间10月9日19:30。超过截止时不把后来获取的信息伪装成原赛前证据；10月20—22日人工体验可使用下列原档回放。

零网络回放：

```powershell
python -m football_agent.learned_voi_replay var/submission/p3-single-match/live-03 --output var/submission/reproduce-replay
```

同状态、同种子动作对照：

```powershell
python tools/check_footballmind_single_case.py --run var/submission/p3-single-match/live-03 --output var/submission/reproduce-choice-check.json
```

直接阅读 `var/submission/p3-single-match/live-03/report.md`；完整数值轨迹为 `run.json`。报告最前面明确常数模型和条件学习未完成，逐步显示网页返回、正文抽取与实际入模，不能仅用概率变化判断正文是否被消费。

## 验证与未完成事项

同状态相对选择实验已经实际运行：`var/submission/p4-review/conditional-choice/`。它固定同一f，在B4个有比较的比赛上按整场留一拟合菜单居中收益，C6场仅作诊断。选中岭0.01，候选配对正确率0.60317（零基线0.5）；但同一已发现菜单、共同候选、读取其他正文后的12个关系中，真实排序反转2次，模型捕捉0次。该相对模型不能授权读取或停止，未接入live。

复现这一实验（无需联网、主f/g不重训）：

```powershell
python -m football_research.acquisition.conditional_choice_diagnostic --model var/submission/p2-historical-context/real-open-menu-v4/model.json --cases var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json --output var/submission/reproduce-conditional-choice
```

模型/标题投影/策略30项、报告32项、解析135项针对性回归通过。旧v6原档仍精确一致；新v7实际导语段落已进入可见状态。未读正文/未发现标题隔离、原始来源/赛果绑定、独立比赛选模、C不参与B选模与零网络回放均已验证。测试数量不是研究效果证据。

工程循环已修复，核心条件选择能力仍未通过。接下来只优先研究同状态候选相对收益与独立的绝对停止值，以及f是否具备有效的正文条件信号；不开展来源扩容或新界面。当前内容可供审查实际缺陷，不能称作最终人工验收版本，也不以常数数值可计算降低原核心要求。
