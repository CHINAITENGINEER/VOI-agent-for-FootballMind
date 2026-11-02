# 固定终端问题与信息的实际影响

`decision-impact-inquiry-v1` 在已有信息选择、来源关系、返回预期与续读预算之上，增加一个固定的来源报告问题。Agent 仍自主提出和选择查询；程序把查询前的报告状态与实际返回后的报告状态对应起来。

本接口回答“所取得来源对指定命题表达了什么立场”。它不把来源声明当作实际轮换或首发真值，也没有建立开放查询的收益分布。

## 任务配置

CLI 配置增加 `inquiry_protocol: decision-impact-inquiry-v1` 和 `decision_problem`。后者包括：

| 字段 | 含义与约束 |
|---|---|
| task_id | 与调查任务完全一致 |
| decision_id | 固定报告决策标识 |
| target | 完整 hypothesis：ID、question_id、原始 statement、类型、主体、范围、范围键及所需证据 |
| semantics | 固定为 source_position_not_world_truth |
| state_definitions | positive 与 negative 两类声明状态的解释 |
| loss | report_supported、report_challenged、withhold 对两类声明状态的非负有限损失 |
| loss_basis | 说明任务偏好及其来源，避免冒充测量费用或经验校准 |

当前固定端点只支持 `rotation_intent`、`tactical_intent`、`starting_lineup`，所需证据须恰好为该类型的既有最低集合。资格、实际轮换及需要复合时点推理的命题可以继续作为辅助 hypothesis，但不进入这一窄来源报告端点，避免绕过其有效时间和推理要求。

第一份有效回复需包含与 target **完整相同**的 hypothesis，并提供对应问题。随后保留这一端点。只匹配球队、比赛、类型不足以区分“计划轮换”与“计划不轮换”，因此 statement 和 ID 同样固定。Agent 可以自行增加辅助问题。

## 报告立场

程序对已接受的 source_claim、精确目标关系以及类型和作用范围做原有投影，再得到：

- 直接支持证据覆盖所需类型，且没有直接反对：`report_supported`。
- 直接反对证据覆盖所需类型，且没有直接支持：`report_challenged`。
- 同时有两种方向，或证据不足：`withhold`。

`withhold` 表示保留判断，不是反向结论。一个新请求失败不会删除既有支持；一条资格规则也不会成为本场教练意图。调查是否 query/stop/defer/wait，与当前报告属于哪个动作分别保存。程序没有从报告方向自动推出应结束调查。

上述结果仍依赖 LLM 对原文含义、身份和范围的提取。原文可追溯不代表已经独立验证语义、时效或事实真伪。

## 查询前的条件说明

每条候选增加 `terminal_effect`：

| 字段 | 含义 |
|---|---|
| route | direct_evidence / discovery / indirect / unknown |
| could_change_to | 例示可能到达的报告动作，可为空 |
| condition | 哪种具体来源内容会支持这种变化 |
| further_information_needed | 还需哪些阅读、核对或其他信息 |

直接证据必须是针对精确终端命题、具有适用类型的 `read_url` 提议。搜索属于发现，搜索摘要不会直接成为命题事实。上述条件不是穷尽返回策略，没有强制自动执行条件中的答案。

空动作列表或间接路线不会被赋零价值，也不会被程序过滤。规则、积分或身份核对可能与后续材料互补；页面还可能含有事前未预料的直接声明。Agent 应按终端问题比较路线，而非遵循“先补完所有背景再看直接证据”的固定次序。

## 数学上界及边界

给定声明损失 L、当前报告动作 d0，程序计算 `max_y [L(d0,y) - min_d L(d,y)]`。因为任何最终策略在状态 y 下的损失都不低于该状态的最小动作损失，任意返回分布下的期望改善都不超过这个最大值。

这是全动作的最大可能改善空间，**不是 EVSI、当前风险、置信度或候选排序分数**。它不使用自然语言列举的动作子集，不会因 Agent 漏列一种答案而缩成零。所有候选仍保留期望收益、货币成本和损失单位获取成本未知，未将操作返回概率乘以上界当作信息价值。

例如错误报告损失1、保留未知损失0.2时，当前 withhold 的上界是0.2；当前 report_supported 的上界是1。这种变化不说明取得证据后风险上升。两条查询也可能上界相同但实际收益一正一负。

该数值仅在任务声明的 positive/negative 两类损失状态内成立。无来源、冲突、时效错误或误导性材料并未自动成为完整模型中的新增状态；它不提供全局开放世界停止证书。

## 真正返回后的对应

每次查询前，先合并并检查本通提取的 claims，再把当前 `terminal_state`、固定问题哈希和候选条件写入查询合同。这避免将已经掌握的信息误记为新查询贡献。

下一次有效提取后，记录 `decision_impact_reviews`：

- 查询前后报告动作是否改变。
- 新提取且只引用本次文档的目标方向性 claim。
- 来自旧文档或混合引用的其他新增目标 claim。
- 变化与本次材料并存、属于联合/其他证据变化，或尚未归因。

同一文档可对辅助问题提供支持，却不改变固定终端问题，因此不把总体 source_support 标签当成终端变化。计划中列出的动作是否出现也只是描述性核对；条件含义和最终报告正确性仍未独立验证。

这些记录不给未选项补返回，不把答案改变当成风险降低，不将同一调查的多个步骤视为独立比赛样本。`terminal_state`、派生上界和影响记录均纳入现有投影回放，不新增另一套审计体系。

代码：[inquiry_impact.py](../../football_agent/inquiry_impact.py)。数学设计及反例：[decision-relevance-design.md](../parallel/round30/decision-relevance-design.md)。本轮实际结果：[第30轮报告](../parallel/round30/README.md)。
