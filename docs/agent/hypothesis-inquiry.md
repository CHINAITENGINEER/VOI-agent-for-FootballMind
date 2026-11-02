**假设驱动的开放信息调查协议。** `hypothesis-inquiry-v1` 在既有真实调查入口增加三个对象：待核实假设、查询前的答案分支、取得文档后的证据反馈。问题、假设和查询顺序仍由 Agent 根据当时可见信息提出；运行时检查引用、证据类型和适用范围，保留推断与未知。

入口在 [inquiry_hypotheses.py](../../football_agent/inquiry_hypotheses.py)。未配置 `inquiry_protocol` 的任务继续使用原协议。新的任务配置示例为 [hypothesis-inquiry-task.json](hypothesis-inquiry-task.json)；其中日期及比赛身份是已发生的实际运行配置，新运行应使用核验后的真实任务，不能自动平移日期复用。

```powershell
python -m football_agent.inquiry_cli --config NEW_TASK.json --root NEW_RUN_DIRECTORY
python -m football_agent.inquiry_verify --root EXISTING_RUN_DIRECTORY --run-id RUN_ID --output VERIFICATION.json
```

每次循环的数据流是：

1. Agent 看到任务、已取得文档、既有假设状态和此前查询反馈；尚未取得的正文不会出现在输入中。
2. Agent 提出或保留假设，更新带引文的来源表述，并为候选查询填写目标假设、预期证据类型、范围及三种答案分支。
3. 运行时校验引用和结构，计算证据状态，保存本次选择。查询前将当时状态与查询目的冻结到请求意图中。
4. 执行真实工具，按实际尝试计数，归档正文与时间。下一次模型调用处理返回，重新选择后续动作。
5. 按实际返回文档与事前目标计算查询反馈。此前的反馈不会被之后的新证据改写。

`supports_change`、`challenges_change`、`no_answer_change` 要求 Agent 说明不同答案怎样改变下一步决策。它们是待检验的定性分支，不是三个分支的概率，也不是数值收益。尚未经过下一次模型处理的返回保持未评审；用完推理预算不会把它记成零价值。

| 对象 | 关键字段 | 用途 |
|---|---|---|
| 假设 | `hypothesis_id / statement / hypothesis_type` | 将具体待核实命题与问题绑定，定义创建后保持固定；新解释使用新 ID |
| 适用范围 | `subject / scope / scope_key` | 区分球队、赛事窗口和具体比赛；同样涉及一支球队不代表适用同场比赛 |
| 证据要求 | `required_evidence` | 对首发、实际轮换、轮换意图等保留保守的证据类型要求 |
| 来源表述 | `evidence / hypothesis_links / citations` | 记录来源表述对哪项假设提供支持、反向证据或背景 |
| 查询意图 | `inquiry_intent` | 在请求之前登记目标、预期证据及答案可能改变的下一步 |
| 查询反馈 | `query_reviews` | 将该请求的真实文档与当时目标对齐，保留背景作用、缺失和归因不确定性 |

例如“本场发生实际轮换”至少需要本场首发与相关首发对比。集训名单即使来自官方正文，也只得到背景状态；名单包含谁与谁本场首发是不同命题。仅有首发而缺少对比时保留部分支持。教练表态描述声明的意图，不证明实际战术或最终表现。这些类型下限是该版本的保守协议约束，不是通用的足球事实判定器。

来源表述的 `subject / scope / evidence.kind / relation` 都来自 LLM 提取。程序只确定性检查已声明字段的兼容性，不独立判断引文是否真正蕴含该语义。错误分类仍可能通过类型检查。因此 `complete_source_support` 只表示已提取的来源表述覆盖了声明类型；它不等于事实确认、来源独立、信息新鲜或概率校准。所有投影明确保留 `semantic_verification=unverified`、`source_freshness=unverified`。

相反表述不会用新结论覆盖旧表述。同一假设同时获得适用的支持与反向来源时记录 `mixed_source_signals`。来源的引用段落仍可查。跨多个文档综合形成的表述可进入整体证据状态，但单次请求的反馈会将其列为 `multi_document_attribution_unknown`，不把综合贡献全部归给最后一个请求。

`context_only` 是有意义的结果：它可以使 Agent 改问“是否有本场发布会消息”，而无需直接改变比分概率。`no_usable_extracted_evidence` 只表示对预先声明的目标没有可归因的有效提取；可能是信息缺失、抽取遗漏、范围不符或查询提出了新问题，并不证明信息无用。

这里为信息价值学习准备的是可核对的过程记录：查询前状态、动作、实际返回、查询后状态、后续动作与实际获取成本。当前没有文本到预测风险的可验证映射，也没有各种可能返回的估计分布，所以 `expected_gain / selection_score / counterfactual_gain` 保持为空。查询继续采用 `llm_bounded_exploration`，停止仍是语义判断或资源边界。

后续数值模型需要针对同一决策状态比较“直接决策的风险”与“获得信息后再决策的期望风险及获取成本”。对于背景信息，还要包括它改变后续查询的价值。某次最终赢球、某条来源支持了假设、甚至一次预测恰好变准，都不足以直接充当这种期望差的真值标签。应先建立可验证的返回分支及状态更新，再评估定性查询契约能否帮助估计这些分支，避免把提取数量或查询次数当成创新成绩。

该版本保留原有限工具概率规划器，未据此声称开放文本 VoI 已实现。真实续查及其范围见 [第十七轮报告](../parallel/round17/README.md)；构造干预测试使用同一任务的不同实际返回，检查下一步获取是否随证据改变，不把测试通过当作真实策略收益。
