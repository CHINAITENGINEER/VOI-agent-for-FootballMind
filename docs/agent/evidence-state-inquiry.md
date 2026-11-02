**`evidence-state-inquiry-v1`：区分输入是否已有、来源表述方向和相反表述。**

新任务在配置中设置 `"inquiry_protocol": "evidence-state-inquiry-v1"`。它继承返回分支、证据继承、反思及执行接口，改用新的前提投影与候选说明。历史 `applicability-inquiry-v1` 和 `return-forecast-inquiry-v1` 继续使用原投影，旧轨迹按各自版本重放。

核心实现为 `football_agent/inquiry_evidence_state.py`。基础协议仅增加两个可覆盖的方法入口：`project_requirements` 与 `assess_requirements`，避免复制执行器或更改历史语义。

| 已提取的适用证据 | 输入状态 | 命题方向 | 相反表述状态 |
|---|---|---|---|
| 没有对应证据，或只有其他范围/类型的背景 | 缺失 | 未确定 | 未观察到 |
| 单向正向表述 | 已有 | 支持 | 未观察到 |
| 单向反向表述 | 已有 | 反对 | 未观察到 |
| 正向与反向表述并存 | 已有 | 混合 | 需要核对相反表述 |

“已有”针对声明的输入类型，不代表来源内容完整、语义正确或结论已经验证。单条反向证据不会自动变成冲突；例如明确说明球员没有首发的适用来源表述，也能提供首发输入。新闻页没有提及某球员、请求失败和空搜索结果仍不构成反向答案。

`requirements` 保留原有 `present_types / missing_types / prerequisites_complete` 接口，并明确以下语义：

- `missing_types` 只包含尚无适用输入或有效时间未解决的类型；相反表述本身不再使已取得的输入消失。
- `proposition_direction` 为 `supports / challenges / mixed / undetermined`，由适用的直接 `source_claim` 表述投影。
- `conflict_status` 的 `opposed_source_claims` 表示观察到相反方向，需要核对；它不独立证明两条来源在逻辑上矛盾，也不证明来源相互独立。
- `contested_types` 单独记录同一输入类型的相反直接表述。不同输入类型可能使整体命题出现相反信号，此时整体状态为混合，单个类型未必标记冲突。
- `source_answer_status` 区分输入不全、来源支持、来源反对、相反表述及仍需推理。只有输入齐备且同一方向直接覆盖全部声明类型时，才给出单向来源答案状态；上下文补齐输入本身不建立直接答案。
- `query_focus` 提示补输入、核对相反表述、复用来源答案或继续推理。这是候选解释，查询顺序仍由 Agent 决定。

来源表述仍需引文锚点、匹配主体与范围。赛事级规则及积分继续通过适用关系进入本场前提；积分有效时间未解决时，正向和反向证据都保留时间缺口。适用关系可以提供推理前提，但不会将赛事背景升级为对本场命题的直接支持或反对。

已有 `hypothesis_states.missing_support_types` 描述缺少哪些**支持**，不能当作缺少哪些**信息**。新协议在公开状态中提供 `evidence_state_semantics` 并向模型解释差别；其他消费方应使用新 `requirements` 判断输入可得性。

候选评估继续报告 `potential_missing_fills` 与 `already_present_types`，并带上来源答案、方向与核对需求。负向答案已提供对应输入后，再查同一类型不再被列为填补缺口；时效、可信度或交叉核实仍可能使刷新有意义。程序没有新增“输入齐了就禁止查询”的规则，没有将完整度转为收益分数。

返回模型的 `missing_after` 也使用新语义：适用负向答案可以填补输入，相反答案并存可以使输入齐备但仍需核对。七种返回类别内仍有范围、完整度和时间差异，分支预测并不因此成为精确状态转移模型。

**本轮诊断入口。**

```powershell
python -m football_agent.probe_evidence_state --root NEW_PROBE_DIRECTORY
python -m football_agent.probe_evidence_state --root EXISTING_PROBE_DIRECTORY --verify-only
```

第一条命令先固定五个构造输入与源码，然后通过现有 CLI 提供器执行五次真实模型调用，最多两次并行、无自动重试。四个新协议场景为正向、反向、相反表述并存和只有背景；另保留一个旧协议反向场景。固定输入中有两个同等相关的人员问题，另一个伤停问题始终缺信息。

该诊断检查真实模型提出的下一行动，校验参数但不执行 HTTP。初始文档和提取表述均为明确构造资料，未测试开放网页提取质量；成功执行命令也不代表查询策略优于其他策略。旧/新反向配对同时改变协议提示与状态投影，且每种只调用一次，适合发现问题，不构成因果效果估计。

第二条命令只重验固定输入、状态投影、响应与提议，不再次调用模型或网页。既有比赛的真实轨迹仍使用 `football_agent.inquiry_verify` 核验。第二十轮的结果、未改善项及证据路径见 `docs/parallel/round20/README.md`。
