**跨范围证据与后续信息选择。** `applicability-inquiry-v1` 接在已有真实调查流程上，将“有一条背景材料”与“这条材料可作为本场推理的输入”区分开。它复用已归档正文，保留原始观察时间，并让 Agent 在取得计算后的缺口状态之后再选择外部查询。

实现为 [inquiry_applicability.py](../../football_agent/inquiry_applicability.py) 和 [inquiry_memory.py](../../football_agent/inquiry_memory.py)。旧的 `hypothesis-inquiry-v1` 保持原有匹配与回放语义；新协议使用独立版本。真实任务配置保存在 [applicability-inquiry-task.json](applicability-inquiry-task.json)，其中比赛身份和截止时间固定，复用配置时须先核验任务身份及真实时间。

此前 `hypothesis_states` 只检查来源是否直接支持同一主体、范围和范围标识下的命题。这能避免把集训名单当成首发，却也会将赛事级规则一直留在背景中。新协议保留这一直接来源状态，另用 `requirements` 表示推理前提的覆盖情况：

| 输入情况 | 直接命题状态 | 前提状态与下一步 |
|---|---|---|
| 只有赛事级晋级规则 | 资格结论仍未知 | 先核实该规则是否适用于本场 |
| 规则与本场之间有带引文的适用关系 | 资格结论仍未知 | 规则前提已有；积分等其余输入仍可缺失 |
| 积分数据存在，但缺少其有效时间 | 资格结论仍未知 | 标为 `time_unresolved`，获取日期不替代有效时间 |
| 规则、关系和具备时间锚点的积分均存在 | 输入仍不直接等于结论 | 标为 `inference_required`，后续仍要进行资格推理 |

`context_bindings` 指向一条已锚定的来源表述和一个具体比赛假设，并保存支持适用关系的原文引文。当前只开放资格假设中的规则和积分关系：规则通过 `competition_rule_applies_to_match` 关联，积分通过 `standings_describe_match_context` 关联。声明主体须相同，来源保留其赛事范围。教练总体目标、集训名单不会经这条关系直接变成本场轮换或战术结论。

范围关系、证据类型、引文含义与积分有效时间仍由 LLM 抽取。程序检查类型、引用、已获取性、主体、时间顺序及不可变标识；它不独立证明原文语义，也不保证来源最新。`semantic_verification` 与 `freshness_verification` 均保留未核验状态。某份规则属于哪一届、哪一组的判断仍需检查引文的真实含义，而不只看结构合法。

候选查询的结构反馈包含：

- `potential_missing_fills`：若返回符合所声明类型、范围和时间，可补哪些当前缺口。
- `already_present_types`：哪些预期输入已经具备，Agent 应说明进一步查询用于刷新还是核实。
- `requires_applicability_link`：哪些背景仍需证明适用于当前任务。
- `could_complete_inputs`：该返回在条件成立时是否可能补齐所有声明前提。

这些字段属于集合与依赖关系计算，不是概率或价值排序。补齐更多字段并不保证风险下降更多；已具备的输入仍可能需要时效更新或独立核实。`selection_score` 和 `expected_gain` 保持为空，状态为 `OUTCOME_AND_RISK_MODEL_REQUIRED`。

`reflect` 是新增的运行步骤。Agent 可以先提交引用、假设和关系，运行时计算 `requirements`，然后下一次模型调用读取这一状态再选择查询。它消耗一次模型调用，外部 HTTP 为零；没有新表述、假设或关系时，重复反思会被拒绝。它不代表无成本推理，也没有证明反思步骤的经济收益。用户任务可要求先整理继承材料，再进行有预算的自主获取。

证据继承通过配置的 `inherit_evidence` 完成，包括父运行位置、运行 ID、固定事件头哈希和明确的文档 ID。系统先离线复验父运行及其原始归档，仅继承成功页面，并记录本次导入时间。原始观察和可用时间保留；未来证据、未完成运行、修改过的父记录、重复导入和循环继承均会被拒绝。继承不会复制父运行的假设结论，也不会将之前支付的获取成本改记为本轮新请求。

```powershell
python -m football_agent.inquiry_cli --config NEW_TASK.json --root NEW_RUN_DIRECTORY
python -m football_agent.inquiry_verify --root EXISTING_RUN_DIRECTORY --run-id RUN_ID --output VERIFICATION.json
```

父运行及其原始归档需要一并保留，核验器会递归检查固定来源。导入不刷新来源；若新任务需要更近时点的信息，Agent 仍需选择明确的刷新动作。当前继承入口只在新运行第一次推理前开放，原运行不会被回填。

研究上的下一条连接是条件价值：同一规则文档在尚未识别赛事时、规则已知但积分缺失时、发生规则冲突时，作用会不同。现有历史条件价值估计器绑定历史状态/动作及其训练支持，有限分布规划器也绑定已知的返回分布。新文本查询尚无这种支持，因此本协议只提供可审查的前提状态和候选返回作用，未将历史数值直接套用到开放文本。

后续应以有版本和可核验语义的返回分支连接风险更新，再研究这些依赖关系能否帮助估计条件价值、识别信息互补性和决定停止。当前“输入齐全”仅为进入推理的条件；完整资格判断可能还需要剩余赛程、同分规则等。LLM 声明的最小输入清单不保证对任何资格命题都充分。
