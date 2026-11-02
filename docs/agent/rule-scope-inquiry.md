# 来源声明的规则适用关系

`rule-scope-inquiry-v1` 是新的可选协议，继承候选信息路径比较、返回预期及后续处理预算。它允许 Agent 在保留联赛规则原主体的情况下，用已取得证据提出“这份规则可用于这个球队的这场比赛”的关联，再根据剩余未知选择下一步行动。

该协议补充来源信息的结构表达，未配置数值 VoI、独立实体消歧或足球预测。资格和轮换仍需各自的证据与推理。

## 使用与边界

配置 `inquiry_protocol: rule-scope-inquiry-v1`，继续使用已有的 continuation_policy 和可选 source_catalog。新回复在旧字段之外增加 `rule_scope_bindings` 数组；没有足够见证时可为空。旧 context_bindings、原规则主体、原协议版本和旧失败轨迹均保持原语义。

每条关联含以下端点：

| 字段 | 约束 |
|---|---|
| binding_id | 不可变且不得与旧 context_bindings 的 ID 冲突 |
| claim_id | 来源陈述，类型 competition_rules、campaign 范围 |
| fixture_claim_id | 单独的来源陈述，类型 other、match 范围，主体与比赛键等于目标假设 |
| hypothesis_id | qualification 类型、match 范围，仍需 competition_rules 和 standings |
| competition_id / campaign_key | 精确等于规则陈述原 subject / scope_key |
| team_id / fixture_id | 精确等于目标假设的 subject / scope_key |
| season | 完整四位年份或年度范围，如 2026、2026-27、2026/2027 |

规则陈述和比赛背景陈述针对该资格假设的关系均须为 context。新关系不把通用规则转换成资格已确定的支持证据，更不允许用规则关联替代教练表态。

## 四个角色的见证

每个角色有 document_id、原文 quote 及下列实体表面名。quote 最长1200字符，必须在当前取得的可用页面中；原 claim 自己的引文仍遵守原有500字符上限。

| 角色 | 要出现在自己引文中的表面名 | 来源约束 |
|---|---|---|
| rule_edition | competition_mention、season_mention | 规则陈述引用的文档；还须与该陈述的赛事定义引文直接包含或被包含 |
| team_in_edition | competition_mention、season_mention、team_mention | 三者在同一引文中，来自规则陈述引用的文档 |
| fixture_in_edition | competition_mention、season_mention | 比赛背景陈述的引文，直接包含或被包含 |
| team_in_fixture | team_mention | 比赛背景陈述的引文，直接包含或被包含 |

同一页的另一场比赛不能仅因与当前比赛共用文档，就补作 fixture 角色见证。同一句真实比赛声明可以承担后两个不同角色，但不是两份独立佐证。

跨角色的 competition/team 表面名经 NFKC、大小写及空白归一化后须相同，每个表面名同时须在其自己的引文中逐字出现。season_mention 须等于声明的完整 season；数字边界检查防止把 2026 作为 20260 或 2026-27 的年份前缀，也防止 League 1 匹配 League 12。未添加跨语言模糊别名匹配；合法但不同的别名暂保留关联缺口。

## 状态与查询流程

新关系独立保存在 rule_scope_bindings。投影暂时将通过校验的关系交给既有输入覆盖计算，之后恢复原 context_bindings。规则和比赛 claim 的主体、范围、正文、引文不会改写。

requirements 中新增路径 `via=witnessed_rule_scope`。规则可以成为 `present` 的**来源声明输入**，仍带有身份、语义、章节归属未核验标记。standings 的有效时间检查保留；资格命题的支持/反向状态由原来源陈述推导，不因新关系改变。即使全部输入取得，也仍需推理，不能从覆盖率推出资格成立。

Agent 可在同一回复形成关系并提议下一次查询，也可先 reflect 检查计算后的缺口。仅新增关系也构成一次合法反思；重复相同关系且没有其他新状态不允许继续反思。反思花费一次模型调用，不发 HTTP。此协议没有规定规则、积分、教练表态的固定查询次序，也没有用填空数量作为效用。

后续查询合同中的 `rule_scope_before` 冻结该次选择前针对目标的关联。离线 verifier 从原回复和实际正文重放它们，并保持知识前缀约束。继承旧正文时仅带入原始资料与观察时间，不继承旧回复中被拒绝的 claim、关系或返回标签。

## 校验的实际含义

程序确认字段端点、不可变性、引用可追溯性及词面一致性。canonical ID 对应哪支现实球队、新闻相对日期对应哪场比赛、某段规则究竟属于哪个章节、来源当前是否有效，仍含未核验的语义判断。

尤其是同页混引：Agent 若给一个 claim 引用 K1 标题及另一赛事的操作条款，形式关联仍可能通过。现有 edition 引文锚定减少无关见证，尚非规则章节解析器。K1 条文本身也可能合法提及 K2；全文出现另一赛事名称不能直接作为拒绝条件。实际报告须保留这些边界，不能称程序已经认证适用关系。

本轮实现与结果见 [第29轮报告](../parallel/round29/README.md)，设计边界见 [设计审阅](../parallel/round29/rule-scope-design-review.md)。下一项研究仍是这些来源输入如何改变可行动结论和后续信息获取，而非提高结构检查通过率。
