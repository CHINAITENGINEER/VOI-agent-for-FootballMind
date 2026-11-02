# 同状态候选信息路径比较

`information-choice-inquiry-v1` 扩展 `evidence-state-inquiry-v1`，沿用问题、假设、来源陈述、适用关系和七类返回计划。新增 `choice_review` 记录 Agent 在获取之前怎样比较自己的候选。它没有替代数值 selector，也没有将语言解释标为已校准 VoI。

## 运行契约

通过 `inquiry_cli` 配置 `inquiry_protocol: information-choice-inquiry-v1`，同时配置现有 `llm-continuation-budget-v1`。可选 `source_catalog` 是至多12个静态入口的列表，每项包含：

```json
{
  "source_id": "league_reference_entry",
  "url": "https://example.com/reference",
  "publisher": "Source publisher",
  "discovery_topics": ["competition_rules"],
  "limitations": "This entry may only discover a season-specific document."
}
```

入口 URL 必须属于既有 allowed_hosts 并在 task.seed_urls 中声明；目录本身不拓展工具权限。目录说明属于策展信息，未取得页面前不进入 claims。来源主体、适用性、可读性和本场真值仍需分别核实。

真实配置例见 [第27轮调查配置](../../var/parallel/agent/round27/information-choice-v1/config.json)。历史配置含当时截止时间，复用时应创建新任务和新目录，保留旧记录。

## 自主提出并比较

每次回复至多两个候选，零个或一个均合法。每个候选沿用原 `inquiry_intent` 和 `return_model`，不复制目标或补造返回分布。

新增 `choice_review`：

- `alternatives`：每个 candidate_id 恰好一项；说明 `information_advantage`、`source_limitations`，以及实际参与比较的已取得文档 `evidence_document_ids`。
- `not_selected_reason`：原提议选择的候选必须为 null，其余候选写出本次暂不优先的理由。终止、等待、暂缓或反思时，所有提出的查询都要说明未选原因。
- `tradeoff`：说明当前选择相对于其他路径的取舍。
- `missing_routes`：说明当前目录或发现能力还覆盖不到什么，不把有限候选视为整个开放网络。

文档 ID 只允许来自当前已取得材料。失败记录可以支持操作性的路径调整，不能作为足球命题的否定证据。目录、模型记忆、未获取页面和未来返回均不作为已有来源陈述。

## 程序计算与实际执行

程序对每个候选执行原工具的 prepare，使用 prepared action 的 digest 区分物理查询；换 candidate_id、仅改变 URL fragment 或参数顺序不产生新的物理路径。入口匹配也使用相同身份计算，保持执行、脱敏保存和回放一致。

每个候选的成本下界从现有 MINIMUMS 派生，统一使用当前回复已经计费后的余量：

| 工具 | 来源请求下界，含当前获取 | 当前提议之后的模型调用下界 |
|---|---:|---:|
| read_url | 1 | 1 |
| search_web | 2 | 2 |

这些是既定工具流程的计数下界。读取资料目录或新闻列表后，可能还要发现、读取具体文件；若返回计划写 reassess，也未给出完整交付路径。因此 `expected_total_cost` 保持 null，不将下界叫作完整任务成本。最低计数足够也不保证返回成功、证据适用或查询有正价值。

投影保存每条路径的执行身份、目录入口身份、此前同动作观察、最低开销和 `within_count_budget`。重复请求、截止与最终执行准入仍由现有工作流处理；计数下界通过并不单独等于完整准入。

当前轮所有候选在比较时都标为 `UNOBSERVED_AT_CHOICE`。所选候选随后是否实际执行，以 acquisition intent 和 receipt 为准；程序守卫覆盖原提议时保留原提议。未选候选没有当前返回标签，历史同动作记录单独列出。选择概率、预期收益和总成本未知，均为 null。

## 验证与研究口径

同状态比较由同一公开输入、同一模型回复和获取前决定记录绑定。离线重放重新验证目录、候选及引文约束，重新计算路径计数和投影，并核对已归档决策。旧协议配置和旧投影不变。

真实验收应分别记录：是否提出不同候选、是否同时满足计数下界及其他执行限制、实际选择并获取哪一条、后续是否处理真实返回。若只有一条能执行，就不是两条可行路径之间的选择。两个不同动作也不保证其信息内容独立。

这些记录支持研究候选生成与选择行为，不单凭比较解释证明最优性。未选择的来源没有反事实收益；后续不同状态下补查同一来源，仍不能倒填为此前状态下的已观察返回。

实现：[inquiry_choice.py](../../football_agent/inquiry_choice.py)。行为验证：[test_inquiry_choice.py](../../tests/agent/test_inquiry_choice.py)。
