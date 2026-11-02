# 受限 LLM 候选提案

`football_agent.proposer.LLMProposer` 仅负责从当前合法目录提出候选动作。它不访问归档、FeatureIndex、完整快照或赛果，不执行查询，也不输出数值信息价值。数值价值由研究估计器独立计算。纯 LLM 对照可以消费候选及显式停止；混合策略再经过数值门控。

```python
from football_agent.proposer import LLMProposer

proposer = LLMProposer.from_env(
    ledger_path="var/parallel/agent/round3/llm/calls.json",
)
result = proposer.propose(
    visible_state, executable_action_catalog,
    request_id="episode-id:decision-0",  # 每次决策稳定，恢复沿用
    remaining_calls=5,                  # 父运行统一预算剩余额度
)
```

`configuration()` 仅含 provider、model、公开 endpoint、prompt/version、输出上限、请求上限、超时及 configured 布尔值；凭据与 ledger 路径不进入 prompt。输入状态严格限制为公开字段，公开组必须带已揭示 mask。嵌套目标标签、隐藏组和归档路径等输入拒绝。调用方仍须从平台 `VisibleState` 构建公开状态，这一检查不是把完整档案自动脱敏的接口。

动作按 `action_id` 精确匹配、排除当前不可执行目录项。原生平台 `capability_status`、`expected_cost` 和 `OBSERVED` 等 mask 状态已接通，`NOT_IMPLEMENTED/NEEDS_CONFIGURATION` 不作为合法候选。`parameters_schema`（兼容 `input_schema`）使用 JSON Schema 2020-12 验证；未知顶层参数拒绝，远程 schema 引用拒绝。平台当前空参数动作使用内建严格空字典校验；带参数 schema 在默认 Python 3.13 使用已有 `jsonschema`，平台 Python 3.11 venv 缺该依赖时返回显式错误，未修改共享依赖清单。

响应只接受 `proposals`、`stop`、`reason`。每个 proposal 必须且只含 `action_id`、`parameters`、`reason`，目录外、重复动作和额外 `expected_gain/net_gain` 字段拒绝。理由作为提案依据展示，`reason_is_value_estimate=false`。文字中模型自行声称的价值不参与数值门控。

默认最多 5 次，硬上限亦为 5 次；每次默认 512 输出 tokens，显式配置上限 2048。标准请求使用 `max_completion_tokens` 与 JSON mode；返回仍需本地完整验证，包括终止原因。OpenAI 官方文档明确 JSON mode 只保证 JSON 格式，并不保证 schema，因此本模块不把成功解析等同动作合法。[结构化输出官方文档](https://developers.openai.com/api/docs/guides/structured-outputs)

配置优先级如下，各值均为服务器环境设置，HTTP 客户端不传 provider 配置或凭据：

| 设置 | 首选 | 兼容既有名称 |
|---|---|---|
| 模型 | `FOOTBALL_LLM_MODEL` | `OPENAI_MODEL` |
| 凭据 | `FOOTBALL_LLM_API_KEY` | `OPENAI_API_KEY` |
| endpoint | `FOOTBALL_LLM_BASE_URL` | `OPENAI_BASE_URL`，默认 OpenAI v1 |
| provider 标识 | `FOOTBALL_LLM_PROVIDER` | 默认 `openai-compatible` |
| 请求上限 | `FOOTBALL_LLM_MAX_CALLS` | 默认 5 |
| 单次输出上限 | `FOOTBALL_LLM_MAX_OUTPUT_TOKENS` | 默认 512 |

模块只读取进程环境，不读取桌面账号令牌或扫描用户凭据文件。项目当前没有既有 LLM 配置模块或 `.env`；使用上述兼容变量接入。非 OpenAI provider 须支持 Chat Completions 请求字段；没有静默模型替换或自动兼容重试。

返回 `status/proposals/stop/reason`、provider/model/prompt version、provider request ID、响应 model、实际请求数与时延。usage 用 `unit='tokens'` 和 `input_count/output_count/total_count`，避免旧审计字段脱敏器把 token 计数字段当凭据删除。费用采用 `actual_fee={status:'unknown',amount:null,currency:null}`；缺配置或未调用为 `not_incurred`。输出额度是预留上限，不是实际 tokens 或货币费用。

每次网络调用前以原子写预留 ledger 项，复用 Agent 跨进程锁。相同 request ID 与输入恢复返回已有结果，`replayed=true,requests=0,original_requests=1`；原 usage/fee 仅供历史展示，父预算应跳过这次重放的累计。崩溃导致预留但无结果时返回 `INTERRUPTED_OUTCOME_UNKNOWN`，保留预算且不自动重试。HTTP 429、超时或错误同样消耗一次请求，actual fee 未知；不自动追加请求。父运行还须在调用前预留统一 LLM 预算、在配置固定后恢复。

2026-09-22 验收：实际环境模型/凭据未配置，真实 provider 调用 0；独立 proposer 回归 32 项通过。`var/parallel/agent/round3/llm/validation.json` 分别记录构造测试与未配置检查；构造 provider 回包不标记为真实 LLM 成功。运行 `python -m pytest tests/agent/test_round3_proposer.py -q` 验证边界与恢复。端到端运行和浏览器验收由研究运行入口记录。
