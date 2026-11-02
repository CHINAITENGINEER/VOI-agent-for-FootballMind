# Agent 第四轮：LLM 候选提出与数值价值门控

更新时间：2026-09-22

本轮把 Agent 的职责拆成两个可审计阶段：LLM 或候选提案器只能从平台给出的 `ActionSpec` 目录提出合法 `action_id`、受限参数和文字理由；研究侧 `ValueEstimator` 独立计算预期损失改善、成本换算、净增益和不确定性，最终获取或停止只由数值门控决定。LLM 理由不会写入 `net_gain`，也不会覆盖保守风险规则。

## 已接通的入口

`football_agent.acquisition.AcquisitionWorkflow.start` 接受四个策略名称及兼容别名：

| 展示名称 | 存储值 | 行为 |
| --- | --- | --- |
| `fixed-order` | `fixed` | 固定顺序工程基线 |
| `numeric-all-candidates` | `all_candidates` | 对当前全部合法动作估值并选择 |
| `llm-proposal+numeric-gate` | `llm_gate` | LLM 提出子集，数值估计器审计遗漏并决定 |
| `pure-llm` | `pure_llm` | 仅在真实 provider 有模型和凭据时调用；缺配置为 `LLM_NEEDS_CONFIGURATION` |

`risk_mode=mean` 使用 `net_gain`；`risk_mode=conservative` 使用 `lower_gain - cost_in_loss_units`。每次 `value_estimated` 事件保存 `proposal_audit`（合法候选数、提案数、支持数、正价值数、正价值遗漏和召回率），以及每个候选的 `expected_loss_improvement`、`net_gain`、`conservative_net_gain`、`uncertainty`、`support`、估计器版本和成本分量。

失败、空结果、超时、限流、过期、冲突、重复和 `no_change` 只标记当前动作实例，并按 `action_id + parameters + evidence_version + refresh_window + cutoff` 形成幂等键；下一步会重新列出其它合法动作。相同版本进入退避，新的证据版本或截止点形成新的可执行实例。父级预算、LLM 调用预算、估计器计算预算、模拟费用和线上预留分开记账。

## LLM 边界和持久化

`football_agent.proposer.LLMProposer` 的 prompt 只包含 `VisibleState`、合法目录、当前预测、动作历史和预算。目录外动作、参数 schema 越权、远程 `$ref`、隐藏字段、目标标签和文件路径在请求前拒绝。provider 响应保存 `provider_response_sha256`、provider/model/prompt 版本、token 用量、延迟、请求 ID 和费用状态；原始响应正文不写入运行记录。缺少凭据时 provider 调用数和预留费用均为零。

## 已接通真实研究服务

本轮生成的配置是 `var/parallel/agent/round4/backend-v3-fold-2.json`，读取平台冻结历史环境和 active-information-v3 fold 2：

- 研究目录：`var/parallel/baselines/round3/active-information-v3`，证据等级为 `exploratory_already_seen_season_controlled_reveal`。
- 比赛：`654a90d2-65a9-5cae-8494-f2fb685a122b`，截止 `2025-12-29T19:30:00+00:00`，信息模式 `historical_reconstruction`。
- 预测器版本 `partial-subset-logistic-1.0:fb968fc75bd79f62de8d8ef5abe8c57a37aae93c15222c5b08d059110e33721a`。
- 估计器版本 `empirical-mask-action-shrinkage-v1:3f28ebabe006605999da`，训练截止 `2025-11-29T12:00:00+00:00`。
- 配置 SHA256 `10220d142f67072418fcb3cf407e38efc6b7af3307b538aae412a2c742af86fd`；完整冻结导出 SHA256 `e0a3aa4c9c750e4aa8453c5654c2466de97c7e3ffbd803fc543e1c706793e497`。

真实历史 mean 运行 `8dca9b25-80b5-4b0a-8739-fb5a118e546b` 首轮比较了 `results_view` 和 `xg_view`：xG 的估计改善 `0.0083338731`，模拟成本损失 `0.0075`，`net_gain=0.0008338731`，于是取得 xG；公开概率从冻结先验主胜 `0.51219512195`、平 `0.21138211382`、客胜 `0.27642276423` 更新为主胜 `0.4839828774`、平 `0.2223008773`、客胜 `0.2937162453`。取得后剩余 `results_view` 的净增益为 `-0.00611759397`，价值停止，模拟费用 `0.03`。

保守运行 `585c5233-2137-483c-abc4-04e4e77583e9` 使用同一比赛、预算和估计器；xG 的保守净增益为 `-0.0310631593`，首轮即以 `VALUE_STOP_NONPOSITIVE_CONSERVATIVE_GAIN` 停止，模拟费用为零。固定顺序运行 `2a9193a6-15b8-4b57-9e72-a32b8c46ec55` 取得两组信息并完成预测，作为同预算工程对照。

真实浏览器入口：`http://127.0.0.1:8771/`（服务 PID 15756）。浏览器核心报告为 `var/parallel/agent/round4/browser-v3/browser.json`，8 项检查通过，JavaScript 错误数为零；保守模式的 HTTP 运行记录为 `var/parallel/agent/round4/real-v3/conservative-ui-run.json`。

## 合成协议验收

`var/parallel/agent/round4/synthetic/mode-matrix.json` 分开保存构造协议场景：四种策略、均值/保守门控、纯 LLM 缺配置、LLM 提案遗漏正价值动作、预算和审计哈希。汇总索引为 `var/parallel/agent/round4/acceptance.json`。构造提案器标为 `constructed_protocol_and_fault_mechanisms`，不计作 LLM 效果证据。`tests/agent/test_round4_active_information.py` 覆盖：

1. LLM 只提议 xG 时，数值门控仍记录正价值遗漏和候选召回率，并按数值结果执行。
2. 均值净增益为正而保守下界为非正时，保守策略停止且不收费。
3. `NO_DATA` 后重新选择其它合法动作，重复动作不计费。
4. provider 完整响应哈希、prompt/参数边界和原始正文不落盘。
5. 策略别名存储为稳定内部策略值。

## 启动与验证

从 `D:\a1-足彩` 执行：

```powershell
python -X utf8 -m demo.acquisition_server --host 127.0.0.1 --port 8771 --mode real `
  --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda `
  --research-artifact var/parallel/agent/round4/backend-v3-fold-2.json `
  --store-dir var/parallel/agent/round4/ui-runs

python -m pytest tests/agent/test_round4_active_information.py tests/demo/test_round3_server.py `
  tests/agent/test_round3_acquisition.py tests/agent/test_round3_proposer.py -q
python -X utf8 tests/demo/browser_round3.py --url http://127.0.0.1:8771 `
  --output var/parallel/agent/round4/browser-v3
```

本轮定向回归为 68 项通过（新增 6 项加原有 HTTP/Agent 62 项）；Node `--check demo/static/acquisition/app.js` 通过。真实 provider 未配置时，`pure-llm` 记录 `LLM_NEEDS_CONFIGURATION`、`llm_calls=0`、线上预留为零；天气和伤停动作保持平台 `NOT_IMPLEMENTED` 状态。研究 v3 的统计结论沿用研究任务报告，Agent 侧只报告执行、隔离、成本和回放，不把单场轨迹称为策略效果。

模式标签在 `acceptance.json` 中分开保存：`historical_reconstruction` 表示历史来源和固定 cutoff，`simulated_reveal` 表示研究环境内的受控揭示和模拟费用，`constructed` 表示协议/故障机制测试；本轮没有新发起 `real` live 刷新，前两轮真实刷新失败记录继续保留。
