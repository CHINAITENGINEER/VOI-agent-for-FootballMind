# Agent 第七轮：相同免费上下文的预测器接线

本轮给 `ResearchServices.load` 增加显式的 `partial_predictor_kind`。省略此字段时仍加载原 `PartialPredictor`，等价于 `partial-subset-v1`；设置为 `contextual-partial-v1` 时加载新的 `ContextualPartialPredictor`。两个模型族使用各自的序列化 schema 和内容哈希，配置错误不会触发自动猜测或替代加载。

新预测器在空信息掩码和各个已获取信息子集中都可使用相同的免费比赛上下文。它的空掩码输出因此也可以依球队身份和赛程变化，不再把某一时间折的所有比赛都压成同一个频率先验。免费上下文的合法性、训练集词表与跨季 provider 身份映射由研究模块负责；Agent 接口只传入已有的可见状态，不开放未获取的 Elo、xG 等付费组。

价值估计器继续使用第六轮已实现的显式加载协议：`contextual-value-v1` 对应 `ContextualValueEstimator`，`source-identity-value-v1` 对应 `SourceIdentityValueEstimator`。当前预测器输出会进入同一可见状态的预测缓存，再传给信息价值估计器；概率、免费上下文说明、未知身份提示和映射审计保留在原生结果内。

示例配置结构如下。示例 lambda 仅展示格式，正式配置按各折、各策略、预算 `.05` 的验证集所选值生成：

```json
{
  "partial_predictor": "models/fold-0-predictor.json",
  "partial_predictor_kind": "contextual-partial-v1",
  "value_estimator": "models/fold-0-source-identity.json",
  "value_estimator_kind": "source-identity-value-v1",
  "lambda_cost": {"money": 0.25}
}
```

`verify_round7_contextual.py` 使用实际 `AcquisitionWorkflow`、`FrozenResearchBackend` 和已训练工件。每个时间折按 kickoff、match ID 固定选择最早测试比赛，分别验证报告中存在的 `contextual-prediction-only` 与 `contextual-source-identity`，不依据赛果、信息收益或动作数量挑选比赛。

验证器检查免费上下文下的初始预测与该比赛 `base-only` 行相同，随后检查动作序列、最终概率、模拟费用与对应策略研究行相同。它同时核对预测器和估值器的研究工件哈希、实际配置加载哈希和执行事件审计。正式使用 `equal-context-v1` 研究产物完成七折、两种策略的 14 条桥接轨迹，全部通过，复用 7 场已有比赛；新增独立比赛数为 0，外部 HTTP 和 LLM 调用为 0。这只证明研究策略与执行系统接线一致，不是另一批比赛上的效果验证。

正式结果在 `var/parallel/agent/round7/equal-context-bridge-v2/acceptance.json`，状态为 `ROUND7_EQUAL_CONTEXT_AGENT_BRIDGE_PARITY_VERIFIED`。第一次数值重放前遇到 Windows 产物路径反斜杠和查找键正斜杠不一致，已修复路径规范化并增加重复路径拒绝测试；`equal-context-bridge-v1/failure.json` 保留了执行病例数为 0 的失败记录。修复仅影响桥接验证器，研究模型、固定首场选择和数值策略均未改变。

固定选择的 14 条轨迹包含 9 条直接停止和 5 条查询一次后停止。研究主效用未显示明确收益时，这些路径仍按原样接受验证；本轮不挑选效果更好的比赛替代停止或负结果。总体预测质量和成本效用差异由完整研究报告评估，不能从这 14 条接线轨迹推断。

运行方式：

```powershell
python -m pytest tests/agent/test_round7_research_bridge.py --import-mode=importlib -q
python -m football_agent.verify_round7_contextual --research var/parallel/baselines/round7/equal-context-v1 --output var/parallel/agent/round7/equal-context-bridge-replay --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda
```

新增单测 13 项通过，与第六轮桥接单测合跑为 24 项通过。测试覆盖旧默认加载兼容、显式新旧类型、误配和未知类型拒绝、新预测器序列化后与直接调用相同、免费身份上下文改变空掩码概率、价值估计使用对应当前预测、固定首场选择、逐策略 lambda，以及跨平台工件路径。重放命令要求新的输出目录，已有验收工件保持冻结。
