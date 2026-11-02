# Agent 第八轮：联合返回模型中的一步与二步风险规划

本轮将研究侧和 Agent 执行侧接到同一个 `JointRiskPlanner`。研究侧拟合的 `JointReturnPredictor` 提供 `P(Y,Z_results,Z_xg | H)`；规划器从该联合分布得到每个可能工具返回的概率及其后的赛果后验，先对所有可能返回积分，再决定查询或停止。它不使用尚未获取的真实工具返回，不用单场已知赛果衡量查询是否值得。

## 物理预算与效用单位

`football_research/acquisition/joint_planner.py` 接受如下独立参数：

```python
planner = JointRiskPlanner(model, horizon=2, lambda_cost={"money": 0.5})
decision = planner.plan(state, actions, budget={
    "limits": {"money": 0.05, "requests": 2},
    "spent": {"money": 0.0, "requests": 0}
})
```

`money` 是既有历史环境的模拟费用，`requests` 是投影调用次数。`lambda_cost` 把费用转换为预测对数损失的效用惩罚；它不改变物理预算。即使 lambda 为零，余额不足的工具仍被排除，二步分支也受剩余请求次数限制。Agent 的 `max_simulated_cost/max_actions` 与已使用量被投影到该物理预算。

对每个候选第一步，规划器记录当前风险、一步 EVSI、各返回后的预测、每分支最优后续行动、期望终端风险、计划期望累计费用和净价值。每个返回分支都可以立即停止。二步中的最优行动依实际可能返回改变，未使用平均返回或虚构数值填入未购买特征。

## 同一候选值贯穿研究与执行

新 `football_agent/joint_bridge.py` 给旧执行器返回：

- `expected_loss_improvement`：完整条件计划的期望终端损失减少。
- `cost_in_loss_units`：完整条件计划的期望累计费用，经 lambda 换算后的惩罚。
- `net_gain`：用于行动排序的净价值。
- `research_estimate`：未舍入净值、一步 EVSI、累计费用、分支后验、后续行动、预算及参数等完整数值依据。

因此，二步计划不会被适配器降为“一步收益减第一笔费用”。真实执行仍只调用当下所选第一步，取得真实返回后重新计算；平台和执行器继续按单次真实模拟费用记账。

两个获取顺序在数学上相同时，浮点求和可能相差约 `1e-16`。研究与执行统一采用 `selection_score=round(planning_net_gain,12)` 排序，得分相同按 action ID 排序，得分不大于零则停止。未舍入风险、费用和净价值完整保留；排序数值和“收益减惩罚”之间至多约 `5e-13` 的舍入差被明确记录。该规范用于确定性复现，不是概率校准或调参。

## 信息边界和有限模型假设

真实 `predict_partial` 与规划根入口拒绝 `joint_return_hypothesis`。只有显式 `predict_hypothetical`、`enumerate_action(..., hypothetical=True)` 支持内部假设分支。分支由类别 token 表示，不伪造 Elo、xG 等数值；实际 Agent 状态还经过原有公开字段和获取掩码检查。审计中的分支标注 `hypothetical=True`，不会混进真实证据状态。

两个工具之间的依赖由同一个联合返回分布保留，不因工具名称不同而当成独立证据。已取得或已尝试的一次性工具不再查询；显式相同证据 ID 也会去重。模型的 `summary_missing` 指压缩统计摘要缺失，和提供商失败是不同事件。`FAILED/TIMEOUT/PROVIDER_ERROR` 不会被重新解释为缺失类别来更新后验。

新增 `JointAcquisitionWorkflow` 继承原执行流程，补充一次必要的缓存失效：从未查询变为 `summary_missing` 时，数值组和 Boolean mask 都可能不变，但已观测的返回类别已改变。此时明确记录 `joint_conditioning_changed` 并在下一步重新预测，避免最后报告仍展示查询前的概率。提供商失败则明确进入 `JOINT_OBSERVATION_MODEL_MISMATCH`，不将它当成训练过的摘要缺失。旧 `AcquisitionWorkflow` 源码保持原样。

模型内的一步对数评分 EVSI 等于 `I(Y;Z | 已有证据)`；减少与赛果无关的潜在状态熵不会产生目标信息价值。人工 XOR 联合分布测试展示了单项价值为零、二项联合有价值的情形。正式历史模型的这些概率与收益仍取决于拟合的联合返回模型是否可信；本轮没有把精确积分等同于真实预测效用提升，也没有把二步最优称为无限步最优。

## 显式模型配置与保守模式边界

```json
{
  "partial_predictor": "models/fold-0-joint.json",
  "partial_predictor_kind": "learned-joint-returns-v1",
  "value_estimator": "models/fold-0-joint.json",
  "value_estimator_kind": "joint-risk-planner-v1",
  "planner": {"kind": "finite-joint-bellman-v1", "horizon": 2},
  "lambda_cost": {"money": 0.5}
}
```

预测与估值必须使用内容哈希相同的联合模型工件。旧模型和旧配置默认加载行为保持兼容。模型参数不确定性没有在本轮估计，因此配置明确只支持 `mean`，所有 `lower_gain/uncertainty` 为 null。若外部要求旧执行器的 `conservative` 模式，工作流以 `ESTIMATE_UNAVAILABLE` 停止、获取次数为零，审计保留 `UNSUPPORTED_JOINT_PARAMETER_UNCERTAINTY`，不会制造置信界。

## 验证范围

`verify_round8_joint.py` 按每个时间折最早 kickoff、match ID 固定选择首场，分别验证 `evsi-one-step` 与 `evsi-two-step`，使用各自验证集所选 lambda 和 `.05` 物理预算。它除了核对最后概率、动作、费用、配置哈希和事件审计，还逐获取步核对所有可行候选的即时 EVSI、分支后验、计划累计成本、终端风险和决策得分。研究和 Agent 即使最后恰好输出相同概率，内部二步估值不一致也会失败。

正式使用 `var/parallel/baselines/round8/joint-returns-v2` 完成七折、14 条实际桥接，全部通过。验收在 `var/parallel/agent/round8/joint-bridge-v1/acceptance.json`，状态为 `ROUND8_JOINT_AGENT_CANDIDATE_AND_EXECUTION_PARITY_VERIFIED`。共核对 40 个候选值及其条件分支；复用 7 场已有比赛，新增独立比赛数为 0，不产生真实 HTTP、LLM 调用或新的真实费用观测。

模型内积分一致和 Agent 接线成功并不表示实测预测改善。正式研究报告在预算 `.05` 时的 log loss 为：零查询基线 `1.080229`、一步 EVSI `1.119578`、二步 EVSI `1.116358`，后两者更差且产生查询成本。本轮保留该负结果，不更换首场示例或把模型内信息量恒等式当成真实收益证明。后续重点应是联合返回分布估计与失配诊断，而不是仅增加规划步数。

本任务新增 31 项测试通过：17 项规划器测试、14 项 Agent 接线与执行测试，覆盖物理预算、二步互补、后验鞅、无关信息、相关证据、缺失与失败、类型与哈希、保守模式显式不可用、missing 后重算，以及候选成本被篡改时验收拒绝。全项目回归由第八轮总验收记录。

```powershell
python -m pytest tests/research/acquisition/test_joint_planner.py tests/agent/test_round8_joint_bridge.py --import-mode=importlib -q
python -m football_agent.verify_round8_joint --research var/parallel/baselines/round8/joint-returns-v2 --output var/parallel/agent/round8/joint-bridge-replay --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda
```
