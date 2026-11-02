# Agent 第九轮：经验校准后的单步查询门控

第八轮已经验证联合模型内的 EVSI 积分、一步/二步规划和真实执行的一致性，但模型内的正信息量没有转化成历史测试中的预测改善。第九轮保留第八轮的联合预测器和原始规划器，新增一个独立的经验校准器：用严格前向 OOF 的实际预测更新损失变化，校正原始模型 EVSI 的系统性偏差，然后决定是否支付下一次查询费用。

此处的 `g_hat` 是固定预测器在相似历史情境下的经验单次更新收益，允许为负。它不是“Bayes 最优信息价值为负”，也不声称求出了二步计划、联合模型参数后验或长期全局最优策略。

## 独立配置与工件绑定

第八轮源码与工件保持冻结。新增入口在 `football_agent/calibrated_joint_bridge.py`，通过独立工厂接到既有 `FrozenResearchBackend`：

```python
from football_agent.calibrated_joint_bridge import make_calibrated_joint_backend
from football_agent.calibrated_joint_workflow import CalibratedJointAcquisitionWorkflow

backend = make_calibrated_joint_backend(
    bundle_path="published-evidence-bundle",
    model_artifact="backend-round9.json",
)
workflow = CalibratedJointAcquisitionWorkflow(backend, "runs/round9")
run = workflow.start(match_id, prediction_cutoff,
                     budget={"max_simulated_cost": 0.05, "max_actions": 2})
final = workflow.run_to_completion(run["run_id"])
```

模型配置核心如下；正式配置还包含已有平台 `experiment`、固定场次和研究发布版本：

```json
{
  "partial_predictor": "models/fold-0-joint.json",
  "partial_predictor_kind": "learned-joint-returns-v1",
  "value_estimator": "models/fold-0-calibrator.json",
  "value_estimator_kind": "residual-evsi-calibrator-v1",
  "gate": {"kind": "calibrated-joint-one-step-v1", "risk_level": "conservative"},
  "lambda_cost": {"money": 0.5}
}
```

`risk_level` 可取 `mean` 或 `conservative`。执行工作流默认采用配置中的值；显式传入不同值会报错。`horizon=2` 等额外 gate 参数也会被拒绝，防止把经验单步校准器误接到二步规划解释上。

联合预测器和校准器是两个不同工件，具有不同内容哈希。加载时校准器必须明确绑定联合预测器的版本、SHA256 和训练时点；不沿用第八轮“预测器与估值器文件相同”的检查。运行预测和估值时，模型及校准参数拟合时点均须不晚于比赛预测截止时点。`diagnostics_as_of` 原样保留，指独立时间尾段的诊断可用时点，不冒充模型拟合时点。

## 单步门控及不确定性解释

研究校准器固定使用 action、已获取动作掩码、EVSI 桶 `[.02,.05,.10]` 定义情境，局部残差向 action+mask 父层收缩，收缩强度为 20，最少支持比赛数为 12。局部不足而父层足够时明确使用父层；父层不足时停止。取得 `summary_missing` 也算已获取动作，不能借 Boolean 特征掩码仍为 false 而错误复用“什么都没查询”的校准支持。

均值门控使用 `g_hat - lambda·单次费用`，LCB 门控使用 `lower_gain - lambda·单次费用`。下界由按 UTC 周聚合的残差均值标准误得到，属于描述性均值估计摘要。它不保证时间漂移、策略选择之后的覆盖，也未纳入拟合联合模型参数不确定性；不足两周时下界为 null，LCB 明确停止。负 `g_hat` 与负下界完整保留，不截断成零。

现金/模拟费用预算与请求次数预算独立于 lambda；lambda 为零不解除物理预算。连续运行中每次取得真实返回后，重新估计下一次单步查询价值。最多两次实际查询的预算不等同于求解二步 Bellman 计划。

为确保可重放，研究与执行都按 `round(net_gain,12)` 排序并在同分时按 action ID 选择，得分不大于零时停止。原始均值、下界和标准误保持原值；新执行器只统一排序得分的舍入规范，不修改冻结的旧执行器。

## 公开状态与审计

每个候选在 `research_estimate` 中保留原始模型 EVSI、`g_hat`、上下摘要界、标准误、local/parent/effective support、fallback、成本惩罚、门控原因及原生得分。审计包含“有支持但收益非正”和“支持不足/下界缺失”的候选；后者不是低价值的实测证据。

真实状态拒绝隐藏赛果、未购买特征及 `joint_return_hypothesis`，包括嵌套 token。`summary_missing` 后继续使用第八轮的 `JointAcquisitionWorkflow` 缓存失效逻辑，刷新后验；提供商失败单独进入 `JOINT_OBSERVATION_MODEL_MISMATCH`，不解释为训练过的摘要缺失。实际运行不调用外部 HTTP 或 LLM。

## 固定首场验收

`verify_round9_calibrated.py` 从每个时间折的既有 `base-only/.05` 赛程，按 kickoff 和 match ID 固定选择最早一场。每场分别验证 `calibrated-mean` 与 `calibrated-lcb`，使用各策略在各折验证集选择的 lambda，不依据收益、动作或最终赛果选择展示案例。

逐步验收包括每个物理可行候选的原始校准字段、支持与 fallback、单步惩罚、排序分数、最终查询动作、实际模拟费用、概率、工件绑定和事件链。无支持候选也必须一致。正式结果写入 `var/parallel/agent/round9/`，包含 7 场既有比赛的 14 个工作流；新增独立比赛数、真实外部调用数和 LLM 调用数均为 0。结果用于工程一致性验收，效用结论以同轮完整历史研究报告为准。

正式研究目录 `var/parallel/baselines/round9/residual-evsi-v1` 已通过实际桥接，验收文件为 `var/parallel/agent/round9/calibrated-joint-bridge-v1/acceptance.json`，状态 `ROUND9_CALIBRATED_JOINT_AGENT_CANDIDATE_AND_EXECUTION_PARITY_VERIFIED`。14 条工作流共核对 30 个原生候选：第 0 和第 6 折的 mean 首场各执行一次查询，其余工作流立即停止。所有概率、动作、费用、校准值与审计均一致；STOP 案例完整保留。

完整研究在同一批反复探索过的 290 场历史测试比赛、预算 `.05` 上得到：

| 策略 | Log loss | 每场模拟费用 | 固定费用权重下的 utility loss |
| --- | ---: | ---: | ---: |
| 零查询 | 1.080229 | 0 | 1.080229 |
| 原始一步 EVSI | 1.119578 | 0.030586 | 1.134871 |
| 原始二步 EVSI | 1.116358 | 0.032931 | 1.132823 |
| 校准 mean | 1.077224 | 0.011862 | 1.083155 |
| 校准 LCB | 1.080229 | 0 | 1.080229 |

mean 在这批历史数据上避免了原始 EVSI 的大部分不利更新，但相对零查询的 log loss 变化约 `-0.003005`，描述性按周 bootstrap 区间跨零；计入固定费用权重后 utility loss 反而增加约 `0.002926`。LCB 在全部 290 场停止，输出与零查询相同。上述结果不支持“已经学会稳定盈利或稳定提高预测效用”的结论，也不是独立新测试集上的确认。

专项测试使用明确标注的构造数据，验证真实模型和校准器的序列化绑定、类型和训练时点、负收益/无支持停止、mean 与 LCB 的实际动作分歧、单周下界缺失、父层回退、预算、missing 后验刷新、failure 边界和篡改候选值导致验收失败。

本轮新 Agent 专项 27 项全部通过。Agent 目录曾完整回归 380 项通过；其后仅为新模块补充 HTTP 错误历史拒绝与 1 项专项，新专项再次全部通过。最终全项目结果以第九轮总验收为准。第八轮桥接验收记录中的六个旧源码 SHA256 均保持一致。

```powershell
python -m pytest tests/agent/test_round9_calibrated_bridge.py --import-mode=importlib -q
python -m football_agent.verify_round9_calibrated --research var/parallel/baselines/round9/residual-evsi-v1 --output var/parallel/agent/round9/calibrated-joint-bridge-v1 --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda
```
