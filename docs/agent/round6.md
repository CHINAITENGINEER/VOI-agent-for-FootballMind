# Agent 第六轮：条件信息价值的精确参照机制

本轮新增 `football_agent/decision_value.py`，把“下一步查询哪个信息、是否停止”变成可以精确积分和逐步回放的小型决策问题。它是有限状态参照求解器，证据等级固定为 `SYNTHETIC_ASSUMPTIONS`。先验、结果条件分布、工具观测分布和成本全部人工预先指定；本模块没有从真实比赛训练这些参数，也没有真实足球收益提升的结论。

`football_agent/verify_round6.py` 的六组预先固定构造、两个观察范围和五个随机种子共生成 60 条模拟轨迹。另有 2 条真实调用平台投影函数的构造跨层轨迹。模拟轨迹不是 60 场真实独立比赛；实际独立比赛数量为 0。输出目录为 `var/parallel/agent/round6/exact-evsi-v1/`，其中保存全部参数、代码及文件哈希、轨迹、逐步重算结果和机制验收。

## 数学对象和决策范围

令 `theta` 表示完整潜在情形，例如“是否轮换 × 替补强度”。当前后验是 `b(theta)`，每个潜在情形下有人工给定的 `P(Y | theta)`。工具 `a` 给出可能观测 `z`，拥有 `P(z | theta,a)` 与成本 `c_a`。

当前比赛预测为：

```text
q(y) = sum_theta b(theta) P(y | theta)
P(z | b,a) = sum_theta b(theta) P(z | theta,a)
b_z(theta) = b(theta) P(z | theta,a) / P(z | b,a)
```

使用对数评分时，最优预测的期望损失为 `R(b)=H(q)`。一步信息价值在观察返回之前对全部可能返回积分：

```text
EVSI(a | b) = R(b) - sum_z P(z | b,a) R(b_z)
```

在本模块的概率模型正确且条件结构成立时，使用对数评分的上述量等于 `I(Y;Z | 已有证据)`。它不是 `I(theta;Z)`，也不是收到某个好消息后事后计算的收益。模块另支持多类别 Brier 评分，其 Bayes risk 为 `1-sum_y q(y)^2`，此时不再使用对数评分的信息量单位。

成本和预算严格使用期望损失的同一单位。对数评分的单位为 `expected_log_loss_nats`；Brier 的单位为 `expected_multiclass_brier`。当前构造成本不是美元、调用次数或者“百分之几”。若将真实 API 费用接入，将来需要明确且校准的效用换算系数。

有限步 Bellman 递推如下：

```text
V_0(b,B,A) = R(b)
V_h(b,B,A) = min(
    R(b),
    min_{a in A, c_a <= B}
      [c_a + sum_z P(z | b,a) V_(h-1)(b_z, B-c_a, A\{a})]
)
```

每个分支都可停止，并在分支内重新选择下一工具。实现显式只接受 `horizon=0,1,2`；二步结果只是在剩余至多两次获取、指定模型和预算内最优，不表示任意长计划的全局最优。每种工具为一次性观测；动态刷新、过期和持续时间成本属于已有动态环境及后续研究。

## 六个构造情景的预期行为

下列数值均来自人工构造，单位是期望对数损失 nats，不能作为足球实测提升报告。

| 情景 | 一步决策 | 关键计算与含义 |
| --- | --- | --- |
| 已出线，轮换及替补强度未知 | 查询阵容 | 阵容 EVSI 约 0.031761，成本 0.015；选择来自预测风险变化，不是“出线就减实力”的规则 |
| 已出线，已有证据假定替补能力足够 | 停止 | 相同阵容查询 EVSI 约 0.001597，小于 0.015；已知内容改变剩余查询价值 |
| 规则假定打平足够，实际战术未知 | 查询战术证据 | EVSI 约 0.027809，成本 0.01；防守先验 0.65 明确是人工假定，未被当成已观测事实 |
| 新闻本身有一比特不确定性，但与比赛结果无关 | 停止 | 新闻使潜在新闻状态完全确定，却不改变任何赛果概率；目标 EVSI 为零 |
| 两家来源转述同一个报告 | 查询一次，再停止 | 初次 EVSI 约 0.094362；取得报告后另一转述的条件 EVSI 为零 |
| 纯数学 XOR 互补测试 | 一步停止，二步获取两项 | 两项单独 EVSI 均为零；二步净价值约 0.422607。这是求解器机制检查，不是足球互补规律 |

第一组还体现信息次序：如果查询发现主力出场，替补强度对当前预测没有增量价值；如果发现替补出场，下一步可以查询替补强度。系统不是预先固定“阵容、伤病、天气全部查一遍”。

球队追求“晋级”“第一名”“节省体能”的效用，与本系统使用预测评分控制获取成本的目标是不同对象。本轮把球队动机视为对潜在战术/阵容假设有影响的情境，不让预测 Agent 替球队选择进攻或防守。构造的“打平足够”没有指定为中国对阿联酋等真实赛事事实，具体赛例核实由项目总报告记录。

## 相关证据、可见性与事后结果

模型假设给定**完整** `theta`，结果和不同工具观测条件独立。边际相关通过共享潜变量保留：重复报告情景将共同报告本身放入潜在状态，两次观测会得到同一内容。尚未被潜变量表达的来源相关性会使模型假设失效，不能仅因网站不同就宣称独立。

工具还支持 `evidence_id`。即使工具名称不同，同一证据 ID 在获取一次后就被排除；同 ID 对应不一致观测分布也会被拒绝。两种机制分别检查已知同一证据的身份去重和模型层面的条件相关性。

概率向量、键集合、成本、单位、后验与观察历史一致性都有校验。未知工具、未知返回、模型下零概率返回会明确报错；不会将零概率事件塞入后验。零概率观测提示当前观测模型不完备，真实系统需由上层处理模型失配，不应把报错转为高置信度结论。

最终胜负仅在模拟结束后生成，放在 `evaluator_only`，不参与 `evaluate` 或 `observe`。相同种子的潜在情形、每种工具潜在返回与最终赛果采用分离的随机流，使两个策略可使用相同构造环境。修改事后胜负不改变逐步重算的政策轨迹或先验；单场大比分取胜也不直接证明赛前信息没有价值。

## 跨层接线

`competition_context_integration()` 实际调用 `football_platform.information.competition_context.project_competition_context`，使用合法的赛前时间和获取账本。结构化 owner 持有资格与两种可能阵容之一；在支付前只公开资格，阵容槽为 `unknown`，球员列表不进入公开状态。

有限模型依据已声明的人工假设选择 `lineup`，付出 0.015 后，owner 返回 `starters` 或 `reserves`，同时公布含 11 个构造球员 ID 的 `starting_xi` 证据。收到返回后才执行 Bayes 更新，平台阵容状态从 `unknown` 变为 `hypothesis`；轮换、实际战术仍然为 `unknown`，没有从球员列表自动推断。两种隐藏阵容产生完全相同的获取前决策，获得后才出现不同预测。

该示例没有隐含“资格状态到概率”的映射器。数值模型如何解释观测由完整参数 manifest 预先声明；平台上下文不会自行修改数值后验。真实球员名单到轮换/强度观测的建模、误差估计和训练数据还需要后续实证工作。

## API 与验证

```python
from football_agent.decision_value import synthetic_scenarios

case = synthetic_scenarios()["qualified_rotation_unknown"]
model = case["model"]
state = model.initial_state()
decision = model.evaluate(state, remaining_budget=.08, horizon=2)
# 实际执行 decision["action_id"] 对应工具并取得返回以后：
state = model.observe(state, "lineup", "reserves")
followup = model.evaluate(state, remaining_budget=.065, horizon=1)
```

```powershell
python -m pytest tests/agent/test_decision_value.py tests/agent/test_competition_value_integration.py --import-mode=importlib -q
python -m football_agent.verify_round6 --output var/parallel/agent/round6/exact-evsi-v1
```

32 项新增验证覆盖精确 EVSI 数值、对数评分与目标互信息恒等式、Brier、预算与停止、二步互补、未来结果隔离、零概率返回、重复报告及证据 ID、随机复现、所有模拟轨迹重算和平台获取前后可见性。跨项目完整回归由第六轮总验收记录。本模块未修改旧 Agent、模型评估流程或已有历史实验。
