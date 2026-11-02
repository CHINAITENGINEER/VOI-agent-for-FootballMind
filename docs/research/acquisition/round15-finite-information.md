# 第十五轮：可枚举真分布中的条件信息获取机制

本轮把两个问题分开：既有 Agent 的数值规划与工具执行机制是否正确；学习得到的足球信息/收益模型是否足以支持现实查询。前者用真分布已知、能够完整枚举的受控环境检验，后者仍保留先前历史实验的零或负结果，不用合成场景替代。

运行前设计为 `round15-design.md`，SHA256 `aa70d9bd6ac92c04722935cdf021d29deaaa811e6f496f5d40fd420aab37a64b`。正式目录 `var/parallel/baselines/round15/finite-information-v1/` 的状态为 `OK`：五个场景、十二个场景/模型组合、五种策略，共六十种配置、1,560 条真实生成分布世界回放。五个场景合计 120 个正质量世界、40 种不同工具返回元组。

这里的“真实”指预声明合成生成分布，与现实足球真相无关。生成分布和模型参数都没有学习或搜索，噪声、费用、预算没有根据结果调整。

## 冻结场景和错误模型

输出 Y 有三个类别。信息通道 K 的正确返回概率为 .9，其余每类 .05。除了 irrelevant 的 Y 先验为 (.7,.2,.1)，其他场景的 Y 都均匀。context 返回独立二值 G，概率各 .5。

| 场景 | 已冻结机制 | 工具费用 | 费用预算 | 检验点 |
|---|---|---|---:|---|
| gated_stop | context 说明 signal_a 是否有信息，另一情况下 signal_a 与 Y 独立 | context .02、signal_a .20 | .22 | 零即时结果信息的上下文能否支持后续查询或 STOP |
| gated_route | G 决定 signal_a 或 signal_b 哪个使用 K，另一个无关 | context .02、signal_a/b 各 .20 | .42 | 同一首次查询的不同真实返回能否选择不同下一工具 |
| complementary | 两个独立三值返回之和 mod 3 通过 K 决定 Y | signal_a/b 各 .12 | .24 | 单独无信息，合并有信息，能否规划两步 |
| redundant | signal_a、signal_b 返回完全相同的 K 信号 | signal_a .04、signal_b .06 | .10 | 边际有信息但条件冗余时停止 |
| irrelevant | 两工具各自独立于 Y，也相互独立 | signal_a .02、signal_b .03 | .05 | 即使返回准确也不浪费费用 |

每个动作消耗一个请求，最多两次；费用换算系数 λ=1，其他费用单位为零。gated_route 虽有三工具且费用预算容纳两个信号，但请求数上限阻止三者全查。

每场景同时定义 `correct` 和 `independent` 模型。correct 与生成分布相同；independent 保留 Y 和各工具返回的边际概率，却去掉所有信息依赖。两个 gated 场景还定义 `swapped_context`，交换 G 的含义。错误模型同时用于预测和规划，所以错误实验反映二者共同失配，未把它解释为旧足球实验误差的确定原因。

模型行只保存正质量项，所有被评分的真实生成世界在各声明模型下都有支持、真实 Y 的预测概率大于零。不会用默认后验填补不可能观测。

## 使用既有规划器与强基线

myopic 和 adaptive-two-step 原样复用冻结 `JointRiskPlanner` 的 horizon 1 和 2，每次实际返回后按剩余物理预算重新决策。预测始终只使用已获得的类别。base-only 不查询。

`best-fixed-subset` 在返回发生前，枚举所有可负担、长度至多二的子集，按声明模型的期望损失加费用选最优，然后完整执行。最终预测仍使用实际返回。

`best-fixed-order-stop` 在返回前枚举全部长度至多二的固定动作顺序，并允许执行中按已获内容提前 STOP，但后续工具身份始终受该预选顺序限制。它比“查固定子集到底”的基线更强，能把内容相关的停止收益与内容相关的工具选择收益分开。

固定基线按最小的 12 位舍入期望风险选择，再依次偏好较短列表、字典序较小列表；工件保留全部候选及最优舍入平局。执行的查询/STOP 遵守原规划器正净值及动作 ID 平局规则。没有用生成分布下的实际评价结果选择错误模型的策略。

## 数据边界与工件

新增 `finite_information.py` 提供冻结场景与 `FiniteJointModel`。实际输入状态只有 `observations` 和 `action_history`；任何 outcome、world_id、完整隐藏 returns 或未知顶层字段都会被拒绝。规划假设类别放在专用 `joint_return_hypothesis` 中，只由显式假设接口接受，实际预测与规划入口拒绝该标记。

模型本身持有声明的完整概率分布，这是先验知识；它从未收到“目前执行的是其中哪一个世界”。具体世界的全返回元组仅由环境保存，只有动作选定后才揭示相应返回。赛果只在轨迹形成后用于评分。

正式输出包含 `spec.json`、全部 `scenarios.json`、逐场景 spec、十二个模型工件、每个生成世界的 episode 和完整 actual-policy trajectory、`fixed-policy-selections.json`，以及声明模型自身的全分布期望 `model-expectations.jsonl`。数学分支标明 hypothetical，实际 receipt 和费用分别记录。source/model/spec 文件均绑定 SHA256。

`metadata.training_as_of` 仅兼容旧规划器协议，另有明确标记指出它是冻结说明的日期，不是学习训练或真实历史可用性声明。

## 正确模型下的精确期望

效用损失定义为自然对数 log loss 加实际模拟费用，越低越好。这里完全枚举生成分布，表中为精确有限总体期望的浮点计算，没有抽样置信区间。

| 场景 | base-only | myopic | adaptive-two-step | best-fixed-subset | best-fixed-order-stop |
|---|---:|---:|---:|---:|---:|
| gated_stop | 1.098612 | 1.098612 | .866505 | .966505 | .866505 |
| gated_route | 1.098612 | 1.098612 | .614398 | .966505 | .866505 |
| complementary | 1.098612 | 1.098612 | .634398 | .634398 | .634398 |
| redundant | 1.098612 | .434398 | .434398 | .434398 | .434398 |
| irrelevant | .801819 | .801819 | .801819 | .801819 | .801819 |

gated_stop 的自适应策略先付 .02 查询与 Y 独立的 context。在 G=0 时继续 signal_a，G=1 时 STOP；平均查询数 1.5、费用 .12。它相对查固定子集节省 .10，但与允许按内容 STOP 的固定顺序基线完全相同。因此该场景验证自适应停止，没有证明动态改换工具的额外收益。

gated_route 同样先查 context，G=0 查 signal_a，G=1 查 signal_b。平均两次查询、费用 .22、最终 log loss .394398；相对允许内容相关 STOP 的强固定顺序基线，效用损失下降 **.252107**。固定顺序选 `context → signal_a`（对称的 `context → signal_b` 构成同分最优顺序之一），它在另一上下文下只能停止而非换工具。

complementary 中，一步策略因为每个工具单独没有 Y 信息而停止。两步策略查询两个信号，最终 log loss .394398、总费用 .24。但最佳固定子集和固定顺序同样达到该值，所以这是规划深度与信息互补的机制结果，不是内容路由优于固定计划的证据。

redundant 选择较便宜的 signal_a 后停止；signal_b 条件信息为零。irrelevant 中所有合理策略都不查询。这些相等结果完整保留，没有为了让自适应策略处处胜出而改变场景。

## 错误模型的代价

下表为 adaptive-two-step 的模型内预期与生成分布下实际期望，两者都包含费用：

| 场景/模型 | 模型内预期损失 | 生成分布实际损失 | 相对真分布两次查询最优值的 regret |
|---|---:|---:|---:|
| gated_stop / correct | .866505 | .866505 | 0 |
| gated_stop / independent | 1.098612 | 1.098612 | .232107 |
| gated_stop / swapped_context | .866505 | 1.685444 | .818939 |
| gated_route / correct | .614398 | .614398 | 0 |
| gated_route / independent | 1.098612 | 1.098612 | .484215 |
| gated_route / swapped_context | .614398 | 2.252275 | 1.637877 |

independent 模型在全部场景选择 STOP，漏掉互补与冗余场景的有益查询，但在 irrelevant 中正确停止。swapped_context 会反向理解上下文，查询真正无关的信号并据错误模型更新概率；实际效用甚至显著劣于不查询。机制能够正确优化给定模型，不等于给定模型正确。

regret 的生产参考是 correct 模型下现有 horizon-two 的精确规划值；根任务另用不依赖生产模型或生成器的穷举策略树核验。约 1e−16 的微小负 regret 或概率和误差是浮点舍入，不解释为优于最优值。

## 验证与现实边界

10 项新增研究测试通过，覆盖冻结质量与 world/tuple 数量、所有真实世界的模型支持、independent 边际保存、各一条观测状态下的后验归一与鞅关系、零即时信息的两步查询、实际 G 变化导致不同第二工具或 STOP、强固定顺序基线、冗余和无关停止、隐藏字段与非法类别/失败状态拒绝、分布损坏及保存回读、动作前缀和请求上限。

Agent 将覆盖正确模型下全部 40 种正质量返回元组，每种只运行一次持久工作流。代表元组按排序选择，环境只收到全 returns，不收到 Y；实际预测/规划只收到已取得 observations。不同 Y 共享一个返回元组时，所有研究路径应相同。根任务独立重建因素分布、穷举策略树和核对精确指标，最终验收单独记录。

正式 40 个 Agent 工作流已全部通过，验收见 `var/parallel/agent/round15/finite-information-bridge-v1/acceptance.json`：152 个候选、418 个模型分支、106 次预测、66 次构造环境的实际揭示、6.96 模拟总费用、1,328 条审计及 116 张卡片一致。这些是对所有返回模式各执行一次的合成检查计数，不是按生成质量加权的总体平均，也不是现实信息源的请求次数。

具体执行为 gated_stop 三种返回模式继续、三种停止；gated_route 九种路由到 signal_a、九种路由到 signal_b；complementary 九种均查两次；redundant 三种均只查一次；irrelevant 四种均不查。每个元组的所有 Y 研究轨迹先核对一致，再只执行一次 Agent。Agent 新增专项 26 项及含相关旧回归的合并测试 60 项通过；旧规划器与执行器保持冻结。

本轮正结果仅建立受控机制证据：在模型和费用已知、动作集合有限时，既有数值 Agent 能购买对后续选择有用的上下文，按实际内容改换工具，并在无信息时停止。它没有新增足球比赛证据，没有训练出出线/轮换/阵容或战术信息模型，也没有把先前足球实验改写为成功。错误模型结果说明，后续现实研究的关键仍在于可用信息及其条件价值是否被可靠估计。

```powershell
python -m pytest tests/research/acquisition/test_finite_information.py --import-mode=importlib -q
python -m football_research.acquisition.experiment_round15 --output var/parallel/baselines/round15/UNUSED-NEW-RUN --design-note docs/research/acquisition/round15-design.md
```
