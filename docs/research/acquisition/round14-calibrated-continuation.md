# 第十四轮：首查是否值得购买后续自主选择的机会

第十三轮固定了第一步，仅在取得返回后比较是否使用实际内容估值。本轮将后续 QUERY/STOP 选择的预期贡献提前纳入第一步：Agent 可以因为未来某些返回将使另一条信息值得获取，而决定先查当前信息。

冻结设计为 `round14-design.md`，SHA256 `9739f3f33abdb7fcdefd532b2f63f2f027a93ced692cb785ef01114e4e8eefc7`。正式目录为 `var/parallel/baselines/round14/calibrated-continuation-v1/`。赛果 joint 模型、原 same-version 残差校准器、五折、校准/验证/测试 ID、工具目录及全部参数均保持冻结。

## 两个预声明规划规则

首查动作 a 的即时净值是原一步校准收益减去费用 `n_a=g_hat(a,s)−λc_a`。对首查的每一种可能返回 z，使用原 joint 的概率 w_z 和原条件校准器得到剩余动作 b 的净值 `d_z=g_hat(b,s+z)−λc_b`。

`calibrated-adaptive-two-step` 的首查总分为

\[
Q_{\mathrm{adaptive}}=n_a+\sum_z w_z d_z\,\mathbf{1}\{\operatorname{round}(d_z,12)>0\}.
\]

每个分支独立选择查询或 STOP，负分支不继续。实际第一条返回到来后，重新执行原条件均值的一步规则。

`calibrated-commit-two-step` 先对所有分支的原校准收益求平均，再决定是否继续：

\[
\bar d=\sum_z w_z\hat g(b,s+z)-\lambda c_b,\qquad
Q_{\mathrm{commit}}=n_a+\bar d\,\mathbf{1}\{\operatorname{round}(\bar d,12)>0\}.
\]

它的后续选择不使用第一条实际返回的内容，实际返回后使用第十三轮 content-marginal 一步规则。真实返回仍更新赛果概率；运行时工具失败、可用性和剩余预算继续受检查，不因预期承诺跳过这些条件。

两者都是把冻结经验收益估计和冻结返回分布组合起来的**近似规划分数**。它们不是原第八轮的 raw EVSI 二步规划，也不是一个经过证明一致的 Bayes 风险分解。逐分支最大化所增加的量称为 surrogate option gap，避免把模型内优势直接命名为实际信息收益。

## 支持、预算和数值边界

全部候选仍使用固定 λ=.5；费用 results=.02、xg=.03，每次一个请求；预算 0/.02/.03/.05，最多两次请求。首动作按总 Q 的 12 位正值排序，平局按动作 ID。负即时净值、正总 Q 可以执行，说明本次首查被模型中的后续机会支持。

如果剩余工具不可用、已尝试、重复同一 evidence，或任一资源单位的预算不足，就没有 continuation bonus。物理预算逐分支检查，使用实际当前已花费用加首查费用再判断第二次费用；预期后续费用不会成为提前扣费，也不会放宽某个实际分支的预算。

所有正质量返回类别保留，包含 missing-summary。任何未来分支缺少支持时，两种模式都取消全部后续贡献，不删除或重加权该分支；首查仍可凭自身支持足够的正收益执行。首查即时收益缺少支持则候选不支持。提供方失败仍属于未建模状态，与源数据缺失分开。

## 计划值与单次证据的审计结构

新类型 `calibrated-continuation-planner-v1` 的工件绑定 joint 与源校准器各自的版本、SHA256、时间和模式，未新增拟合。每折复制原 joint 和 same-version 校准器，生成 adaptive、commit 两个 wrapper，共 20 个模型/绑定工件。

候选的 `expected_gain` 是计划路径上校准收益的期望总和，`cost_penalty` 是整条计划期望费用在损失单位中的总和。单次实际工具费用仍保存在 `cost_vector`。`immediate_estimate`、即时收益/费用/净值、后续贡献、预期后续 gain、预期后续四单位费用和总计划费用向量分别保存。

每个返回分支记录概率、类别、原模型 EVSI、原校准收益、费用、支持、adaptive/commit 的独立继续标志与本模式的继续决定。顶层 EVSI、下界和标准误为 null，源一步估计的描述性统计保留在内部 audit，未构造计划置信区间。

实际动作结束后，另生成 `realized-transitions.jsonl`，逐步比较真实查询前后概率的对数损失，并核对一步、二步收益之和恰好等于 base minus final 的损失变化；扣去实际费用后也必须与总效用变化相等。这些回顾性标签在完整策略轨迹形成之后计算，从未作为运行估值输入。

## 评价与验证

只新增两种策略的 1,600 次测试、1,704 次验证回放；第十一、十二、十三轮参考策略从冻结文件读取。主比较提前指定最大预算下 adaptive 减 commit 的效用损失 `LL+.5×money`；次比较为两模式相对对应一步规则及 base、stage/tool 均值。

采用同场配对，随机参考策略先在每场内平均种子，再按比赛等权；排序 UTC 周 bootstrap 2,000 次、种子 1729。验证只是描述，未据结果调整模型、费用或支持门槛。小预算缺少合法第二次查询，因此每种新策略的动作、费用、最终概率和损失必须与对应的一步参考精确一致。

10 项新专项测试通过，覆盖负即时正总值的执行、adaptive/commit gap、分支逐资源预算、不可用或重复证据、未来不支持但首查仍可执行、实际返回后的条件/抹除内容差异、missing 与失败语义、隐藏数据与假设 token 边界、mean-only 规则、模式和双源 hash 绑定。研究 runner 保存全部初始候选的两种 bonus、option gap 及实际路径，用于独立复算。

根任务独立重建原联合张量和原校准统计，验证每条新轨迹、候选和汇总，并分析模型内部机会与真实访问路径的区别。Agent 固定每折最早比赛运行两个模式，共十个新工作流，引用相同比赛的旧一步卡片；不会因为新策略是否查询两次而更换案例。

## 正式结果

正式实验状态为 `OK`，完成 1,600 次新测试回放和 1,704 次新验证回放。20 个模型/绑定工件包括五个原 joint、五个原校准器和十个新 planner wrapper；没有新增拟合。运行完成后，记录的 17 个实现文件 SHA256 再次全部匹配。

两种新策略在所有预算、测试及验证回放中都与对应的一步参考产生**完全相同的实际动作路径**。最大模拟预算 .05 的结果如下：

| 策略 | Log loss | 平均模拟费用 | 每场查询数 | 效用损失 |
|---|---:|---:|---:|---:|
| base-only | 1.109339 | 0 | 0 | 1.109339 |
| tool-prior-mean | 1.116457 | .01200 | .400 | 1.122457 |
| stage-prior-mean | 1.119416 | .02155 | .850 | 1.130191 |
| frozen-version-mean | 1.121521 | .01980 | .760 | 1.131421 |
| content-marginal-mean | 1.121521 | .01980 | .760 | 1.131421 |
| calibrated-adaptive-two-step | 1.121521 | .01980 | .760 | 1.131421 |
| calibrated-commit-two-step | 1.121521 | .01980 | .760 | 1.131421 |

adaptive 减 commit 的主比较，以及各自减对应一步参考的 LL、费用和效用差，均为零，周 bootstrap 区间均为 [0, 0]。原因是本轮每场配对差都为零；此退化区间只描述这些冻结回放，不证明总体等效或后续规划永远没有价值。

测试中每种新策略在 .05 下分别查询 results 60 场、xg 92 场、初始 STOP 48 场。验证分别为 results 74 场、xg 89 场、STOP 50 场。所有实际查询后均 STOP，无二次查询、无初始 STOP 改查，也没有选择即时净值为负的首查。低预算对应参考的 1,200 次测试、1,278 次验证路径、费用与最终概率精确相同。

每种模式在 .05 下的 152 次真实测试查询，平均实际对数损失收益为 **−.016029**，总实际收益 **−2.436337**；扣除模拟费用后的总收益为 **−4.416337**，即每场平均效用损失相对 base-only 增加 .022082。回顾性逐次收益与 base 到 final 的总损失、效用变化相互核对，未将假设类别配上原赛果打分。

## 模型内后续机会与实际选择

规划值不是全部为零。生产轨迹的只读初始审计显示：.05 测试预算下 400 个初始候选中，122 个有数值上大于 1e−12 的 adaptive bonus，72 个有正 commit bonus；122 个有正 surrogate option gap。最大 adaptive bonus 为 .029430，最大 option gap 为 .018037。这里 1e−12 仅用于区分数值零的描述，不是新决策阈值；执行仍遵守设计中的 12 位舍入正值规则。

实际选中的 152 个候选里，仅 1 个有正 adaptive bonus，大小约 .000259；其余正机会主要存在于最终未选的首动作。全部初始候选的后续分支均有支持，故零路径变化不是未来不支持回退造成的。规划分数有所变化，但变化未超过初始动作排序或 STOP 门槛；实际观察返回后也未产生第二次查询。

这些数值由 native audit 汇总，独立验证任务将从原张量与残差记录复算。它们支持的解释是：当前已拟合环境中的后续机会多数不足以改变此成本和候选集合下的实际政策。它们没有证明“信息内容无用”，也没有证明模型内部 option gap 是现实世界的真实价值。本轮保留零策略差和实际查询负收益，不据此调整门槛来制造更多二次查询。

真实 Agent 的十个固定首场案例已全部通过，见 `var/parallel/agent/round14/calibrated-continuation-bridge-v1/acceptance.json`：26 个候选、16 次初始或真实返回后概率、6 次真实历史揭示、.16 模拟费用、218 条审计及 26 张卡片一致。158 个数学分支单独审计，其中 140 个用于未来续查规划、18 个属于 commit 返回后即时估值的内容边缘化；它们均未成为实际 receipt。两个模式每折案例分别是 xg、results、STOP、STOP、xg，原第十一和十三轮各五个案例以 hash 核验后引用。Agent 新增专项 34 项、合并回归 118 项通过；真实样本没有负即时净值查询，相关门禁能力通过独立机制测试验证，未拿测试 fixture 冒充研究结果。

## 结论边界

当前目标是证明和检验 Agent 能否在首查前显式考虑“看到内容后再选择”的机会，同时如实评价其效果。模型内部正 continuation bonus 不保证该场实际查询有益；已有收益校准和返回分布的误差都可能被规划放大。

测试仍是此前探索的 200 场旧历史，两个工具是同一缓存来源的投影，费用是模拟值。假设分支不是实际查询，也不是独立比赛，不把原赛果贴到虚构返回上评价反事实。区间不包括拟合、反复研究选择、多重比较或环境变化不确定性；无结果优势也完整保留。

```powershell
python -m pytest tests/research/acquisition/test_calibrated_continuation.py --import-mode=importlib -q
python -m football_research.acquisition.experiment_round14 --source var/parallel/baselines/round11/frozen-version-v1 --source-round12 var/parallel/baselines/round12/state-ablation-v1 --source-round13 var/parallel/baselines/round13/content-marginal-v1 --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --earlier-bundle var/parallel/collectors/round3/data/runs/fc298603-aff8-4f3e-bc4f-6c91db871036/exports/understat-2024-historical_reconstruction-ec78dfa8670887cf --output var/parallel/baselines/round14/UNUSED-NEW-RUN --design-note docs/research/acquisition/round14-design.md
```
