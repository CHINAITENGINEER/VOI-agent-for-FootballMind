# Agent 第十四轮：为后续查询或停止的选择机会付费

本轮执行冻结设计 `docs/research/acquisition/round14-design.md`，SHA256 为 `9739f3f33abdb7fcdefd532b2f63f2f027a93ced692cb785ef01114e4e8eefc7`。研究进一步检验：首个查询的价值是否包含“看见其返回后决定要不要查第二个工具”的机会。

固定第十一轮的 joint 预测器、same-version 残差校准器、历史来源、时间折和费用，新增 adaptive、commit 两种最多两步的规划方式，不重新拟合或选择阈值。它们将经验单次更新收益与冻结返回概率组合成近似规划奖励模型，不是严格一致的 Bayes 风险分解，也不是已经证明的真实信息价值。

## 规划值与真实收费分开

首个动作的即时净值为即时校准收益减去该次费用。在所有合法正概率返回分支上：adaptive 逐支决定下一次查询或 STOP；commit 对后续收益先平均，再作统一的继续或停止决定。每个分支都有自己的物理预算检查。若一个正概率分支缺少支持，两种模式都停用整个后续奖励，同时保留有支持的即时收益。缺失摘要类别保留在积分里，提供方失败则在模型覆盖范围之外。

新 `CalibratedContinuationServices` 复用既有公开状态和执行转换，但将以下字段明确解释为计划累计量：

| 字段 | 含义 |
| --- | --- |
| `expected_gain` / Agent `expected_loss_improvement` | 条件计划内预计累计的校准更新收益 |
| `cost_penalty` / Agent `cost_in_loss_units` | 条件计划内预计累计费用乘固定 lambda 的损失惩罚 |
| `mean_net_gain` / `selection_score` | 整条条件计划的净值及 12 位选择分数 |
| `immediate_expected_gain/cost_penalty/net_gain` | 当前单次查询的对应量 |
| `expected_continuation_gain/cost/cost_penalty` | 尚未执行的后续查询期望量 |
| `continuation_bonus` | 后续查询或停止带来的规划净贡献 |
| `expected_plan_cost_vector` | 预计整条计划的费用、请求数等资源消耗 |
| `cost_vector` 和实际工具收据 | 当前动作的单次目录费用，实际执行时才扣款 |

执行器按累计净值选动作，即使即时净值为负，也可因总计划净值为正而选择首个动作。预算扣除始终取实际工具返回收据，数学积分和未来预计请求不算真实请求。标准误、下界、顶层 EVSI 均为 null，源单步估计的描述统计只作为原始审计。

实际取得首次返回后，adaptive 使用原条件 mean；commit 使用第十三轮 content-marginal mean。两者的赛果预测仍使用实际返回内容；commit 不忽视运行中改变的合法性、失败或剩余预算。

## 新配置与工件绑定

新增 `football_agent/calibrated_continuation_bridge.py`，提供 `load_calibrated_continuation_services`、`make_calibrated_continuation_backend`、`CalibratedContinuationAcquisitionWorkflow` 和 `build_calibrated_continuation_cards`。旧服务、默认工厂和旧工件保持冻结。

配置示意如下，正式路径按研究 `fold.planners`、`joint_model` 和 `source_calibrator` 清单给出：

```json
{
  "partial_predictor": "models/fold-2-joint.json",
  "partial_predictor_kind": "learned-joint-returns-v1",
  "source_calibrator": "models/fold-2-source-calibrator.json",
  "source_calibrator_kind": "residual-evsi-calibrator-v1",
  "value_estimator": "models/fold-2-adaptive-planner.json",
  "value_estimator_kind": "calibrated-continuation-planner-v1",
  "gate": {
    "kind": "calibrated-continuation-two-step-v1",
    "risk_level": "mean",
    "mode": "adaptive"
  },
  "lambda_cost": {"money": 0.5}
}
```

commit 配置使用对应工件和 `mode="commit"`。路径相对于配置文件，实际配置还包含 experiment、fixtures、费用和研究版本信息。预测器、原校准器和 planner 分别作版本/字节哈希绑定，planner 的 mode 必须与配置相符，最大 horizon 明确为 2。运行门禁拒绝隐藏结果、客户端假设 token、模型时点晚于查询时点、失败返回、模式混用和非法预算。

卡片沿用已审计事件快照，新增 `plan_value` 展示即时贡献、后续贡献、累计收益/费用以及 adaptive/commit surrogate option gap。`model_only_value_audit.branches` 展示后续数学分支；commit 在真实首次返回后的估值边际化另记 `immediate_valuation_branches`。两类分支都设置 `counts_as_acquisition=false`，与实际 `receipt` 和返回后真实概率分开。

## 固定首场验收与测试

`football_agent/verify_round14_continuation.py` 从已绑定哈希的原赛程选择五个目标折各自最早 kickoff、match ID 的比赛，每场执行 adaptive、commit 两种模式，总共 10 条新历史工作流。原第十一轮五份条件 mean 卡片和第十三轮五份 content-marginal 卡片按文件哈希、运行审计、预测器/源校准器和时间核对后引用，不重新执行。

逐候选比较包含所有即时量、后续量、总规划量、分支概率/校准估计/继续决定及预算；每次实际返回后的概率、动作顺序、STOP、费用和事件/blob 审计均与研究轨迹核对。研究、Agent 首场的 STOP 或负结果全部保留，不为展示第二次查询而重新挑场。

```powershell
python -m pytest tests/agent/test_round14_continuation_bridge.py --import-mode=importlib -q
python -m football_agent.verify_round14_continuation --research var/parallel/baselines/round14/calibrated-continuation-v1 --output var/parallel/agent/round14/calibrated-continuation-bridge-v1 --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --conditional-agent var/parallel/agent/round11/frozen-version-bridge-v1 --marginal-agent var/parallel/agent/round13/content-marginal-bridge-v1
```

专项覆盖预定四个预算、两模式、单次剩余查询、lambda 为零的物理限制、运行门禁、独立哈希、原卡片复用与数学/真实收据区分。关键构造测试把即时收益设为 `-.005`、后续收益设为 `.1`：两模式都选择即时净值为负而总计划净值为正的首个 results 查询。预计计划费用 `.05`，第一笔收据只扣 `.02`；缺失返回更新后验后才执行第二笔，提供方失败则只扣首笔并延迟处理。另一构造测试保留全部缺失/unsupported 分支，停用后续奖励而允许有支持的即时查询。这些参数是独立测试夹具，不进入研究指标。

## 正式验收与当前实测结果

正式研究工件位于 `var/parallel/baselines/round14/calibrated-continuation-v1`，新增两策略测试轨迹 1,600 条、验证轨迹 1,704 条。Agent 验收位于 `var/parallel/agent/round14/calibrated-continuation-bridge-v1/acceptance.json`，状态为 `ROUND14_CALIBRATED_CONTINUATION_AGENT_PARITY_VERIFIED`。

10 条固定首场工作流全部通过，共核对 26 个候选、16 次初始/真实返回后概率、158 个模型数学分支。其中 140 个是首动作后的规划分支，18 个是 commit 真实首次返回后的即时估值边际化分支；两者都不计真实请求。实际发生 6 次历史揭示，总模拟费用 `.16`，218 个审计事件。10 份卡片文档共 26 张查询/终止卡；原第十一轮和第十三轮各五份卡片验证后引用，未再执行。

| 目标折 / 参考模型折 | adaptive | commit |
| --- | --- | --- |
| 2 / 0 | xG → STOP | xG → STOP |
| 3 / 1 | results → STOP | results → STOP |
| 4 / 2 | STOP | STOP |
| 5 / 3 | STOP | STOP |
| 6 / 4 | xG → STOP | xG → STOP |

每个候选的即时收益、后续奖励、计划累计收益/费用、原生分支及最终动作均与研究轨迹相符；每次真实后验、费用、版本/哈希和事件/blob 审计均通过。服务验证累计量恒等关系后，保留原 planner 的 `selection_score` 舍入结果，避免浮点重新组合在 12 位边界产生不同的择优顺序。

验收文件 SHA256 为 `bedbb0be7d4de762c356c946c6eabeac9c60f0904313a8f7483ff59573bca645`，绑定的全部源文件哈希已重新读取核对一致。新增专项 34 项通过，与冻结第九、十一、十二、十三轮相关专项合并为 118 项通过。

完整测试和验证中，两新策略均未改变对应单步策略的实际查询路径；没有第二次实际查询，也没有实际选中负即时净值的首查询。预算 `.05` 的 200 场测试中，两模式均查询 results 60 次、xG 92 次，Log loss 为 `1.121520518`，每场模拟费用 `.0198`，固定 `.5` 费用权重效用损失为 `1.131420518`，与原条件 mean 及 content-marginal mean 一致。零查询的效用损失为 `1.109338831`。

预定主比较 adaptive 减 commit 的逐场效用差均为零，描述性区间因此为 `[0,0]`。该结果反映这批实际到达路径没有产生两模式差异，不证明新场景等效，也没有支持规划带来实际收益提升。构造测试的执行能力、模型分支中的选择机会、真实到达状态和实现收益是四个不同层次；只有真实执行路径用于实际效用统计。

此轮仍使用同一个历史数据响应的 results/xG 两个视图及模拟费用，新增独立比赛、真实外部请求和 LLM 调用均为 0。数学分支不增加独立比赛数，假设返回不与原赛果配对作为反事实收益。资格、阵容、轮换和战术证据尚未接入这组研究；前瞻来源和真实开放工具价值仍需后续独立验证。
