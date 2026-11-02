**FootballMind 逐轮算法与实验原始记录索引**

生成时间：2026-10-07T15:17:22.449235+08:00。完成态截止 Round196，该轮于北京时间15:17完成验收后追加。

本附录按轮次提供原始说明入口、开头说明与结果边界摘录。摘录是各轮当时的历史记录，不是当前状态；同一比赛在多轮重复出现，不增加独立样本。算法与效果的综合解释见同目录主报告。

Round 编号包含工程接线、故障修复、数据采集和研究实验，196 个编号不等于 196 项独立效果实验。下列来源文档可能包含当时的后续计划，计划不算已实现能力。

**R1　真实运行与验证结果：2026-09-20**

原始记录：[results.md](<D:/a1-足彩/docs/research/baselines/results.md>)

最终验收目录：`var/parallel/baselines/real-asof-final/`。预测截止为 **2026-09-20 11:22:02 UTC / 北京时间19:22:02**。状态是 **FUNCTIONAL_VALIDATION**。

**R2　第二轮：三个现有任务并行接通真实数据链路**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round2/README.md>)

发布：2026-09-20。协调任务按用户指令核查三个既有任务后分发。本轮是当前执行基准；旧 COMMON/提示词中的八角色划分仅作历史背景，冲突处以本文件为准。保留已有有效实现和第一轮冻结证据。

当时的结果或限制原文摘录：

> - 来源原始观测、规范化版本、特征、标签和冻结预测分别留存；新观测不改写过去截止点的结果。
> - operational 仅使用系统当时实际可用的信息；historical_reconstruction 明示比赛结束/公布延迟等假设、来源事后修订风险及重建范围。历史样本可以支持假设明确的重建实验，结果与真实前瞻实验分别报告。
> - Understat 已有真实证据在 var/parallel/collectors/understat-research/。tests/fixtures 内为测试副本；正式导入读取有时间与哈希的原始研究档案，经显式登记保存来源链，不将夹具伪装成新生产观测。
> - Understat 单场 xG 与比分属于赛后数据；只能进入后续比赛的历史特征。来源 forecast 字段含义和生成时间未核实，隔离于赛前特征及市场基线。赛季总计也应按时点重算。
> - 真实外部请求、额度和数据库服务由平台任务统一管理。Agent 实测通过该服务发起少量有上限的调用；基线默认消费冻结导出。配置值脱敏，已有数据库和旧数据保留。
> - 第一轮的 real-asof-final、replay-final、acceptance 等证据保留。本轮新增产物使用 round2 目录。构造测试、真实档案、真实联网、PostgreSQL 和实际评分分别报告。

其他说明：[01-platform-collectors.md](<D:/a1-足彩/docs/parallel/round2/01-platform-collectors.md>)；[02-baselines-evaluation.md](<D:/a1-足彩/docs/parallel/round2/02-baselines-evaluation.md>)；[03-agent-integration.md](<D:/a1-足彩/docs/parallel/round2/03-agent-integration.md>)；[platform-interfaces.md](<D:/a1-足彩/docs/parallel/round2/platform-interfaces.md>)；[platform-runbook.md](<D:/a1-足彩/docs/parallel/round2/platform-runbook.md>)

**R3　第三轮：从工具工作流进入主动信息获取研究**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round3/README.md>)

发布时间：2026-09-22。依据用户明确的项目核心：Agent自己判断下一步取什么信息、是否值得以及何时停止。研究主线见 docs/research/ACTIVE-INFORMATION-RESEARCH.md。本轮覆盖此前任务书的下一步安排，保留前两轮成果和冻结记录。

当时的结果或限制原文摘录：

> 所有成本单位显式，离线模拟费用与在线实际费用分开；LLM/价值估计器自身成本计入。一个真实请求返回多个信息组时如实处理成本依赖。科学效果以研究冻结报告为准，演示单场轨迹只证明流程。

其他说明：[01-evidence-environment.md](<D:/a1-足彩/docs/parallel/round3/01-evidence-environment.md>)；[02-value-research.md](<D:/a1-足彩/docs/parallel/round3/02-value-research.md>)；[03-agent-information-policy.md](<D:/a1-足彩/docs/parallel/round3/03-agent-information-policy.md>)；[platform-interfaces.md](<D:/a1-足彩/docs/parallel/round3/platform-interfaces.md>)

**R4　第四轮：验证信息价值估计是否真正成立**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round4/README.md>)

发布时间：2026-09-22。第三轮已经完成“可见状态 → 候选动作 → 受控揭示 → 概率更新 → 价值停止”的第一版闭环。本轮目标从工程可运行转向算法研究验证，重点回答：**价值估计器能否在当前状态下可靠预测下一项证据的风险改善，并在成本、时效、来源依赖和估计不确定性存在时作出更好的停止决策？**

当时的结果或限制原文摘录：

> - `historical_reconstruction`：复用 Understat 2024/25 与 2025/26 COMPLETE 数据和明确延迟假设。已看过的 2025/26 290 场继续标为探索性；新的报告要区分训练、调参、探索性时间保留段。
> - `simulated_reveal`：在完整历史证据上按掩码揭示，成本来自冻结配置，属于受控机制实验；不能表述为真实供应商节省。
> - `live`：真实请求/刷新另记请求、延迟、费用和内容是否变化；生产成本未知时保持 unknown。没有有效新证据时不伪造 live 成功。

> 各策略共享同一比赛、同一基础状态、同一候选动作、同一成本表、同一预测器和同一随机种子集合。报告以比赛或时间块为统计单位，不能把同一比赛的多个掩码、提前量和策略轨迹当作独立比赛。

> - 改变未获取字段或未来标签，不会改变获取前状态、概率和动作。
> - 重复/过期/来源矛盾会改变可用性和价值，而不是被当作新信息。
> - 真实来源一次响应绑定多个组时，物理成本与模拟投影成本不会被重复计算。
> - LLM 即使提出了动作，最终获取与停止仍由合法动作和数值净价值门控。
> - 本轮策略若没有稳定改善，报告负结果并保留第三轮作为诚实基线。

其他说明：[round4-risk.md](<D:/a1-足彩/docs/research/acquisition/round4-risk.md>)；[01-evidence-reliability.md](<D:/a1-足彩/docs/parallel/round4/01-evidence-reliability.md>)；[02-risk_aware_voi.md](<D:/a1-足彩/docs/parallel/round4/02-risk_aware_voi.md>)；[03-llm_value_gate.md](<D:/a1-足彩/docs/parallel/round4/03-llm_value_gate.md>)；[platform-interfaces.md](<D:/a1-足彩/docs/parallel/round4/platform-interfaces.md>)

**R5　第五轮接续交付：信息价值估计、动态动作、Agent 对齐**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round5/README.md>)

日期：2026-09-26。状态：`ROUND5_VERIFIED_EXPLORATORY_NO_SUPERIORITY_CLAIM`。

当时的结果或限制原文摘录：

> - 最终跨模块回归：**556 passed，112.16 秒**。范围为 research/acquisition、platform/information、agent、demo。
> - 默认 pytest 导入模式曾被环境中安装的同名 `tests` 包遮蔽，第一次收集失败记录保留。最终使用 `--import-mode=importlib`，没有修改依赖或跳过该模块。
> - 两个正式实验各 **16,240 runs、290 场独立测试比赛、7 个时间折、21 个模型工件**。运行次数、动作数、不同 mask 和随机种子均不增加独立比赛数。
> - 单赛季：25,881 条可见状态检查、9,641 次模拟动作；双赛季：27,089 条可见状态检查、10,849 次模拟动作。重新计算分数、检查概率、预算、模型哈希与时序隔离均通过。
> - 动态环境：**10 个场景、18 个不同 action_id**，确定性重放；研究与 Agent 各消费 19 次状态转换；0 HTTP、0 数据库写入。
> - Agent 接线：每个时间折按开球时间/ID选首场，分别运行 mean/conservative，共 **14 条真实历史受控轨迹**。动作、模拟费用、最终概率与研究端逐条一致；审计链通过。未按效果挑选比赛。
> - 构造的组合价值测试验证了负的一步净值仍可启动有利的二步序列；该测试不证明足球数据中存在可学习的互补收益。

> risk-mean 相对 base-only 的平均 log-loss 差为 -0.020609，按 UTC 周配对 bootstrap 的 95% 区间为 [-0.049124, 0.006091]。相对 entropy 差为 +0.032182，区间 [-0.013131, 0.074814]。尚未证实优越性或非劣性。成本较低并不等于同质量节省成本。

其他说明：[round5-calibration.md](<D:/a1-足彩/docs/research/acquisition/round5-calibration.md>)

**R6　第六轮：比赛情境、条件获取价值与精确观测积分**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round6/README.md>)

最终状态：`ROUND6_VERIFIED_CONTEXTUAL_ACQUISITION_EXPLORATORY`。总验收：`var/parallel/round6/acceptance.json`。

当时的结果或限制原文摘录：

> 冻结验证目标 `log_loss + 0.5 × simulated_cost` 的纠错策略相对熵阈值差为-0.002459，区间[-0.031692,0.024627]；相对全部获取差-0.012618，区间[-0.041940,0.015212]。点估计体现费用与质量的取舍，区间不支持整体效用优越性。[固定效用比较](../../../var/parallel/baselines/round6/contextual-v4-identity-utility.json)。

其他说明：[round6-context-and-incentives.md](<D:/a1-足彩/docs/research/acquisition/round6-context-and-incentives.md>)

**R7　第七轮：同等免费信息下，主动查询还剩多少增量**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round7/README.md>)

日期：2026-09-26。本轮并行复用三个工作代理，分别处理研究实验、Agent 接线、真实杯赛证据；根代理独立检查新收益标签、策略轨迹、验证集选择及最终集成。没有创建新的用户侧聊天。

当时的结果或限制原文摘录：

> 本轮核心发现：所有策略的赛果预测器获得相同免费球队身份与赛程上下文后，当前条件信息价值策略相对不查询的 Log loss 差只有 **−0.001901**。周聚类区间跨零，预设成本效用的点估计为 **+0.004806**，未证明主动查询的独立收益。此结论限定于当前两个历史统计投影和重复探索的赛季，不否定真实阵容、晋级目标或动态消息的潜在价值。

其他说明：[round7-equal-context.md](<D:/a1-足彩/docs/research/acquisition/round7-equal-context.md>)

**R8　第八轮：查询前预测信息价值，并检验这种价值是否可信**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round8/README.md>)

日期：2026-09-26。复用研究、Agent、平台三个既有工作代理；根代理独立复算联合分布、分支规划、验证集选择、测试轨迹及失配诊断。本轮没有新增用户侧聊天。

当时的结果或限制原文摘录：

> **本轮已实现“先估计工具可能返回什么，再判断是否值得查询”的完整机制。但测试结果显示，当前返回模型高估了信息价值，主动查询劣于直接使用免费信息。算法效果按负结果封存。**

> - 二步相对不查询：Log loss差 **+0.036129**，95%区间 **[−0.010880, 0.082793]**；效用损失差 **+0.052595**，区间 **[0.004988, 0.100434]**。
> - 二步相对全部获取：效用损失差 **−0.024180**，区间 **[−0.042323, −0.007712]**。这只说明当前失配模型下少获取一些信息减少损害，不代表优于停止。
> - 二步相对一步：效用损失差 **−0.002048**，区间 **[−0.007664, 0.004202]**。尚未建立加深规划的收益。

其他说明：[round8-learned-joint.md](<D:/a1-足彩/docs/research/acquisition/round8-learned-joint.md>)；[round8-probabilistic-consistency.md](<D:/a1-足彩/docs/research/acquisition/round8-probabilistic-consistency.md>)

**R9　第九轮：校准信息获取收益与停止决策**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round9/README.md>)

日期：2026-09-26。复用三个既有工作代理，根代理独立重算；本轮未创建新的用户侧聊天。总验收文件为 `var/parallel/round9/acceptance.json`。

当时的结果或限制原文摘录：

> **本轮把“模型认为有信息”推进为“过去类似查询是否真的帮助这套预测器”。均值校准减少了原始 EVSI 策略的过度查询，但计入成本后仍未建立优于零查询的证据；经验保守策略全部停止。实现通过验收，效果保留为探索性结果。**

> - 760场输入，7个时间折，仍为已探索的290场测试比赛。新增独立测试比赛为0。
> - 23个新前向 OOF 联合返回模型，690个目标比赛、2,760条新转换；每场4个合法单步转换不计为4场独立比赛。
> - 最早40场没有前向 OOF 基础预测；其后的首个30场目标块没有更早可用的联合返回训练样本，显式跳过。没有复用第七轮预测器的旧增量收益标签。
> - 每个外层训练段的 OOF 比赛按时间取前80%拟合，后20%只诊断；相同开球时刻一起划分，诊断段首次预测时点之后才可用的训练标签被移出拟合段。
> - 拟合/诊断独立比赛数依次为292/74、320/80、340/89、390/99、430/110、460/117、490/125。跨折重复不计新样本。
> - 校准参数固定；仅成本系数在原外层验证段、原网格中选择。尾段诊断不拟合残差、不选成本系数。
> - 9种策略、4个预算，随机策略每场5个种子，其余每场1次，共15,080条轨迹、60个模型工件。
> - 第八轮12,760条旧策略轨迹、除计算延迟外的所有结果字段及验证集选择全部精确重现，外层联合预测器字节一致。

> - 均值校准相对原始一步：效用损失差 **−0.051716**，描述性95%区间 **[−0.095763, −0.009567]**。
> - 均值校准相对零查询：Log loss差 **−0.003005**，区间 **[−0.021727, 0.017756]**；效用损失差 **+0.002926**，区间 **[−0.015369, 0.023826]**。
> - 均值校准162场初始停止、128场查询一次：results40次、xg88次。没有查询两次的比赛。
> - 下界策略在所有预算全部停止，逐场等于免费信息预测。它证明停止机制可执行，没有证明主动获取优越性。

其他说明：[round9-residual-calibration.md](<D:/a1-足彩/docs/research/acquisition/round9-residual-calibration.md>)；[round9-value-calibration-rationale.md](<D:/a1-足彩/docs/research/acquisition/round9-value-calibration-rationale.md>)

**R10　第十轮：查询边界、成本标尺与校准支持**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round10/README.md>)

日期：2026-09-26。根代理与研究、Agent两个既有工作代理并行完成。本轮未启动采集分支，没有新增用户侧聊天、外部请求、拟合模型或独立比赛。

当时的结果或限制原文摘录：

> - 同一批290场已探索测试比赛，7折、4个物理预算。
> - mean与经验LCB两种固定成本策略，共2,320条测试轨迹。
> - 在原验证分区重算2,184条轨迹，与上一轮已记录的lambda=0.5候选效用完全一致。
> - 新独立比赛0，新拟合模型0，真实外部调用0。
> - 零查询、原校准均值、原校准LCB读取已冻结工件；没有重新训练基线。

> 把当前更新收益写成 **g*(H,a;ψ)**：它同时依赖已知信息H、查询动作a和实际预测更新器版本ψ。旧OOF模型产生的收益标签，即使匹配相同动作和EVSI桶，也尚未证明可直接用于后来版本。

> 根代理独立验收包括：2,320条新测试轨迹、2,184条新验证轨迹、2,674个测试公开事件、354笔跨预算查询收费、3,053个可负担候选、852个独立联合公开状态；4,640条新旧策略核算、663条跨预算选中查询分解、5,800条初始候选分解。以上重复记录不计新独立比赛。

其他说明：[round10-design.md](<D:/a1-足彩/docs/research/acquisition/round10-design.md>)；[round10-support-audit.md](<D:/a1-足彩/docs/research/acquisition/round10-support-audit.md>)

**R11　第十一轮：固定预测器版本下的自主信息选择**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round11/README.md>)

日期：2026-09-26。状态：`ROUND11_VERSION_BOUND_ACQUISITION_VERIFIED_NO_CONFIRMED_BENEFIT`。

当时的结果或限制原文摘录：

> **本轮已经完成自主查询策略的对照实验和实际 Agent 工作流验收。它验证了选择工具、获取历史证据、更新概率、重新估值并停止的执行机制，但没有证明“同版本收益校准”提升策略收益。负结果和全部预定预算均已保留。**

> 同一条信息对不同预测器的更新效果可能不同。第十轮发现，原收益标签由多个旧 OOF 模型生成，而部署时使用后续模型；这一差异值得控制，但当时没有证明它是失效原因。

> 同时需要保留一个限制：同版本组第 2 折 50 场和第 6 折 41 场全部先查 xG，第 3 折 42 场中 41 场先查 results。其他折存在更明显的逐场 STOP 差异。整体动作多样性部分来自折间模型和校准差异，尚未证明精细逐场判断优于“当前时段固定偏好某类工具”。这一点直接影响项目关于自主信息选择的研究主张。

其他说明：[round11-design.md](<D:/a1-足彩/docs/research/acquisition/round11-design.md>)；[round11-frozen-version.md](<D:/a1-足彩/docs/research/acquisition/round11-frozen-version.md>)

**R12　第十二轮：自主信息选择相对历史均值的增量价值**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round12/README.md>)

日期：2026-09-26。状态：`ROUND12_STATE_ABLATION_VERIFIED_NO_CONDITIONAL_ADVANTAGE_ESTABLISHED`。

当时的结果或限制原文摘录：

> 新增 1,600 条测试、1,704 条验证回放；涉及 200 场既有测试比赛和 213 场验证比赛。新增 10 个均值估计工件，赛果预测器没有重新拟合。旧测试回放 10,400 条与新增回放合计 12,000 条，仅用于完整比较，不构成新的独立比赛。

> 当前没有建立条件估值优越性。区间跨零也不表示等效性已被证明。工具均值点估计较低，同样未建立相对零查询的收益优势：工具均值减零查询为 +0.013118，区间 [−0.012953, +0.052919]；阶段均值减零查询为 +0.020852，区间 [−0.016905, +0.064198]。

> 这种结果区分了两个事实：条件策略使用了当前状态；当前状态引起的额外动作变化还未表现出明确效用增益。它没有证明状态信息在所有任务中无效，也没有证明未来更合适的条件表示一定无效。

其他说明：[round12-design.md](<D:/a1-足彩/docs/research/acquisition/round12-design.md>)；[round12-state-ablation.md](<D:/a1-足彩/docs/research/acquisition/round12-state-ablation.md>)

**R13　第十三轮：返回内容是否改变自主查询与停止**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round13/README.md>)

日期：2026-09-26。状态：`ROUND13_CONTENT_SENSITIVITY_VERIFIED_NO_REALIZED_POLICY_DIFFERENCE`。

当时的结果或限制原文摘录：

> 这里先对每个分支应用原 EVSI 分桶残差校准，再求平均。将平均 EVSI 送入校准器是另一个估计量，本轮没有这样实现。缺失摘要也是完整类别；任一正概率分支缺乏支持时，整个候选保留为不支持，不丢弃该分支后重新归一化。

> 400 个上下文来自 200 场比赛，每场两个可能的首查工具；它们不构成 400 场独立比赛。38 个真实类别下的策略差异全在原策略没有访问的首查路径上。122 个会发生类别决策分歧的上下文中，121 个对应先查 results，1 个对应先查 xG。

> [Agent 验收](D:/a1-足彩/var/parallel/agent/round13/content-marginal-bridge-v1/acceptance.json)；[根代理独立复验](D:/a1-足彩/var/parallel/round13/agent-card-independent-verification.json)。新增独立比赛、外部 HTTP 请求和 LLM 调用均为 0。

其他说明：[round13-content-marginal.md](<D:/a1-足彩/docs/research/acquisition/round13-content-marginal.md>)；[round13-design.md](<D:/a1-足彩/docs/research/acquisition/round13-design.md>)

**R14　第十四轮：第一次查询是否值得购买后续选择机会**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round14/README.md>)

本轮已完成两步自主信息获取策略、Agent 接入、冻结历史回放和根代理独立复算。结论是：在当前冻结模型及既有样本上，后续查询机会提高了部分候选的数值评分，实际首查询、停止、预测与收益仍与对应的一步策略相同。尚未建立自主获取的收益优势。

当时的结果或限制原文摘录：

> 三列均越低越好。两个新策略主预算均首查 results 60 场、xG 92 场、直接 STOP 48 场；152 次实际首查之后均 STOP。其余预算和验证集也与对应的一步策略路径一致。自适应减 commit、自适应减原条件一步、commit 减原内容平均一步的逐场差均为 0，周分块 bootstrap 区间为 [0,0]；这只反映本次冻结回放逐场相同，未证明总体或新比赛上的策略等效。

> 自适应两步相对零查询的综合损失差为 +0.022082，描述性 95% 周分块区间为 [−0.011886, +0.064847]。目前点估计更差，区间跨零；结果未建立收益改进，也未建立总体劣势。所有区间继续使用原 2000 次、seed=1729 的设置，未覆盖反复研究、模型拟合、多个比较和时间漂移的不确定性。

> 本轮范围是数值驱动的有限工具自主获取研究，新增独立比赛、网络请求、LLM 调用均为 0。历史重建、同源工具投影及模拟成本不等同于真实开放信息环境部署。最终可交付的核心仍应是：Agent 能说明下一步为何查、查到不同内容为何改变下一步，以及停止后实际决策净收益是否更好。

其他说明：[round14-calibrated-continuation.md](<D:/a1-足彩/docs/research/acquisition/round14-calibrated-continuation.md>)；[round14-design.md](<D:/a1-足彩/docs/research/acquisition/round14-design.md>)

**R15　第十五轮：让实际返回决定下一步查什么**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round15/README.md>)

本轮完成了条件信息基准及完整 Agent 执行。预先固定的已知分布中，Agent 会购买即时预测价值为零的情境线索，再按它的实际返回选择不同工具或停止；重复信息和无关信息触发停止。独立枚举全部有限策略树、全部真实生成世界及 40 种返回组合后，现有规划器和实际执行记录一致。

当时的结果或限制原文摘录：

> **下一轮的必要工作。** 条件信息的选择与停止机制已有覆盖完整分支的构造证据。下一轮优先回到真实条件收益估计：把反复使用的历史样本标为开发诊断数据，明确固定预测器下的查询前状态、返回、更新损失和来源时点，评价估计值对实际增量收益的适配；同时确定一个可取得真实时点证据的最小情境链。未来的独立比赛用于确认收益，当前构造概率不迁移成真实球队规则。

其他说明：[round15-design.md](<D:/a1-足彩/docs/research/acquisition/round15-design.md>)；[round15-finite-information.md](<D:/a1-足彩/docs/research/acquisition/round15-finite-information.md>)

**R16　第十六轮：真实 LLM 根据开放网页证据重新决定下一步调查。**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round16/README.md>)

本轮落实 2026-09-27 项目复盘确定的方向，新增真实自主调查入口，没有增加历史胜平负评分实验。正式轨迹发生在 2026-09-26 16:55—16:59 UTC，即北京时间 9 月 27 日 00:55—00:59，早于任务截止时间 18:45 UTC。

当时的结果或限制原文摘录：

> 正式运行共 **5 次完成的真实 CLI 模型调用、4 次项目外部 HTTP 尝试，其中 3 次 HTTP 200、1 次 ConnectionError**。模型服务内部的网络请求数未暴露，货币费用未报告。25 个持久化事件及全部原文、模型公开输入和结构化输出可以重放。

> 该轨迹证明已接通真实语义提问、开放页面获取、证据引用和根据实际返回重规划。它没有证明获取顺序最优，也没有建立收益优势。尤其最终暂缓由证据不足与预算边界共同形成，不是数值 VoI 已确定非正。

> **接入故障也保留。** 正式运行之前有三次 CLI 接入尝试，均未形成有效调查决策、没有请求足球来源。第三次保留的服务错误明确指出输出 Schema 不支持 `uniqueItems`；前两次缺少完整服务失败日志，保留失败/结果未知记录。修正了服务 Schema 子集以及错误事件分类，并保留本地完整校验。全部尝试分别在 `open-inquiry-v1` 至 `open-inquiry-v3`，未覆盖旧账本。完整本轮为 8 次 CLI 调用尝试、5 次有效结构化决策，调用尝试也不等于计费或推理成功。

**R17　第十七轮：让实际返回改变待核实假设，并记录每次查询原本想解决什么。**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round17/README.md>)

本轮实现并运行了 `hypothesis-inquiry-v1`。Agent 自主提出具体假设，为每次候选查询说明支持、反向和无答案三种返回会怎样影响下一步；取得文档后，程序根据带引文的来源表述及其类型、主体和适用范围计算证据状态，并保存该请求的实际反馈。核心实现见 [inquiry_hypotheses.py](../../../football_agent/inquiry_hypotheses.py)，接口与边界见 [hypothesis-inquiry.md](../../agent/hypothesis-inquiry.md)。

当时的结果或限制原文摘录：

> 本轮真实运行 `05c2f62a-aed4-44d2-bd30-5e19dc546a99` 是上一轮英格兰—西班牙案例的明确续查。初始赛程及种子链接引用原运行和原观测，继承的赛程引文已与原正文校对；三条文章链接仅作为发现线索，正文开始时未读。它不是新增独立比赛，也不是策略收益测试样本。任务预先固定 4 次 HTTP 尝试、5 次模型调用，全部发生在 2026-09-26 **17:25:45—17:29:50 UTC**，早于 18:45 UTC 截止时间。

> 实际共 **5 次成功完成的 CLI 模型调用、4 次 HTTP 尝试**：两份可用正文、一份 HTTP 200 的空搜索结果、一份连接失败。形成 3 项假设、5 条带引文的来源表述、4 份查询反馈。反馈分别为背景信息、无可用返回、获取失败、背景信息。没有在预算之外补抓网页，也没有补跑有利结果。实际货币费用及响应模型标识未由服务报告；调用次数不等于服务内部 HTTP 次数。

> - **数值 VoI 尚未接通。** `expected_gain / selection_score / counterfactual_gain / prediction` 仍为空，真实查询按 `llm_bounded_exploration` 运行。查询反馈只是后续估计返回分布与状态变化的过程数据，不是预测风险下降或反事实效用标签。
> - **语义与时效仍待核验。** 引文出现在正文中、字段类型兼容，不保证 LLM 对主体、范围、证据类型或支持关系的解释正确；来源发布时间和来源独立性也未自动验证。
> - **跨范围证据组合仍偏保守。** 本次已经取得赛事级规则，但规则尚未经过明确的适用关系与本场、积分状态组合，因此仍保留为背景，资格假设的直接支持类型仍显示缺口。下一步需要可审查的“赛事规则适用于该场次”等关系与组合更新，不能仅通过扩大字段匹配来制造确认。
> - **主动停止的经济性尚未证实。** 此次暂缓主要受资源边界及证据不足影响，未证明剩余查询的价值低于成本。也尚未判断为何读赛事说明比读剩余集训名单更值得；目前该选择只有可查的定性理由。

**R18　第十八轮：组合已取得的证据，让计算后的信息缺口进入下一次实际查询。**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round18/README.md>)

本轮解决上一轮真实运行暴露的问题：已有赛事级规则因范围与比赛级假设不同，被长期留在背景层。新增 `applicability-inquiry-v1`，用带引文的适用关系把规则作为本场推理的前提，同时保留资格结论未知。Agent 通过一次 `reflect` 读取程序计算后的前提状态，再选择外部查询。它可以复用已经核验归档的旧正文，原始观察时间保持不变。

当时的结果或限制原文摘录：

> 第二次决策的实际候选反馈为：规则已存在，积分是可能补齐的缺口，若返回相关、带时间锚点且无冲突的内容，可补齐当前声明的输入。Agent 的公开理由明确引用了这个 `reflect` 结果。本次轨迹因此验证了“证据组合 → 计算缺口 → 模型读取 → 实际查询”的执行链。它并未证明补查积分比查阵容更有价值，亦未将首轮赛程直接等同于完整的资格推理。

> 运行发生于 **2026-09-26 17:49:24—17:52:20 UTC**，早于原 18:45 UTC 截止。共有 **4 次完成的真实模型调用、2 次新增 HTTP 尝试、2 份继承正文、20 个事件**。两次新增请求均为 HTTP 200，其中搜索无可用结果，另一份为新闻索引。整理得到 2 项假设、4 条带引文的来源表述、1 条适用关系和 2 份查询反馈。没有重新请求继承的两篇文章，也没有在预定预算之外补跑结果。调用费用与响应模型标识继续按服务实际暴露情况保存，未填造值。

> 引文锚点和类型检查也不独立验证语义与时效。比如哪届规则适用哪场、积分有效时间是否解释正确，仍依赖提取质量。当前前提列表只是声明的最小输入，完整资格推理还可能需要剩余赛程、同分规则等。实际停止由预算及证据不足触发，未证明继续查询的净价值为负。

**R19　第十九轮：把“这条信息可能有用”变成查询前声明、查询后可检查的分支预期。**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round19/README.md>)

本轮实现 `return-forecast-inquiry-v1`。Agent 为每个候选查询预先说明不同返回下预计还缺什么、下一步准备查什么或暂缓；实际请求之后，程序检查这些预期与提取状态、后续行动提议是否一致。该机制进入真实 LLM 与外部 HTTP 的执行链，保留失败和计划偏差，作为开放信息获取研究的过程观测。

当时的结果或限制原文摘录：

> 第一次反馈后的第二次查询确有 `inquiry_acquisition_intent` 与实际返回，并非只有提议。第二次反馈后的暂缓也有终止事件。协议一般比较的是下一次提议，执行事实在导出的观测记录中另列，以避免把预算拦截后的提议计为实际调用。

> 正式运行 `a9567b3f-47d7-4401-a056-270b51f1da8b`，发生于 **2026-09-26 18:17:37—18:20:38 UTC**，早于原截止。共有 **3 次完成的真实模型调用、2 次新增 HTTP 尝试、3 份继承文档、16 个事件**，提出 1 项假设、整理 4 条继承来源表述、留下 14 份事前分支计划与 2 条实际返回反馈。两次新增请求都失败；14 个计划分支不等于 14 次实验，只有 2 条观测，且都属于失败类别。本次也没有增加独立比赛样本。

**R20　第二十轮：反向答案提供信息，来源冲突单独进入下一步查询判断。**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round20/README.md>)

本轮修正第十九轮发现的状态问题。旧前提投影将每条反向表述计为 `contested`，再把它列入 `missing_types`。即使来源明确表示某球员没有首发，也可能继续显示“首发信息缺失”。这会向后续查询选择与价值模型提供错误的状态解释。

当时的结果或限制原文摘录：

> **结果没有建立效率优势。** 旧协议在反向对照中也选择了伤停，模型已经能借助正文保留负向事实，未被旧缺口字段完全误导。因此这组结果支持状态解释得到修正、新状态可被真实模型使用；不支持“少查一次”或“新策略更优”的结论。只有一组旧/新反向配对，而且提示与投影一起改变，也不构成单一字段的因果效应识别。

**R21　第二十一轮：决策价值进入 Agent 的实际信息获取**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round21/README.md>)

**本轮交付：LLM 提出查询候选后，程序按明确的终止决策损失比较其信息价值，选择实际工具；实际返回更新信念，再决定继续或停止。** 重点从证据状态描述推进到了行动选择，未新增预测模型或追逐胜平负分数。

**R22　第二十二轮：把真实查询历史整理成返回模型研究数据**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round22/README.md>)

本轮检查了现有证据是否足以学习“给定任务状态，下一次查询可能返回什么”。结论是：已具备可重放的获取记录，但样本集中在同一场比赛，且未登记选择概率，不适合直接拟合并宣称已校准的开放文本返回模型。

当时的结果或限制原文摘录：

> 两项工作并行推进：一是实现多个声明模型共同评价同一个可执行策略，量化查询选择依赖哪些假设；二是为真实查询加入事前登记的有限探索与条件选择概率。没有把单场的 12 次请求伪装成独立大样本，也没有从 HTTP 成功率推出足球决策收益。

其他说明：[prospective-query-design.md](<D:/a1-足彩/docs/parallel/round22/prospective-query-design.md>)；[return-evidence-audit.md](<D:/a1-足彩/docs/parallel/round22/return-evidence-audit.md>)；[uncertain-value-design.md](<D:/a1-足彩/docs/parallel/round22/uncertain-value-design.md>)

**R23　第二十三轮：返回模型不确定时，评价并执行共同查询策略**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round23/README.md>)

本轮将 Agent 的数值获取依据从单一声明模型扩展到有限模型集合。关键约束是所有模型评价同一个实际可执行的“查询—返回—终止决策”策略，而非给每个模型分配知道其真实身份的答案。模型集合属于声明假设，未赋予统计置信覆盖意义。

当时的结果或限制原文摘录：

> 目录必须全新，以保留已有实验。脚本提供器与真实模型提议分开计数，工具调用也明确区分构造传输和外部请求。

**R24　第二十四轮：为未知查询记录有范围的探索概率**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round24/README.md>)

本轮将预先声明的有限探索接入实际 Agent 工作流，并把选择设计登记到后续学习数据中。它解决的是“将来如何知道实际选择了什么、当时还有什么可选、样本怎样产生”，没有把随机探索等同于正信息价值，也没有用连接成功率代替决策收益。

当时的结果或限制原文摘录：

> 共 3 次完成的真实 CLI 调用、2 次真实来源尝试，未使用剩余的一次来源请求和一次模型调用额度。新增两条物理获取失败观察，新增足球事实为零。两次失败没有被解释为首发尚未公布或球员没有首发；暂停也不表示所有剩余信息价值为零。

> 旧 12 条选择概率保持未知，新两条记录条件概率 1。生成概率、独立真实标签、反事实收益及实际决策收益保持未知。截止前状态不读取本次后续返回；未选择候选仍没有返回标签；继承材料不会再算为新请求。详见 [v2 数据设计](return-dataset-design.md)。

> 1. 先取得可独立核验的实际任务正文与相应真值标签，区分工具可达、内容适用、来源正确和终止决策收益。新目标仍处于早期查询窗口，时钟到达不等于已获得首发；本轮未建立自动唤醒任务。
> 2. 只在确有不同且合理的路径时比较候选，记录“为何提出此查询、它可能改变哪项决定”。不添加无关候选制造随机比较，不将单候选 p=1 当成探索覆盖。
> 3. 按来源、任务与发布时间描述返回规律及依赖，随着多任务的前瞻证据积累，评估模型集合是否覆盖真实返回。当前 14 次自适应尝试还不足以为任意新闻估计可靠似然。
> 4. 有了受支持的返回模型后，再比较相同初始状态、截止和成本下的 Agent 策略；区分选择候选的能力、风险估值的能力和停止的能力。多步互补、时间演化及探索/利用组合需要新的政策类和实验，不由本轮一步保证自动外推。

其他说明：[exploration-design.md](<D:/a1-足彩/docs/parallel/round24/exploration-design.md>)；[live-inquiry-audit.md](<D:/a1-足彩/docs/parallel/round24/live-inquiry-audit.md>)；[return-dataset-design.md](<D:/a1-足彩/docs/parallel/round24/return-dataset-design.md>)

**R25　第二十五轮：让行动取决于目标、证据适用范围与可得性**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round25/README.md>)

本轮围绕 FootballMind 的核心问题：**给定当前任务和已知材料，Agent 是否还需要信息，需要哪种信息，什么时候可以结束当前任务。** 三条并行工作分别完成来源诊断、两步条件策略设计、实验独立审查；主线程实现并运行真实模型的成对提议实验。没有向用户指定的历史 Codex 对话发送任务，本轮使用现有子 Agent 协作。

当时的结果或限制原文摘录：

> 仍有一项重要缺口：首发案例知道当前早于设定比赛日期，却提出立即搜索；本轮未比较现在查询、等待信息发布及暂缓交付的价值。每例只允许一通模型调用，因此该候选七种返回分支的后续动作全部写为 `defer`，其中发现链接分支明确说没有剩余模型调用来规划读取。这个诊断预算限制与现有协议会影响后续计划的表达，不能把全部暂缓都归因于真实任务中没有有价值的继续动作；相同动作标签下拟议的答案内容仍有差别，也不等于信息价值为零。搜索摘要在当前工具协议下只作发现，也没有自动升级成首发证据。后续应把获取、解析与后续调用所需的总预算，以及来源发布时机放进可执行的条件计划；本次结果尚未证明“选中的查询现在值得付出成本”。

> 正文原始观察时刻为 `2026-09-26T17:26:25.577503+00:00`。父运行重放核对原始响应、正文提取、时间及头哈希，继承没有刷新观察时刻。页面自报发布时间与独立核验发布时间分开处理。一次归档正文复用于不同输入条件，不增加独立比赛样本或足球观测数。

其他说明：[contingent-policy-design.md](<D:/a1-足彩/docs/parallel/round25/contingent-policy-design.md>)；[relevance-probe-review.md](<D:/a1-足彩/docs/parallel/round25/relevance-probe-review.md>)；[source-recovery.md](<D:/a1-足彩/docs/parallel/round25/source-recovery.md>)

**R26　第二十六轮：为实际信息获取保留后续处理能力**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round26/README.md>)

本轮围绕“Agent 自己决定下一步获取什么、取得后如何更新、何时暂缓”完成执行链升级，并运行了一条真实调查。来源接口、预算机制和独立审查并行推进。**真实运行从取得赛程正文、提出资格前提假设，到处理搜索失败并自主暂缓，已经形成实际获取与重规划的闭环；业务问题仍保留未知。** 数值 VoI、等待收益和足球结果预测收益均未建立。

当时的结果或限制原文摘录：

> 最终共有3次完成的真实 CLI 回合、2次物理来源请求、1份新增可用文档、1条带引文来源陈述、3项调查假设、2条查询反馈和2条返回计划复核。候选数为1、1、0，未进行真实多候选收益比较。三次预算准入均通过，最后的 DEFER 来自真实模型回复，程序没有覆盖其选择。

> ### 截止与真实请求边界

> 如果解释或状态处理期间跨过截止，产生明确的运行性停止。来源请求登记前及实际工具调用前再次检查时间；登记后到时则记为0 HTTP的取消并释放预留。导出时排除截至当时已经确认取消的请求；更早截点仍把未知结果的意图记为censored，避免提前使用未来取消信息。

其他说明：[continuation-design-review.md](<D:/a1-足彩/docs/parallel/round26/continuation-design-review.md>)；[continuation-review.md](<D:/a1-足彩/docs/parallel/round26/continuation-review.md>)；[transport.md](<D:/a1-足彩/docs/parallel/round26/transport.md>)

**R27　第二十七轮：同状态多候选信息路径比较**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round27/README.md>)

本轮重点是 Agent 自己提出多个下一步信息路径，在当前证据与资源条件下比较并选择，然后处理真实返回。新增 `information-choice-inquiry-v1`，保留原证据状态及返回计划机制；数值 VoI、收益概率和期望完整成本仍未校准。

当时的结果或限制原文摘录：

> 实际运行时间为 **2026-09-27 01:37:04—01:44:29 UTC**。最终为DEFERRED，共5次完成的真实CLI回合、4次来源请求、26个事件，候选数依次为2、2、2、1、0。五次准入均通过，程序未替换原始提议。实际行为为：

其他说明：[choice-design-review.md](<D:/a1-足彩/docs/parallel/round27/choice-design-review.md>)；[choice-review.md](<D:/a1-足彩/docs/parallel/round27/choice-review.md>)；[source-coverage.md](<D:/a1-足彩/docs/parallel/round27/source-coverage.md>)

**R28　第二十八轮：查询前返回预期、拒绝轨迹与真实适用关系瓶颈**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round28/README.md>)

核心仍是 Agent 决定下一步获取什么信息、如何使用返回、何时继续或暂缓。本轮增加可核对的主观返回概率契约，并在真实续查中发现了一个比拟合概率更紧迫的问题：取得通用联赛规则后，当前结构关系尚不能把它用于参赛球队的单场资格问题。该次调查因响应校验拒绝而中断，没有完成真实的概率评分闭环。

当时的结果或限制原文摘录：

> 实际时间为 **2026-09-27 02:17:44—02:21:22 UTC**，共2次完成的真实CLI回合、1次来源请求、11个事件。过程如下：

> 1. Agent 根据继承正文提出资格确定与本场轮换意图两个问题。它比较了已发现的官方大赛规程链接与教练定向搜索，两条路径不同且均满足当时的最低调用数。选择规程，说明应先核清资格判断条件。程序未改写此选择。
> 2. 两条候选的返回概率均为 abstain；没有数值分布。所选入口 `https://www.kleague.com/about/competition.do` 来自继承积分页中实际取得的链接，而非臆造路径。
> 3. 唯一真实请求返回 HTTP 200，原始响应1,222,541字节，公开文本上限18,000字符且标记截断。正文包含2026赛季规程。Agent 在第二通回复中提取相关规则，引用4处条款，另以3处见证解释赛事、双方参赛及目标比赛的联系，并拟下一步搜索有效积分。
> 4. 原验证器拒绝整份第二回复，原因是 `Unsupported applicability scope, subject or evidence type`。规则陈述主体为 `KLeague1`，资格假设主体为 `GangwonFC`；旧适用关系要求两者 subject 相同。第二回复中的规则陈述、binding 和积分查询都未进入已提交状态，也未执行第二次来源请求。
> 5. 终态为 `DEFERRED / INVALID_INQUIRY_RESPONSE`，仍余2次来源请求和2次模型调用。它由校验错误触发，不是 Agent 主动停止、预算耗尽或“剩余信息不值得获取”的证据。

其他说明：[forecast-design-review.md](<D:/a1-足彩/docs/parallel/round28/forecast-design-review.md>)；[probability-review.md](<D:/a1-足彩/docs/parallel/round28/probability-review.md>)

**R29　第二十九轮：把已有规则用于当前问题，再由 Agent 选择信息**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round29/README.md>)

本轮解决第28轮真实调查暴露的阻断：联赛规则的主体是赛事，本场资格问题的主体是球队，两者需要有来源依据的适用关系。现在 Agent 可以用已取得正文建立赛事、赛季、球队与比赛的联系，把规则作为当前推理输入，再自行选择下一条信息。旧的主体一致校验和第28轮失败记录保持原样。

当时的结果或限制原文摘录：

> 真实调查完成了 `reflect → query → defer`：Agent 从四份继承正文中提取规则与比赛背景，建立一条适用关系；随后比较有效积分与教练轮换表态两条搜索路径，选择积分。搜索返回 HTTP 302、没有正文，Agent 在尚余请求额度时暂缓。资格和轮换结论仍未知，数值信息价值尚未估计。

> 实际执行时间 **2026-09-27 02:52:52—02:57:20 UTC**，3次完成的真实CLI回合、1次物理来源请求、15个事件。CLI回合内提供器的底层请求数未报告，不把一次CLI等同于一次底层API请求。

> 终态为 **`DEFERRED / LLM_EVIDENCE_INSUFFICIENT`**，剩余2次来源请求、1次模型回合。所有决定通过准入，没有响应拒绝、预算强制停止或人工覆盖。它的解释是剩余模型回合不够完成新的搜索、读原文、解释返回链条。这个解释对发现后续读路径符合计数下界；某些已知URL的一步读取仍可能满足计数要求，因此并未证明所有可选行动都不值得执行。`early_stopping_with_unused_budget_observed=true` 仅描述未用尽额度时的模型暂缓，不表示最优停止或低VoI已获证明。

其他说明：[rule-scope-design-review.md](<D:/a1-足彩/docs/parallel/round29/rule-scope-design-review.md>)；[rule-scope-review.md](<D:/a1-足彩/docs/parallel/round29/rule-scope-review.md>)

**R30　第三十轮：围绕固定终端问题选择信息，并核对真实返回的影响**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round30/README.md>)

本轮将真实开放查询与一个明确的来源报告决策相连：事前固定要回答的命题，Agent 说明候选返回在什么条件下会改变答案，取得材料后再核对实际来源立场是否改变，以及新增证据来自哪里。候选获取继续由 Agent 选择，没有按字段覆盖率或返回预测分数排序。

当时的结果或限制原文摘录：

> 合计 **3次研究诊断HTTP、0次模型调用、0条新增比赛事实**。这些是研究人员配置的诊断，不是Agent自主选择的动作，也不进入策略收益或独立比赛样本。既有官方页面已暴露新闻入口及详情链接，正式Agent仍可按任务从这些真实链接选择读取。

> 实际运行 `f3bd9d5d-4575-4eed-a4c5-f55e346b89bc`，时间 **2026-09-27 03:55:37—03:58:08 UTC**，事件头 `072adc2499a96efee0ec156af2188d0c8d22d5f54af57a6008c56656b291f019`。共2次完成的真实CLI回合、1次物理来源请求、11个事件，无结果驱动的运行重试或运行后产品修改。第二次CLI内部有两条流重连记录，之后正常完成；它们不记为新的Agent回合，底层服务请求总数仍未知。

> 终态 **DEFERRED / LLM_EVIDENCE_INSUFFICIENT**，仍余 **3次来源请求、3次模型回合**。这次额度足以再完成一条发现、阅读、解释路径，退出不是预算或协议拒绝。它展示了模型在看到具体返回后自主暂缓，却没有证明已穷尽值得查的信息。首次候选数为1、最后为0，本轮也没有产生同状态多候选收益排序证据。

其他说明：[decision-impact-review.md](<D:/a1-足彩/docs/parallel/round30/decision-impact-review.md>)；[decision-relevance-design.md](<D:/a1-足彩/docs/parallel/round30/decision-relevance-design.md>)；[source-discovery.md](<D:/a1-足彩/docs/parallel/round30/source-discovery.md>)

**R31　第三十一轮：独立检查 Agent 暂缓后的未读路径**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round31/README.md>)

本轮把第30轮的一项未验证判断变成实际可检查的对象：Agent 认为新新闻与旧活动稿同题，暂缓继续取证；我们事前固定两条未读详情路径，随后独立读取，检查是否存在会改变原来源报告的新内容。原 Agent 运行保持封存。

其他说明：[content-comparison.md](<D:/a1-足彩/docs/parallel/round31/content-comparison.md>)；[source-inventory.md](<D:/a1-足彩/docs/parallel/round31/source-inventory.md>)；[stop-evaluation-design.md](<D:/a1-足彩/docs/parallel/round31/stop-evaluation-design.md>)；[stop-evaluation-review.md](<D:/a1-足彩/docs/parallel/round31/stop-evaluation-review.md>)

**R32　第三十二轮：事前冻结候选正文，检验 Agent 的有限范围自主选源**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round32/README.md>)

本轮把前几轮的开放式“下一步查什么”推进到一个可复现实验：在同一个固定来源报告问题上，先登记三条候选公开路径并在 Agent 调用前分别冻结正文；Agent只看到来源角色和URL，正文对其保持隐藏，只能读取一条或暂缓。选择之后，独立评价器才同时查看三个快照，分别计算每条路径对终态来源报告的影响。

当时的结果或限制原文摘录：

> 本轮不支持以下结论：

> - 没有证明 Agent 估计了数值 VoI、查询成本或当前风险；
> - 没有证明 `direct-player-report` 在开放网络中是全局最优来源；
> - 没有证明停止规则已经学会，或者剩余所有信息都没有价值；
> - 没有证明足球比赛真实世界的首发真值，更没有证明胜平负预测收益；
> - 三条候选由研究人员事前策展，尚不足以估计所有可见 URL 的选择分布；
> - 预取成本与 Agent 成本分开记录，因此本轮不是端到端成本节省实验。

其他说明：[prospective-review.md](<D:/a1-足彩/docs/parallel/round32/prospective-review.md>)

**R33　第三十三轮：多任务条件下的自主选源与证据范围测试**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round33/README.md>)

Round32 证明了一个有限菜单内的单任务选源动作。Round33 将同一协议复制到三个目标卡片：正文快照保持相同并在选择前隐藏，目标陈述分别改变方向或比赛范围。这样可以观察 Agent 是否只记住“官方单场报道优先”，还是会在目标改变后重新解释证据范围。

当时的结果或限制原文摘录：

> 这是一个**目标条件化**测试，而非三个独立比赛样本。前两个任务是同一场比赛的正反命题，第三个任务是日期范围改变；三者共享一批正文，所以结果适合检验证据边界与目标绑定，暂时不适合估计跨联赛泛化率。

> 所有 Agent 和评价回执的工具事件为空；网络正文来自事前冻结快照。批处理摘要中的 `expected_query_value`、`selection_regret` 和 `optimal_menu_choice_proven` 都保持未知或未证明。

> - 任务共享同一冻结菜单，尚未形成独立比赛样本；
> - 三条候选由研究人员策展，尚未估计开放网络候选分布；
> - 三次选择都命中同一来源，尚未测出选择多样性或选择错误率；
> - 未计算数值 VoI、查询成本收益或全局停止最优；
> - 来源报告终态不等于比赛世界真值或胜平负收益；
> - 评价器仍是语言模型，语义标注需要后续人工或规则双审。

其他说明：[multi-task-review.md](<D:/a1-足彩/docs/parallel/round33/multi-task-review.md>)

**R34　第三十四轮：把时间可得性纳入 Agent 自主选源（2026-09-27）**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round34/README.md>)

`ROUND34_TEMPORAL_ELIGIBILITY_GATE_DEFERRED_RETROSPECTIVE_SOURCE_VALUE_UNMEASURED`

当时的结果或限制原文摘录：

> 第一次真实 Codex CLI 回合耗时约 61.8 秒，返回了明确的策略判断：唯一可用的 spftll 来源只能确认比赛识别和赛程/市场上下文，缺少晋级状态与小组排名证据，因此选择 `defer`；它把三个不可用候选分别标为 cutoff 之后、范围隔离和请求失败，并没有读取或猜测它们的正文。候选正文没有暴露给 Agent，工具事件为 0。

其他说明：[temporal-gating-review.md](<D:/a1-足彩/docs/parallel/round34/temporal-gating-review.md>)

**R35　第三十五轮：时间门禁接入动态 inquiry（2026-09-27）**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round35/README.md>)

`ROUND35_TEMPORAL_GATE_INQUIRY_INTEGRATION_DYNAMIC_CUTOFF_CONSTRUCTED`

当时的结果或限制原文摘录：

> 第三十四轮的门禁先作为有限菜单协议运行。本轮把它接到既有 `InquiryWorkflow` 的真实控制点：LLM 返回候选之后、HTTP 预留之前。这样 Agent 仍负责提出下一步路线，但时间和范围门禁在执行层生效，事后来源不会因为语言模型认为“相关”而进入请求。

> 这证明了一个关键的动态性质：同一 Agent 路线的可执行性由信息时点决定。它仍然没有证明该路线有正的数值 VoI；第二条轨迹中的 HTTP 是构造请求，未访问外部网络。

**R36　第三十六轮：时间门禁与有限 EVSI 的连接（2026-09-27）**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round36/README.md>)

`ROUND36_TEMPORAL_MASKED_EVSI_UNAVAILABLE_DISTINCT_FROM_ZERO_VALUE_SYNTHETIC`

**R37　第三十七轮：从来源返回日志到条件似然，再接入时间门禁 EVSI（2026-09-27）**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round37/README.md>)

`ROUND37_RETURN_LIKELIHOOD_BRIDGE_CONSTRUCTED_POINT_ESTIMATE_UNCALIBRATED`

**R38　第三十八轮：留出校准诊断与跨来源依赖审计（2026-09-27）**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round38/README.md>)

`ROUND38_HOLDOUT_CALIBRATION_DIAGNOSTIC_DEPENDENCE_GUARD_OPERATIONAL_SCOPE_LIMITED`

**R39　第三十九轮：随机来源探测臂与 Agent 自适应决策臂分离（2026-09-27）**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round39/README.md>)

`ROUND39_RANDOMIZED_SOURCE_PROBE_ARM_SEPARATED_FROM_ADAPTIVE_AGENT`

当时的结果或限制原文摘录：

> 后续接入真实采集时，随机探测应使用独立任务或独立比赛集合，并在来源返回前冻结：

**R40　第四十轮：查询前状态标注与随机探测绑定**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round40/README.md>)

本轮把信息价值决策里的“查询前状态”从一个裸字符串提升为可审计对象。对象同时记录状态变量、取值、标注依据、来源 ID、来源可得时间、观察时间、预测 cutoff、作用范围和可见性。它服务于 FootballMind 的核心问题：Agent 选择信息以前，评价器究竟依据了什么状态；这些依据在当时是否已经可得；不同语义是否被错误地压成一个标签。

**R41　第四十一轮：状态假设到候选查询的语义路由**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round41/README.md>)

本轮继续围绕 FootballMind 的核心创新：Agent 根据当前未知状态，决定下一步查询哪一种信息。Round40 已经解决了查询前状态的来源、时间和 provenance；本轮增加中间语义层，把状态假设和候选查询明确连接起来。

**R42　第四十二轮：查询价值状态与范围化停止证书**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round42/README.md>)

本轮把“当前存在未知状态”和“查询的数值信息价值已经测得”拆成两个层次。Round41 的 `StateQueryBoard` 负责回答候选路径针对哪个未知状态；本轮的 `QueryValueGate` 只接收已经明确声明返回模型的数值结果，并为停止或暂缓动作生成范围化证书。

当时的结果或限制原文摘录：

> 本轮只建立了价值状态和停止证书，所有数值仍来自构造的有限模型。它没有证明某个真实来源对足球预测有多少收益，也没有证明 Agent 策略优于固定策略。下一步应使用已通过返回似然、依赖审计和留出诊断的候选，进行同一菜单、同一 cutoff、同一预算下的策略回放。

**R43　第四十三轮：来源探测似然到有限信息价值的桥接**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round43/README.md>)

这条链只允许通过显式状态、时间、选择概率、依赖簇和来源探测设计门禁的返回行进入有限模型。它仍然保留 `NOT_CALIBRATED`，所以点估计可以用于构造的有限模型参考计算，却不被写成真实来源的部署校准。

当时的结果或限制原文摘录：

> - `SUPPORTED_*` 且 `likelihoods` 存在时，读取有限模型候选行的 `immediate_evsi` 和 `cost`，生成 `MODELED_DECLARED_ASSUMPTIONS`；
> - 估计的 `calibration_status` 必须仍为 `NOT_CALIBRATED`，结果写入 `ASSUMPTION_ONLY` 和 `return_likelihood_point_estimate_NOT_CALIBRATED`；
> - `INSUFFICIENT_SUPPORT`、自适应选择、依赖重复、时间越界等状态直接生成 `VALUE_UNMEASURED`；
> - 不支持的估计不会提供 `expected_value` 或 `net_value`。

**R44　第四十四轮：同一隐藏环境下的自主信息选择策略对照**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round44/README.md>)

本轮把项目的核心创新从“会调用工具”推进为可比较的实验对象：在同一个有限假设模型、同一候选菜单、同一预算、同一时间截止点和同一隐藏状态/赛果随机流下，回放四类信息策略：

当时的结果或限制原文摘录：

> 这些数字只用于验证构造机制：自主策略在本组机制中以更少查询接近全量获取的风险水平，并优于固定优先级与基础停止；它们不是足球预测准确率、真实来源价值或投注收益结论。构造样本也不支持把一次回放的损失差异写成统计显著性。

**R45　第四十五轮：停止时的语义未知状态审计**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round45/README.md>)

第一，调用过工具不等于状态已经确定。查询可能返回模糊表述、互相冲突的报道、样本不足的结果或低质量来源。因此 `policy_audit.py` 将以下概念分开：

**R46　第四十六轮：平台证据状态接入主动信息审计**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round46/README.md>)

Round45 已经把查询动作、状态解决、晚到和失败状态分开。本轮把这些语义接到平台已有的证据字段：observed_at、operational_available_at、expires_at、quality、evidence_refs、content_hash 和 dependency_key。

**R47　第四十七轮：AcquisitionResult 到候选证据的适配**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round47/README.md>)

Round46 已经可以把平台证据字段送入语义审计。本轮继续向采集层靠近，新增 platform_evidence_adapter.py，直接消费平台公开的 AcquisitionResult 和 EvidenceGroup 结构，生成 evidence_binding 可识别的候选证据行。

**R48　Round48：追加式 Evidence Ledger 与 cutoff 快照**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round48/README.md>)

本轮把 Round47 的平台采集结果落入一个可校验、可追加、可回放的 evidence ledger。账本保存 episode、target、candidate、action、request、sequence、来源、质量、可用性、观测时间、运营可用时间、预测截止点、证据引用、内容摘要、依赖簇和显式 resolution status；每条记录带有稳定 entry_id 与 record_sha256。

**R49　Round49：Point-in-Time Evidence Replay**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round49/README.md>)

本轮收紧 Evidence Ledger 的时间边界。一个决策点的快照只读取 `captured_at <= prediction_cutoff` 的采集尝试：决策前已经发起、但平台在截止点后才提供结果的记录保留为 `late`；截止点后才发起的查询完全留在未来账本记录中，不能进入当时的决策状态。

当时的结果或限制原文摘录：

> 本轮只验证时间切分和账本回放，没有接入实时来源、真实比赛标签、LLM 部署或真实来源校准。后续可以在同一接口上记录真实查询请求的发起时间、返回时间和 source version，再评估信息价值估计是否具备校准性。

**R50　Round50：可审计的 Pre-action VoI Trace**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round50/README.md>)

本轮把 Agent 的信息价值判断从运行时计算结果提升为可持久化的决策轨迹。`VoITrace` 保存查询前的 belief、prediction、expected information value、expected cost、expected net value、selection basis、model version 和查询后的 posterior；当 Agent 停止时，记录 `STOP`、风险状态和 stop reason。

**R51　Round51：VoI 到 Point-in-Time Evidence 的桥接**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round51/README.md>)

本轮把 Round50 的 pre-action VoI trace 与 Round48/49 的 Evidence Ledger、cutoff 快照连接起来。每条查询通过 `episode_id + target_key + request_id + action_id + candidate_id` 关联平台证据，再检查证据是否在 prediction cutoff 前可用。

**R52　Round52：请求时间与 Source Version 贯通**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round52/README.md>)

本轮补齐真实来源接入所需的两个元数据：请求发起时间 `requested_at` 和来源版本 `source_version`。它们从 `EvidenceLedgerEntry.from_row()`、`append_acquisition_result()` 进入追加式账本，再由 `VoIEvidenceLink` 输出到 VoI—Evidence 审计结果。

**R53　Round53：来源返回与 VoI 校准诊断**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round53/README.md>)

本轮在 Round52 的时间与版本契约之上增加离线校准层。`calibrate_source_returns()` 按 `source_version` 和时间桶统计 `available`、`late`、`failed`、`duplicate` 的返回频率，并报告依赖簇重复度；`calibrate_voi_links()` 按 source version 比较 Agent 的 expected information value 与 observed posterior risk reduction。

**R54　Round54：未来时间切分与来源 Holdout**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round54/README.md>)

- 来源分组键升级为 `source_id + source_version`，同名版本来自不同来源时保持独立。
- 支持量按照独立 `dependency_key` 计算，重复请求不会增加有效样本量。
- `linked_unusable` 的晚到、失败和 cutoff 后证据只保留运营状态，不计算 posterior 风险下降或 VoI 校准误差。

**R55　Round55：真实采集器事件元数据接入 Evidence Ledger**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round55/README.md>)

把已经跑通的 `xx.spftll.cn` 真实采集链路接入来源返回校准与未来 holdout 所需的统一事件格式，同时保持原始响应归档和业务解析边界稳定。

当时的结果或限制原文摘录：

> 本轮记录的是采集事件元数据，不把 HTTP 成功直接当成语义正确，也不把实体映射状态隐藏在解析器内部。`VALIDATED` 需要显式的 `mapped` 状态；只有 `operational_available_at <= prediction_cutoff` 的事件进入截止时点可用证据。这样，后续 VoI 评估可以用真实来源的请求成本、返回时延、失败率和映射不确定性做校准。

**R56　Round56：从 NormalizedBatch 自动推导实体映射质量**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round56/README.md>)

把 Round55 中由调用方传入的实体映射状态，推进为由标准化结果自动推导的状态。这样来源返回校准使用的是实际解析结果和实体质量，而不是人工标签。

**R57　Round57：把采集事件接入 pipeline run manifest**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round57/README.md>)

让真实采集平台的处理结果携带实体映射质量和解析器版本，并把这些信息写入每次 source run 的 manifest，作为 Agent 后续来源返回校准的输入。

**R58　Round58：Pipeline Manifest 到决策 Ledger 的截止时间重绑定**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round58/README.md>)

把采集运行 manifest 中的 `collector_events` 接入具体决策 episode。采集运行结束时间只描述来源到达审计；Agent 必须用自己的 `prediction_cutoff` 重新计算证据可用性，然后才能写入 Evidence Ledger。

**R59　Round59：采集 Manifest 到 VoI 轨迹的端到端绑定**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round59/README.md>)

让一次真实来源查询可以完整进入 Agent 的可审计链路：来源 HTTP 请求 → collector event → 按决策 cutoff 重绑定 → Evidence Ledger → VoITrace evidence link。

**R60　Round60：动作级多请求证据聚合**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round60/README.md>)

为一个需要多个来源请求共同完成的 Agent action 建立明确的聚合语义，同时保留每个底层来源的独立 dependency，避免 VoITrace 因多条同身份 Ledger entry 产生歧义。

**R61　Round61：截止点安全的动作级证据聚合**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round61/README.md>)

把多来源信息动作从“收到几条返回就直接合并”推进到可审计的依赖状态机：同一依赖的重试需要归并，替代来源需要按 `any_usable` 选择，必需来源需要按 `all_required` 显式声明；聚合结果还要沿同一个 prediction cutoff 连接回 VoI trace。

**R62　Round62：依赖感知的动作下一步信号**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round62/README.md>)

让多来源证据聚合结果直接服务于 Agent 的下一步规划。该层只描述证据状态和可行动依赖，把“查什么、重试什么、等待什么、停止当前动作”变成机器可读信号；信息价值、成本和开放世界未知仍由 VoI 门控负责。

**R63　Round63：readiness 驱动的候选工具路由过滤**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round63/README.md>)

把证据状态进一步接到 Agent 的实际工具候选集：同一个 dependency 可以声明 primary、retry 或 alternative 路由，Agent 先按证据 readiness 过滤，再把剩余候选交给 VoI 和成本门控排序。

**R64　Round64：readiness 到 VoI 查询门控的接线**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round64/README.md>)

让证据依赖状态真正影响查询决策入口：readiness 先生成裁剪后的 `StateQueryBoard`，随后复用现有 `QueryValueGate` 进行预算、成本和信息价值判断。

**R65　Round65：上下文条件化的信息价值**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round65/README.md>)

把小组赛和晋级条件纳入 Agent 的信息价值判断，而不是把它们当作固定规则：已提前出线不会自动推出轮换，平局足够出线不会自动推出防守；上下文只通过显式假设改变未知状态的先验、后验和决策风险。

**R66　Round66：平台证据绑定到动作 readiness**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round66/README.md>)

让真实采集器产生的 `EvidenceBinding` 直接影响 Agent 的依赖状态，同时保持运营可用性、证据质量和赛事预测概率的边界分离。

**R67　Round67：重复证据的独立来源分流**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round67/README.md>)

把平台层的 `duplicate` 证据与 provider failure 分开处理。重复记录表示已有相同依赖的证据，当前动作缺少独立支持；它不应被算作可用，也不应简单触发同一报告的重复重试。

**R68　Round68：真实采集 fixture 到 readiness 的桥接**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round68/README.md>)

使用已有的 `xx.spftll.cn` 脱敏真实采集 fixture，验证“原始观测 → 标准化 → 实体映射 → EvidenceBinding → ActionReadiness”的完整路径，并保持传输状态、语义质量和赛事概率边界分离。

**R69　Round69：真实来源的来源级 readiness 到 VoI 查询**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round69/README.md>)

区分动作级 `any_usable` 与来源级质量：一条稳定来源可以让动作获得可用证据，但同一动作中的 `CONFLICT`、`UNVERIFIED` 来源仍然应分别进入替代来源查询，而不是被整体停止信号掩盖。

**R70　Round70：时效感知的来源选择与重复查询停止**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round70/README.md>)

本轮把证据有效期接入 Agent 的信息选择闭环。预测截止点之前已经过期的 `VALIDATED` 证据现在被重分类为 `STALE`，并保留 `failed` 可用性和过期原因；它进入 readiness 后会触发独立来源或刷新路线，而不是继续把旧结果当作当前事实。

**R71　Round71：未来安全的经验 VoI 校准**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round71/README.md>)

本轮新增 `adaptive_voi_calibration.py`，把历史查询回报接入候选工具的 `QueryValueAssessment`。校准按 `source_id + source_version` 分区，只读取 `as_of` 之前的查询；共享 `dependency_key` 的重试聚成一个独立支持单元，避免同一信息被重复计权。

**R72　Round72：上下文条件化的经验 VoI 与跨场景隔离**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round72/README.md>)

上一轮的来源校准能够估计一个来源的总体查询价值，但总体平均值可能把不同比赛状态混在一起。本轮新增 `contextual_adaptive_voi.py`，要求每条历史 VoI 记录带有查询前的公开上下文指纹，并只使用与当前上下文完全一致的历史记录估计 expected information value。

**R73　Round73：上下文优先的分层自主信息选择**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round73/README.md>)

本轮新增 `context_query_readiness.py`，把“当前上下文是否足够”从经验 VoI 中单独抽成一个规划层。每个候选路线声明 `required_context_keys`、`route_kind` 和 `provides_context_keys`；上下文路线负责补齐公开状态，领域路线只有在其前置上下文已知后才进入 QueryValueGate。该层只产生 route readiness，不填写 expected value。

**R74　Round74：哈希链上下文与决策回放**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round74/README.md>)

本轮新增 `context_transition.py`，将上下文状态和 Agent 查询选择写入 append-only、hash-chained JSONL 账本。事件类型包含 `INITIALIZED`、`QUERY_SELECTED`、`CONTEXT_UPDATED`、`REPLANNED` 和 `STOPPED`；每个事件保存前一上下文、后一上下文、变化字段、触发原因、查询候选、来源证据引用和 prediction cutoff。

**R75　Round75：真实采集元数据到 ContextState**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round75/README.md>)

本轮增加 `collector_context_adapter.py`，把已有 `xx.spftll.cn` 脱敏归档经过标准化后的结果投影为 Agent 可见的 `ContextState`。适配器采用结构化白名单，只允许比赛 ID、赛事/球队来源 ID、开赛时间、赛前状态、排名是否存在、赔率是否存在和距离截止点的时间等赛前事实进入上下文；比分、半场比分、结果、概率和结束状态不会进入上下文。

**R76　Round76：真实采集上下文接入规划与 VoI**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round76/README.md>)

本轮新增 `collector_context_bridge.py`，把 Round75 的真实 `xx.spftll.cn` ContextSnapshot 接入上下文优先规划。detail 与 odds 脱敏 fixture 先通过 `merge_context_snapshots()` 合并，再由 bridge 增量写入 `ContextState`；赔率覆盖和报价数量使用单调更新，普通字段继续保留冲突语义。

**R77　Round77：查询执行 Envelope 与上下文再规划**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round77/README.md>)

本轮补齐了从 `QueryValueGate` 选中候选到真实上下文更新之间的最小执行接口，新增 `query_execution_envelope.py`。它不负责联网，只接收本地工具 fixture，记录一次查询的 `query_id`、目标、prediction cutoff、预期上下文键、成本、选择依据、工具版本、请求哈希、结果哈希、证据引用和执行状态。

**R78　Round78：自主信息获取策略对照评测**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round78/README.md>)

1. `qualification-high-value`：晋级状态会显著改变当前预测，天气信息价值很低。
2. `weather-low-value`：所有候选信息的净价值都低于获取成本。
3. `qualification-late`：晋级查询的操作可用时间晚于 prediction cutoff。
4. `qualification-ambiguous`：查询返回来源歧义，没有可接纳的上下文更新。

当时的结果或限制原文摘录：

> 评测层保存一个私有响应 tape 和一个真实结果标签，用来计算单条构造轨迹的 Brier 决策损失、获取成本、查询次数和净效用。它们位于 `evaluator_only` 区域，未进入请求体、`ContextState`、转换事件或策略可见场景清单。这个实验展示机制差异，不宣称真实足球样本上的统计收益；后续需要用按时间切分的历史数据估计返回模型和置信区间。

**R79　Round79：上下文条件经验 VoI 接入 Gate**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round79/README.md>)

本轮把第71—72轮已有的经验校准真正接到查询选择入口。新增
`contextual_query_gate.py`，先按来源 ID、来源版本、请求时间、独立依赖和上下文指纹计算
保守的风险降低下界，再把**精确上下文**的估计转换为 `QueryValueAssessment` 交给
`QueryValueGate`。全局来源平均值只作为诊断字段保留，不跨上下文迁移。

**R80　Round80：足球语境中的上下文路由回放**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round80/README.md>)

- 已出线但轮换未知：`qualification=already_qualified` 只说明晋级状态，Agent 仍需在 `reserve-strength` 和 `tactical-intent` 中按价值选择；`lineup` 因前置状态未齐被暂缓。
- 平局足够但战术未知：`qualification=draw_sufficient` 不直接等价于防守，Agent 先查询 `tactical-intent`；是否防守由后续证据决定。
- 晋级状态未知：先查询 `qualification-refresh`，下游轮换、战术和阵容路线保持前置条件未满足。

**R81　Round81：真实采集到多步自主上下文 Loop**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round81/README.md>)

本轮把 Round76 的真实 `xx.spftll.cn` 脱敏 fixture 和 Round80 的足球上下文路由接入同一条可回放 loop。初始上下文包含真实比赛 ID、赛事 ID、开赛时间和赔率覆盖；`qualification`、`reserve_strength`、`tactical_intent` 与 `lineup` 仍保持未知。

**R82　Round82：答辩用自主信息选择回放报告**

原始记录：[README.md](<D:/a1-足彩/docs/parallel/round82/README.md>)

本轮新增 `autonomous_loop_report.py` 和渲染脚本，将真实采集多步 loop 转成评审者可直接阅读的 Markdown/JSON 报告。报告逐步列出：

**R83　Round83：自主信息选择策略对照评估**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round83/README.md>)

* `fixed_all`：固定获取所有候选信息，作为成本上界基线；
* `heuristic`：沿用 Round78 的首个候选启发式（天气先查），代表没有 VoI 停止标准的工具调用；
* `voi_gate`：使用声明的候选价值、成本和 `QueryValueGate` 做选择与停止；
* `oracle_upper_bound`：评估器侧的 hindsight oracle，枚举候选子集，使用私有实现结果寻找最大净效用。它只作为上界和 regret 参照，未进入 Agent 可见输入。

当时的结果或限制原文摘录：

> 输出指标包括最终 Brier 风险、log loss、top-class accuracy、confidence error/reliability bins、查询成本、查询比例、零查询率、净效用、相对 oracle 的 regret，以及成本—风险 Pareto 前沿。真实结果标签和响应 tape只用于评估计算；场景 public manifest、查询请求、上下文和回放投影不含这些字段。

其他说明：[historical-autonomous-evaluation.md](<D:/a1-足彩/docs/parallel/round83/historical-autonomous-evaluation.md>)

**R84　Round84：历史 as-of 协议与自主信息选择边界审计**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round84/README.md>)

本轮把自主信息选择评估放到按比赛开球时间切分的历史协议中，先验证 Agent 在每一个 `prediction_cutoff` 看到的上下文是否可得，再把结果指标留在评估器侧。协议的目标是保护 FootballMind 的核心研究问题：Agent 需要判断下一条信息是否值得获取，而不是把赛后信息倒灌到一次性预测特征中。

**R85　Round85：历史 paired policy metrics 与配对审计**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round85/README.md>)

本轮把 Round84 的历史 as-of 协议连接到“Agent 是否值得获取下一条信息”的策略比较。评估器在同一 `match_id + prediction_cutoff + budget_money` 上配对各策略，重新计算 Brier、log loss、实际成本和请求量，并以比赛为聚类单位给出 bootstrap 区间。运行时 Agent、公开 ContextState 和查询响应仍然与结果标签分离。

当时的结果或限制原文摘录：

> 1. episode summary 没有显式 `lead_hours` 和 `league/competition` 字段；当前只能从时间戳和轨迹元数据派生。
> 2. 顶层 `actual_external_requests` 全部为零；真实请求量必须使用嵌套 `cost.requests`。
> 3. 当前 test match IDs 来自曾经探索的 Round9 历史归档，尚未构成独立 holdout。
> 4. 下一次实验需要在训练折拟合来源/上下文 VoI 与停止参数，在验证折选阈值，冻结后只运行测试折。

**R86　Round86：自主信息获取实验协议与中断恢复**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round86/README.md>)

磁盘中 Round61–85 的目录、代码和回放记录仍在，恢复时 status/agent.md 的最新交付为 Round85。
Round86 当时只有初步协议代码、6 项定向测试和两个回放文件，缺少完整回归与验收。
截图停在 Round60 并显示连接中断；本次只核验本地成果，没有把截图当作工程版本，也没有推断服务端聊天记录的同步原因。

当时的结果或限制原文摘录：

> cost.money 是归档设计中的 simulated currency，cost.requests 是回放中的获取次数；均不是实际 API 计费或真实外部网络请求量。
> 旧记录中的“实际成本/实际请求量”应按这个限定理解。历史诊断的 checks 全真表示成功识别缺陷，不表示旧归档通过独立实验准入。

**R87　Round87：冻结配置约束下的动态信息获取循环**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round87/README.md>)

原有 autonomous_context_loop 依赖事先填写的候选价值，且每次门控使用固定的 1.0 预算。
本轮连接现有上下文规划、历史价值校准、查询门控和执行账本：每次获得新上下文，重新估计剩余信息的价值，只执行选中的工具，并持续扣减预算。
这直接服务于 Agent 下一步获取什么、是否值得获取以及何时停止的研究问题。

当时的结果或限制原文摘录：

> 同时重读 Round81 已采集并脱敏的 spftll 比赛 2996160 初始结构化上下文。没有编造校准收益；匹配历史为空，因此返回 VALUE_UNMEASURED / DEFER，工具调用数为零。
> 本轮没有新增外部网络请求、独立比赛样本或能支持实际预测收益的实验。

**R88　Round88：未知上下文到开放查询的有界探索桥接**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round88/README.md>)

Round87 在缺少相符历史证据时保留 DEFER，这避免了编造信息价值，但尚未连接到开放查询工作流。
仓库已经有 InquiryWorkflow、LearningExplorationSelector、UniformInquiryExploration 和 return_dataset。
本轮新增 context_inquiry_handoff.py，将首次、零查询的未知价值 DEFER 接到这些现有模块；没有另建爬虫或重新实现探索算法。

当时的结果或限制原文摘录：

> 子任务严格沿用原 prediction_cutoff，不以新任务延后截止时间。调用方声明的初始上下文 available_at 须不晚于当前运行时钟和截止时间；冻结时间须早于执行和 cutoff。
> 子工作流已有真实运行时钟检查。如果模型提议期间超过截止时间，后续来源请求不执行；对应测试已覆盖。

> - 本轮是真实本地工作流的机制执行，外部网络请求为 0，没有新增真实比赛收益样本。
> - 探索机制此前已存在；本轮新增的是从上下文价值门控到该机制的连接、预算绑定和恢复流程。
> - 随机化只在当前已生成且合格的候选内；不提供全开放空间动作概率，也不意味着候选已经穷尽。
> - 桥接传递未知状态范围，但新文本问题的语义相关性仍依赖既有提议协议和后续审核。
> - 子任务获取的来源表述不自动更新父预测或数值概率；语义核验与可训练风险变化标签仍待补齐。
> - 时间和冻结哈希校验提交内容的一致性，不证明调用方声明的外部可用时间真实存在。

**R89　Round89：开放查询结果返回父规划状态**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round89/README.md>)

本轮解决 Round88 子查询结束后与父规划状态脱节的问题。新增 football_agent/context_inquiry_return.py，复用现有引文、来源归档、假设适用范围和事件链校验，将已获得的信息状态返回父上下文。此前的证据状态算法继续沿用，本轮新增的是父子绑定、按时点投影和规划接入。

当时的结果或限制原文摘录：

> 实际发生 **1 次模型调用尝试，120 秒后 TimeoutExpired，0 次完成的模型回合，0 次足球来源请求**。任务留在 DEFERRED / PROVIDER_FAILED_OR_RESPONSE_UNKNOWN，已保留原 run_id、预算、失败事件和空返回数据集；没有通过重建任务重新发放额度。命令正常退出只说明失败记录及导出完成，不表示真实采集成功。

**R90　Round90：修复自主查询的模型运行时配置冲突**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round90/README.md>)

日期：2026-09-29。研究主线仍是 Agent 按信息价值选择下一项查询、利用返回调整判断并决定停止；本轮解决这条链路进入真实运行前的模型调用故障，不把工程测试或健康响应当成信息选择收益。

当时的结果或限制原文摘录：

> 修复后的独立健康检查实际完成，返回 status=ready，用时约 14.5 秒，过程只有模型消息而没有工具事件。这确认新配置可以完成最小结构化模型调用；它不是完整足球查询、不是开放文本语义正确性证明，也不能单独排除其他超时成因。

> 每次调用均在执行前登记；同一脚本再次运行会拒绝重复执行。首次诊断之后因明确发现日志缺项，追加了单独登记的一次补充诊断，没有覆盖首次记录。包括原失败调用在内共 4 次 CLI 调用，本轮新增 3 次；CLI 内部请求次数和实际货币费用未知。失败调用没有退款或重分类为成功。

> - football_agent/inquiry_provider.py：修复 host 配置、版本化配置、超时诊断、异常保留和失败日志脱敏。
> - football_agent/inquiry_provider_diagnostics.py：记录有界进程事件、长度、哈希、错误特征与响应文件状态，不根据超时推断远端完成、不自动重试。
> - football_agent/inquiry_provider_errors.py：只提取结构化公开错误项，过滤 reasoning；错误消息限制长度/数量并脱敏常见凭据与 URL。此过滤不是任意秘密检测器。
> - 两个 test_inquiry_provider_* 测试文件：覆盖部分输出、bytes/text、异常日志写失败、晚到响应、工具隔离和流程预算保留等行为。
> - diagnose_provider.py、diagnose_error_item.py、verify_provider_health.py：三个分别登记的实际运行入口，重跑保护阻止重复调用。

**R91　Round91：真实自主信息选择与采集，返回后推理仍待打通**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round91/README.md>)

Agent 在实际模型回合中生成两个候选，比较 Gangwon FC 俱乐部新闻入口与 K League 球队页，选择前者。其问题是先确认具体比赛身份与时间，再决定是否值得追查轮换或战术。选择理由明确保留联赛入口作为替代路径，没有同时抓取全部候选。

当时的结果或限制原文摘录：

> 以上是依次调整后的操作排障，不是随机对照实验；不能据此证明简化必然提升速度或决策质量。全部诊断均未再次观测到 Round90 的 host 禁用错误，当前超时根因仍未建立。

> 已有行为支持“Agent 可提出问题并选择信息路径”的操作事实，尚不支持“Agent 已知道什么信息最有价值”。当前选择依据是定性、有界探索，预期风险下降和数值 VoI 没有校准。未选来源的返回仍未观察，不给它负价值标签。

> 1. 复用已归档正文和原观察时点，先做离线回放，不重复请求同一页面。
> 2. 设计保留原文位置与覆盖范围的证据窗口：由 Agent 选择继续读哪个段落或链接，所有截断明确标注，未展示部分不解释成没有信息。源文档与精确引文仍可核验。
> 3. 重新验证读取后能作出一条有效决定：追查具体场次、换源或明确暂缓。先做到这一点，再加入完整分支评估。
> 4. 独立登记后续回放/实际调用的预算和时限，并承接本次累计成本；不修改这三条已冻结轨迹。
> 5. 完成真实操作闭环后，再设计同预算固定、随机、启发式与 VoI 策略对照。HTTP 成功、字段数量和测试数都不是信息价值收益。

**R92　Round92：真实模型自主选读归档证据并决定暂缓**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round92/README.md>)

本轮实际完成的是「看导航预览 → 自主选择正文窗口 → 阅读返回 → 重新判断 → 暂缓并提出下一信息需求」。复用 Round91 已完成的来源获取，不把 Round91 后续超时推理视为成功，也不延长其原任务截止时间。

当时的结果或限制原文摘录：

> 本轮登记独立阅读任务，截止时间为 2026-09-29 11:43:15.971114 +08:00，原始页面观测时间仍为 2026-09-29 11:14:03.363180 +08:00。新阅读任务不构成新来源观测或独立比赛样本。

> 本轮证明真实模型能够根据局部返回调整信息需求并自主暂缓，没有证明该选择最优、停止点最优或降低了预测风险。当前选择依据是定性判断，未校准数值 VoI；不更新比赛概率、不声称完整联网获取闭环或独立策略收益。输入缩小后两次成功也不构成上轮超时根因的因果证明。

**R93　Round93：真实自主换源、按需阅读与返回后再决策**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round93/README.md>)

本轮把 Round92 的下一信息需求接回统一动作循环，实际完成「比较继续读归档/查询新来源/暂缓 → 自主选择来源 → 获取 → 自主选读正文 → 再次判断」。旧需求是上下文中的提议，不是强制执行的指令；系统没有预设下一步一定换源。

当时的结果或限制原文摘录：

> 1. Agent 认为旧新闻余文仍可能含线索，但当前主要缺口是场次身份和时间。它从已经可见的页面链接中，自行选择尚未获取的 Gangwon FC 官网首页，而非继续读旧新闻或直接暂缓。调用前说明可能有新赛程、延期、仅有旧内容或获取失败等不同返回如何改变后续选择。
> 2. 首页真实请求成功，HTTP 200。系统先展示元数据、链接和导航预览；Agent 看到下一主场线索后，自主选择读取首页唯一的 188 字符正文窗口。没有把预览直接提升成可引用事实。
> 3. 已读正文包含 NEXT HOME MATCH、10 月 09 日 16:30 及对手富川的提示。Agent 明确区分「下一主场」与「所有赛事中的下一场」，保留正文未标年份、完整赛程及 9 月 27 日实际赛况未确认的问题，选择 AGENT_DEFER。

> 以上描述的是已归档页面文字及 Agent 的解释，不是独立核实的赛程公告；不擅自补全年份或把预定日期已过当作比赛实际完成。只提取到 188 字符也不证明完整覆盖网页动态内容。

**R94　Round94：将真实信息选择返回接回上下文与价值接口**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round94/README.md>)

本轮复用 Round93 的真实获取、阅读和再次判断记录，完成导航返回适配；没有新增模型调用或网络请求，没有延长旧任务或重新分配额度。它解决操作闭环与既有上下文/价值门控之间的接口断点，属于离线投影与诊断，不宣称新的在线策略收益。

当时的结果或限制原文摘录：

> Agent 上次实际完成的序列为：获取自选官网首页 → 选读新正文 → 暂缓。现在逐步保留获取前可见上下文、预期哪类答案会改变判断、实际返回、后续需求变化与分项成本。后续评估能区分“是否拿到数据”“Agent 是否调整信息需求”“任务是否真的改善”。

**R95　Round95：隐藏策略轨迹的任务证据评审**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round95/README.md>)

本轮实际运行三次隔离评审，回答：新增信息是否足以支持最初的任务结论，而不只是让 Agent 的理由或下一步需求发生变化。三次都返回有效结构化意见，新增网络请求为 0；原 Agent 获取任务没有重启。

当时的结果或限制原文摘录：

> 三次评审费用单独记账，不回填到原导航成本。此前获取链路累计 13 次 CLI 尝试，本轮评审增加 3 次，合计 16 次；来源请求仍为此前 2 次。真实金额与 CLI 内部请求数未知。具体耗时、原始输入输出绑定和验证结果见 review-report.json、review-verification.json、execution-record.json、acceptance.json。

**R96　Round96：固定任务下的场次对象绑定**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round96/README.md>)

本轮直接处理 Round95 暴露的问题：评审中的 supported 可能指向一场旧比赛，而下一阶段的 partial 指向另一场下一主场。这样的标签变化没有固定比较对象，不能作为信息获取收益。

当时的结果或限制原文摘录：

> 三次真实模型调用均返回有效暂定评语，没有未确认返回。三次耗时约 154.8、173.3、62.2 秒；本轮新增来源获取为 0。历史获取链路 13 次 CLI、上一轮评审 3 次、本轮评审 3 次，合计 19 次 CLI 尝试；来源尝试仍为 2 次。实际金额及 CLI 内部请求数未知。

> 这次交付消除了一个明确的错误入口：以不同场次的支持标签变化计算收益。它尚未测得真实信息收益，也没有证明候选越多越好。

> 将“目标尚未确定”作为下一次实际 Agent 任务中的显式未知，而不预先指定伤停、天气等固定特征。Agent 需要自己判断补哪类证据能解决这个未知、查询是否值得以及是否暂缓。回顾性评审不可回填到原任务输入；未来调用必须使用当时可用的状态和单独登记的任务/预算。

**R97　Round97：把信息获取假设接回实际 Agent**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round97/README.md>)

- 新增 football_agent/target_navigation.py：沿用既有取页、精确正文窗口、来源限制、预算和持久化执行器；加入固定任务契约、多文档继承、目标状态和剩余未知。
- Agent 在读取或取页之前提出 information_test，明确针对哪个未知、什么答案会有用、什么返回不足以改变判断。这里规定任务维度，不规定必须先查哪类信息或哪个页面。
- 返回后，return_assessment 绑定此前问题和实际文档；取页预览仍是 unread，读正文之后才允许带精确引文判断 partial、resolved、irrelevant 或 conflicting。
- 新窗口评估必须引用此次实际打开的窗口；失败返回保留为信息缺失，不变成不存在伤停、没有客场等反向结论。
- proposed_target 是模型提出的目标，非已核实比赛真值；未知仍在时可 defer。没有要求补齐任意 UTC 偏移，也没有把固定特征列表当作自主获取。
- 新增 target_navigation_verify.py：重建实际可见输入、契约、行动、正文暴露和消耗。为保留冻结历史代码，本轮版本化复用旧回放的状态转换规则；原回放器与旧轮源码未修改。
- 新增 28 项机制测试，与既有导航及场次契约测试共 82 项定向验证；未执行全量测试。

当时的结果或限制原文摘录：

> 注册上限为 5 次模型、2 次 HTTP、3 次正文读取，不要求用满。第一笔真实 CLI 请求约 180.1 秒后超时，日志出现 TLS handshake eof、连接重试与网络等待，未观察到 turn.completed，也没有可接受的响应文件。日志是传输故障信号，不足以定位完整根因。

> **实际新增：1 次模型尝试，0 次确认模型返回，0 次信息选择，0 次来源请求，0 次正文阅读。**保留 pending 与未知远端结果，未退款、未自动重复调用。历史获取及评审累计 20 次 CLI 尝试，来源尝试仍为 2 次；内部请求数和实际费用未知。

**R98　Round98：真实自主取信息、返回检验与显式纠错**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round98/README.md>)

在保留 Round97 未确认调用及原任务目标的前提下，恢复实际模型通道，继续验证 Agent 能否自主决定查什么、读什么、以及返回是否解决原先未知。执行器没有预设来源调用顺序，也没有替 Agent 补写有效响应。

当时的结果或限制原文摘录：

> - 本轮真实模型尝试 4、确认返回 4，包含被拒绝的一次响应；来源请求 1、正文读取 1。四次模型延迟约 62.0、45.4、111.7、34.5 秒。
> - 加上 Round97 未知调用，原任务累计模型尝试为 5，达到原上限；失败没有退款。历史累计 CLI 尝试（含评审和失败）为 24，来源尝试为 3。
> - 原参考时点北京时间 13:10:20 保留。Round98 截止时间 13:50:04 保留，未通过纠错延长；原 Round97 结束状态与 pending 保留。
> - 最终 defer 同时受到“新信息未解决关键未知”和“没有后续复判模型预算”影响，不能据此宣称学习到了最优停止或所有剩余信息均无价值。
> - 真实金额及 provider 内部网络请求数未知；CLI 尝试数不等于底层请求数。

> 最终候选支持还存在语义缺口：support_claim_indices=[0] 指向排名摘要，而不是富川场次事实；候选名称又包含“2026 年（年份待核）”。精确引文与索引有效只证明形式关联，尚未证明候选得到该证据支持。此观察保存在 information-test-observation.json，未倒灌给已执行 Agent，也未改写模型输出。

**R99　Round99：候选场次原文绑定与问题级来源反馈**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round99/README.md>)

新增 football_agent/candidate_grounded_navigation.py，提供 CandidateGroundedWorkflow 与配套 CandidateGroundedProtocol。复用既有导航动作、真实证据窗口、成本账本和执行器；历史协议与运行文件保持冻结。

当时的结果或限制原文摘录：

> 输入为 Round98 已冻结的实际模型响应和正文；未重开原任务，未追加模型或 HTTP 调用。

> 验收脚本核对冻结源码、Round98 原始验收、实际归档回放和定向测试，不调用外部模型或获取新来源。

**R100　Round100：带来源反馈的真实自主信息选择**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round100/README.md>)

本轮验证新候选原文绑定协议如何进入实际信息选择，继续围绕“Agent 决定当前最值得查什么、返回是否回答原问题、何时暂缓”。信息增益、策略收益与现实比赛真值尚未标定，工程通过与研究有效性分别报告。

当时的结果或限制原文摘录：

> 本轮实际模型尝试 3、确认返回 2、执行器接受的选择 1；其中第二次返回因摘要哈希转录错误被拒绝。实际来源请求 1，域名解析失败，新增正文阅读 0。前两次模型延迟分别约 102.0 秒、100.9 秒。累计 CLI 尝试包含评审和失败增至 27，来源尝试增至 4，真实金额和 provider 内部请求数未知。

> Agent 已真实完成“参考旧问题级反馈 → 选择不同资料入口 → 发起请求”。第二次返回包含把失败视为缺失及另行搜索的意图，但被协议门控拒绝，搜索未执行。纠错后未取得模型响应，因此尚未完成新资料正文读取、重新判断与自主停止闭环。此处不把超时或执行器终止视为 Agent 的价值停止。

> 本目录 navigation-report.json 保存真实轨迹，information-selection-observation.json 区分已接受动作与被拒绝提议，provider-verification.json 和 navigation-verification.json 核对实际调用与逐事件状态。acceptance.json 单列工程通过状态和真实闭环完成状态。

**R101　Round101：执行器持有任务身份，模型专注信息选择**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round101/README.md>)

本轮针对 Round100 的真实失败移除模型自由复述长摘要哈希的要求。核心决策字段保持：当前未知、候选证据、哪条信息可能改变判断、获取前条件、实际返回评价，以及查询、阅读或暂缓的选择。

当时的结果或限制原文摘录：

> 该单次调用约 180.1 秒后超时，未取得模型响应；诊断未提供明确的鉴权、限流、连接或重连信号，根因仍待定位。真实新协议返回尚未确认，未声称真实自主信息获取闭环完成。累计 CLI 尝试含评审和失败为 28，足球来源尝试仍为 4；本轮另有 2 次 DNS 操作。调用内部请求数和真实金额未知。

> 本轮只解决执行绑定及隔离错误来源，尚未证明数值 VoI、最优停止或独立策略收益。未来迁移到新任务须显式登记，保留累计成本；旧轮次不覆盖。

**R102　Round102：最小通道与归档负载的有界诊断**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round102/README.md>)

本轮只处理 Round101 留下的运行时不确定性：此前一次新的执行协议核验在 180 秒后超时，但没有鉴权、限流、连接或重连信号。为了避免继续盲目重复长调用，登记了两个独立计费、无足球来源请求的诊断阶段。

当时的结果或限制原文摘录：

> ## 实际结果

**R103　Round103：执行器绑定下的真实自主信息选择闭环**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round103/README.md>)

本轮是一个新的前瞻任务，参考时间在本轮登记时生成，任务身份与 Round98/100 不同。早期轮次只作为 Agent 可见的范围化记忆，旧任务的预算没有恢复，历史评审标签没有注入，旧材料也没有被当作当前比赛事实。

当时的结果或限制原文摘录：

> 实际预算是 4 次模型调用、2 次来源请求、1 次正文窗口读取。4 次模型返回均带有执行器生成的请求绑定，原始响应没有要求或注入 task contract digest。来源回执记录了一个成功的赛程页面和一个失败的搜索请求；失败被当作缺失信息，没有被解释为“没有比赛”。

> 它没有证明数值 VoI、最优停止、预测准确率提升或独立策略收益。官方页面的语义和现实赛程真值也没有被本系统自动确认；“下一主场”仍只是来源陈述。第三次模型调用接近 180 秒上限但最终返回，说明此前的长延迟问题仍需在更多任务上观察，不能仅凭本轮一条轨迹解释。

**R104　Round104：上下文条件化 VoI 审计**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round104/README.md>)

* “已经出线”只进入公开上下文和先验，不直接推出不上主力；首发、替补质量仍是未知量。
* “打平即可出线”改变决策目标和风险，但防守/进攻意图仍需单独查询，不能由赛制规则自动填充。

当时的结果或限制原文摘录：

> 本轮是 `SYNTHETIC_ASSUMPTIONS` 机制审计，不是足球样本预测实验。它证明了上下文条件化和有限前瞻在当前声明模型下的行为，尚未证明真实来源的概率定律、LLM 自主策略收益、比赛预测提升或开放世界全局最优停止。下一步应把这些接口接到冻结的历史 as-of 数据，在训练折估计来源价值，在独立验证/测试折评估策略成本和风险。

**R105　Round105：源质量不确定性下的鲁棒 VoI**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round105/README.md>)

这一轮把上一轮的上下文条件化 VoI 再推进一步。Round104 假定候选工具的观测规律已知；开放信息环境里这个假定偏乐观：阵容、伤停或战术报道可能及时且准确，也可能过期、缺失或把同一条旧消息重新包装。若 Agent 只按名义模型计算信息价值，就可能因为过度相信工具而浪费预算，甚至把低质量返回当成证据。

**R106　Round106：把来源健康度作为二阶信息价值**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round106/README.md>)

Round105 已经说明：来源质量不确定时，Agent 不能只按名义信息价值调用工具。本轮继续问一个更具体的问题：**在读取阵容或战术正文以前，是否值得先获取来源健康度、发布时间或新鲜度状态？**

**R107　Round107：来源健康的时间切分校准**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round107/README.md>)

本轮把 Round106 的二阶来源健康探针接到 append-only `EvidenceLedgerEntry`。目标是给 Agent 的信息选择器提供一个可回放的**运行层诊断**：在固定的历史训练窗口估计来源是否能按时返回，再在后续 holdout 窗口检查可用率是否漂移。它服务于“下一步查什么、是否值得查、何时停止”的门控，但本轮不把 HTTP 成功当作语义正确，也不把来源健康直接当成比赛结果概率。

当时的结果或限制原文摘录：

> 本轮证据类为 `OPERATIONAL_METADATA_ONLY_WITH_SYNTHETIC_MECHANISM_CASES`，`global_claim=false`。合成状态、日期和漂移阈值是机制测试；它们没有证明开放世界中的来源质量、语义准确率、预测收益、LLM 策略优越性或全局最优停止。

**R108　Round108：来源可用性参与自主信息选取**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round108/README.md>)

本轮接续 Round107，将来源运行回执接入有限一步 VoI 策略。研究对象仍然是 Agent 在当前未知状态下选择下一条信息，并在获得信息后重新决定查询、暂缓或停止。

当时的结果或限制原文摘录：

> 训练/评估频率只是压力场景，不是置信区间，更不是正文语义准确率。不同 dependency_key 只表示不同分组，并未建立统计独立性；因此本轮明确不使用 Round107 的 legacy lower-bound 字段。最新依赖状态的分组频率向下一次请求转移，还需要端点、请求契约、总体和稳定性假设。

**R109　Round109：真实返回是否回答当前未知**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round109/README.md>)

Agent 自主选择下一条信息，必须区分两个问题：来源是否成功返回，以及实际阅读到的内容是否回答查询前声明的未知。本轮复用 Round103 的真实执行档案，将已有问题级判断投影为可核对的观测记录，并准备与原 Agent 判断隔离的复核材料。没有新增网络或模型调用，没有改写历史选择，也没有切换生产策略。

当时的结果或限制原文摘录：

> ## 尚未证明的部分

**R110　Round110：问题级返回的真实隔离盲评**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round110/README.md>)

Round109 已把“工具成功返回”与“返回是否回答查询前的未知”分开，并留下 1 份正文复核材料。本轮实际执行 1 次隔离模型评审，得到可引用、可回放且有独立发布时间的复核建议。原 Agent 的判断、理由、后续动作和挑选的支持引文都未作为评审输入；评审者看到原问题、原信息测试及完整可见窗口。

当时的结果或限制原文摘录：

> ## 实际结果

> 该结论与原 Agent 的窗口级判断一致。这里没有将 1/1 的标签一致性报告为准确率；同一个默认模型可能共享偏差，单条历史轨迹没有识别出策略因果收益。判定范围仅限这条查询前声明的问题及这段已读正文，未证明网站整体缺少赛程、来源通用无效或真实比赛状态。

**R111　Round111：同预算、返回后重规划的信息获取对照**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round111/README.md>)

本轮复用 decision_value.py 的信念更新与有限步 EVSI，新增共同预算比较器。目标是检验“下一步选什么、什么时候停止”，不是增加胜平负刷分项目。

当时的结果或限制原文摘录：

> - football_agent/budgeted_acquisition_comparison.py：固定顺序、均匀随机顺序、冻结初始 VoI 排序、一步自主 VoI、两步自主 VoI，共用先验、观测模型、更新器、成本和最多两次查询的预算。随机策略遍历全部顺序，不挑选随机种子。
> - 隐藏状态与完整返回带只在评估器内。策略先选择动作，执行器再暴露选中返回，更新信念并重规划；各策略对照使用同一返回带。
> - 按已有六个构造场景，加上来源退化、全部失败和紧预算条件，共九个配置。所有分支精确积分，不把分支数量算成独立比赛样本。
> - 保留失败请求成本、无目标一步信息价值成本、预测未变化成本以及停止后仍未解决的问题。互补信息可能单步价值为零，不能据此认定查询无用。

其他说明：[report.md](<D:/a1-足彩/var/parallel/agent/round111/report.md>)

**R112　Round112：真实同预算信息获取试验**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round112/README.md>)

执行前冻结两项新任务、共同预算、固定顺序、模型协议、源文件哈希和 25 分钟总截止时间。四臂为江原自主/固定、蔚山固定/自主；同上限而非同实际花费。每臂最多 2 次 HTTP、3 次 CLI、2 个窗口，窗口为前 6000 个提取字符。模型单次超时 90 秒；失败不自动重试。

其他说明：[report.md](<D:/a1-足彩/var/parallel/agent/round112/report.md>)

**R113　Round113：人工体验入口与真实自主信息获取闭环**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round113/report.md>)

本轮交付一个可供用户直接试用、查看证据并提交反馈的本地应用。核心体验是提出问题后，观察 Agent 选择下一条信息、检验返回、重新决定获取或停止。这里的工程验收不表示论文级研究问题已解决。

当时的结果或限制原文摘录：

> 最后一次模型选择仍存在实际超时，公开来源也存在不可读和相关性不足的情况。本轮把这些状态作为可见、可恢复、可反馈的体验路径交付，并没有证明模型通道或公开互联网始终可用。

**R114　比赛驱动入口修正**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round114/report.md>)

用户要给出比赛，由系统发现未知、评估信息价值、选择获取动作并分析。原页面让用户先提出研究问题并提供来源，实际交付成了问题核验助手。原入口的产品方向不符合目标。

当时的结果或限制原文摘录：

> - 168 项定向回归通过，日志 regression.log。
> - 8 项构造浏览器检查通过，见 browser-final/acceptance.json；没有真实请求。桌面和 390px 移动端截图保留。
> - 首次浏览器检查发现 Windows 状态保存时文件被读取占用的 PermissionError；失败 fixture 位于 browser/fixture-data，保留未覆盖。锁修复后重新检查通过。
> - 真实试运行输入仅为用户既有例子“日本 vs 泰国，亚运会男足”，未编造日期。系统自行生成比赛身份、实际参赛阵容与实力、晋级形势与人员安排三项未知，执行 search_web 后执行 ask_user，仅确认日期。
> - 真实开销为 2 次模型调用、2 次 HTTP；搜索返回无关摘要，比赛身份未识别，结果 needs_input/unresolved。没有将其记为完成比赛分析、最优信息选择或真实 VoI 收益。原始输入、输出、调用回执与来源归档在 live-smoke/runs；不自动追加重试。
> - Round113 冻结的 148 项产物验收仍通过，当前工作代码与旧快照差异属本次修正。

**R115　信息获取运行底座：候选生成与策略选择解耦**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round115/report.md>)

- acquisition_boundary.py：冻结事前状态和候选，独立 SelectionPolicy 接口，明确策略选择/停止输出、配置身份和绑定校验。未知价值沿用 VALUE_UNMEASURED，未知成本不填零。
- experience.py：接入可替换候选 provider 与 selector，保留原预算、归档、取消和恢复；逐次记录原始候选、实际选择、执行请求和证据的关联。
- match_proposals.py：一次模型调用返回共享分析和多个可执行候选；模型定性推荐与策略最终选择分别保存。
- acquisition_replay.py：离线重建选择，核对原始输出、配置身份、事前快照、请求回执与可见证据；不调用模型或网络。
- runtime_cli.py：无界面创建、查看、推进、停止和回放任务；默认目录 var/runtime。
- test_acquisition_boundary.py / test_match_proposals.py：新增策略替换、独立停止、过期或跨任务决策、恢复配置变化、预算、归档语义及篡改检查。

当时的结果或限制原文摘录：

> 200 项定向回归通过，见 regression.log。新增测试属于接口和执行行为检查，没有证明任何算法优越性。

> 另进行一次真实候选协议检查，保留在 live-proposal。计入 1 次模型尝试、0 次来源请求。CLI 出现 TLS 握手 EOF 与重连信号，60.29 秒后超时；没有完整返回，没有候选被接纳，没有重试或退回预算。该观察不能确认新结构化协议的真实返回兼容性，也不证明协议本身被拒绝。远端完成状态与计费仍未知。见 live-proposal-result.json 及 provider 诊断。

**R116　已有结构化足球数据接入 Agent 运行底座**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round116/report.md>)

用户只输入比赛，系统决定获取信息。为后续独立算法准备可选工具和可靠输入，本轮将已有足彩接口接入新运行链，不增加固定研究维度、不调整预测模型。

当时的结果或限制原文摘录：

> 真实来源检查于 2026-09-30 21:54:29—21:54:30（UTC+08:00）进行，人工预设两个目录入口，实际 2 次 HTTP、0 次模型调用、0 次 Agent 自主选择。近期赛程返回 200，解析得到 88 条来源内比赛记录，12,000 字符窗口展示 15 条；足彩选场返回 200 空数据，质量标记 no_data，未解释为不存在比赛。每个来源只请求一次，无重试；原始记录和离线重建见 live-source-check/journal.json。

> 已有近期、选场、详情及赔率的脱敏真实采集样本亦完成离线解析，未作为新增采集或生产入库。详情和赔率的当前在线可用性尚未在本轮请求确认。

**R117　Round117 验收报告：已采集结构化证据的本地定位与分页**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round117/report.md>)

本轮解决的问题是：来源的完整响应已经归档，但展示窗口只包含源顺序前几条，Agent 无法在不重复发起网络请求的情况下定位目标比赛。实现了一个受预算约束的 query_acquired 动作，让 Agent 自主决定是否按源比赛 ID、球队文字条件或分页读取已经获得的结构化证据。

当时的结果或限制原文摘录：

> - 258 项相关回归通过，含本轮本地查询、分页、预算、恢复、篡改检测、协议兼容和离线重建测试。
> - 构造运行使用一次构造 HTTP 获取 80 条比赛记录；展示窗口不含目标记录。Agent 之后选择 query_acquired，HTTP 已耗尽仍定位源比赛 ID 1079，再停止。
> - 构造运行与离线回放均为 0 次真实模型调用、0 次真实来源请求；回放重建平台文档和本地查询文档。
> - Round116 的一份真实归档响应仍保留 88 条规范化记录。本轮用离线的显式 ID 和分页检查其重建结果，0 次新请求、0 次 Agent 选择，因此这不是策略收益实验。
> - numeric_voi_calibrated 保持 false。当前仍是支持自动信息选择研究的可审计底座，不宣称已完成数值 VoI 或预测效果。

**R118　Round118：模型/provider 返回边界诊断（2026-09-30）**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round118/report.md>)

本轮继续建设开放信息环境中的 Agent 底座，处理一个会直接影响自主信息选择研究可信度的问题：模型调用没有形成可验证决策时，系统不能把它当作 Agent 选择了停止，也不能把未知的远端状态当作可安全重试。

当时的结果或限制原文摘录：

> - `tests/agent tests/demo`：1866 项通过；完整测试路径仍会误收集 `var/parallel` 中的冻结源副本，并因环境没有 psycopg 产生收集错误，因此本轮使用项目既有的定向回归入口。
> - 构造运行覆盖 provider turn failure：1 次逻辑选择预算被扣除，0 次 HTTP，0 条决策，状态暂停，远端结果保留 unknown，自动重试关闭。
> - 构造运行覆盖“先接纳 ask_user 决策、再发生非 JSON provider 流”：第一条 turn 保留，第二次预算独立计入，状态暂停，没有伪造第二条 turn。
> - 构造运行没有真实模型、真实来源请求或人工体验；数值 VoI 仍未校准，未宣称算法收益。

**R119　Round119：来源返回观测与语义判断分离（2026-10-01）**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round119/report.md>)

本轮处理开放信息 Agent 的一个基础错误边界：来源返回了内容，不等于内容回答了当前未知；HTTP 200、页面标题、命中球队文字和结构化记录，也都不等于信息价值已经实现。若底层把这些状态混在一起，后续 Agent 很容易把“拿到数据”误判为“应该停止”，VoI 研究也没有可审计的输入。

当时的结果或限制原文摘录：

> - `tests/agent tests/demo`：1870 项通过，命令为 `python -m pytest tests/agent tests/demo -q --disable-warnings`。
> - 相关边界回归覆盖正文返回、结构化部分覆盖、搜索发现元数据、失败返回、运行时持久化以及本地回放篡改检测。
> - 构造运行覆盖四类返回，0 次真实模型调用、0 次来源请求、0 次 Agent 选择；所有 `information_gain` 均保持 `null`，数值 VoI 未校准。
> - 本轮没有人工体验，也没有把返回观测当作策略收益或足球预测结果。下一阶段可以在这个稳定输入上研究：当前未知与候选信息之间的相关性评估、事前 VoI 估计、成本/延迟归一化和停止规则。

**R120　Round120：实际返回的问题级评估**

原始记录：[report.md](<D:/a1-足彩/var/parallel/agent/round120/report.md>)

本轮完成的是信息获取闭环中的执行反馈接口。用户只输入比赛，Agent 自主生成未知问题、有用返回判据和候选行动；实际来源返回后，下一次正常规划调用对照原问题评估，再选择继续或停止。评估只消耗已有规划调用，不额外唤起模型。

当时的结果或限制原文摘录：

> 真实模型调用 0，真实来源请求 0，真实 Agent 策略选择 0。构造运行验证执行接口，不证明模型实际遵循新协议或带来信息收益。引用检查验证出处与可见性，不验证语义蕴含、矛盾或世界真值；所有返回评估都是 ACTOR_PROVISIONAL_NOT_GOLD，realized_decision_gain=null，numeric_voi_training_eligible=false。

**R121　Round121：真实 Provider 契约回执与服务绑定**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round121/README.md>)

本轮围绕开放信息 Agent 的底层可信边界，补齐真实 Provider 调用前的契约审计。回执现在绑定公开输入、协议版本与协议指纹、响应 Schema、提示词摘要、流式响应、输出文件、事件摘要和能力开关；工具事件保持空集，语义判断继续交给 ExperienceService。

当时的结果或限制原文摘录：

> - 定向回归：70 passed。
> - 标准范围全量回归：1909 passed。
> - 真实模型调用：0。真实来源请求：0。
> - 根目录全量收集仍会触发历史快照重复模块名和可选 psycopg 依赖；本轮采用项目既定 tests/agent tests/demo 口径。

**R122　Round122：本地价值选择策略**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round122/README.md>)

本轮把核心创新从“Agent 生成候选”推进到“本地策略决定是否执行候选”。新增 value-aware-selection-v1，接收同一批 Agent 候选和 QueryValueAssessment，结合当前可见资源预算做可回放选择。

当时的结果或限制原文摘录：

> - 定向回归：51 passed。
> - 标准 tests/agent tests/demo 回归：1913 passed。
> - 策略审计：四个场景全部通过。
> - 真实模型调用：0；真实来源请求：0。
> - numeric VoI、世界真值和实际决策收益：仍未校准。

**R123　Round123 将本地价值选择策略接入 headless match runtime。**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round123/README.md>)

核心链路：Agent 生成候选 -> CandidateValueModel 读取已注册规则 -> ValueAwareSelectionPolicy 选择、暂缓或停止。

**R124　Round124: context-sensitive candidate value model**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round124/README.md>)

Status and validation totals are recorded in acceptance.json after successful validation.

**R125　Round125: current-return review in the same value selection**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round125/README.md>)

Status: ROUND125_CURRENT_RETURN_VALUE_SELECTION. The authoritative frozen receipt is acceptance.json.

**R126　Round126: separate control actions from information value**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round126/README.md>)

Status: `ROUND126_STOP_BOUNDARY_AND_RESOURCE_GATES`; the frozen receipt is `acceptance.json`.

**R127　Round127：历史策略身份的显式迁移**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round127/README.md>)

Round126 introduced `value-aware-selection-v2` and made the declared resource
cap part of the persisted policy identity. That correctly prevents a changed
cap from consuming another provider request, but it also means an older v1 run
would pause even when its saved semantics can be checked. This round adds one
explicit migration boundary in the resume path.

**R128　Round128：迁移后的历史选择回放**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round128/README.md>)

Round127 made an auditable v1-to-v2 identity migration. A run can already have
accepted v1 choices before it is resumed, so replay must preserve those
historical bindings after the state configuration becomes v2. This round adds
the smallest lineage rule: replay accepts an old selection fingerprint only
when a `selection_policy_migrated` event points from that fingerprint to the
current policy fingerprint and version.

**R129　Round129：运行时正式接入 QueryValueContract**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round129/README.md>)

Round126—128 已经把候选价值、资源门控、停止范围和历史策略回放分别固定下来，
但比赛运行时的 `ValueAwareSelectionPolicy` 仍只写入自有诊断字段，
`autonomous_context_loop` 使用的 `QueryValueDecision` / `StopCertificate` 则是另一套投影。
本轮只做协议对齐，不改变候选排序、停止条件或数值估计：

当时的结果或限制原文摘录：

> 验证：受影响契约回归由 `regression-result.json` 记录，七项形式化审计由
> `formal-contract-audit.json` 记录。审计全部使用构造数据，真实模型请求和真实来源请求均为 0，
> 数值 VoI 仍是声明模型，不构成校准收益证据；完整仓库套件不在本轮重跑。

**R130　Round130：动作级成本—证据—后续选择账本**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round130/README.md>)

- 选择前：候选级声明价值、选择理由、策略范围、独立资源预算和 `prospective_digest`；
- 执行中：规划调用与实际 HTTP / local-read 预约的独立成本；
- 返回后：技术 `return_observation`、来源覆盖、原问题和 useful-if；
- 再选择：暂定返回评估、剩余未知和下一次选择操作。

当时的结果或限制原文摘录：

> 失败及 pending 预约按实际预约计入资源槽位，HTTP 调用和本地读取不相加。`observed_information_gain`、`realized_decision_gain` 和数值 VoI 训练资格保留为空/false；暂定语义评估仍不是 gold label。

**R131　Round131：自主信息选择的离线评估契约**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round131/README.md>)

本轮把动作账本交给独立离线评估器时的边界固定下来。`value-evaluation-contract-v1` 将一次运行拆成四类证据：

当时的结果或限制原文摘录：

> 本轮不重跑完整仓库套件；Round124 的 1945 项全量结果仍是历史记录。真实模型调用、真实来源请求和人工体验均为 0/后置，数值 VoI 未校准。

**R132　Round132：动作账本到离线评估的桥接**

原始记录：[README.md](<D:/a1-足彩/var/parallel/agent/round132/README.md>)

本轮建立 Round131 评估契约与运行时动作账本之间的最小稳定接口。目标是让后续 evaluator runner、场景夹具、策略基准和演示都消费同一种 episode，而不把评估结果倒灌到 Agent 的事前状态。

当时的结果或限制原文摘录：

> episode 没有 evaluator sidecar 时保持 `partial_observation`。Agent 的 `ACTOR_PROVISIONAL_NOT_GOLD` 不会成为独立标签，`numeric_voi_training_eligible` 固定为 `false`。本轮只完成契约接线，没有真实来源请求、模型调用、比赛结果标签、VoI 校准或人工体验。

**R133　Round133：离线评估、情景夹具与策略回放**

原始记录：[round133.md](<D:/a1-足彩/docs/parallel/status/round133.md>)

本轮完成了从动作账本到离线机制评估的闭环：账本 episode 经过 public-boundary bridge 后接入 evaluator sidecar，再交给 policy benchmark 回放。运行时不读取 sidecar，actor 也看不到 evaluator-only 的 expected policy、结果标签和损失。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round133/README.md>)

**R134　Round134：as-of 与来源可靠性底座**

原始记录：[round134.md](<D:/a1-足彩/docs/parallel/status/round134.md>)

本轮围绕“Agent 自主判断下一步获取什么信息”补齐底层时钟和审计能力，保留
Round133 的 actor/evaluator 隔离。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round134/README.md>)

**R135　Round135：来源门控、平台时间适配与自主动作轨迹**

原始记录：[round135.md](<D:/a1-足彩/docs/parallel/status/round135.md>)

本轮把“Agent 自己决定下一步获取什么信息”落成一条可审计的离线执行链：平台
记录先转换为明确的 operational 或 historical reconstruction 时间视图；来源门
在 prediction cutoff 前过滤未来记录并区分证据可用性与动作可执行性；决策轨迹再
依据有限的 value/cost 估计选择 query、refresh、defer 或 stop。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round135/README.md>)

**R136　Round136：公开上下文到自主采集动作桥**

原始记录：[round136.md](<D:/a1-足彩/docs/parallel/status/round136.md>)

本轮把平台时间与来源证据转换成 Agent 可以检查的公开上下文，再接入
Round135 的 provenance gate 和 query-value contract。用户只需要提供比赛身份；
候选信息、查询/刷新动作、预算和停止条件由机器边界处理。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round136/README.md>)

**R137　Round137：执行回执到上下文重建**

原始记录：[round137.md](<D:/a1-足彩/docs/parallel/status/round137.md>)

Round136 已经选择了下一条信息动作，本轮把这条动作接到 provider 返回边界。回执
分为 `accepted`、`completed`、`failed`、`late` 和 `ambiguous`，并记录声明成本与
实际资源花费的分离值。只有 cutoff 内完成且带有公开证据行的回执，才有资格重建
actor context。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round137/README.md>)

**R138　Round138：独立评估回执到经验 VOI**

原始记录：[round138.md](<D:/a1-足彩/docs/parallel/status/round138.md>)

本轮把 Round137 的 provider execution receipt 接到经验信息价值评估。动作回执
本身只说明信息是否返回、是否按 cutoff 可用以及消耗了多少资源；它不等于价值。
价值样本必须来自独立 evaluator 对同一决策在获取前后的损失测量。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round138/README.md>)

**R139　Round139：历史 episode 上的自主 VOI 策略对照**

原始记录：[round139.md](<D:/a1-足彩/docs/parallel/status/round139.md>)

本轮把信息价值创新落到可比较的离线实验上。相同的历史 as-of episode 同时回放三
条策略：`autonomous_voi` 使用当前状态条件的 `current_voi_gate`；`fixed_all` 使用
公开候选菜单的预算基线；`fixed_order` 按预先声明的菜单顺序逐项获取。三条策略共享
同一 episode、预算、候选菜单和独立 evaluator sidecar，因而可以在配对条件下观察
资源消耗、决策损失和净效用。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round139/README.md>)

**R140　Round140：比赛上下文分层的自主信息选择评估**

原始记录：[round140.md](<D:/a1-足彩/docs/parallel/status/round140.md>)

本轮把 Round139 的单场配对比较扩展为按公开比赛语境分层的多 episode 评估。每个
episode 仍然带有 prediction cutoff、kickoff、公开候选菜单和独立 evaluator sidecar；
新增的分组标签来自 cutoff 时刻已经存在的公开 context，例如：

当时的结果或限制原文摘录：

> 分层标签只来自 public context，不读取赛果或 evaluator loss；evaluator sidecar 只用于
> 回放后的评分。Round140 仍是冻结/合成离线机制实验，没有真实来源请求、数据库访问或模型
> 调用，也没有声称不同语境下已经取得真实比赛收益。跨比赛的置信区间与独立校准留给后续
> 多赛事样本阶段。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round140/README.md>)

**R141　Round141：上下文候选提议与 receipt 后自主重规划**

原始记录：[round141.md](<D:/a1-足彩/docs/parallel/status/round141.md>)

本轮针对“Agent 自己判断下一步该获取什么信息”的核心链路做了边界收紧和真实运行接入。上下文提议器只读取预测截止时刻的 actor-visible 公共信息，例如 `qualification_state`、`tactical_layout`、`rotation_risk` 和 `dominant_uncertainty`。它生成的是候选查询的审计优先级，不是 VOI 数值，也不替代后面的状态绑定估计器。

当时的结果或限制原文摘录：

> 后续边界复核补充了一个状态一致性测试：上下文绑定只读取 `acquisition_state.snapshot.source_observations`，不读取未纳入摘要的 convenience projection，避免状态摘要与实际绑定上下文分离。复核结果：`138 passed`，无失败；本轮没有真实来源请求、模型调用或独立 gold 标签，`numeric_voi_calibrated=false`。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round141/README.md>)

**R142　Round142：状态绑定的上下文条件 VoI 区间估计**

原始记录：[round142.md](<D:/a1-足彩/docs/parallel/status/round142.md>)

本轮把“上下文候选排序”和“当前状态下是否值得获取”正式拆成两层。候选生成器仍然只产生 proposal；`football_agent/state_bound_context_value.py` 才负责数值估计，并且每个估计同时绑定当前 `state_digest`、准确的上下文单元、候选路由和估计器快照。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round142/README.md>)

**R143　Round143：泄漏安全的反事实策略回放**

原始记录：[round143.md](<D:/a1-足彩/docs/parallel/status/round143.md>)

本轮把 Round142 的区间门接入离线 episode 回放，用来检验“Agent 是否少获取无价值信息”这一研究命题。模块是 evaluator-only 的实验边界，不进入运行时数据采集路径。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round143/README.md>)

**R144　Round144：episode 配对与 block bootstrap 统计**

原始记录：[round144.md](<D:/a1-足彩/docs/parallel/status/round144.md>)

本轮为 Round143 的离线回放结果增加统计层。统计单位固定为 episode，而不是一次查询；同一 episode 中的查询序列、成本和最终损失先形成配对差值，再进入 bootstrap。这样可以避免把一场比赛中的多次动作当成相互独立样本，从而高估 Agent 自主选取信息的效果。

当时的结果或限制原文摘录：

> 本轮仍然是离线统计基础设施：没有调用真实数据源、模型或独立比赛标签，也没有声称已经完成实际赛事上的显著性检验。真实运行前必须先构造满足 Round143 transition、cutoff、sidecar 和独立 evaluator gold 约束的 episode 集合。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round144/README.md>)

**R145　Round145：历史 as-of 到反事实回放适配层**

原始记录：[round145.md](<D:/a1-足彩/docs/parallel/status/round145.md>)

本轮把已有的 `historical_asof_episode` envelope 接入 Round143 的逐步反事实回放。新增 `football_agent/replay_episode_adapter.py`，入口是 `build_replay_episode_from_historical_envelope(...)`。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round145/README.md>)

**R146　Round146：比赛情境证据绑定层**

原始记录：[round146.md](<D:/a1-足彩/docs/parallel/status/round146.md>)

本轮把“提前出线、平局即可出线、必须取胜、防守布局、轮换风险”等真实比赛情境，落成可审计的公开 context 证据层。它服务于自主信息获取策略：情境只改变候选信息的优先级，不直接生成比分、概率或结论。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round146/README.md>)

**R147　Round147：采集器到情境绑定适配层**

原始记录：[round147.md](<D:/a1-足彩/docs/parallel/status/round147.md>)

新增 football_agent/source_context_adapter.py，入口为 normalize_source_context_observations(...)、build_source_backed_context(...) 和 apply_source_backed_context(...)。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round147/README.md>)

**R148　Round148：候选查询菜单编译层**

原始记录：[round148.md](<D:/a1-足彩/docs/parallel/status/round148.md>)

本轮把 Round147 的来源适配结果接到 Agent 的候选查询菜单。系统现在可以在一个比赛状态下明确列出：有哪些信息可查、每个查询对应哪个 source/action route、资源成本是多少、哪些公开情境使它更相关。

其他说明：[README.md](<D:/a1-足彩/var/parallel/agent/round148/README.md>)

**R149　Round149：候选动作注册与执行回执重规划**

原始记录：[round149.md](<D:/a1-足彩/docs/parallel/status/round149.md>)

本轮把 Round148 的 proposal-only candidate menu 接入已有 state-bound VoI 选择器，并把选中的 candidate 映射到显式 provider action。新增 football_agent/candidate_action_registry.py，入口包括：

当时的结果或限制原文摘录：

> - 注册表不计算数值 VoI，选择仍由 select_acquisition_action 完成；
> - 执行请求不发起真实来源请求，保留真实执行接入口；
> - completed receipt 才能进入下一状态，failed/late/ambiguous 只返回需要重规划的边界；
> - evaluator sidecar 不进入 actor context、menu、registry 或 state；
> - 每个 route 都保留 candidate_id 到 action_id 到 source_id 的可审计链。

**R150　Round150：执行动作适配与回执完成边界**

原始记录：[round150.md](<D:/a1-足彩/docs/parallel/status/round150.md>)

本轮新增 football_agent/execution_action_adapter.py，把 Round149 生成的 execution request 接到 provider 返回数据的规范化边界。

**R151　Round151：自主信息获取运行时编排**

原始记录：[round151.md](<D:/a1-足彩/docs/parallel/status/round151.md>)

1. 从 actor-visible context 和 acquisition state 进入 StateBoundContextValueEstimator；
2. 由 state-bound VoI 选择 acquire、defer 或 stop；
3. acquire 时将 candidate_id 路由为 action_id/source_id；
4. 通过 provider adapter 完成 execution receipt；
5. completed receipt 重建公开上下文、候选菜单和下一轮 state；
6. 在 stop/defer、失败回执或 step cap 时留下显式审计轨迹。

当时的结果或限制原文摘录：

> 这轮仍然不发起真实网络请求，也不让用户提出查询问题。运行时接口要求外部注入 execute_fn，未来 collector 或 MCP 只需实现该 provider 回调。

**R152　Round152：多步自主信息获取的预算与重规划修复**

原始记录：[round152.md](<D:/a1-足彩/docs/parallel/status/round152.md>)

本轮直接修复既有运行链路。目标是让 Agent 在一次比赛分析内连续选择信息、接收证据、更新状态，并依据最新状态与剩余资源决定继续、停止或暂停。

当时的结果或限制原文摘录：

> - `run_information_loop` 原先每一步都传入初始预算。现在按回执中的实际资源成本逐项扣费；HTTP、模型令牌、本地读取各自记账，资源成本与决策损失效用保持独立。预估成本未全部消耗时释放余额，实际超支则记录差额并停止后续调用。
> - 失败、迟到或歧义回执原先强制要求证据。现在这些终态允许空证据并保留实际成本；成功回执仍要求证据。回调抛异常、延迟迭代响应失败或响应校验失败时，保存已接受的回执、保留预留预算并暂停，不把未知费用写成零，也不自动重试。
> - 原先重规划可能丢掉初始公开证据，并将只有 `observed_at/ingested_at` 的已接收数据漏出情境上下文。现在沿用已有 actor 证据，从平台截止时间过滤后的 actor projection 构建下一轮情境。
> - 菜单转换保留 `action_kind`、前置依赖、证据聚合模式及候选完整性。已声明可执行、但尚未查询的工具不会仅因尚无返回而退化成未知；已有可用证据与工具是否可调用分别处理。
> - 每个实际查询携带目标 `match_id`，在 dispatch 前核对状态、注册表、菜单与比赛身份，响应中的明确 canonical 目标冲突也会被拦下。provider 原生 fixture ID 和历史特征中的其他比赛身份保留独立语义。
> - 后续动作的接受时间至少达到前一次回执捕获时间。到达执行截止边界后停止调用。opaque provider 的真实网络/模型调用数保持未知，只记录确切的回调调用次数。

> 本轮是受影响模块回归，全仓测试未重跑。实际外部来源请求、真实模型调用、独立比赛结果标签均为 0。上述轨迹证明控制链路满足这些合成场景，不构成真实比赛上的数值 VoI 校准或预测提升证据。

**R153　Round153：状态绑定的 VoI 策略接入比赛运行入口**

原始记录：[round153.md](<D:/a1-足彩/docs/parallel/status/round153.md>)

`runtime_cli` 新增显式 `state-bound-voi` 选项，接入现有 `SelectionPolicy` 边界。继续由 `ExperienceService` 管理候选生成、持久化、真实工具执行、HTTP/本地预算、恢复和回放；没有再启动第二套运行循环。默认 `baseline` 保留，原 `value-aware` 声明模型也保留。

当时的结果或限制原文摘录：

> 受影响回归结果以 `var/parallel/agent/round153/regression-result.json` 为准；没有重跑全库套件。`entry-scenario.json` 与 `runtime/` 保留实际持久化入口的离线记录：候选 A/B 中选择 B，获得新返回后在 A/C 中选择 C，随后 D 区间低于成本而停止。HTTP 预约 2 次、模型 stub 调用 3 次，未询问用户研究什么，回放验证全部选择和请求绑定。

> 另外覆盖无支持、跨成本区间、同场排除、来源版本不匹配、非规范比赛身份、时间异常、合成包隔离、策略变更与排队恢复。所有本轮经验和来源返回均为合成测试；真实来源请求、真实模型调用、独立金标准标签均为 0。

**R154　Round154：比赛文字输入接入来源身份与 VoI 选择**

原始记录：[round154.md](<D:/a1-足彩/docs/parallel/status/round154.md>)

日期：2026-10-02。目标仍是“用户给出比赛，Agent 自主选择下一条信息、更新判断并决定何时停止”。本轮处理身份前提和执行闭环，不新增预测榜单或固定研究问题清单。

当时的结果或限制原文摘录：

> 本轮真实来源请求、真实模型调用和独立金标准标签均为 0，`numeric_voi_calibrated` 仍为 false。合成实验只证明链路及约束，不证明策略在真实足球任务上的收益。

**R155　Round155：大赛程本地检索与覆盖证明接入身份解析**

原始记录：[round155.md](<D:/a1-足彩/docs/parallel/status/round155.md>)

日期：2026-10-02。目标仍然是让 Agent 面对开放、不完整、持续变化的信息时，自主判断下一条信息是否值得获取，并在证据不足时停止。足球比赛只是这个机制的验证环境。本轮没有增加固定的预测特征清单，也没有把比赛结果刷成分数榜。

当时的结果或限制原文摘录：

> 已另外使用当前代码复核 Round154、Round155 的既有持久化运行目录，各重建 3 次已接受决策和 3 个身份快照，回放没有发起真实请求。记录见 `saved-run-replay.json`，可用 `record_saved_replay.py` 重做。

> 最终受影响回归 524 项、完整 `tests/agent` 测试 2,612 项均通过，失败、错误与跳过均为 0，测试期间源码哈希一致。记录见 `var/parallel/agent/round155/regression-result.json` 与 `agent-suite-result.json`，对应日志与 JUnit 文件同步保存。完整 Agent 测试不代表全仓库测试范围。真实来源请求、真实模型调用和独立金标准标签均为 0，`numeric_voi_calibrated` 仍为 false。这些结果证明的是自主查询、覆盖验证、状态绑定和可回放链路，不代表真实足球预测准确率或 VoI 已经校准。

**R156　Round156：来源陈述接入有主体、有时效的自主信息选择**

原始记录：[round156.md](<D:/a1-足彩/docs/parallel/status/round156.md>)

本轮延续“用户提供比赛，Agent 自主提出候选信息，再由价值与成本决定获取或停止”的目标。新增的是来源陈述到运行时状态的接口，以及显式采用这种语义的数值选择协议；没有增加必须查询伤停、天气或轮换的固定顺序。

当时的结果或限制原文摘录：

> 该持久化场景使用现有定性 baseline 选择采集动作，**不是数值策略完成冷启动探索的证明**。另有独立数值机制场景：相同比赛和候选菜单，分别只给出“已出线”或“必须取胜”的来源正文，合成经验包令选择在路线 a/b 间切换；保持证据而提高成本到区间上界以上时停止。路线收益是构造值，未证明真实足球信息的效果。

> 受影响回归 **611 项全部通过**，失败、错误、跳过均为 0，测试期间源码哈希一致。详细计数及源码哈希见 `var/parallel/agent/round156/regression-result.json`；验收汇总见同目录 `acceptance.json`。本轮不重复完整 Agent 套件，Round155 的 2,612 项完整 Agent 结果只作此前代码的历史记录，不宣称覆盖本轮修改。真实来源请求、真实模型调用和独立 gold 标签均为 0；数值 VoI 未完成真实校准。

**R157　Round157：未知状态下的有限探索接入持久化信息选择**

原始记录：[round157.md](<D:/a1-足彩/docs/parallel/status/round157.md>)

目标仍是用户只提供比赛，由 Agent 提出候选信息，由算法决定获取与停止。本轮补上 Round156 的运行缺口：必需上下文全未知或经验不足时，显式启用的探索额度可以支持最初几次获取；得到有来源、主体和适用区间的陈述后，继续使用原数值策略。

当时的结果或限制原文摘录：

> 首次探索菜单有两条未知路由，Agent 推荐较贵候选，算法选择较便宜候选；后续陈述进入同次状态，正值路由再次覆盖模型推荐。限制次数为一次或声明成本为 0.1 时，均在正文请求前暂缓。以上数值都是机制夹具，未证明真实赛事收益。

> 正式结果见轮次工件 `var/parallel/agent/round157/regression-result.json` 和 `acceptance.json`。受影响回归 **653 项全部通过**（包括上述 42 项），失败、错误、跳过均为 0，测试期间源码哈希一致。Round154、155、156 的保存运行均用本轮代码复验。完整 Agent 套件本轮不重复，Round155 的 2,612 项仅作为历史记录。

> 真实来源请求、真实模型调用和独立 gold 标签均为 0；数值 VoI 未完成真实校准。回放重新验证选择绑定、预算预约和原始来源链，不独立重估历史区间或验证来源陈述的世界真值。在线投影信任本地 state，手工篡改/磁盘损坏的预约由离线 verify 检出；本轮没有新增每次调用前的完整档案重建。

**R158　Round158：Agent 新查询接入工具能力、成本与有限探索**

原始记录：[round158.md](<D:/a1-足彩/docs/parallel/status/round158.md>)

目标仍是用户给出比赛，Agent 自行提出信息需求，由选择策略决定下一次获取和停止。本轮处理 Round157 的开放环境缺口：执行器已经支持新搜索和已发现链接，但数值选择层只接纳经验包内精确登记的路由。现在可以显式配置工具成本，让新提出的可执行动作进入共享的有限探索额度，保持收益未测量。

当时的结果或限制原文摘录：

> 最终受影响回归 **722 项全部通过**，包括本轮新增 69 项；失败、错误、跳过均为 0，测试期间源码和测试文件哈希保持一致。Round154、155、156、157 的已保存运行通过本轮代码回放。Round133—157 验收所引用的历史工件均有原文件或匹配备份；完整 Agent 套件本轮未重跑，历史完整套件数字不作为本轮结果。审查前启动的回归已主动停止，正式结果仅取修正后完整运行。

> 离线回放重建绑定、预算和原始来源链；历史收益区间仍属于策略诊断，回放没有独立重新估计历史收益或证明因果效用。在线投影继续信任本地持久化状态，手工篡改的预约由离线 verify 检出。本轮真实来源请求、真实模型调用和独立 gold 标签均为 0，数值 VoI 尚未完成真实校准。

**R159　Round159：本场结构化来源支持进入自主信息选择**

原始记录：[round159.md](<D:/a1-足彩/docs/parallel/status/round159.md>)

本轮处理 Round158 明确留下的错误来源状态问题：引文存在、出现同名球队，并不证明材料讨论本场，也不证明支持 Agent 填写的谓词和值。现在新增一条显式的严格路径，让执行器从来源记录直接产生数值上下文；Agent 继续自主提出信息需求、生成候选查询和选择下一步，不要求用户提出研究问题。

**R160　Round160：Agent 按需查询已采集的本场信息**

原始记录：[round160.md](<D:/a1-足彩/docs/parallel/status/round160.md>)

本轮接通比赛身份确认后的普通本地信息获取。用户仍只输入比赛；Agent 提出与当前未知事项相关的候选，策略可以选择读取已归档但窗口尚未显示的本场赔率，也可以选择其他网络信息，或在没有足够预算/价值支持时暂缓。没有新增固定赔率步骤，也没有要求用户提出研究问题。

**R161　Round161：独立证据状态评估契约**

原始记录：[round161.md](<D:/a1-足彩/docs/parallel/status/round161.md>)

本轮围绕 FootballMind 的核心问题继续推进：Agent 获取到的信息是否真的能进入当前比赛状态，以及读取行为是否应该与决策收益分开评估。实现范围限定为离线 evaluator 旁路，不改变 Round160 的线上选择、预算、执行和回放语义。

当时的结果或限制原文摘录：

> - Round161 契约测试：**16 passed**；
> - 五个条件共用同一基础摘要：`044894c2f44d9e4a4791009bc47884df41b4bd0c5a9b071d3abeabd33bd1723d`；
> - `complete` 唯一提升上下文；其他四类均触发重新规划；
> - `missing` 没有独立损失标签，因此收益保持未知；
> - `stale` 的读取成功不会被解释成正收益；
> - `conflict` 的负损失变化保留为负值，没有截断；
> - Round160 生产源码哈希再次核对为零差异；
> - Round160 的 876 项回归和16项验收仍保持通过；
> - 本轮真实来源请求、实时模型调用和独立金标准标签均为0。

> 本轮证明的是：系统能够把“读取成功”“证据是否可用”“是否进入当前状态”“损失是否变化”拆开，并对过时、冲突、缺失和迟到证据产生不同的重新规划信号。它没有证明实际 LLM 已经学会 VoI，也没有进行收益校准；样例 sidecar 只是机制验证材料，不能作为训练数据或真实市场效果。

**R162　Round162：匹配公开 episode 的反事实策略回放**

原始记录：[round162.md](<D:/a1-足彩/docs/parallel/status/round162.md>)

本轮把信息价值研究推进到可成对比较的策略回放。每个比赛 episode 的公开快照、候选菜单、预测截止时间和查询预算保持不变，只替换信息获取策略：固定菜单顺序、一次性全量查询、以及根据当前状态区间选择下一项并在上界低于成本时停止。

当时的结果或限制原文摘录：

> - `evaluation_scope=synthetic_structural_mechanism`；
> - `synthetic_or_offline_only=true`；
> - `numeric_voi_training_eligible=false`；
> - 真实来源请求为 0；
> - 实时模型调用为 0；
> - 独立金标准标签为 0。

> 本轮验证了反事实策略回放和数据隔离契约，没有验证区间估计已经校准，也没有证明真实 Agent 已经学会 VoI。下一轮应接入带有截止时间、来源链和独立结果标签的真实历史 episode，先完成小规模人工审计，再做时间滚动的 paired evaluation。真实评估必须继续区分：查询成功、证据进入状态、决策损失变化以及最终策略收益。

**R163　Round163：真实历史赛前 episode 与独立结果绑定边界**

原始记录：[round163.md](<D:/a1-足彩/docs/parallel/status/round163.md>)

本轮把已有的历史 as-of 构造器接入 Round162 的成对策略回放格式。重点不是制造一组“看起来完整”的预测数据，而是把真实来源能够证明的内容、需要独立评估来源才能证明的内容分开。

**R164　Round164：独立赛果绑定与真实采集时间修复**

原始记录：[round164.md](<D:/a1-足彩/docs/parallel/status/round164.md>)

本轮完成此前中断的实现和验收。用户的目标保持为：只输入比赛，由 Agent 自主选择下一条信息、更新判断并根据价值和成本停止。赛果绑定是评估这一机制的前提，不能替代信息收益评估。

**R165　Round165：逐动作 VoI episode 证据完整性契约**

原始记录：[round165.md](<D:/a1-足彩/docs/parallel/status/round165.md>)

Round164 解决了独立赛果能否绑定到公开比赛的问题，但这还不足以评估信息价值。本轮只补一个必要的底层边界：一次 Agent 运行是否留下了可以回放的逐动作证据。

**R166　Round166：动作账本接入逐动作证据契约**

原始记录：[round166.md](<D:/a1-足彩/docs/parallel/status/round166.md>)

本轮新增 `football_agent/round166_ledger_adapter.py`，把已验证的
`action_value_ledger-v1` 投影到 Round165 的逐动作 Episode 证据契约。

**R167　Round167：实际入口的返回准入、请求绑定与截止时间边界**

原始记录：[round167.md](<D:/a1-足彩/docs/parallel/status/round167.md>)

本轮把“Agent 选择了什么”和“来源返回是否真的可以进入当前状态”在实际
`ExperienceService` 入口上接通。请求结束后先保存原始返回，再依据任务本地
准入时刻检查取消和任务截止；被取消或超截止的返回保留 `result.json`、请求
回执和资源成本，但不会进入 Actor 的 `documents` 或 `pending_return`。已准入的
历史文档不会因任务后来结束而从公开状态消失。

当时的结果或限制原文摘录：

> 本轮仍是离线结构和边界验证：真实来源请求、实时模型调用和独立比赛金标准均
> 为 0；没有校准数值 VoI、概率准确率或收益结论。`ExperienceService` 与
> `autonomous_runtime_loop` 是两个分别验证的入口，下一阶段应在此底座上补齐持久
> episode、独立赛果/损失 sidecar，再做同初态、同预算的自主选择与固定顺序对照。

**R168　Round168：逐动作 Episode、独立损失与 paired replay 闭环**

原始记录：[round168.md](<D:/a1-足彩/docs/parallel/status/round168.md>)

本轮补齐了从“Agent 选择下一条信息”到“事后能够公平评价该选择”的底层闭环。
核心仍是让 Agent 根据当前不确定性、候选信息的预期价值和获取成本自主选择，
并在剩余价值不足时停止；本轮没有把问题改写成固定特征的足球预测。

当时的结果或限制原文摘录：

> - `tests/agent` 与 `tests/demo`：**3206 passed，0 failed，0 skipped**；
> - Episode、ledger、Experience、independent outcome 和 paired replay 定向测试均通过；
> - 无真实网络请求、实时模型调用或真实比赛金标准写入；
> - `numeric_voi_training_eligible` 在缺少独立证据时保持 `false`。

**R169　Round169：运行时 Episode 接线与独立结果投影加固**

原始记录：[round169.md](<D:/a1-足彩/docs/parallel/status/round169.md>)

本轮继续只做底层，不处理前端。目标是让“Agent 自主选择信息”在真实运行入口中可选地留下逐动作审计记录，并阻断不满足独立来源与时间证据的伪金标准进入 paired replay。

当时的结果或限制原文摘录：

> - 定向底层测试：**65 passed**；
> - `python -m pytest -q tests/agent tests/demo`：**3218 passed，0 failed，0 skipped**；
> - `python -m compileall -q football_agent tests/agent`：通过；
> - 没有新增真实网络请求、实时模型调用或真实比赛金标准；
> - 没有生成准确率、收益或数值 VoI 校准结论。

**R170　Round170：运行时概率后验绑定**

原始记录：[round170.md](<D:/a1-足彩/docs/parallel/status/round170.md>)

本轮继续只处理底层。目标是把 Agent 主动选择信息的运行时状态，与真正的比赛概率后验绑定起来，使 Episode Journal 能记录信息获取前后的认知变化，同时保持 VoI 选择逻辑和时间边界独立。

当时的结果或限制原文摘录：

> - 后验接线及受影响 runtime 测试：**17 passed**；
> - `python -m pytest -q tests/agent`：**3170 passed，0 failed**；
> - `python -m pytest -q tests/demo`：**53 passed，0 failed**；
> - `python -m compileall -q football_agent tests/agent`：通过；
> - 合并计数：**3223 passed，0 failed**（agent + demo）；
> - 没有新增真实网络请求、实时模型调用或独立比赛金标准；
> - 没有宣称准确率、收益或数值 VoI 校准结果。

**R171　Round171：研究级 EVSI/VoI 契约与 AIC 参赛就绪度冻结**

原始记录：[round171.md](<D:/a1-足彩/docs/parallel/status/round171.md>)

本轮从“工程底座继续加固”切换到“把核心科学主张变成可验收算法”。重点仍是 Agent
自主选择下一条信息，而不是增加数据源数量或把 LLM 工具调用包装成创新。

当时的结果或限制原文摘录：

> 本轮只验证 EVSI/NetVOI 的输入、计算和停止契约，没有真实来源请求、实时模型调用或
> 独立比赛金标准；`numeric_voi_training_eligible` 仍为 `false`。数学期望值只有在响应
> 模型、预测器和成本模型用 cutoff-safe 历史数据校准后，才可进入真实策略比较。

**R172　Round172：首个真实历史 Episode 与未知候选 VoI 纵向切片（2026-10-05）**

原始记录：[round172.md](<D:/a1-足彩/docs/parallel/status/round172.md>)

Round172 完成了 AIC 研究路线的第一个纵向切片：使用 Round163 已冻结的
2026-09-26 斯洛文尼亚 vs 苏格兰赛前公开快照，将独立的苏格兰足协赛后页面绑定为
`INDEPENDENT_GOLD_RECONSTRUCTED` 结果标签，并生成一个只读的 Expected VoI 选择轨迹。

当时的结果或限制原文摘录：

> - 独立结果来源归档在
>   `var/parallel/agent/round172/independent-source/`，保存 URL、抓取时间、原始 HTML
>   的 SHA-256、结果和重建时钟声明。
> - `var/parallel/agent/round172/episode.json` 通过 Round163 bridge 校验；赛果只在
>   `evaluation-sidecar.json`，没有进入 Actor 可见的 `public.json`。
> - 新增
>   `football_research/evaluation/voi_vertical_slice.py`，在一个冻结的公开 Episode 上
>   调用严格 `expected_voi` 契约，输出候选价值分解、选择、停止/继续决定和状态摘要。
> - `candidate-menu.json` 已重建为 `lineup_context`、`qualification_context`、
>   `tactical_context` 三个人工声明的待查方向，尚未证明其对应实时可调用的数据工具。它们的 source id 均不在
>   actor 已见的 `source_observations` 中，且 `voi-trace.json` 在开发响应模型下选择了
>   `lineup_context`；晋级信息的 EVSI 为零，战术信息的 EVSI 小于获取成本。这是机制演示，
>   不是校准后的比赛收益。
> - 新增 `tests/agent/test_round172_real_episode.py` 和
>   `tests/research/evaluation/test_voi_vertical_slice.py`。

> Round172 现在证明的是“真实赛前边界 + 未观测候选菜单 + 独立赛果 sidecar + 自动候选
> 选择”可以放在同一份 Episode 中而不泄漏。它还没有证明 FootballMind 的预测优于固定
> 顺序策略；该结论必须等待校准 response model、固定概率基线、逐动作损失和成对回放。

**R173　Round173：真实赛前先验与未知信息价值接线（2026-10-06）**

原始记录：[round173.md](<D:/a1-足彩/docs/parallel/status/round173.md>)

本轮完成可重复执行的底层切片：原始赛前响应 → 经摘要校验的公开赔率 → 固定市场基线
→ 绑定公开状态的初始概率 → 显式未知的候选 VoI → 独立结果评分。
核心问题仍是 Agent 在不完整、动态信息环境中选择下一条信息及停止时机。

当时的结果或限制原文摘录：

> ## 实际结果

> 可用 `--output` 指定独立输出目录。输出包括公开状态、模型工件、初始概率、VoI 轨迹、
> 未执行的状态转换、评估 sidecar 和 report。验收命令及实际结果记录在同目录 acceptance.json。

**R174　Round174：学习型 VoI 接入受控历史信息获取**

原始记录：[round174.md](<D:/a1-足彩/docs/parallel/status/round174.md>)

复用已经训练好的 `JointReturnPredictor`、原有历史数据加载器和
`ControlledEpisode`，将它们接入 Round171 的严格 EVSI 契约。没有新增爬虫、前端、
人工指定的重要性分数，也没有重新拟合或针对测试比赛调整模型参数。

当时的结果或限制原文摘录：

> 本次仅有两类工具且预算足够，固定顺序查完与全部查询逐场相同，不能算两份独立的
> 优势证据。VoI 相比不查询变差 0.035612；本轮没有证明模型获得正的实际信息收益。

> 1. 保留本轮负结果与原冻结文件，先定位响应分布/后验失配。复用已有 OOF、残差诊断
>    和校准代码，严格区分“模型 EVSI”“经验收益门控”；不能把校准后的单个收益分数
>    冒充满足分支一致性的联合概率模型。
> 2. 用仅早于预测截止的训练与验证样本确定修正方案，再用新的冻结时间段检验。当前
>    290 场只用于定位问题，不再作为支持新算法有效性的独立测试。
> 3. 扩展到首发/伤停、赛制约束、战术等开放候选时，先采集带时间与真实返回的轨迹，
>    为响应和证据条件后验建立训练支持；已有预算探索器可服务冷启动，但探索不等于
>    正 VoI，LLM 不可直接自报数值收益。国家队样本必须与目前国内联赛模型支持区间
>    分开处理。

**R175　Round175：信息更新强度修正与冻结新赛季检验**

原始记录：[round175.md](<D:/a1-足彩/docs/parallel/status/round175.md>)

Round174 跑通了自主选择信息的闭环，但模型预计的信息收益与获得信息后的实际
变化不一致。本轮直接处理响应分布与后验更新过强的问题，保持原有信息成本、预算、
初始预测、候选动作和支持度门限，随后用冻结参数接入新获取的数据检验。

当时的结果或限制原文摘录：

> 日期：2026-10-06。代码、原历史回放、新来源入库、新时期检验和验收均已实际完成。

> 每个冻结模型只在原来的时间验证集上选择一个 s，目标是战绩、xG、两者联合三个
> 已获取集合的平均 log loss。每场比赛等权，三个状态不计为三场独立比赛。
> 一维凸目标用端点导数与确定性二分求解；平坦目标选 0。验证标签必须早于评估
> 预测时点，训练比赛和适配比赛在推断时仍被拒绝。测试集没有进入参数选择。

> 相对原策略，平均损失差为 -0.021097，按 UTC 周成对重采样的描述性区间为
> [-0.055531, 0.012456]；相对不查询差为 +0.014515，区间为
> [-0.007955, 0.039930]。这些区间不支持确定的优势结论，且没有修正此前反复研究
> 这个赛季、模型拟合或多重比较的影响。研究重点仍是信息获取机制，不是优化这份分数表。

其他说明：[round175-design.md](<D:/a1-足彩/docs/research/acquisition/round175-design.md>)

**R176　Round176：学习型 VoI 与未知信息探索的实际执行闭环**

原始记录：[round176.md](<D:/a1-足彩/docs/parallel/status/round176.md>)

日期：2026-10-06。沿用 Round171 的 A/B/C/D 工作包，推进 C 运行时集成与 B 返回数据记录；
没有重开项目路线，也没有调低查询阈值来制造更多动作。本轮交付是无界面的研究运行入口，
不是完整 AIC 效果验收，也尚未替换现有体验服务。

当时的结果或限制原文摘录：

> - **A 算法包：仍未完成。**数值 VoI 的响应分布尚不能充分区分比赛上下文；
>   本轮均匀探索不是新的信息价值算法，也没有解决开放内容的概率响应学习。
> - **B 数据包：新增在线查询返回记录。**有来源字节、真实获取时钟和选择概率，
>   但缺少比赛相关实质内容、响应映射及独立赛后结果，不能计作已完成的数值 VoI 训练样本。
> - **C 集成包：完成这一段实际执行。**已把学习型 VoI、未知探索、返回再规划接在一起；
>   用户不需要人为指定调查问题。当前是受限来源的无界面研究入口。
> - **D 评估包：完成机制回放，效果验收仍未完成。**没有改写 Round174/175 的负结果，
>   尚不满足七策略公平比较或 AIC 研究优势主张。

> 本轮 `aic_research_claim_ready=false`、`open_world_policy_benefit_proven=false`。
> 已通过的是执行与回放验收；自主信息选择的实际预测收益尚未证实。

**R177　Round177：比赛入口、正文追问与条件信息价值的联合交付**

原始记录：[round177.md](<D:/a1-足彩/docs/parallel/status/round177.md>)

日期：2026-10-06。本轮沿用已有 A/B/C/D 工作包，并行完成来源发现、正文证据、
条件响应模型和共同评估，由总控接入同一后端入口。用户只需提供比赛身份。
当前不做前端；没有更换 Round175 冻结的默认概率模型。

当时的结果或限制原文摘录：

> 共同评估使用旧开发集 290 场、7 个时间切分、相同初态/工具/成本/预算，比较不查询、
> 全查、固定顺序、随机和 strict-VoI。初次比较为 15,660 次策略运行；残余关联实验新增
> 5,800 次，复用 5,800 条旧轨迹。大量重复策略运行不增加独立比赛样本数。

**R178　Round178：正文条件响应学习与数值 VoI 执行（2026-10-06）**

原始记录：[round178.md](<D:/a1-足彩/docs/parallel/status/round178.md>)

本轮完成学习模型、数据审计、来源诊断、共同评估四项并行工作，并接入既有后端。
用户入口仍只需要一场比赛。本轮没有新增前端或让用户指定调查问题。

当时的结果或限制原文摘录：

> ## 实际完成的核心变化

> 相应 `football_agent/text_voi_policy.py` 已接入 `LearnedVOIRuntime` v3。每次调用前
> 保存模型和分支；每次返回后保存实际时钟、正文摘要、对应后验和下一次选择。
> 原始网页归档、文本模型和分析结果可由现有 replay 路径零网络重建。v1/v2 保留原语义。
> 生成初始概率的模型和文本模型都必须在实际调用前训练完毕。

> 该模型是有序获取过程，未证明不同查询顺序具有相同联合分布。当前词袋表示也不是
> 语言语义或信息真实性验证。VoI 理论、文本聚类本身不是本项目的新理论贡献。

**R179　Round179：学习信息发现的后续价值与真实多场赛前轨迹**

原始记录：[round179.md](<D:/a1-足彩/docs/parallel/status/round179.md>)

本轮交付直接服务于“用户给出比赛，Agent 自己选择下一步信息”。已经将训练轨迹学习的
候选发现转换、预算约束两步 VoI、实际网页执行与回放接通；同时用当前公开数据建立了
三场真正赛前的市场初态与获取轨迹。没有启用合成模型处理真实比赛。

当时的结果或限制原文摘录：

> 固定查询的纯预测损失略低，两步策略付出更少成本，综合目标更好；不是所有指标都获胜。
> 本设计只有一个初始发现动作，两个后续来源提供相关转载，因此固定与随机在此等价。
> 实验验证发现的后续价值和选择性停止，没有证明一般性的多工具排名能力。
> 所有负更新、失败、未知与不获胜单元保留；浮点舍入产生的微小负数与小于 −1e-12 的
> 实质负收益另行统计。最终 16 项机制检查全部通过。

> 三份真实运行与 6 份原始返回均已零网络回放，run/analysis/report 字节一致。
> 当前为 3 个独立比赛、6 条实际返回、0 个独立赛果、0 条合格真实正文训练样本。
> 缺少市场先验证据的问题已消除，剩余资格缺口是独立赛果及一个入口页类型未知。
> 即使赛果齐备，这批数据也未提供有效新闻带来正收益的证据。

**R180　Round180：真实比赛身份、正文获取、实际追问与独立赛果接入**

原始记录：[round180.md](<D:/a1-足彩/docs/parallel/status/round180.md>)

本轮把真实数据缺口合并处理。用户仍只提供比赛，部署配置提供来源和预算；程序核实身份、
生成候选、获取实际页面、读取已发现链接、根据返回生成后续查询，最终保存分析与完整回放。
不做前端，不改默认 Round175 数值模型，不把尚未训练的真实正文查询赋予虚构 VoI。

当时的结果或限制原文摘录：

> 真实策略仍是 UNKNOWN VoI 下的已有有限均匀探索；本轮取得正文并不等于已学会真实信息
> 价值。三场概率均未改变，不能宣称胜率改善或成本收益获胜。Round179 与 Round180 查询
> 的是同三场比赛，合并时仍只有三个独立比赛，不能计作六个样本。

> - `var/parallel/agent/round180/live-batch/`：三场实际运行、分析报告、市场与身份原始归档。
> - `var/parallel/agent/round180/replays/`：本轮零网络回放。
> - `var/parallel/agent/round180/episodes-pending.json`：正文响应与独立赛果资格审计。
> - `var/parallel/agent/round180/identity/`：身份绑定、赛果核验、真实请求与解析契约。
> - `var/parallel/agent/round180/discovery/`：来源诊断、实际文章与新旧投影对比。
> - `var/parallel/agent/round180/verify_round180.py`、`acceptance.json`：独立可重算总验收。
> - `docs/agent/verified-prospective-runbook.md`：采集、回放、待真实完赛后接入结果的命令。

> 不重置用户的十轮预算。本轮之后只保留两次主交付：下一次冻结后端、现有实验、案例及 AIC
> 材料；最后一次只修阻断交付的问题。赛事尚未结束不新增等待轮次，不启动后台任务。
> 真实正收益仍由未来冻结评估决定，现阶段按可运行研究原型交付，完整披露合成机制结果、
> 真实历史负结果、真实赛前轨迹与待结算数据，不用新增协议或测试数量替代研究结论。

**R181　Round181：后端、证据与 AIC 材料封版**

原始记录：[round181.md](<D:/a1-足彩/docs/parallel/status/round181.md>)

本轮完成上一轮明确的“冻结可用后端、既有策略比较、案例和答辩材料”。使用三个现有
子任务分别实现统一入口、独立实验证据重算和研究材料，总控负责源码审阅、依赖冻结、
发布包及独立目录联合验收。没有新增用户对话，也没有向旧对话发送未授权消息。

当时的结果或限制原文摘录：

> 旧 290 场开发比赛在多轮被复用；主预算 0.05/2 下，strict-VoI 未优于不查询。
> 较低预算 0.03/1 的上下文策略 log loss 1.079009 略好于 1.080229，但成本加权目标
> 1.085854 较差。冻结新时期 50 场中修正策略不查询，与基线同为 1.090590。
> Round178/179 的合成固定全查在纯损失上略优，两步 VoI 的优势是声明费用下的综合目标。
> 真实三场八次请求没有独立标签和经真实文本训练验证的正数值 VoI 行为。

**R182　Round182 最终阻断修复及交付验收**

原始记录：[round182.md](<D:/a1-足彩/docs/parallel/status/round182.md>)

本轮执行 Round181 约定的最后一次阻断修复与验收，原十轮预算不重置，剩余常规交付
轮次为零。工程收敛完成的范围是已声明的后端研究原型；真实 VoI 收益尚未证实。

当时的结果或限制原文摘录：

> 这次实际调用发生在修复前，源代码与 Round181 包逐项核对相同，出处保存于
> `live-source-reference.json`。之后的兼容入口直接接受旧记录，未回填新的 ID 或文件
> 绑定。新增 UUID 行为通过完整零网络集成验证，未为多取一次成功页面而重试真实比赛。
> 这两个请求属于工程检查，不计入旧冻结效益表，也不增加独立比赛、标签或训练样本。

**R183　Round183：开放正文的经验信息价值闭环**

原始记录：[round183-text-risk.md](<D:/a1-足彩/docs/research/acquisition/round183-text-risk.md>)

本项是用户在原十轮工程收敛结束后授权的一项算法攻坚任务。原十轮剩余轮次仍为零，
不重启常规工程排期。工作范围是“给定比赛，由 Agent 自主决定查询、更新与停止”；
没有前端、额外爬虫框架或结构化模型竞赛。Round182 发布包保持原样。

当时的结果或限制原文摘录：

> 完整逐场轨迹、分箱和负结果在 `var/parallel/agent/round183/evaluation/`；
> `verify_results.py` 独立重算比分损失、成本、时间映射、身份与模型绑定，不重训。

**R184　Round184：真实赛前采集与有限候选响应面板**

原始记录：[round184-prospective-panels.md](<D:/a1-足彩/docs/research/acquisition/round184-prospective-panels.md>)

完成时间：2026-10-06。原十轮工程收敛已在 Round182 结束，本项是其后单独授权的真实信息获取研究任务，不重置原轮次预算。

当时的结果或限制原文摘录：

> 项目要回答的是 Agent 在当前证据下应查询什么、是否值得查询、何时停止。上一项已有有符号风险改善学习接口，但缺少真实监督，合成开发复验未证明策略优势。继续调整旧分数不能补齐这个缺口。

> 本项补的是可追溯的真实响应覆盖。原日志已有实际菜单、选择概率和已选返回；未选择请求的返回一般未知。新增有限面板先固定候选，再单独采集响应，让未来比较能在明确、相同的有限信息集合中逐步读取。它不重建整个开放互联网，不是历史完整反事实，也不是一个新的学习算法。

> ## 实现与实际结果

**R185　Round185：真实正文、来源语言关联与语料资格闭环**

原始记录：[round185-real-text-corpus.md](<D:/a1-足彩/docs/research/acquisition/round185-real-text-corpus.md>)

交付日期：2026-10-07，北京时间。核心仍是由 Agent 选择信息；本项完成数据处理包，
不拟合新模型，不改变旧开发集分数，也不把处理成功当成 VoI 收益成立。

当时的结果或限制原文摘录：

> f_training_candidate 表示“可进一步形成离线正文预测语料的候选”。标签缺失时
> supervised_text_eligible=false；即使以后标签具备，也不自动成为在线 VoI 校准序列。
> 同场、转载、多个运行和不同投影版本必须按 match_id 一起划分训练/评估，不能把两篇
> 文章算成两场独立比赛。来源适用描述不作为有用性筛选或权重。

**R186　Round186：信息选择的历史泛化检验**

原始记录：[round186-history-generalization.md](<D:/a1-足彩/docs/research/acquisition/round186-history-generalization.md>)

本项接续 Round185 数据处理包，处理一个影响自主信息选择的算法瓶颈：模型对已经
读取的信息采用精确有序匹配，未见过的查询顺序可能让下一步信息价值全部变为
UNKNOWN。研究对象仍是给定比赛后机器选择下一步信息，而非要求用户指定问题。
原十轮工程收敛已经结束，本项是单独授权的研究任务，不重置原计划。

当时的结果或限制原文摘录：

> 1. `(channel, request_cluster, response_token)` 的完整多重集合一致，重复次数保留。
> 2. 每个来源/请求簇内部的响应次序一致。
> 3. failure、no_data、index、discovery 的有类型事件清单和相互次序一致。
> 4. 完全同序，或仅一次相邻交换；不支持更长距离和删除历史事件。

> 历史核为 `K_h = exp(-0.5 * adjacent_swap_count)`，再乘已有请求相似度和先验概率核。
> 每场比赛的多条记录仍共享一个权重上限，原独立比赛及结果类别支持门槛不降低。
> 另以每场历史权重均值之和检查绝对支持量，防止权重统一衰减后，经归一化的有效
> 样本数仍把它当作完整支持。例如 15 场纯邻居仅提供约 9.10 的历史权重质量，不能
> 满足 12 的门槛。它是保守支持指标，不是新的独立比赛数或置信保证。

> 确认集保留正文重排、已见中性事件移动、重复转载、同请求冲突版本、跨请求顺序
> 敏感反例，以及未知正文、来源、请求和低支持来源。相同比赛的参考与压力历史
> 只是配对机制探针，不增加独立比赛数量。随机种子、费用、门槛及训练覆盖不按
> 确认结果调整。数据生成或运行实现错误必须保留原冻结文件和修复说明。

**R187　Round187：固定预测器的信息获取价值校准**

原始记录：[round187-transfer-risk.md](<D:/a1-足彩/docs/research/acquisition/round187-transfer-risk.md>)

本项处理 Round186 已发现的负迁移：相邻历史满足结构匹配，不代表当前请求会带来
正的实际预测改善。目标是在给定比赛及已有证据后，由机器选择下一步信息；本项
不增加前端、不扩大爬虫框架，不用已看过的确认结果继续调参。

当时的结果或限制原文摘录：

> 历史中的正文统一标为 article，取消精确正文簇组合这一分格条件；来源、请求身份、
> 重复次数和顺序继续保留。failure、no_data、index、discovery 仍分别表示，不删除。
> 局部组使用已有的概率置信度两档和模型 EVSI 四档；样本不足只向同一父组收缩。
> 最少独立比赛支持 12，收缩强度 20，预先固定，不根据确认集得分调整。

> 正式实验只运行一次，预测器拟合次数为零。600 场校准形成 2,400 条原始转移，
> 来自每场两个独立模拟分支；不是 2,400 场独立比赛。440 条读前受支持转移来自
> 421 场比赛，其中 117 条为实质负更新；1,960 条读前 UNKNOWN 保留审计。
> 本次正式校准内没有“读前支持、实际更新 UNKNOWN”的记录，其阻断逻辑另由测试验证。

> 新策略减原策略的平均目标差为 -0.0205923。以比赛为单位、固定种子重采样 2,000
> 次的描述性百分位区间为 [-0.0617936, 0.0129114]，包含零，不支持稳定优势声明。
> 该区间不是开放环境覆盖保证。192 场中只有 7 场改善、6 场恶化、179 场相同；
> 变化集中在矛盾消息组，其余七组没有变化。两请求预算下两个价值策略仍产生相同
> 决策，第二步大多缺少模型支持，不能据此主张多步规划成立。

**R188　Round188：真实结果反馈与正文驱动的后续查询**

原始记录：[round188-real-feedback-and-articles.md](<D:/a1-足彩/docs/research/acquisition/round188-real-feedback-and-articles.md>)

本项处理两个直接阻断自主信息获取学习的问题：真实赛前响应缺少独立赛果监督，
以及新正文解析器输出的 article 无法进入旧证据投影。未重新拟合构造模型或调整
Round187 的确认结果；原十轮工程收敛约束不重置。

当时的结果或限制原文摘录：

> 下一阶段仍须按真实时序积累独立比赛：先训练并冻结真实正文预测器，再使用更晚、
> 身份不重叠的真实响应学习信息价值，最后在再后的比赛确认。不能把两条响应当作
> 两场比赛，也不能在同一批标签上同时训练预测器和校准器。

**R189　Round189：把信息获取接到事先登记的真实学习阶段**

原始记录：[round189-prospective-learning.md](<D:/a1-足彩/docs/research/acquisition/round189-prospective-learning.md>)

本项目标是使现有开放信息采集器在未来真实学习中使用正确的预测器和信息价值校准器，
并保留能够回放的实验边界。用户仍只给比赛，研究用途由实验监督层登记，
不要求用户选择伤病、天气或查询问题，也不把分组用途传入 Agent 的候选生成输入。

当时的结果或限制原文摘录：

> 截至北京时间 2026-10-07 02:37:20，旧市场归档中有 15 场距离开球超过半小时，
> 其中 14 场在未来 72 小时，但没有可直接新增的合格独立比赛。
> 美国—加拿大、芝加哥—温哥华虽有历史常规时间赔率，均超过初始化时的一小时新鲜度约束。
> 02:32:55 的实际市场更新返回 HTTP 200、`data=[]`；这不算新增数据。

**R190　Round190：真实训练入口与一个新的独立采集场次**

原始记录：[round190-real-training-and-new-fixture.md](<D:/a1-足彩/docs/research/acquisition/round190-real-training-and-new-fixture.md>)

本项把原始响应到冻结正文预测器的训练入口接通，并取得一场具有新鲜初态、
独立官方身份的新比赛，完成预登记后的实际 Agent 获取和精确回放。
用户仍只提供比赛；查询由现有策略与预算机制选择。

当时的结果或限制原文摘录：

> 审查发现旧桥在前序 receipt 无法读取时，可能将后续响应按缩短的历史处理。
> 本入口核对真实选中请求步骤与重建 episodes 的完整对应，缺步时完整排除该运行，
> 同时保存全部原始诊断。中性返回、失败和未知的处理依据已有协议，不只保留好看的正文。

> 固定使用既有 `min_matches=12`、`min_class_matches=1` 等参数。至少 12 个独立合格
> 比赛并覆盖主胜、平局、客胜，只是可拟合的必要条件；不意味着未来每个来源、查询、
> 概率状态与历史都有条件支持，也不证明获取信息能降低实际风险。
> 不足时仅输出 `not_ready`，不创建模型；满足条件后调用既有 v1 或显式邻域 v2，
> 保存并重新加载核对完整 payload，仍保留 `production_ready=false`。

> 北京时间 2026-10-07 03:17:52 已对真实旧 9 个 run 执行此入口：登记身份 6 场，
> 合格监督只有 1 场、2 条响应，类别为主胜 0／平局 1／客胜 0。因此 `not_ready`、
> `model=null`、未拟合、无冻结模型文件。当前仍没有可以投入真实校准的 f。

**R191　Round191：正文发现、导航隔离与真实失败轨迹**

原始记录：[round191-content-discovery.md](<D:/a1-足彩/docs/research/acquisition/round191-content-discovery.md>)

本项围绕“用户提供比赛，Agent 自主获取下一条信息”修复信息入口。
查询内容没有改成用户问题或人工指定的伤停、天气清单。
信息可读取、属于本场、具有正信息价值，是三个分别核验的问题。

当时的结果或限制原文摘录：

> 追加验证只使用已经获得的原始字节，不能改写原 Actor 的 NO_DATA，也不计为新的
> 实际查询、有效信息收益、独立比赛或监督样本。运行时源码副本在 source-at-live
> 保存，新增版本与原执行版本的配置分别记录。

> 累计仍是 7 场独立比赛，实际运行 11 次、响应 32 次；独立有赛果比赛 1 场，
> 合格监督响应 2 条。当前训练支持不足，未生成真实预测器或 critic。
> 本轮没有等待或轮询尚未到期的五场结果；加拿大常规时间结算规则继续未知。

**R192　Round192：接通真实赛果与查询前候选上下文**

原始记录：[round192-results-and-query-context.md](<D:/a1-足彩/docs/research/acquisition/round192-results-and-query-context.md>)

本项围绕 Agent 的自主信息选择补齐两处实际输入缺口：将到期官方赛果关联到原始赛前采集链；保留查询前已读页面中的候选标签及出处。没有训练新模型、改变模型支持门槛或将标题直接转换成价值分数。

当时的结果或限制原文摘录：

> 现有 11 次实际运行、32 次实际响应仍对应 7 场物理比赛。真实训练入口重验全部登记、身份、市场初态及完整有序历史后，可用独立比赛由 1 增至 6，监督响应由 2 增至 26。类别为主胜 0、平局 2、客胜 4，故返回 `not_ready`、`model=null`，未输出冻结模型。26 条不等于 26 篇有效新闻：其中 article 5 条，覆盖 3 场，其余是可编码的失败、无数据或目录响应。正文内也存在跨比赛重复及无关内容。

> 集成验证使用真实采集/归档/投影/运行/回放代码与构造响应，检查先看到父页标签、后请求子文章，未读正文不出现在先前公共状态，保存后的逐字回放一致。这类测试不计实际 HTTP 或训练样本。最终相关回归1204项全部通过；11条真实原轨迹在当前代码下全部零网精确回放，新的赛果关联和26条训练输入逐项重建一致。正式真实数据训练报告在 `var/parallel/agent/round192/real-training/training-report.json`；最终零网核验运行：

> 当前 f 和 critic 尚未使用新标题上下文；本项没有证明查询选择更好或风险下降。下一项应扩大预先登记的独立赛前样本，覆盖可重复来源、成功正文及负响应，再按既有 A/B/C 分期冻结 f、校准查询收益并作独立确认。不得依据已知赛果挑选样本以填满类别，亦不得把多次查询或同场重跑计为新独立比赛。

**R193　Round193：比赛输入、持久任务与可见状态**

原始记录：[round193-live-match-workflow.md](<D:/a1-足彩/docs/research/acquisition/round193-live-match-workflow.md>)

2026-10-07，北京时间。执行既有剩余计划的 M1，并并行建立 W3 的输入接口。本项没有重置原计划，也没有宣布整个 M1 已通过真实端到端验收。

当时的结果或限制原文摘录：

> 批次先固定输入顺序和配置，再逐个执行，最多20个输入。失败和不支持保留在原批次，不自动补成功案例。`task-tick` 是一次有界扫描，执行到期工作；本项没有安装或启动后台常驻服务，也不声称任务正在等待某个离线执行器。

> 准备→登记→Actor→完整回放→到期结果→training_eligible 已由禁止联网的构造集成验证；不完整回执、篡改cycle、重复比赛预算、过期报价、失败批次和结果来源绑定分别覆盖。真实外网只通过不支持状态与恢复验收，不能用构造测试冒充真实新比赛成功。

> 旧 Round192 零网验收再次通过：11条真实轨迹精确回放、32条实际响应、6场/26条合格监督输入、原始文件和旧验收不变。当前真实 f 仍为 not_ready / model=null；没有新 critic 或正VoI结论。

**R194　Round194：共享内容预测器与真实新比赛入口**

原始记录：[round194-context-predictor-and-live-source.md](<D:/a1-足彩/docs/research/acquisition/round194-context-predictor-and-live-source.md>)

2026-10-07，北京时间。继续执行 `docs/research/remaining-execution-plan-2026-10-07.md`，不重启计划或轮次承诺。

当时的结果或限制原文摘录：

> 新增 `context_predictor.py` 的共享可见内容预测器 f，以及 `context_training.py` 的原始轨迹到训练前缀桥。
> 新增限定范围的芬兰官方身份适配，统一入口实际完成 IF Gnistan–FC Inter 的赛前准备、登记、采集及精确回放。
> 真实采集暴露的日期型文章漏选，采用候选发现 v8 修复；原 v7 运行不改写，派生菜单不算实际执行。

> 固定模型形式为 `softmax(log(p_market) + phi(H) @ W)`。H 只含当前已读正文、已观察标题、来源和有序获取状态。
> 不读取未选候选正文，不把当前后验、未来赛果、比赛 ID、成本或预算当作预测内容。
> 字符 TF-IDF 使用物理比赛分组拟合，训练目标按物理比赛等权、再在不同特征前缀间等权；固定 L2=10。
> 重复运行或重复前缀不会增加独立比赛支持。空内容或完全未知词元精确回到市场先验。
> 对新来源只能声明外推；代码可运行并不代表校准已成立。

> 联合语料为12次真实运行、8场物理比赛、35次实际响应及前后配对。
> 35个带标签原始前缀来自6场独立结果，另有12个待结算前缀；新芬兰比赛贡献4个待结算前缀，没有新增标签。
> 模型特征去重后带标签前缀32个，独立比赛类别仍为主胜0、平局2、客胜4。
> 真实训练为 `not_ready/model=null`；12场且三类齐全只是必要门，不是充分样本量或有效性证明。

**R195　Round195：正文、独立赛果与有符号查询收益**

原始记录：[round195-article-body-result-and-signed-gain.md](<D:/a1-足彩/docs/research/acquisition/round195-article-body-result-and-signed-gain.md>)

日期：2026-10-07。继续执行 `docs/research/remaining-execution-plan-2026-10-07.md`，不重置研究目标或阶段。

当时的结果或限制原文摘录：

> Round194 的 Gnistan 首页已包含实际文章链接。candidate8 可生成这些候选，但原 run 已结束，直接重新调用 Runtime.run 会清空历史并重置预算。因此没有把这次补采写成原 Actor 的继续执行。

> 旧 12 run 已在新代码下零网络精确回放，原 35 响应重投影一致。527 份冻结文件 SHA 不变。语料七个核心 JSON 中六个逐字一致；唯一差异是芬兰 PENDING 结果的派生核验行，因正式结果适配器加入而由“不支持规则”变为“该历史结果归档缺少已登记规则原文”，标签和身份未变。

> 1. 按已登记时点执行今日比赛的结果核验。北京时间 10 月 8 日 00:00 开球，02:10 为任务首个允许结果检查点；查询仍须观察到实际完场证据。未启动常驻进程，当前任务未被提前推进。
> 2. 将已验证的正文/身份/赛果能力用于预先确定顺序的新比赛采集，扩大独立比赛、类别和来源覆盖。当前芬兰身份范围仍窄，不能把一个球队对的成功当持续 20 场供给已完成。
> 3. 达到真实 f 的数据条件后，按物理比赛与时间留出训练、验证及冻结；“12 场且三类齐全”只是必要门槛，不保证可靠性。
> 4. 对新 B 比赛接入 ContextPredictor 的采集登记，使用本轮公共桥产出实际 signed delta，训练 g，再验证其估值、排序、成本约束选择与停止。


**R196　上下文条件查询、实际执行与新增真实样本**

原始记录：[round196-context-query-interaction-and-runtime.md](<D:/a1-足彩/docs/research/acquisition/round196-context-query-interaction-and-runtime.md>)

本轮沿用既有执行计划，目标是“输入比赛后，机器根据已有信息决定下一步查询”。交付包含查询收益模型、现有执行器的 f/g 接线、B 阶段正式采集入口，以及两场新比赛的真实 A 阶段采集。没有训练或发布真实 f/g，也没有声称已具备 AIC 实验效果结论。

本轮完整原文如下，区分机制验收、实际失败和下一步计划：

# Round196：上下文条件查询、实际执行与新增真实样本

本轮沿用既有执行计划，目标是“输入比赛后，机器根据已有信息决定下一步查询”。交付包含查询收益模型、现有执行器的 f/g 接线、B 阶段正式采集入口，以及两场新比赛的真实 A 阶段采集。没有训练或发布真实 f/g，也没有声称已具备 AIC 实验效果结论。

## 1. 科学问题与本轮算法修正

f 是冻结的可见状态预测器；g 学习一次实际查询对这个同一 f 的有符号改善：

`target = log f(s_after)[y] - log f(s_before)[y]`

结果 y 只在赛后独立监督侧出现。g 的输入只含查询前可见的正文、来源、历史状态、预测概率、已观察父页标签、工具与声明成本。未读取的目标正文、赛果、URL 具体路径和比赛编号均不作为输入特征。正收益、负收益和失败轨迹都保留。每场物理比赛总训练权重相同。

初版纯加性 ridge 虽会随状态改变分数，但比较同成本候选时，共同状态项会抵消，不能充分表达“不同状态优先查不同信息”。本轮在保留固定正则强度 20 的基础上加入 **128 维固定有符号 TensorSketch 双线性交互**：

`g(s,a) = linear[visible_state, candidate] + linear[sketch(normalize(state) ⊗ normalize(candidate))]`

状态向量包含已知文本、状态统计、当前 f 概率、先验和已知来源；候选向量包含已观察标签、工具来源和声明成本。哈希由固定 SHA-256 坐标及固定种子决定，无人工主题权重、模型搜索或根据实验效果调参。交互层最多增加 128 个回归参数。它提供候选排序随状态改变的表达能力；哈希碰撞、词面支持和模型误差仍是限制。

`ContextVOIPolicy` 只在支持充分时以 `g(s,a) - λ·cost(a)` 排序，成本扣一次。缺乏支持返回 `UNKNOWN/None`，只允许独立预算下的探索。所有已知候选净值非正且菜单不完整时 `defer`；只有声明完整菜单时才能按经验均值停止，仍没有统计停止证书。

构造交叉验证预先固定 48 个独立 B 身份、3 个日期、2 个已知状态和 2 个同 host/tool/cost 候选，每格 12 条。两个候选标题在两个状态下都已可见；before 的 f 均为同一先验；target 由同一实际 f 对 before/after 重新计算。结果两个状态的优先候选分别为 0、1。四个 gross 估计约为：

| 已知状态 | 候选 0 | 候选 1 | 相对优先 |
|---|---:|---:|---|
| 状态 0 | -0.0039713181 | -0.0039923085 | 候选 0 |
| 状态 1 | -0.0039857798 | -0.0039714616 | 候选 1 |

这只证明相对排序能改变。四个值均负，不能据此主张应该查询，更不能算真实效果。另一条完整构造运行用实际拟合 g 验证了 UNKNOWN 探索 → 获得正文 → 正经验收益查询 → 保存与精确回放，成本设置明确用于机制检验。构造数据不混入真实语料。

## 2. 接到现有运行器，而非另外建立演示流程

新增显式运行版本 `learned-voi-open-execution-v4`，继续复用原候选生成、工具、请求前预算预留、原始回执、有限探索和离线回放。新版流程为：

1. 按实际决策时刻过滤已取得文档，运行 f 后再生成和估值候选。
2. 用查询前可见上下文运行 g；未训练 g 的 B 阶段只有限探索。
3. 实际工具返回后先核验并加入文档、attempt 和有序历史，再用同一 f 更新概率。
4. 保存 `public_after`，时点严格为实际 `returned_at`。
5. 下一次决策重新计算当前状态与概率。等待引起的内容老化不归到上一次查询的即时收益中。

f/g 在第一次请求之前持久化，文件摘要、版本、实际拟合时间和物理比赛身份均绑定。旧 v1–v3 路径及原档不改写。v4 的 FAILED/NO_DATA 仍记录费用和历史；它们不强制零收益。倒退时钟拒绝，物理或探索子预算实际超支保留真实花费并停止。返回超过信息截止或实际超支时不制造合法 `public_after`，监督中明确排除其收益而保留成本账本。

## 3. B 阶段已经有正式采集入口

`collect_prepared_batch` 新增 `context_registration_path`、`context_predictor_path`；CLI 对应 `--context-registration`、`--context-predictor`。注册文件由上一轮 `prepare_context_calibration` 生成。

采集前重验原始身份、冻结 f、实际拟合与注册时间、A/B 物理分组。注册内容及 f 字节在查询前保存，监督 metadata 不进入 Agent public state。原生回放重新核验它们。`build_context_gain_from_runs` 经过原始数据重放、独立结果与即时 before/after 生成监督。B 轨迹不能再通过旧 exposure import 改写为 A。

正式 C 阶段的预登记和公开 collector 入口仍未开启；本轮仅验证 g 在现有 runtime 中可执行和精确回放。真实 f 未就绪时，不能开始真实 B，也不能将此前 A 轨迹补称 B。

## 4. 两场新比赛、失败原因与传输修复

通用芬兰冠军组身份适配器覆盖已核验的六队映射，使用官方原生 `veikkausliiga:2026:match:<id>`，不把原生编号未经证明地改称足协 ID。旧 Gnistan–Inter 路径保留；新适配器拒绝该旧球队对，避免换命名空间重复计数。准备入口 v3、source profile4、candidate9 与官方独立赛果流程已接通，旧版本保留精确行为。

真实新 A 批次：

| 比赛 | 市场 ID | 官方事件 | 开球（北京时间） | 最早本轮结果核验点 |
|---|---|---|---|---|
| HJK–VPS | 3085211 | 4353553 | 10 月 8 日 23:00 | 10 月 9 日 01:10 |
| KuPS–AC Oulu | 3085212 | 4353563 | 10 月 9 日 00:00 | 10 月 9 日 02:10 |

市场准备 4 次请求全部 200；Actor 每场 4 次，共 8 次。实际 6 次官方网页请求在 Python/environment 证书验证阶段失败，2 次 Bing RSS 返回 302。因此本批 0 可用正文、0 新标签、概率保持市场先验。失败原档已封账，不重跑、不回填，也不将两场说成信息价值成功案例。

新增显式 `source-route-selection-v2`，只有可信配置给官方 host 设置 `system_tls` 才启用 Windows Schannel。请求仍做严格 TLS 和默认撤销核验、无凭证、无重定向跟随、0 重试；文章必须来自已观察链接，候选不能改变路由。旧 transport 配置逐字恢复为 v1。

独立研究核验预算 1 次，实际官方首页 **HTTP 200、97,851 字节、80 个保留链接**，包含实际新闻链接。该原文独立归档，不增加 Actor、A/B 前缀或收益目标。首页成功证明传输链打通；本轮没有宣称目标文章正文已在真实 Actor 中读取成功。未来 live 配置已显式启用该 host 的新通道并补入当前两场及旧芬兰任务的已登记身份，防止再次采集同一场。

## 5. 当前库存和科学边界

正式合并语料为 **14 个 run、10 场物理比赛、43 条实际返回、35 个有标签原始前缀、22 个待标签前缀**。独立有结果的比赛仍只有 **6 场，主胜/平局/客胜 = 0/2/4**。真实 f `not_ready`，真实 B 为 0，真实 g 未拟合。增加失败前缀不会增加独立标签数。

14 条原始轨迹均已零网精确回放；R195 冻结的 527 份文件未变。最终合并测试数量与当前源码摘要以 `var/parallel/agent/round196/acceptance.json` 为准。

g 当前拟合的是已有记录策略下的条件收益，并非严格 Bayes EVSI，没有因果反事实识别、条件置信区间或开放世界最优停止保证。局部来源/词面支持只是训练覆盖检查。算法机制完成不等于真实实验证明。

## 6. 下一步按收益排序

1. 使用已修复传输采集尚未登记的新比赛；优先实际赛前文本和可独立结算的来源，不再重复本批失败 Actor。已找到后三场真实 provider ID，报价需新鲜获取。
2. 到时核验已登记比赛的官方结果，更新同一真实语料。缺结果或规则证据时维持 pending，不从预测反推标签。
3. 在独立比赛和类别覆盖满足必要门槛后冻结真实 f，并检查校准及有效内容覆盖；通过门槛本身不等于模型有效。
4. 使用已接通入口登记未来 B，收集实际 signed gain；再训练带交互 g。增加真实条件排序/停止实验、成本敏感比较和等预算基线。
5. 接入正式独立 C 确认入口，在未参与 A/B 及选择设计的比赛上评估。届时才讨论预测风险—查询成本改进是否成立。

无需前端才能推进这些工作。用户输入仍只需比赛；能力来源和预算属于部署配置，不要求用户提出研究问题或指定正文。

零网复核：

```powershell
python var/parallel/agent/round196/verify_round196.py
```

新两场到期检查（提前运行返回 `not_due`，不会请求网络；已完成或不确定的请求不自动重试）：

```powershell
python var/parallel/agent/round196/settle_new_fixtures.py
```

不要再次运行 `collect_new_fixtures.py` 或 `verify_news_transport_live.py`，它们是已封账的单次真实采集。
