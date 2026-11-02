# 第四轮接续验收与旧结论更正

日期：2026-09-26。此次接续发现共享工作区已经有第五轮校准修复、两份正式实验和 Agent 接线，直接消费这些已就绪能力。未新建线程、未覆盖前三轮或 round4/round5 冻结实验。新审计产物写入 `var/parallel/baselines/round4/`。

## 结论

风险估计器的区间语义、时间隔离与二步控制流得到修正；预测效果及条件价值可靠性尚未建立优势。相同训练历史和终端预测器下，双赛季 risk-mean 比第三轮算法 v1 少查询，但 log loss 较差。二步结果与均值策略一致；七折各自只有一种初始价值向量，尚未实现空状态下逐比赛的差异化价值判断。质量/依赖机制有效不等于已学得来源可靠度。

原 round4 `risk-aware-v4` 为历史负结果，保留原文件。原始 v1 以单场残差放大均值区间，且二步先要求正的一步价值，均影响研究解释。旧 `verify_round4` 没有实际检查预算，历史文档中的“概率/预算检查”需按本说明更正。新 `verify_round5` 检查预算、分数重算、时序和哈希；新 `audit_calibrated` 进一步实际调用冻结预测器/估计器重放决策。

## 实际复验

- `resume-tests.xml`：**829 passed, 1 skipped，113.20 秒**。包含研究 baselines/evaluation/acquisition、平台 information、Agent、demo。跳过项为既有可选 PostgreSQL 依赖。
- `resume-same-season-verification.json`：21 模型、16,240 运行、290 测试比赛；25,881 可见事件、9,641 动作。
- `resume-two-season-verification.json`：21 模型、16,240 运行、290 测试比赛；27,089 可见事件、10,849 动作。
- `resume-behavior-audit.json`：预算 .05、seed 0，mean/conservative/two-step 的 **1,223 个决定**逐条重放，动作/停止原因与冻结轨迹一致。
- 每场由已揭示完整状态构造全部 8 子集：**2,320 个概率检查**；分别把各组标为 source_missing/stale：**1,740 个概率检查**。掩码重新从有效可见组派生。未知组数值、额外标签、未来快照、路径和候选伪造 result 不改变获取前概率、估值、排序或停止。此审计验证接口隔离，不声称 Python 对象构成进程权限隔离。
- `resume-behavior-audit-repeat.json` 与首次审计逐字节一致，SHA256 `2832c8c2585490bb235841585a12de0fcdaec6b7a076cc71768aecd57fe29e1d`。实际重放不含计时噪声；模型/种子由冻结文件加载，不把本次加载称为重训。
- 已重读平台 round4 交接、平台/Agent 最新状态。Agent 已有 calibrated-risk-v2 的 14 条七折 mean/conservative 对照轨迹，研究端动作/概率/成本一致，记录在 `var/parallel/agent/round5/integration-v2/acceptance.json`。

## 数据、划分与版本

2024/25 较早赛季 380 场进入训练；2025/26 380 场做滚动训练/验证及已看过的 290 场探索性测试。不是新独立确认集。双赛季七折的 train/validation/test 数分别为：440/30/35、470/30/55、500/55/50、560/50/42、610/41/37、651/37/30、689/30/41。验证区间可以随时间成为后折训练历史；每折内部互斥，测试比赛唯一。同场掩码/随机种子不增加样本数。

模型与代码采用已冻结的 `round5/{same-season-v2,two-season-v2}`：partial-subset-logistic-1.0、冻结空集 prior、empirical-mask-action-shrinkage-v1、risk-aware-conditional-value-v2、two-step-feasible-mask-v2。21 个外层模型包括每折部分预测器、v1 增益、v2 风险估计器；OOF 预测器另存。加入较早赛季后已重建 OOF 监督，未混用只在旧单赛季预测器上计算的 Delta。

模型 JSON 记录训练/校准 ID、时点、正则化/收缩参数、版本和 hash。风险均值标准误按周聚类，描述性乘子 1.64；单场增益 80% 预测区间单独使用更晚校准段残差，不参与均值下界膨胀。lambda 来自验证配置，按策略/预算/折冻结，不用本次审计重新选择。

## 相同 290 场、最大模拟预算 .05 的结果

|策略|Log loss|Brier|准确率|平均模拟费|平均动作|
|---|---:|---:|---:|---:|---:|
|base-only|1.108492|0.673332|40.34%|0|0|
|all-available|1.060894|0.638023|46.55%|.050000|2.000000|
|random / fixed-order|1.060894|0.638023|46.55%|.050000|2.000000|
|entropy|1.055700|0.635264|46.55%|.040069|1.668966|
|disagreement|1.108492|0.673332|40.34%|0|0|
|第三轮算法 v1，以双赛季历史重训|1.073134|0.649710|44.83%|.019379|.810345|
|risk-mean|1.087883|0.658745|41.72%|.012172|.548276|
|risk-conservative|1.106459|0.671605|38.97%|.003621|.120690|
|two-step|1.087883|0.658745|41.72%|.012172|.548276|

当前 disagreement 以当前预测对初始预测的差异作为条件，初始差异恒为零，故全部停止。它是退化的对照，不能据此解释真正模型间分歧的价值，也不能当作第三轮同名策略的复现。所有完整策略/预算/种子结果仍在 report.json 中；random 使用五个种子，各策略共享确定性环境。

risk-mean − base-only 的 log-loss 差 −0.020609，95% UTC 周配对 bootstrap 区间 [-0.049124, 0.006091]；相对 entropy 差 +0.032182，区间 [-0.013131, 0.074814]。这些描述性区间未校正多重比较或模型拟合不确定性。

v2 单场增益区间经验覆盖 79.83%（目标80%，1,160转移/290场），不是条件均值或序贯停止覆盖保证。增益符号准确率47.59%，均值偏差 −0.028823；最低估值箱预测 −0.045648、实际平均 +0.024595，说明存在过度低估的状态。单场增益符号有赛果噪声，不是条件期望的无噪声真值。

新增十等宽箱 top-label 概率校准 ECE：v1 .048828、risk-mean .095844、risk-conservative .086221、entropy .081450、all .094083。该有限样本描述性统计只用预算 .05 / seed 0；它不是三类分别校准或显著性检验。分箱明细与点估计成本—质量前沿在新审计 JSON 中，旧实验图表保留。

## 保守系数、失败状态与成本

固定 risk-mean 已记录的初始状态、验证 lambda 和预算，只改变均值下界乘子，290 场首步获取数分别是：0 → 143；0.5 → 35；1 → 35；1.64 / 2 / 3 → 0。这是固定状态门控敏感性，不是重新运行完整策略。正式 conservative 使用它自己的验证 lambda，因此有35次动作，与此结果不矛盾。该审计不选择新乘子。

两种真实失败例子（均为离线事后诊断，不进入运行态）：

- `b8fd871c-9936-5954-a9cb-0fc19b746af3`（fold3，2026-02-07）：risk-mean 以 nonpositive_net_gain 停止，loss1.184134；all loss .288321。可行后续最佳损失改善 .925044，体现过早停止风险。
- `a60cca85-aa16-56da-8e4b-da47149648b4`（fold4，2026-03-03）：购买 results_view 花 .02，loss2.190081；base .806648，all2.474441。新增信息与当前预测器组合可能使单场损失更差，不能预设全信息最好。该场停止 regret 为0也不代表此前获取正确；现有 regret 仅量化停止时遗漏的后续机会。

平台动态场景覆盖重复、陈旧、冲突、失败、共享响应和刷新等控制流；其扰动是构造机制，主实验仍只有固定两动作与单来源。当前研究 bridge 将 FAILED 映射为 source_missing+reason，未训练独立失败模型；部分新增质量字段未进入已冻结风险模型。当前证据支持有效性排除和审计，不支持已学得来源可靠性、真实质量衰减或统计互补性的结论。

money 是 simulation_credit，requests 是 projection_call；底层同一 Understat 响应含三组，一次物理响应的供应商费与两次研究揭示模拟费区分。延迟按本机计量、LLM tokens0、外部请求0、本机货币费用 unknown。预算 .05 下每 episode 估值/选择平均耗时：v1 .2931ms、risk-mean .2886ms、conservative .2216ms、two-step .9002ms。二步增加计算耗时但没有提升预测结果。当前 runner 的 predictor_latency_ms 未覆盖初始预测，不能当成端到端时延；策略比较必须保留这个计时边界。

## 可执行入口

从 `D:\a1-足彩` 执行。输出使用新路径；audit_calibrated 以独占创建方式拒绝覆盖已有审计。

```powershell
python -m football_research.acquisition.experiment_round5 `
  --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda `
  --earlier-bundle var/parallel/collectors/round3/data/runs/fc298603-aff8-4f3e-bc4f-6c91db871036/exports/understat-2024-historical_reconstruction-ec78dfa8670887cf `
  --round3-dir var/parallel/baselines/round3/active-information-v3 `
  --output var/parallel/baselines/round4/reproduce-calibrated-new

python -m football_research.acquisition.verify_round5 `
  --run var/parallel/baselines/round5/two-season-v2 `
  --output var/parallel/baselines/round4/reverify-new.json

python -m football_research.acquisition.audit_calibrated `
  --run var/parallel/baselines/round5/two-season-v2 `
  --output var/parallel/baselines/round4/reaudit-new.json

python -m pytest tests/research/baselines tests/research/evaluation tests/research/acquisition tests/platform/information tests/agent tests/demo --import-mode=importlib -q
```

部分预测/估值/停止的精确 Python import 与调用例子见 [interfaces.md](interfaces.md)。LLM 策略无真实 provider 配置，保持 skipped；本次没有新增前瞻效果结论、真实 API 节费或稳定策略优势声明。
