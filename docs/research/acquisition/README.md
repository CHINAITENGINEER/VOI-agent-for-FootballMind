# 第三轮：部分信息预测与经验增益获取

2026-09-22。可运行链路已经完成，正式实验为`var/parallel/baselines/round3/active-information-v3/`。研究结论：当前低复杂度增益估计未表现出可靠的动作价值排序；学习策略减少模拟获取成本，但预测损失高于全取和熵启发式。保守规则在本探索集全部停止。保留这些负结果，未根据测试成绩重调模型或阈值。

## 真实数据与时间设计

输入是平台已冻结Understat EPL2025/26的380场、20队完整赛季，使用a3b3导出的canonical UUID。研究只读平台归一化和特征，不重新解析来源、不维护Elo评级。`load_episodes`验证COMPLETE、文件hash、match_id+cutoff+information_mode+feature_version连接与特征时点。仅选24h提前量，每场一条，380场成功连接、0场因连接失败排除。赛季开头10场近期/xG等字段缺历史，按缺源处理，未伪造完整信息。

这是historical_reconstruction：赛程假设提前168h可见、结果假设开球后3h、xG假设开球后24h可用；真实observed/ingested仍保留今天采集时间。历史统计可能经过后续修订；数据时区依据及来源单一偏差沿用平台审计。当前目标比赛的结果、单场xG、来源forecast和最终赛季汇总始终隔离。没有合格历史赛前报价，市场覆盖为0，不用今天的赔率补历史。

外层每30天验证/测试，7个fold共290个测试比赛。这290场第二轮已经看过，因此全部为探索性分析，多策略、多mask和多seed不增加独立样本。

| fold | 预测训练比赛 | 增益训练比赛 | 验证比赛 | 测试比赛 |
|---|---:|---:|---:|---:|
| 0 | 60 | 20 | 30 | 35 |
| 1 | 90 | 50 | 30 | 55 |
| 2 | 120 | 80 | 55 | 50 |
| 3 | 180 | 140 | 50 | 42 |
| 4 | 230 | 190 | 41 | 37 |
| 5 | 271 | 231 | 37 | 30 |
| 6 | 309 | 269 | 30 | 41 |

训练/验证标签按下一预测截止点purge；同场全部快照同分区。每fold训练先于其验证、测试；跨fold已结束的早期比赛可进入后期训练。报告保存逐阶段ID、时点、标签可用时间规则与purge数量。

## 模型、估计与成本

部分预测器使用三组：Elo长期实力2字段、近期赛果6字段、rolling xG/xGA4字段。空mask独立冻结Dirichlet(1)联赛频率先验；其余7子集各自训练L2多项逻辑回归（regularization=10），每个子集独立在训练窗拟合插补/标准化。缺类别/样本/整列特征时显式回退先验。区分not_acquired、source_missing、stale，未取得字段不经完整模型隐式插补。

经验g监督来自历史内部向前OOF：初始40场、每30场一块，12个概率模型；每个目标的标签/特征均在其概率模型训练之外。Delta为该目标获取前后log loss差，负值保留；外层g只消费外层训练ID和截止之前已可用标签。条件为可见mask/质量、动作和当前最大概率分箱（0.45）；用mask-action与action父级收缩，shrinkage=20，最小12个独立比赛。低支持回退明确标记。分箱参数固定，测试仅作校准诊断，没有测试集拟合或后校准。

不确定性为自然周聚类SE；保守规则取gain−1.64×SE。单周等不足场景返回不确定性缺失。这是经验下界，未给严格覆盖保证，也未涵盖重新训练和跨赛季迁移不确定性。合成XOR互补测试证明单步规则可能在两步联合收益为正时立即停止。

平台ActionSpec与ExperimentConfig已接通。results_view一次返回Elo+近期赛果，成本0.02模拟货币；xg_view返回xG，0.03。每动作1次模拟投影调用。真实Understat同一响应包含三组，缓存读取外部请求为0；这些费用不代表真实HTTP/API节省。缓存免费与source_bundle一次全组的语义由平台保留，正式曲线仅解释受控模拟揭示。

预算为0/.02/.03/.05，最多2动作；所有策略同一初始状态、模型、目录、预算和确定性环境。随机策略seed为7/19/41/73/101，先在同一比赛内平均。lambda_money在外层验证从0/.25/.5/1/2选择，所有候选都按相同`mean(log_loss + 0.5 * simulated_money)`比较，平局取较大lambda；每fold/预算分别冻结。lambda单位是每模拟货币单位的log-loss nats，其他权重0。0.5是实验事先给定效用权重，不是现实费用标定。

## 实际策略对照

总13,920次策略运行。以下是相同预算0.05、共同290场；Brier为三类误差平方和，loss越低越好。

| 策略 | log loss | Brier | 准确率 | 平均模拟费 | 平均投影次数 |
|---|---:|---:|---:|---:|---:|
| base-only | 1.108492 | 0.673332 | 40.34% | 0 | 0 |
| all-available | 1.087116 | 0.650720 | 46.21% | 0.050000 | 2.0000 |
| random（5 seed均值） | 1.087116 | 0.650720 | 46.21% | 0.050000 | 2.0000 |
| fixed-order | 1.087116 | 0.650720 | 46.21% | 0.050000 | 2.0000 |
| entropy | 1.083401 | 0.650432 | 45.52% | 0.034483 | 1.4828 |
| disagreement | 1.086510 | 0.650041 | 46.21% | 0.049069 | 1.9690 |
| learned-net-gain | 1.102519 | 0.667196 | 43.10% | 0.016379 | 0.6276 |
| conservative-net-gain | 1.108492 | 0.673332 | 40.34% | 0 | 0 |

预算足够时random/fixed/all取得相同全部字段，终态概率一致是预期行为；低预算时报告另列差异。预算不足时all-available按目录顺序取可负担动作，真正全信息参照是0.05那一行。entropy固定阈值1 nat；disagreement比较当前与冻结先验L1，阈值0.04，初始先取目录首动作。这些是预先固定的启发式，没有做测试调优。

learned比全取减少约67.2%模拟费，但log loss增加0.015403；不将此解读为真实节费或整体策略优势。learned执行results_view 71次、xg_view 111次；266场因非正净值停止、24场耗尽动作。保守规则290场均以非正净值停止，损失与base完全相同。熵策略150场阈值停止、140场全取。

以27个自然周做逐场配对块bootstrap（2000次），learned−base平均loss差−0.005973，95%经验区间[−0.026083, 0.014509]；all−base为−0.021376，[−0.067555, 0.031986]；entropy−base为−0.025091，[−0.069444, 0.024615]。这些区间均跨0，未做多重比较或拟合策略不确定性修正。

增益诊断1160条状态/动作转移仍只对应290场：符号准确率49.14%，初始最佳动作排序与实际一致47.93%，实际Delta为负占43.19%。五等频箱的预测/实际均值分别为−.02891/−.00812、−.01023/+.05146、−.00243/+.00738、+.00575/−.00904、+.02554/+.01176，显示明显失配。失败例：比赛`a67f6329-ea2b-52e2-9e0e-9aeacc9a99dd`初态results_view估计+.02358，实际−1.78544；单场实现损失含赛果噪声，完整20例见report，不能把单例当精确反事实期望。

估计器本地均值0.0704ms/候选批；各episode平均批时延的95分位0.1085ms，非严格逐批尾时延。实际外部请求0、LLM tokens0；本地CPU货币成本未计量，报告为null。模拟工具latency=0和本地实测CPU时间分开记录；并行机器负载使不同运行的耗时有变化。

![损失与模拟费用](../../../var/parallel/baselines/round3/active-information-v3/loss-cost-curves.png)

![经验增益分箱校准](../../../var/parallel/baselines/round3/active-information-v3/gain-calibration.png)

## LLM与Agent接入

Agent已提供真实平台执行器与研究bridge，消费第一版同模型产物完成接线；正式v3核验复现。pure-LLM/hybrid未进入本次公平效果表。冻结design中的`SKIPPED_NO_AGENT_REAL_LLM_RUNNER`记录设计冻结时状态；交付前重读Agent的`round3/llm/validation.json`确认代码已就绪，但model=null/configured=false，真实provider验收为NEEDS_CONFIGURATION，实际请求0。后续接入应使用相同预算/预测器/比赛并单列LLM token、钱与推理时延；构造proposer通过不算真实LLM效果。

精确import、公开状态与模型路径见[interfaces.md](interfaces.md)。平台FrozenEvidenceEnvironment实接记录见[platform-bridge.md](platform-bridge.md)。标签及完整特征只留可信环境/离线监督，预测与选择只接序列化VisibleState及目录；改未知组或标签不会改变获取前概率/动作。Python对象边界仍需部署执行器实施进程权限隔离。

## 命令与产物

在`D:\a1-足彩`执行，复跑选择全新输出目录：

```powershell
python -X utf8 -m football_research.acquisition run --platform-export var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --output var/parallel/baselines/round3/reproduce-new
python -X utf8 -m football_research.acquisition.verify --run var/parallel/baselines/round3/active-information-v3 --previous var/parallel/baselines/round3/active-information-v1 --previous var/parallel/baselines/round3/active-information-v2 --output var/parallel/baselines/round3/verification-rerun.json
python -X utf8 -m football_research.acquisition.platform_demo --platform-export var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --experiment var/parallel/baselines/round3/active-information-v3 --output var/parallel/baselines/round3/platform-bridge/environment-rerun.json
python -m pytest tests/research/baselines tests/research/evaluation tests/research/acquisition -q
.\football_spider\.venv\Scripts\python.exe -m pytest tests/research/baselines/test_reconstruction_bundle.py -q
```

正式目录的design/catalog/implementation/data-audit记录配置、公共成本绑定、代码SHA和数据检查；models为7套partial/gain共14个JSON；oof-models为12个历史预测器；oof-supervision仅供离线监督；fold-progress/report保存全部训练、验证、测试ID和lambda；episodes/trajectories/gain-diagnostics分别保存评分、仅可见决策轨迹、离线诊断。v1/v2均保留原产物：v2修正首步预测重复计时与浮点预算容差；v3修正动作历史列表被后续append导致早期日志状态事后改变，事件改为独立拷贝。该日志缺陷不影响当时算法概率/选择/损失，但旧版日志不适合作为精确动作重放依据。三个版本不是新的独立测试集。

前瞻71条预测的新结算单独见[round3-settlement.md](../baselines/round3-settlement.md)：48条/12场评分、20条/5场质量隔离、3条/1场pending，冻结概率hash不变。该12场仅作真实前瞻结算功能验证，与本页历史策略实验分开。

修改范围：新增acquisition/{state,partial,value,policy,data,platform_bridge,platform_demo,experiment,verify,__main__,__init__}.py及独占测试；修复data_adapters/{types,jsonl,platform_history,settlement}.py和evaluation/ledger.py的逐场时钟；新增本目录文档、结算文档及baselines状态/请求。共享依赖、采集、迁移、Agent与展示模块由各自任务维护。

## 证据限度与后续研究

当前g条件粗、早期OOF支持仅20场、单赛季漂移明显；保守下界容易整体停取。下一轮应先在未用于选择的后续时间段预注册验证，或用较早赛季扩充训练后固定方案；新增历史赛季不是未见未来测试。成本换算需要实际部署效用与费用标定。一步停止的互补性缺陷已有构造测试，尚无真实两步策略优势；真实联网获取、前瞻主动策略和LLM优越性均未在本实验建立。
