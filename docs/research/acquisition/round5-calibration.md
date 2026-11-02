# 第五轮信息价值校准

本轮交付 `CalibratedRiskAwareValueEstimator`、`select_two_step_v2`、严格时间校准划分、单/双赛季实验、工件复验、配对比较与初始状态可辨识性诊断。

## 区间含义

- `expected_gain/gain`：前向 OOF 对数损失差的条件均值近似；允许负数。
- `standard_error/uncertainty`：按比赛聚合、按周聚类的均值标准误，考虑局部与父层收缩的重叠影响。
- `lower_gain/upper_gain`：均值的描述性区间；不宣称有限样本或条件覆盖保证。
- `predictive_lower_gain/predictive_upper_gain`：使用严格更晚校准段的单场绝对残差分位数形成的预测区间。该区间不驱动均值门控。
- `support/parent_support`：本地与父层独立比赛数；父层存在不等于本地有支持。
- `gate_reason/fallback_reason`：低支持、未知动作、区间缺失和负值等不同原因。

旧风险模型和第三轮模型保持可加载。第四轮 episode runner 仅增加可选规划器参数，默认行为保持不变；第五轮显式选择 v2 两步近似，所有正式实验都保存当前实现哈希。

## 前瞻近似的边界

旧实现先要求第一步净值为正，遗漏组合价值。新实现先检查预算、支持度和合法性，然后比较第一步与最多一个后续动作的总值。后续状态只是掩码投影、概率保持当前值，实际获取后再预测。构造反例证明控制流可以选择负的一步启动动作；历史双赛季下 two-step 与 risk-mean 仍相同，尚无组合收益证据。

## 结果入口

正式结果使用 `var/parallel/baselines/round5/{same-season-v2,two-season-v2}`。每个实验290场测试比赛、16,240次策略运行，旧赛季加入训练不增加独立测试样本。两组实现哈希、21个模型/组、时序隔离、概率、预算和分数重算均通过。

完整表格、配对区间、初始价值向量恒定问题与复跑命令集中在 `docs/parallel/round5/README.md`。不将较低模拟费用直接解释成同质量省费，也不将单场增益符号当作期望增益的无噪声标签。
