# 上下文条件响应模型：实现与适用边界

实现：`football_research/acquisition/contextual_joint.py`。

该模型解决一个明确的建模限制：原联合响应模型只通过初始胜平负概率 `p0`
区分比赛。如果两场比赛的 `p0` 相同，原模型会给出相同的战绩/xG返回分布，即使
公开球队身份不同。本实现让这种差异可以从历史响应中学习，并保留原全局模型退路。

它仍只覆盖已有的 `results_view` 和 `xg_view`。网页、伤停、晋级动机等尚未形成
受监督的响应类别，不会因为这个扩展自动获得数值VoI。

## 数学形式

令原模型为 `q0(z_results,z_xg | y,p0)`，历史样本核权重为 `k_i(p0)`。
只用公开赛程身份构建上下文权重：

```
u_i(H) = 1[同一赛事] * (1[同一主队角色] + 1[同一客队角色]) / 2
C_y,z(H) = sum_i k_i(p0) * u_i(H) * 1[y_i=y, z_i=z]
q_local(z|y,H) = (C_y,z(H) + tau*q0(z|y,p0)) / (sum_z C_y,z(H) + tau)
q_alpha = (1-alpha)*q0 + alpha*q_local
m_alpha(z|H) = sum_y p0(y|H)*q_alpha(z|y,H)
q_alpha,s = s*q_alpha + (1-s)*m_alpha
P(y,z_results,z_xg|H) = p0(y|H)*q_alpha,s(z_results,z_xg|y,H)
```

`tau`复用原模型的Dirichlet质量。`alpha`候选固定为`0, 0.5, 1`，不按测试表现扩展；
每个候选的`s∈[0,1]`用已有凸优化方法选择。评价目标是过去验证比赛上三种已获取集合
的等权平均log loss，而不是让查询次数增加。保留`alpha=0`和`s=0`退路。

上述所有分布在查询开始前由同一个公开初态建立。实际或规划中的已揭示响应只用来
对该联合分布作条件化，不能重置`p0`、重新选择历史邻居或调整参数。因此每个查询的
响应概率和后验分支保持一致，后验的条件期望等于当前概率。该数学一致性不保证真实
足球分布的校准，也不保证获取信息带来正实际收益。

## 小样本与信息边界

- 局部样本不足时，模型收缩到原全局分布；没有同角色身份样本时严格退回全局。
- 原VoI支持门限和物理成本不变。全局支持继续表示原联合样本覆盖；局部有效数量、
  每类核质量和局部实际权重单独报告，不能解释为独立置信区间。
- 参考训练样本的球队身份必须来自其赛前可用的同一场赛程；新类只读取允许的
  赛程字段，忽略其余特征和赛果载荷。保存的上下文与原模型及选择记录共同哈希绑定。
- 原模型训练标签和过去验证标签可以用于拟合；运行时只接收既有公开状态。
  未来标签、验证身份冒充测试、未查询数值和假设分支冒充实际返回均被拒绝或隔离。
- 本模型沿用数据集的历史重建身份/时间语义，不把事后采集的历史快照改称赛前在线证据。

## 接入

```python
model = ContextualJointReturnPredictor.fit_context(
    reference,             # 冻结的原 JointReturnPredictor
    fixture_rows,          # 所有 reference.records 对应赛程，可能跨赛季
    heldout_rows,          # 对应过去验证段，不能放测试行
    actions,
    as_of=validation_label_cutoff,
)
model.save(artifact_path)
model = ContextualJointReturnPredictor.load(artifact_path)
policy = LearnedJointVOIPolicy(model, lambda_cost=unchanged_lambda)
```

选择过程按三种已获取集合等权，未对真实部署中的选择偏差或未知信息覆盖作推断。
因此这是可检验的统计建模改进，不能单独称为AIC算法创新已经成立。统一实验应在
原时间切分、相同初始概率、相同预算及成本下比较原模型、全局收缩模型与本模型，
同时呈现不查询、随机、固定顺序、全部可负担信息和VoI策略。真实收益为负时应保留
负结果，不通过降低成本阈值制造查询或继续在相同测试比赛上调参。
