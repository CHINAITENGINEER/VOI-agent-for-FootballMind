# FootballMind g v4：平均信息价值与负期望净值风险

本轮完成训练、独立留出评价和训练集机制诊断。结果未支持把 v4
替换到 Agent 主线：验证集没有找到满足预先规定风险偏好、且具有足够
查询覆盖的策略。`stop_only` 是实验失败时的保底输出，不代表证明所有
现实信息都不值得查询。旧 v3 模型、真实 shadow、经验 g gate 均未修改。

## 定义与输入

令 Z 表示 teacher 的完整合成机制，X 表示查询前可观察状态，S 是信息
返回，Y 是比赛结果。教学标签为：

```text
G(Z,a) = E[log f_after(Y) - log f_before(Y) | Z,a]
g(X,a) = E[G(Z,a) | X,a]
r(X,a,c) = P(G(Z,a) - c < 0 | X,a,c)
```

g 使用旧 v3 的 14 维 observable 向量；风险头仅额外加入查询成本。
latent probability、effect logits 不进入两者的训练矩阵。标签可以使用
teacher 的隐藏机制，推理输入不能使用它们。

风险头估计的是**完整机制未知时，查询的期望净值为负的概率**。它并非
某个真实赛果到来后，单次预测 loss 变差的概率。后者还需要对 S、Y、
NO_DATA 的真实分支计算，不得把本轮两个概念合并。

合成的 source reliability、freshness、redundancy、response probability、
updater trust 依然是显式假设，不会因为列入 observable 就变成经过核实
的足球事实。模型保持 `synthetic_research_only`，无在线选择权限。

## 实验设计

- 新 seed：20261021；2,000 个合成比赛组，每组 3 个状态、每状态 4 个动作。
- 共 24,000 条候选；按比赛分为 train 1,000、calibration 400、validation
  300、test 300；同场不同状态和动作不会跨集合。
- 均值头：ExtraTreesRegressor，96 棵树、最大深度 14、叶节点至少 8 条。
- 风险头：ExtraTreesRegressor 对二值标签做概率回归，128 棵树、最大
  深度 14、叶节点至少 12 条；单独 calibration 组拟合 isotonic。
- validation 选择 STOP margin 和风险阈值，随后冻结模型；test 只用于
  最终评价，不因结果不好重调。
- mean-only 和 risk-filtered 共用完全相同的均值模型和测试集。前者的
  margin 也独立在 validation 选择，避免把增加训练数据的效果误归给风险头。
- 风险偏好预设：validation 至少查询 20 个状态、覆盖 15 场比赛、平均
  净值为正，条件负期望净值率不超过 30%。这是研究操作点，不是用户
  风险偏好、数学必然条件或真实环境安全保证。

对完整菜单先过滤风险，再对合格候选比较净值。最高均值候选不合格时，
其他合格候选仍可被选中；输入缺失会报错，不伪装为 STOP。

## 冻结测试结果

测试集为 300 场、900 个状态、3,600 个候选。

| 指标 | 同一均值 g 的 mean-only | 新增风险约束 |
| --- | ---: | ---: |
| 查询状态 | 113 | 0 |
| 查询比赛 | 99 | 0 |
| 所选查询的负期望净值比例 | 82/113 = 72.57% | null，未查询 |
| 每个状态平均 teacher 净值 | 0.00002935 | 0 |
| mean regret | 0.00120419 | 0.00123355 |
| oracle 正净值状态的查询召回率 | 40.23% | 0 |

均值回归 MSE 为 0.000274616，常数基线为 0.000377498。回归误差改善
不等于查询策略已可靠；风险约束在 validation 就没有找到足量合格查询，
因此 test 的零查询不能用来声称风险降低成功。

全候选风险 Brier：原始风险头 0.0206314，isotonic 后 0.0212707，使用
训练集负类占比的常数风险基线 0.0244053。独立校准没有在本次 test 上
改善原始风险头。mean-only 选中子集的平均预测风险 61.97%，实际 teacher
负期望净值比例 72.57%，子集 Brier 0.213825。全候选指标不能替代选择后指标。

风险策略相对 mean-only 的平均净值差为 -0.00002935；按比赛组 bootstrap
的 95% 区间为 [-0.00021949, 0.00015421]。这是同一合成分布的结果，
没有真实足球泛化或因果保证。本轮不能声称策略显著优于 STOP。

频繁出现小额负值、偶尔出现较大正值，仍可能使期望收益为正。因此
“最大化期望信息净值”和“限制负期望净值概率”是两种不同目标。额外的
30% 限制不能被悄悄当成项目的必选目标。

## 训练集机制诊断

诊断只使用 1,000 个 train 比赛组，不重训模型，不调整 test 策略。额外
计算让 teacher 更新器的信任度等于已知真实来源可靠度的反事实，用于
定位机制问题。这不是现实 f 已得到校准。

| 教学机制 | 正值候选比例 | 负值候选比例 | 平均值 |
| --- | ---: | ---: | ---: |
| 原更新器，不扣成本 | 29.63% | 70.35% | -0.00514021 |
| 原更新器，扣原成本 | 3.05% | 96.95% | -0.01808757 |
| oracle 匹配更新器，不扣成本 | 84.69% | 0% | 0.00181893 |
| oracle 匹配更新器，扣原成本 | 4.68% | 95.32% | -0.01112843 |

平均 effective source reliability 为 0.599756，updater trust 为 0.763692，
84.54% 的候选存在 updater overtrust。平均查询成本为 0.0129474 loss units，
明显高于匹配更新器下的平均 gross value。表中近零值以 1e-12 为容差；
未列出的占比为近零，不是丢失记录。

这表明本次压力环境包含大量无价值、昂贵或被过度信任的信息；它可以
测试 Agent 是否拒绝坏信息，但不宜独自承担整个教学分布。诊断没有
证明 g 无法学习，也没有证明更多真实信息都没价值。

下一项训练应预先声明多个来源质量、f 更新误差和成本权衡环境，保留
当前压力环境作为单独测试分层；应同时评价期望净值与可选风险偏好，
不能靠看到 test 后降低成本来制造成功。还需要验证只用公开状态是否
足以描述“该信息会怎样影响当前 f”，而不是只描述查询工具类别。

## 文件与复现

- 实现：`football_research/acquisition/synthetic_g_risk_v4.py`
- 测试：`tests/acquisition/test_synthetic_g_risk_v4.py`
- 诊断脚本：`tools/diagnose_synthetic_g_risk_v4.py`
- 冻结训练产物：`var/submission/p9-g-risk-aware-v4-20261010`
- 独立诊断产物：`var/submission/p9-g-risk-aware-v4-diagnosis-20261010`
- 完整 policy、模型和输入摘要见 `protocol.json`、`model.json`、`manifest.json`。

```powershell
python -m pytest -q tests/acquisition/test_synthetic_g_risk_v4.py tests/acquisition/test_synthetic_g_observable_v3.py tests/acquisition/test_synthetic_g_shared_world.py tests/acquisition/test_synthetic_g_teaching.py

# 如需复现，使用新的空目录。相同 seed 是复现，不算新独立验证。
python -m football_research.acquisition.synthetic_g_risk_v4 --output var/submission/g-risk-v4-reproduction --matches 2000 --seed 20261021
```

本轮实际回归为以上四个文件 24 passed；两个新增 Python 模块编译通过。
没有 HTTP、没有经验 g 训练、没有改另一线程的 runtime/builder/evidence/
transport/shadow 接线。v4 artifact 不应加载到 v3 shadow adapter。
