# 第三轮信息获取桥接

`football_agent/research_bridge.py` 连接平台 `FrozenEvidenceEnvironment` 与研究 `PartialPredictor`、`ValueEstimator`，执行器和提案器只接公开字典。完整历史特征与目标赛果留在环境/验收评分端。研究投影复用 `football_research.acquisition.platform_bridge.to_research_state`，组映射为 strength→elo、recent_form→recent_form、rolling_xg→xg。

服务端构造：

```python
from football_agent.research_bridge import make_backend
backend = make_backend(
    bundle_path="var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda",
    model_artifact="var/parallel/agent/round3/real/backend-fold-2.json",
)
```

配置保存部分预测器、估计器、验证段冻结的 lambda、平台实验成本表、公开fixture列表。模型文件只由服务端传入；运行配置钉住模型/配置/COMPLETE文件哈希、训练截止、目录/费用版本。每次获取先验证目录和空参数schema，再通过平台公开restore恢复公开状态。该冻结缓存环境的restore在环境内部重建历史收据，最终预算等于原状态，没有新增外部请求或新收据。新增一次动作仅新增对应的一次费用。

LLM与策略均不接收环境对象、服务端文件路径、模型工件或完整manifest。公开状态的递归隐藏字段检查、boolean mask检查与研究可用时点检查共同生效。估计结果中的损失改善、支持度、标准误均来自研究服务；净值按冻结lambda与显式成本向量换算。负改善保留。预测数组只转成home/draw/away字典，不改数值。

真实受控揭示已保存三个独立费用/模型情景，均为历史重建、探索性展示，0次HTTP：

| 配置/记录 | 行为 | 费用解释 |
| --- | --- | --- |
| real/backend.json；511ad5c7-5426-4b0d-a1eb-8bfdc5872f2f | fold0预计results_view改善0.01427665低于换算成本0.04，立即停止 | 显式模拟收费；lambda.money=2由研究验证段冻结 |
| real/backend-fold-2.json；39cd4a74-ab46-4e51-a6e9-46c1a8b52b97 | xg_view净值0.00083387为正，获取xG后重新估计并停止 | 1次模拟动作、0.03 simulation_credit；fixed对照c92473ae-4bd8-465d-959f-42681fc07239为2次/.05 |
| real-cache/backend.json；fb0785d6-be2a-4974-a239-0ecad915b6dd | 原fold0模型/lambda不变，缓存成本0，获取results_view后停止 | 已有离线缓存无新增供应商费用，与模拟收费情景分开 |

展示场次依据公开候选值选取，未依赖目标赛果；单场轨迹不证明研究效果。所有原始数值见real/initial-values-by-fold.json、fold-2-validation.json与完整runs目录。

零网络复跑独立验收：

```powershell
python -m pytest tests/agent/test_round3_bridge.py -q
python -X utf8 -m var.parallel.agent.round3.real.verify_bridge
```

桥接测试21项通过。第二个命令检查真实数据上的构造隐藏值/赛果扰动：获取前公开状态、概率、候选及数值全相同，取xG后才改变；另验证真实环境中途恢复、同配置继续、2动作总模拟费.05和完整hash审计。该命令创建新的验收运行，报告覆盖为最近一次验证；原运行目录保留。

`make_backend(mode="synthetic_test", scenario=...)` 为独立固定概率/固定增益测试替身，支持normal、immediate_stop、no_data、cost_exceeds_gain，始终明确synthetic_test。没有真实配置时抛出BackendUnavailable，不转成虚构真实概率。
