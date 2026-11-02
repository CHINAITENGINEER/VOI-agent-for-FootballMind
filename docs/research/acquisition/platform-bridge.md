# Public information protocol → research consumption

2026-09-22。实现入口：

```python
from football_research.acquisition.platform_bridge import (
    experiment_actions, to_research_state, to_research_actions,
)
actions, manifest = experiment_actions(seed=7)
research_state = to_research_state(platform_visible_state)
prediction = partial_predictor.predict_partial(research_state)
research_actions = to_research_actions(platform_action_catalog)
```

`experiment_actions` 实际实例化公共 `ExperimentConfig` 和 `ActionSpec`，返回研究候选和其完整公开成本／动作目录 manifest。`results_view` 一次返回平台 `strength/recent_form`，映射研究 `elo/recent_form`，模拟 money=0.02；`xg_view` 返回平台 `rolling_xg`，映射研究 `xg`，模拟 money=0.03。每动作四分量均显式：requests=1、latency_ms=0、tokens=0，basis全部simulated。公共单位分别为 simulation_credit、projection_call、ms、token；因此lambda.money应为每simulation_credit的log-loss nats，不把money直接减loss。

同一Underlying Understat响应包含三组。这里测量已缓存资料的受控揭示，不等于真实HTTP节费。results_view联合两组收费一次。公共source_bundle模式要求一动作绑定三组；cache_views读取现有缓存无新增外部查询费。两种参考语义和公共类型／config版本已记录manifest。

`to_research_state` 接受公共VisibleState或dict。mask决定可见性；UNACQUIRED整组值完全忽略，FAILED映射source_missing，TIMEOUT等公开失败状态与action_history保留供policy限制尝试次数。源缺失／过期／坏数据的数值不进入模型。目标标签、完整manifest、快照hash、原文路径均丢弃；可见组只接收已注册数值键和惰性证据ID。Operational组检查实际观测／入库／可用时钟；重建组保留特征可用假设。预算只保留4个成本分量的非负有限limits/spent和unknown_components。

`to_research_actions` 从公共能力目录投影。未知成本仍null，不补0；配置缺失保持available=false。无研究支持组的动作不进入数值候选，部分组不支持的动作禁用，避免静默拆分一个计费动作。它不读取目标比赛未知数据或未来缺失状态。

版本：`research-public-information-bridge-v1`。桥接及序列化策略边界机制测试11项通过，命令：

```powershell
python -m pytest tests/research/acquisition/test_platform_bridge.py -q
```

早期证据：`var/parallel/baselines/round3/platform-bridge/catalog-manifest.json` 保存真实公共类型序列化。`public-type-real-history-check.json` 使用真实Understat2025前60场拟合部分模型、对2025-10-02赛前时点构造公共VisibleState/EvidenceGroup并消费，概率分别为基础[0.444444,0.285714,0.269841]、揭示strength后[0.446937,0.355345,0.197718]。该早期文件保留原 `platform_environment_executed=false` 类型联调身份，后续环境实跑在独立报告中记录。

## 已完成：真实 FrozenEvidenceEnvironment 集成

平台环境发布后，实际执行公共 `start → serialized state → bridge → partial → gain → select → acquire → state → stop`。入口 `football_research.acquisition.platform_demo`；报告 `var/parallel/baselines/round3/platform-bridge/environment-integration.json`，其中 `platform_environment_executed=true`。此新增入口未修改 active-information-v2 已冻结的 experiment/partial/value/policy/data/state/platform_bridge 七个文件；运行前后核对其发布SHA256一致。

使用同一真实a3b3冻结Understat导出，`active-information-v2` fold 2模型，固定测试比赛 `654a90d2-65a9-5cae-8494-f2fb685a122b`，截止 `2025-12-29T19:30:00+00:00`。采用该fold验证集既定 `lambda.money=0.25`，预算0.05模拟credit/2投影调用，未在此单场重新拟合／调参。该场用于展示有动作的集成路径，选择不基于目标赛果；这不是额外独立效果样本。

| 步骤 | 概率 H/D/A | 数值选择 |
|---|---|---|
| 空集合 | 0.512195 / 0.211382 / 0.276423 | xg_view估计gain=0.008333873，支持80场；净值=0.008333873−0.25×0.03=0.000833873，执行 |
| 实际揭示 rolling_xg | 0.483983 / 0.222301 / 0.293716 | results_view估计gain=−0.001117594，支持53场；净值=−0.006117594，停止 |

平台 `acquire(xg_view)` 返回真实 `ACQUIRED`，strength与recent_form保持UNACQUIRED。最终停止原因为nonpositive_net_gain。总模拟支出money=0.03、requests=1、latency_ms=0、tokens=0；实际外部联网请求0、LLM tokens 0。获取的数值来自已验证缓存，费用语义仍为模拟揭示，并未变成真实供应商节费。

同request_id重复执行返回repeated_request=true、四分量增量收费0，完整public state保持一致。每个决策步骤还执行owner侧内存反事实检查：改变全部未获取组数值及目标标签，重放已获取动作，所得public state、概率、增益和动作／停止均相等。该检查标为合成机制扰动，正式源文件未修改。完整manifest与标签仅在可信owner侧，`decide_public`函数只收JSON序列化state/catalog及冻结训练模型，没有环境对象参数。该协议边界不等同进程或操作系统权限沙箱。

本次实测预测器调用约0.10/0.25ms，增益估计约0.07/0.12ms；这些是本机两次调用测量，非稳定生产时延分位数。模型版本、完整公开轨迹、收费收据、幂等验证和反事实检查均在报告中。

复跑需新报告路径以保留首份证据：

```powershell
python -X utf8 -m football_research.acquisition.platform_demo --platform-export var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --experiment var/parallel/baselines/round3/active-information-v2 --output var/parallel/baselines/round3/platform-bridge/environment-integration-rerun.json
```

v3修复了实验轨迹中可变动作历史的记录问题；同一平台实链已用`active-information-v3`复跑，报告为`environment-integration-v3.json`。模型、动作、验证lambda和平台成本绑定不变，复跑仍为先取`xg_view`、再以`nonpositive_net_gain`停止，模拟支出0.03/1次投影，实际网络0。
