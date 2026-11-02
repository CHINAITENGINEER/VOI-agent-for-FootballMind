# Agent 第十二轮：检验按当前证据选工具是否胜过历史平均收益

本轮执行预先冻结的 `docs/research/acquisition/round12-design.md`，SHA256 为 `884ac4fe147f006855acbd959b21e824d78e7169c001f1f70a41759dfe2cf8a8`。核心问题是：同一个预测器和收费条件下，按当前证据估值的自主查询，是否优于只知道工具历史平均收益的简单策略。新增两条明确屏蔽状态数值的基线，不调整旧预测器、校准器或工厂默认值。

## 估值屏蔽与预测更新分开

| 策略 | 数值估值读取内容 | 历史收益平均方法 |
| --- | --- | --- |
| `tool-prior-mean` | 动作 ID | 每场每工具先平均两个已记录状态的单次更新收益，再按比赛平均 |
| `stage-prior-mean` | 动作 ID、已获取动作集合 | 对相同动作及获取集合，按比赛平均单次更新收益 |

这两者不把当前概率、模型 EVSI、队名、比赛日期或已获取返回的具体数值输入收益估计。日期仍参与 as-of 门禁，状态仍参与身份、可用性、失败和预算检查；这些检查不构成数值估值特征。`source_missing` 是已经取得的一类响应，所以属于 acquired set。

预测层保持冻结的 `learned-joint-returns-v1`，每次真实历史揭示后仍根据实际返回更新赛果分布。测试明确验证：改变合法的已获取数值和队名会改变预测，但两条基线的同动作/同获取集合估值不变；改变获取集合允许 stage 策略的估值改变。工具平均法给予两个记录状态各一半权重，并非训练得到的部署状态访问分布。

新工件身份为 `historical-gain-prior-v1`。候选 `model_evsi`、`evsi_bin`、`lower_gain` 均为 null，不用零冒充已计算的 EVSI。按周标准误仅保留为审计描述，本轮只有 mean 门控。最小支持量沿用 12 场；支持不足进入 STOP，负收益保留原值。固定 `lambda_money=.5`，报告效用费用权重也为 `.5`；`.05 simulation_credit` 和两次 `projection_call` 是独立物理预算，lambda 为零的专项测试仍严格限制费用。

## 独立接口

新增 `football_agent/gain_prior_bridge.py`，提供 `GainPriorServices`、`load_gain_prior_services`、`make_gain_prior_backend`、`GainPriorAcquisitionWorkflow` 和 `build_gain_prior_cards`。它复用旧加载绑定、公开状态门禁、执行及审计逻辑，显式覆盖配置身份和 mean-only 门控入口；旧加载器继续拒绝这个新 kind，不改变旧行为。

最小模型配置如下，`experiment` 和 fixtures 在实际工作流配置中仍须提供：

```json
{
  "partial_predictor": "models/fold-2-joint.json",
  "partial_predictor_kind": "learned-joint-returns-v1",
  "value_estimator": "models/fold-2-action_only.json",
  "value_estimator_kind": "historical-gain-prior-v1",
  "gate": {
    "kind": "historical-gain-prior-one-step-v1",
    "risk_level": "mean",
    "erasure_mode": "action_only"
  },
  "lambda_cost": {"money": 0.5}
}
```

示例路径用于说明相对配置定位，正式文件名由研究 `fold.gain_priors` 清单给出。stage 策略使用对应 `acquired_set` 工件和同名 erasure mode；声明与工件不匹配时加载报错。校准工件引用的预测器版本、SHA256 和训练时点必须与实际预测器一致。

新卡片 wrapper 调用冻结的第十一轮 builder，并在实际事件映射上添加真实估值 kind 和屏蔽模式。每个候选维持 null EVSI/null lower；查询前概率、公开掩码、候选收益、成本、支持、实际选择、收费收据及返回后概率，均来自当时事件。既有事件链和外部 blob 仍由 `RunStore.verify` 核验。

## 固定首场实际验收

正式研究目录为 `var/parallel/baselines/round12/state-ablation-v1`。其 episode 文件仅保存两条新增策略，本轮验收因此从研究报告已经绑定 SHA256 的第十一轮 episode 文件读取 `base-only/.05` 日程，再按 kickoff、match ID 选择每个目标折的第一场。选场不读取收益、赛果或查询次数。

五个目标折各运行两条新策略，共 10 条实际 Agent 历史工作流。原有五条同版本 mean 卡片核对文件哈希、模型哈希、时间和完整运行审计后直接引用，没有再次执行。固定首场结果：

| 目标折 / 参考模型折 | tool prior | stage prior | 原条件 mean 卡片 |
| --- | --- | --- | --- |
| 2 / 0 | xG → STOP | xG → STOP | xG → STOP |
| 3 / 1 | STOP | results → STOP | results → STOP |
| 4 / 2 | STOP | results → STOP | STOP |
| 5 / 3 | xG → STOP | STOP | STOP |
| 6 / 4 | STOP | xG → STOP | xG → STOP |

验收文件 `var/parallel/agent/round12/state-ablation-bridge-v1/acceptance.json` 状态为 `ROUND12_STATE_ERASURE_AGENT_QUERY_PARITY_VERIFIED`。10 条工作流核对了 26 个候选、16 次初始/返回后概率、6 次实际历史揭示和 218 个审计事件，总模拟费用 `.16`；10 份卡片文档共含 26 张查询/终止卡。全部动作、原生候选数值、支持/回退、后验、STOP、费用、请求预算以及配置/模型哈希与研究轨迹相符。验收清单同时绑定新增源代码、复用代码、研究报告、轨迹及原赛程哈希。

`NO_DATA/summary_missing` 的后验刷新与 provider failure 的延迟处理继续保持区分。专项还检查隐藏结果、假设 token、未来模型、信息模式混用和真实失败状态均在运行门禁被拒绝，而不伪装为正常缺失。

## 结果含义及证据边界

完整研究使用此前已探索的 200 场历史比赛；预算 `.05` 的固定效用结果如下，数值越低越好：

| 策略 | Log loss | 每场模拟费用 | 固定 `.5` 权重效用损失 |
| --- | ---: | ---: | ---: |
| 零查询 | 1.109339 | 0 | 1.109339 |
| 工具平均 | 1.116457 | .012000 | 1.122457 |
| 获取集合平均 | 1.119416 | .021550 | 1.130191 |
| 原条件 mean | 1.121521 | .019800 | 1.131421 |

预定主要比较为条件 mean 减 stage prior，其效用损失差为 `+.001229821`，描述性按周配对区间为 `[-.010434981, .016459984]`。本轮没有显示当前条件估值优于更简单的获取集合平均基线，也没有显示这几个查询策略胜过零查询。区间跨零不等于两种策略等效；估值器的偏差和方差也随方法变化，差异不解释为纯粹的“条件信息”因果效应。

本轮有 0 场新增独立比赛、0 次真实外部请求、0 次 LLM 调用。results 与 xG 仍是同一个 Understat 历史响应的两个受控视图，费用为模拟投影收费。自主选择链路的工程通过，不代表开放工具发现、真实线上信息价值或新赛事域泛化已完成。资格、轮换、阵容和战术仍需未来专门的数据与实验。

## 验证入口

```powershell
python -m pytest tests/agent/test_round12_gain_prior_bridge.py --import-mode=importlib -q
python -m football_agent.verify_round12_state_ablation --research var/parallel/baselines/round12/state-ablation-v1 --output var/parallel/agent/round12/state-ablation-bridge-v1 --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --conditional-agent var/parallel/agent/round11/frozen-version-bridge-v1
```

输出目录采用新建要求，重复验收时使用新的目录保留已有工件。本轮新增专项 20 项通过；与冻结的第九轮、第十一轮专项合并为 62 项通过。项目总回归由根验收记录。专项中用于强制 missing/failure 分支的 `.1` 收益明确为构造测试参数，不进入正式研究指标。
