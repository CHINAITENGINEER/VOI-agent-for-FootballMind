# Agent 第十三轮：已获取内容对下一次查询的作用

本轮按冻结的 `docs/research/acquisition/round13-design.md` 执行，SHA256 为 `dc26af59df5fc7533130c48a2767e643a7d86835f8b8610c16c71ac724309054`。研究问题进一步缩小为：比赛免费先验、已经查询的工具集合和预算相同时，实际返回了什么内容是否改变下一次查询/停止，以及保留这种内容敏感性是否改善实际效用。

新增控制策略 `content-marginal-mean`，不重新训练预测器或残差校准器。它复用第十一轮五个精确冻结的 joint 工件和 same-version 残差校准器。尚未查询时，与原条件 mean 的候选收益及首次决策完全一致；已查一个工具时，使用免费先验联合分布对该工具所有正概率类别逐支计算原条件收益，再按概率平均。实际已经观察到的返回类别不参与这组边际化权重。

这项屏蔽只作用于查询估值。实际预测器仍使用真正揭示的返回更新赛果概率。分类分支是模型内部的数学对象，不代表发生过的足球事件、工具请求或真实证据。

## 独立接线与三工件绑定

新增 `football_agent/content_marginal_bridge.py` 提供：

- `ContentMarginalServices` 和 `load_content_marginal_services`：声明独立 wrapper 身份，复用既有公开状态、时间、失败及预算检查。
- `make_content_marginal_backend`：创建冻结历史环境后端，保留既有工厂默认行为。
- `ContentMarginalAcquisitionWorkflow`：仅允许单步 mean 门控，继续使用实际工具执行和缺失后验刷新机制。
- `build_content_marginal_cards`：复用冻结卡片 builder，另外加入查询估值屏蔽范围和模型类别积分审计。

配置身份如下；各文件路径相对于配置文件解析，正式文件名取研究清单：

```json
{
  "partial_predictor": "models/fold-2-joint.json",
  "partial_predictor_kind": "learned-joint-returns-v1",
  "source_calibrator": "models/fold-2-source-calibrator.json",
  "source_calibrator_kind": "residual-evsi-calibrator-v1",
  "value_estimator": "models/fold-2-content-marginal.json",
  "value_estimator_kind": "content-marginal-gain-v1",
  "gate": {
    "kind": "content-marginal-one-step-v1",
    "risk_level": "mean"
  },
  "lambda_cost": {"money": 0.5}
}
```

实际配置另有 `experiment`、fixtures 和 research release。wrapper 工件绑定预测器版本/哈希、原残差校准器版本/哈希及时间元数据。运行配置暴露四个独立 SHA256：配置、预测器、原校准器和 wrapper。文件格式化导致原字节哈希变化也会被绑定校验检出，哈希绑定不被解释为数据来源真实性证明。

候选顶层 `model_evsi`、`evsi_bin`、`lower_gain`、`standard_error` 和 `uncertainty` 均为 null；另设 `marginal_model_evsi`，避免把平均模型 EVSI 当成当前真实后验下的条件 EVSI。任何正概率分支缺少支持时，整个候选保留 unsupported；分支不会被丢弃后重新归一化。报告的最小分支支持量是原校准诊断的保守摘要，不是额外独立样本数。

固定 `lambda_money=.5` 与报告效用费用权重 `.5`。物理限制仍为 `.05 simulation_credit`、最多两次 `projection_call`，并单独检验 lambda 为零仍遵守费用预算。没有新增 LCB 或平均收益不确定性覆盖声明。

## 卡片中的数学分支与真实收据

查询卡保留 `current.probabilities`、当时公开证据掩码和已观测返回类别。候选新增 `model_only_value_audit`，包括边际模型 EVSI、类别权重、逐支校准收益及支持、source calibrator 身份，并显式声明 `counts_as_acquisition=false`。每个分支保留 `hypothetical=true`。空获取集合的单个概率 1 分支只表示尚无内容需要平均，不算模拟请求。

工具 `receipt` 仍完全由实际 `acquisition_result` 和实际 `prediction_updated` 事件生成，模型分支不会被插入收据、公开 evidence 或真实状态。运行入口继续拒绝客户端提供的 hypothetical envelope。`NO_DATA/summary_missing` 是模型定义的观测类别，需刷新实际后验；provider failure 在拟合模型范围外，执行链保留延迟状态和空后验更新。

## 固定场次验收方法

`football_agent/verify_round13_content_marginal.py` 读取研究报告已绑定哈希的第十一轮赛程，按 kickoff、match ID 固定目标折 2–6 的各自第一场。只执行新控制策略五条工作流，直接验证并引用五份已有条件 mean 卡片，不重新执行旧策略，也不按查询次数、分支差异或结果挑场。

每条工作流检查全部实际原生候选字段及数学分支与研究轨迹相符，检查选择/停止、每次真实后验、费用与请求预算、事件及 blob 审计、模型与校准器身份。卡片还与旧条件卡逐项比较初始概率、候选收益、净值及首次动作，明确边际化只从已经取得返回之后产生潜在差异。

```powershell
python -m pytest tests/agent/test_round13_content_marginal_bridge.py --import-mode=importlib -q
python -m football_agent.verify_round13_content_marginal --research var/parallel/baselines/round13/content-marginal-v1 --output var/parallel/agent/round13/content-marginal-bridge-v1 --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --conditional-agent var/parallel/agent/round11/frozen-version-bridge-v1
```

专项 22 项通过，覆盖类型/三工件绑定、首次决策保持、后验变化与估值内容不变、缺失类别完整积分、预算、失败、时间、隐藏输入和卡片分离。与冻结第九、十一、十二轮专项合并为 84 项通过。专项构造的缺失/失败返回不计入历史研究效果。

## 正式验收结果

研究 `var/parallel/baselines/round13/content-marginal-v1` 正式完成 800 条新测试轨迹、852 条新验证轨迹。Agent 实际回放输出在 `var/parallel/agent/round13/content-marginal-bridge-v1/acceptance.json`，状态为 `ROUND13_CONTENT_MARGINAL_AGENT_QUERY_PARITY_VERIFIED`。

五条工作流全部通过，共核对 13 个候选、8 次初始/真实返回后概率、28 个数学积分分支、109 个审计事件。实际只有 3 次历史揭示，模拟费用合计 `.08`；五份卡片文档含 13 张查询/终止卡。28 个数学分支包含初始状态概率为 1 的记录，它们都不计入查询次数或独立比赛。五份旧条件策略卡片核对哈希后引用，未重复执行。

| 目标折 / 参考模型折 | 新 content marginal | 原条件 mean |
| --- | --- | --- |
| 2 / 0 | xG → STOP | xG → STOP |
| 3 / 1 | results → STOP | results → STOP |
| 4 / 2 | STOP | STOP |
| 5 / 3 | STOP | STOP |
| 6 / 4 | xG → STOP | xG → STOP |

上述五场全部保留原固定首场，无额外挑选。每次原生候选的分支权重、逐支校准量、平均值、支持/回退及后续选择与研究轨迹一致；预测使用真实历史返回，配置、三个模型工件、事件链和 blob 审计一致。正式验收文件 SHA256 为 `0c6cd05b9ebaf368938b5c85a77ef0f766eb6cd7a3e5e20d6aaf6ea2b37d637b`，其实现哈希再次读取确认未变化。

## 这次没有观察到实际路径收益

完整研究的测试 800 条和验证 852 条新轨迹均与原条件策略的动作路径相同，低预算路径、费用和概率也保持一致。预算 `.05` 的完整 200 场测试中，新控制与原条件 mean 都查询 results 60 次、xG 92 次；两者 Log loss 为 `1.121520518`，每场模拟费用 `.0198`，固定 `.5` 费用权重下的效用损失为 `1.131420518`。零查询的效用损失为 `1.109338831`。

预定主比较“条件 mean 减 content marginal”的逐场效用差全部为零，描述性 bootstrap 区间因此退化为 `[0,0]`。它只描述这组固定策略在已实现历史路径上的相同结果，不证明方法在新比赛或其他返回类别下等效，更不证明保留内容提高了实际效用。模型内类别敏感性诊断与实际到达路径的收益分开报告；未到达的分支不获得真实结果标签，也不增加样本量。

两个候选仍来自同一个历史响应的受控视图。本轮新增独立比赛、真实外部请求及 LLM 调用均为 0。对模型类别的穷举用于揭示拟合模型内的内容敏感性，不新增比赛样本，也不把某个虚构返回与原赛果配对解释为因果收益。该轮仍是已反复探索历史上的消融，真实开放来源的在线价值和新赛事域迁移留待后续独立证据。
