# Agent 第十一轮：冻结预测器下的自主查询与决策卡

本轮将“是否值得查、下一步查什么、何时停止”落到实际 Agent 工作流与可核对的事件卡片上。研究侧比较同一冻结预测器产生的后续收益校准，与相同校准比赛 ID 的旧混合 OOF 标签校准；执行侧保持既有预测、校准器加载、预算和工具协议，使用两组冻结工件分别进行查询决策。

这不是通过手工选择某场的查询工具来展示效果。每个目标时间折固定按 kickoff、match ID 选择第一场，三个策略拿到相同基础信息和候选目录，独立使用自己的数值门控选择 `results_view`、`xg_view` 或 STOP。选场不参考实际收益、赛果或是否发生查询。

## 固定模型、固定效用口径

目标时间折为 2–6，分别使用参考折 0–4 的冻结联合预测器。研究工件在 `fold.joint_model` 记录预测器身份，`fold.calibrators.same_version` 与 `fold.calibrators.matched_mixed` 记录两个残差校准器。预测器仍为 `learned-joint-returns-v1`，校准器仍为 `residual-evsi-calibrator-v1`。

实际验收策略：

| 策略 | 校准工件 | 门控 |
| --- | --- | --- |
| `frozen-version-mean` | `same_version` | 经验单次更新收益均值 |
| `frozen-version-lcb` | `same_version` | 描述性按周标准误下界 |
| `matched-mixed-mean` | `matched_mixed` | 同比赛 ID 的混合 OOF 校准均值 |

三个策略统一固定 `lambda_money=.5`，物理预算为 `.05 simulation_credit` 和最多两次 `projection_call`。这次不从各策略验证曲线再次挑选 lambda。每笔候选成本单独乘以 `.5` 转换为 log-loss 惩罚；现金预算与请求次数仍是独立的物理限制。

仍然使用 `CalibratedJointAcquisitionWorkflow` 和独立 `make_calibrated_joint_backend`。本轮没有修改旧工厂默认行为或前几轮冻结源码。新配置中的 `research_release` 明确记录 `reference_fold`、校准组、固定 lambda 来源及报告费用权重。

## 15 条实际工作流验收

新入口 `football_agent/verify_round11_frozen.py` 从完整正式报告的 `base-only/.05` 赛程确定每折首场，执行五折乘三个策略，共 15 条真实 Agent 历史回放。它验证：

- 所用预测器和校准器文件 SHA256 与正式研究清单一致；校准器引用的 joint 版本和哈希与实际加载模型一致。
- 每次估值的全部原生候选字段与对应研究轨迹一致，包括模型 EVSI、校准 `g_hat`、下界、费用、支持量、父层回退、掩码和选择分数。
- 每次实际返回之后的概率、已观测返回类别和模型版本与研究轨迹一致；摘要缺失同样要求刷新后验。
- 自主选择的动作顺序、最终概率、实际模拟费用、调用次数和停止状态一致。
- 事件链及输入/输出 blob 的审计完整通过；真实状态不出现假设返回 token。

没有支持、收益非正或费用不足时，STOP 本身就是正式案例结果。验收不为制造动作展示重新挑场。

## 查询决策卡

检索现有模块后，项目尚无独立的结构化查询决策卡。新增 `football_agent/query_decision_card.py` 提供：

```python
from football_agent.query_decision_card import build_query_decision_cards

workflow.store.verify(run_id)  # 验事件链及外部工具 blob
run = workflow.get_run(run_id)
card_document = build_query_decision_cards(run)
```

builder 本身是无 I/O、无模型调用的纯转换，校验内联事件哈希链和最终快照。发布卡片之前仍需 `RunStore.verify` 完成工具 blob 内容哈希检查；卡片不会把仅验证内联链宣称为验证全部输入/输出。

每个 `value_estimated` 事件产生一张决策卡，字段直接来自该时刻快照与后续实际事件：

| 字段 | 来源与含义 |
| --- | --- |
| `current.probabilities` | 查询前当时已存在的赛果概率 |
| `current.public_mask/group_statuses` | 当时公开证据掩码与缺失状态 |
| `current.observed_return_categories` | 当时已经取得的返回类别 |
| `candidates[]` | 每个候选的 EVSI、经验收益、下界、成本、净值、支持量、回退和费用范围 |
| `decision` | 实际 `action_selected`，或实际 STOP/DEFER 事件 |
| `receipt` | 实际 action/request ID、状态、模拟收费、证据引用、返回后的公开掩码 |
| `receipt.prediction_after_return` | 该返回之后实际执行过的后验更新；没有更新时为 null |
| `audit` | 对应事件序号和哈希，支持回查原始事件 |
| `final_probabilities/actual_actions` | 整条已结束工作流的最终状态，明确与查询前状态区分 |

查询前的掩码和概率来自当时 `value_estimated.snapshot`，不会从最终状态反填已经付费获取的证据。卡片不复制实际特征值、隐藏结果或完整响应正文，只保留公开决策和收据字段。

`NO_DATA/summary_missing` 与 `PROVIDER_ERROR` 分开记录。前者可在 Boolean 特征掩码仍为 false 时改变观测类别并刷新概率；后者没有被模型覆盖，会保留明确的延迟处理和空更新，而不会伪造成缺失类别的成功后验。

终端状态另有一张 terminal 卡，所以因预算停止而没有再执行估值的情况也有记录。卡片没有 LLM 生成的解释、人工补充的因果故事或对实际赛果的事后归因。

## 证据边界

两个候选工具来自同一个 Understat 历史数据响应，只是不同受控视图；它们不是两个独立来源。执行成本是历史环境的模拟费用，没有本轮真实 HTTP 或 LLM 调用。实验仍处于同域时间重建，不是全新在线数据的前瞻验证。

本轮体现了给定合法工具目录中的自主信息选择、数值停止和可重放执行。开放工具发现、新来源真实性判断、联赛到国家队/杯赛的迁移、阵容/动机/战术证据的学习价值均未因此自动解决。LCB 依然是描述性均值摘要，不是参数后验或覆盖保证；模型版本一致性的改进也需要完整实验检验，不能只凭桥接通过宣称有效。

## 本轮验证方式

```powershell
python -m pytest tests/agent/test_round11_frozen.py --import-mode=importlib -q
python -m football_agent.verify_round11_frozen --research var/parallel/baselines/round11/frozen-version-v1 --output var/parallel/agent/round11/frozen-version-bridge-v1 --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda
```

正式报告完成前，已用第九轮第 0 折固定首场的真实 mean/LCB 运行文件验证卡片结构，完整核对 `RunStore` 事件/blob 审计；其他专项使用明确标注的构造轨迹，检查查询前后信息边界、停止、摘要缺失、失败、篡改、模型版本、固定 lambda 与每次后验更新。本轮最终实际桥接结果单独记录在正式 Agent 验收工件中，不以这些构造测试替代。

正式研究 `var/parallel/baselines/round11/frozen-version-v1` 已完成，实际 Agent 验收在 `var/parallel/agent/round11/frozen-version-bridge-v1/acceptance.json`，状态为 `ROUND11_FROZEN_PREDICTOR_AGENT_QUERY_DECISION_PARITY_VERIFIED`。15 条工作流全部通过，共核对 37 个候选、23 次初始/返回后概率、8 个实际历史揭示动作、311 个审计事件；总模拟费用 `.21`。15 份决策卡文档位于 `decision-cards/`，共 37 张查询/终止卡，每份文件的 SHA256 都由验收清单绑定。

固定首场的实际动作如下，顺序由候选数值门控决定：

| 目标折 / 参考模型折 | same mean | same LCB | matched mixed mean |
| --- | --- | --- | --- |
| 2 / 0 | xG → STOP | STOP | results → xG → STOP |
| 3 / 1 | results → STOP | STOP | results → STOP |
| 4 / 2 | STOP | STOP | STOP |
| 5 / 3 | STOP | STOP | xG → STOP |
| 6 / 4 | xG → STOP | STOP | xG → STOP |

该验收复用 5 场已有历史比赛，新增独立比赛数为 0，真实外部调用和 LLM 调用均为 0。LCB 在五个首场都停止；完整 200 场研究中的 LCB 实际查询了 4 次，不能从这五个固定案例推断全研究一律停止。

完整研究在预算 `.05` 下给出的结果如下，数值越低越好：

| 策略 | Log loss | 每场模拟费用 | 固定 `.5` 费用权重的 utility loss |
| --- | ---: | ---: | ---: |
| 零查询 | 1.109339 | 0 | 1.109339 |
| 同版本 mean | 1.121521 | .019800 | 1.131421 |
| 同版本 LCB | 1.112433 | .000400 | 1.112633 |
| 同比赛 mixed mean | 1.111102 | .025550 | 1.123877 |

同版本 mean 的效用损失相对零查询增加约 `.022082`，描述性按周配对区间 `[-.011886,.064847]` 跨零。桥接正确与自主选工具的链路成立，并不意味着校准改动改善了实际预测。本轮更早冻结的预测器、有限校准窗口和反复探索过的 200 场历史样本共同限制结论；两校准组还同时改变 EVSI、收益标签的生成模型与残差分桶，因此不作单因素因果归因。

本轮新增专项 **15 passed**；与冻结的第九轮校准桥接专项合并 **42 passed**。项目全回归由第十一轮总验收记录。
