# 第三轮研究接口 v1.0（可执行，2026-09-22）

本页由研究任务维护。实现目录 `football_research/acquisition/`；领域ActionSpec/VisibleState由平台维护，Agent在边界转为下列只含可见数据的字典。接口不得传完整features、manifest、标签、原始档案路径或可加载隐藏字段的对象。380场真实数据训练、290场滚动受控揭示实验已完成，报告见本目录README；研究效果为探索性负结果。

## 部分概率预测

```python
from football_research.acquisition.partial import PartialPredictor
predictor = PartialPredictor.load('MODEL.json')
prediction = predictor.predict_partial(visible_state)
# prediction['probabilities'] == [p_home, p_draw, p_away]
# 另含三标准概率字段、mask、mask_key、model_version、prior_version与可见血缘
```

训练仅离线调用 `PartialPredictor().fit(rows, as_of=aware_cutoff, dataset_version=version, regularization=10)`。rows为已按历史窗口筛选的监督行；标签仅进入fit。空集合是独立冻结的训练联赛频率先验，其余最多7个子集分别训练正则化模型。输入未获取并非完整模型插补。

visible_state最小研究投影：

```python
visible_state = {
  'match_id': 'CANONICAL_ID', 'prediction_cutoff': '2025-10-31T15:00:00Z',
  'information_mode': 'historical_reconstruction',
  'base': {'competition_id': 'LEAGUE_ID', 'home_team_id': 'HOME_ID',
           'away_team_id': 'AWAY_ID', 'kickoff': '2025-11-01T15:00:00Z'},
  'groups': {
    'elo': {'status': 'observed', 'values': {'home_elo': 1500., 'away_elo': 1480.},
            'available_at': '2025-10-30T12:00:00Z', 'version': 'FEATURE_VERSION',
            'evidence_refs': ['sha256:VISIBLE_GROUP_DIGEST']},
    'recent_form': {'status': 'not_acquired'},
    'xg': {'status': 'not_acquired'}
  }
}
```

状态区分 observed / not_acquired / source_missing / stale。未获取、缺源、过期组的values不供模型使用；future available_at拒绝。mask由有效可见组派生，不信任调用者伪造mask。运行时只接受上述字段投影。evidence_refs仅可见摘要/标识，不含完整特征文件路径。

固定研究字段组（平台可用ActionSpec一次返回多个group_ids，按一次动作计成本）：

- elo：home_elo、away_elo。
- recent_form：双方home/away_rolling_goals_for、rolling_goals_against、rolling_points。
- xg：双方home/away_rolling_xg_for、rolling_xg_against。

## 经验增益与停止

```python
from football_research.acquisition.value import ValueEstimator
from football_research.acquisition.policy import select_or_stop
estimator = ValueEstimator.load('GAIN_MODEL.json')
estimates = estimator.estimate(visible_state, legal_actions, prediction=prediction)
decision = select_or_stop(visible_state, estimates, budget=remaining_budget,
                          lambda_cost=frozen_lambda, conservative=False)
```

研究action投影为 `action_id, group_ids, available, cost, cost_status`，cost向量显式提供requests/money/latency_ms/tokens四分量，单位保留；动作不含未来返回值/目标/未来成功状态。研究私有投影不替代平台公共ActionSpec。`platform_bridge.to_research_actions`已消费公共类型并保存实际目录版本/绑定。

估计返回gain、standard_error、lower_gain、support_matches、cost、uncertainty_kind、version、reason；候选保留动作ID/返回组。estimate阶段net_gain为null，由policy结合冻结lambda计算。policy输出decision='action'或'stop'、action_id、reason、candidates、rejected、净值与成本解释。net_gain=gain-sum(lambda_component*cost_component)，lambda单位是每成本单位对应的log-loss，不直接拿money或ms减loss。budget是剩余额度，省略分量表示该维度无上限；省略lambda分量权重为0。可设deadline_ms_remaining。周块SE只给经验解释，没有严格覆盖保证。

ValueEstimator.fit(samples, as_of=..., dataset_version=...)只接历史训练内部向前OOF监督，样本含state/action/delta/prediction_before/match_id/kickoff/result_available_at/oof_train_cutoff/oof_train_ids。运行estimate无delta或标签。低支持显式回退或停止，负Delta保留。lambda用外层验证段选择后冻结，测试不参与。

## CLI 与实验边界

本轮入口 `python -m football_research.acquisition run --platform-export DIR --output NEW_DIR`。它完成精确特征连接、子集模型训练、向前OOF增益监督、验证lambda、滚动测试与图表。输出目录须全新，保留冻结运行。

正式产物：`var/parallel/baselines/round3/active-information-v3/`。每fold的`models/fold-N-partial.json`与`models/fold-N-gain.json`分别用上面的load读取。`report.json.folds[N].lambda_selection['learned-net-gain/0.05'].selected`为预算0.05下该fold验证冻结的money权重。训练ID/时点、验证ID、测试ID均写在报告与模型metadata/manifest。每个模型仅适用于自身训练时间之后及对应数据ID命名空间；请勿把末fold模型用于早期比赛。

## 公共平台接入与产物版本

```python
from football_research.acquisition.platform_bridge import to_research_state, to_research_actions
state = to_research_state(public_visible_state)  # 平台VisibleState或其model_dump
actions = to_research_actions(public_action_catalog)
prediction = predictor.predict_partial(state)
estimates = estimator.estimate(state, actions, prediction=prediction)
decision = select_or_stop(state, estimates,
    budget={'money': 0.05, 'requests': 2}, lambda_cost={'money': frozen_lambda})
# 只有可信执行器处理decision['action_id']，再返回新的公开状态。
```

公共组strength/recent_form/rolling_xg映射elo/recent_form/xg；ActionSpec ID保持原样。本次训练动作results_view返回elo+recent_form，模拟0.02；xg_view返回xg，模拟0.03。两者各1次模拟投影调用。改变动作ID/绑定/成本需要新的验证配置，不能沿用未经核验的lambda或假设旧g已支持新动作。

partial版本`partial-subset-logistic-1.0:<sha256>`，空先验`partial-frozen-prior-1.0:<sha256>`；gain版本`empirical-mask-action-shrinkage-v1:<hash>`；策略`one-step-empirical-net-gain-v1`；实验`controlled-reveal-forward-v1`。数据为`platform-history-research-v1.0:a3065657f0a6e7e8385e783a9169e23d35e419469c54061e1da91ee4806493ad`，ID为a3b3正式导出的canonical UUID，不混用后续9b013映射增强导出。

旧`football_research.baselines.PredictionService`接口保持兼容。部分预测器不要求报价，额外隐藏值始终由白名单投影丢弃。平台环境对伪造公开状态另做可信恢复校验；Python字典隔离不等于进程权限隔离。

当前380场及已看过的290测试场仅用于探索性历史重建/受控揭示，成本为显式模拟。独立前瞻结算另报；多mask及多seed不扩大独立比赛数。真实LLM策略只有Agent提供真实调用及成本记录后接入，否则明确skipped。

## 第四轮风险感知扩展（v1）

此节是旧接口记录。新消费使用下文 calibrated v2；v1 的均值区间和二步控制流缺陷已在 `resume-audit.md` 列明，保留仅用于复现旧实验。

```python
from football_research.acquisition.risk import RiskAwareValueEstimator
from football_research.acquisition.lookahead import select_two_step

estimator = RiskAwareValueEstimator.load('fold-N-risk.json')
values = estimator.estimate_value(
    visible_state,
    action_specs,
    prediction=prediction,
    risk_level='mean',             # or 'conservative'
)
decision = select_or_stop(
    visible_state, values,
    budget={'money': 0.05, 'requests': 2},
    lambda_cost={'money': frozen_lambda},
    conservative=False,
)
two_step = select_two_step(
    visible_state, action_specs,
    predictor=lambda state: predictor.predict_partial(state),
    estimator=estimator,
    budget={'money': 0.05, 'requests': 2},
    lambda_cost={'money': frozen_lambda},
)
```

`estimate_value` 返回 `expected_gain`、`uncertainty`、`support`、`lower_gain`、`cost_vector`、`net_gain`（由policy填充）、`estimator_version`、`fallback_reason` 和 `quality_context`。质量上下文只来自 VisibleState 中已经揭示的 status/available_at 和 ActionSpec 的 dependency/cache 元数据；未知值保持 `unknown`。运行时不接受 result(a)、标签、完整 FeatureIndex、未来快照或原始路径。

第四轮正式产物为 `var/parallel/baselines/round4/risk-aware-v4/`，实验入口为 `python -m football_research.acquisition round4`。模型 schema `risk-aware-conditional-value-v1`，策略版本 `two-step-mask-lookahead-v1`。风险区间是 OOF 独立比赛残差的经验区间，没有 Bayesian posterior 或严格覆盖保证；二步前瞻只做最多一个 projected continuation，不声称全局最优。

## 当前消费入口：calibrated v2（2026-09-26 复验）

```python
import json
from pathlib import Path
from football_research.acquisition.partial import PartialPredictor
from football_research.acquisition.risk_calibrated import CalibratedRiskAwareValueEstimator
from football_research.acquisition.lookahead_calibrated import select_two_step_v2
from football_research.acquisition.policy import select_or_stop
from football_research.acquisition.platform_bridge import to_research_state, to_research_actions

run = Path('var/parallel/baselines/round5/two-season-v2')
fold = 0  # 根据模型训练截止时间选择，不把末折模型用于更早比赛
predictor = PartialPredictor.load(run / 'models' / f'fold-{fold}-predictor.json')
estimator = CalibratedRiskAwareValueEstimator.load(run / 'models' / f'fold-{fold}-risk.json')
report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
frozen_lambda = report['folds'][fold]['lambda_selection']['risk-mean/0.05']['selected']

# public_state / public_catalog 由可信平台执行器提供，含可见证据和合法候选。
state = to_research_state(public_state)
actions = to_research_actions(public_catalog)
prediction = predictor.predict_partial(state)
values = estimator.estimate_value(state, actions, prediction=prediction, risk_level='mean')
decision = select_or_stop(state, values, budget=remaining_budget,
                          lambda_cost={'money': frozen_lambda}, min_support=12)
```

`remaining_budget` 是执行器提供的剩余额度，沿用 requests/money/latency_ms/tokens，截止期限用 `deadline_ms_remaining`。训练的动作 ID 为 `results_view` 和 `xg_view`，公共组 strength/recent_form/rolling_xg 映射到研究 elo/recent_form/xg。新目录中的额外动作不自动获得支持。平台 FAILED 映射为 source_missing，并保留失败 reason/history；没有独立的 failed 数值子集模型。

返回保留 `expected_gain`、`uncertainty`、`support`、`parent_support`、`cost_vector`、`estimator_version`、`fallback_reason`、`gate_reason`。`net_gain` 在估计阶段为 null，由策略使用已冻结 lambda 计算。`uncertainty/standard_error` 是按周聚类的均值标准误；`lower_gain` 为均值的描述性下界。`predictive_lower_gain/predictive_upper_gain` 是更晚内层校准段的单场增益区间，独立于均值门控；两者均无严格覆盖保证。

使用保守策略时同步设置 `risk_level='conservative'`、`conservative=True`，并取该 fold 的 `risk-conservative/0.05` 验证 lambda。二步入口 `select_two_step_v2(state, actions, predictor=predictor.predict_partial, estimator=estimator, budget=remaining_budget, lambda_cost=..., min_support=12)` 使用对应 `two-step/0.05` 的 lambda。二步最多一个后续动作，保留当前预测做 mask 近似，未对未来返回分布积分；允许净值为负的可行第一步由组合价值启动。

正式版本为 `risk-aware-conditional-value-v2:<hash>` / `two-step-feasible-mask-v2`；全部模型参数、训练/校准 ID 与时点在各 fold 工件中。部分预测器仍是 8 个子集的正则化模型与冻结空集先验，缺值预处理只在对应训练窗口拟合。组版本与摘要在 `visible_lineage`；模型版本保存训练内容 hash。完整特征、赛果和原始路径留在离线监督/执行器之外，不跨入这些推理参数。

质量、陈旧、依赖的合法性机制不等于已学得来源可靠性。当前历史实验主要是固定 Understat 来源与成本目录，缺乏用于估计动态质量效果的变化。跨模式、早于训练截止的调用会被拒绝；新增来源/成本/动作需重新验证。Agent 已有 `calibrated-risk-v2` 后端配置，见 `var/parallel/agent/round5/integration-v2/acceptance.json`。训练和审计命令见 [resume-audit.md](resume-audit.md)。
