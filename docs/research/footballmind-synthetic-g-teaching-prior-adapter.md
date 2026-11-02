# `synthetic-g-teaching-v1` 先验适配器

`football_agent.synthetic_g_teaching_prior.SyntheticGTeachingPriorAdapter`
读取
`var/submission/p9-g-synthetic-teaching-20261010/model.json`，只提供候选动作的
诊断排序分数。它不改变现有的经验 g，也不把合成结果写入
`expected_signed_gain`、`net_voi` 或选择决定。

候选动作可以在已有的 `synthetic_prior_features` 字段中声明
`footballmind-g-teaching-features-v1`：

```json
{
  "schema": "footballmind-g-teaching-features-v1",
  "action": "lineup",
  "scenario": "rotation_sensitive",
  "entropy": 0.897,
  "max_probability": 0.535,
  "time_to_kickoff_hours": 24.0,
  "source_reliability": 0.94,
  "event_relevance": 0.96,
  "tactical_sensitivity": 0.50,
  "cost_units": 0.018
}
```

声明必须在请求前完整出现。适配器不会从 action id、URL、标题、摘要或
返回正文猜测 action、scenario、相关性或成本；缺失或非法字段返回
`status=UNAVAILABLE`。已有 `g_prior_features` 别名和候选菜单辅助函数仍然
可用，运行时会按 schema 选择旧 18 行 bridge 向量或这个新向量，禁止混用。

`prior()` 返回 `support_status=synthetic_teaching_only`、
`evidence_origin=constructed_mechanism`、`production_ready=false`，并把
`expected_signed_gain`/`expected_net_voi`复制到 `prior_score`/
`prior_net_value`。`attach()` 把结果放在 `value.synthetic_prior`；经验值的
原字段原样保留。只有经验估计明确给出 `support_status=supported` 且 gain/net
均为有限数时，`selection_allowed` 才为真；即便如此，动作选择仍由经验 g
策略执行。`rank_diagnostic()` 只产生按合成先验排序的审计结果，不能授权请求。

底层 `SyntheticGModel.estimate()` 也保持同一语义：它输出的是合成机制的
`expected_signed_gain`/`expected_net_voi`，同时固定返回
`support_status=synthetic_teaching_only`、`synthetic=true`、
`calibration_status=not_established` 和 `production_ready=false`。适配器不会
把这些字段翻译成 empirical `supported`，也不会生成 propensity 或
realized gain。

总控还保存了一份同一构造菜单的双契约诊断：
`var/submission/p9-g-training-bridge-20261010/g-prior-dual-ranking-diagnostic.json`。
它并列记录旧 bridge 向量和 `synthetic-g-teaching-v1` 向量的排序，同时把
经验层固定为 `UNKNOWN`、`selection_allowed=false`。这份文件用于检查两个
先验契约确实可区分、且都不会绕过经验 g 门禁，不是足球比赛效果评估。

带适配器的 context/historical 运行会在运行目录保存
`synthetic-prior-model.json` 及其 SHA-256，并在 `configuration.json` 绑定
适配器 schema、模型 schema 和原始来源路径。离线 replay 会验证该绑定后重新
装载适配器，避免依赖后来被替换的外部文件。

该模型来自 48 个合成比赛、192 行机制教学数据，holdout MSE
`0.0010337241`；这些数字只描述合成演示，不能解释为实测足球 VoI、校准、
因果收益或真钱成本。真实 Leeds–Manchester United 单场回放和同场 3 正/3
负的转移仍属于 `historical_reconstruction`，不满足 prospective g 门禁。

来源分离核对记录见
`var/submission/p9-g-training-bridge-20261010/g-provenance-separation-validation.json`。
Leeds–Manchester United 原记录的发布时间为 `2026-01-03T12:10:00+00:00`，
正文采集时间为 `2026-10-07T16:57:07.846521+00:00`，`available_at=null`。
其 `+0.4647850245` 是冻结 f 的历史回放 log-loss 差值，可以用于明确标注
历史版本假设的回溯分析，但不能当成赛前真实采集的前瞻标签，也不能单凭
一个 realized gain 偏离均值就证明条件化 g 优于常数 g。

旧 `g_teacher_prior` 与新 teaching adapter 互相拒绝对方的模型 schema。
临时副本测试还核对：只改 bridge 外层 `one_match_probe` 会改变
`source_sha256`，不会改变 `model_sha256` 或 `load_prior()` 抽取的模型。
这次核对不合并数据集、不重训模型；UNKNOWN 与 supported 两条路径均通过。
