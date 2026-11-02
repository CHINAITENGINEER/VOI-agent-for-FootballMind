# 逐动作 Episode 持久化

`football_agent.episode_persistence.EpisodePersistence` 为每一次 Agent 选择
保存一条不可变 JSONL 记录，文件位于运行归档的 `episode-actions.jsonl`。
记录包含 `prediction_before`、`prediction_after`、`state_before_digest`、
`state_after_digest`、`prediction_cutoff`，以及选择、采集和准入的 digest
绑定。

预测字段可以显式为 `null`。这表示当前未知，系统不会用均匀分布、零或上
一次结果填充。`validate()` 会把缺字段的 episode 标为 `PARTIAL`，并令
`eligible_for_replay=false`；这不等同于失败，也不能升级为数值 VoI 样本。

日志采用记录 digest 和前一条 digest 的链式校验。修改正文、digest、顺序，
或者把 acquisition/selection 绑定到另一动作，都会在读取时拒绝。评估器的
赛果、损失和未来结果字段禁止写入 actor-side journal，独立结果仍须走隔离的
evaluator sidecar。

现有 `action_value_ledger.build_action_value_ledger()` 在检测到该可选日志时
会携带 `episode_persistence`、状态和 replay eligibility；没有该日志的旧
运行保持原有 ledger 结构。

状态绑定 runtime 可以通过显式 `prediction_state_projector` 和
`prediction_adapter` 接入真实后验。接入规则、before/after 的生成时机以及
迟到信息的处理见[Runtime 后验绑定](prediction-runtime-binding.md)。

