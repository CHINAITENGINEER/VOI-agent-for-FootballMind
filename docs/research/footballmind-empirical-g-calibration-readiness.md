# Empirical g calibration readiness

审计产物：`var/submission/p9-g-training-bridge-20261010/empirical-g-calibration-readiness.json`

## 当前结论

当前 `g` 仍是 `fitted_research_prototype`，不能进入在线选择。模型的 B 集只有 5 个独立比赛和 46 条 transition，且模型记录的参考门槛是至少 12 个独立比赛、至少 2 个决策日期。当前选择结果是常数基线，校准状态仍为 `not_established`。

`p8-g-stream-v3` 的 62 条文档虽然有 62 条 publication timestamp 早于开赛，但这不等于 Agent 在开赛前看到了正文。实际可见性为 0 条 before-kickoff、62 条 unknown；discovery 为 0 条 before-kickoff、31 条 after-kickoff；attempt 为 0 条 before-kickoff、62 条 after-kickoff；context snapshot 为 0 条 before-kickoff、26 条 after-kickoff。原始 Agent 菜单也没有被证明。相对先前包没有新增独立比赛。

因此，历史重建只能用于回放、机制解释、泄漏检查和展示单场 before/after，不能生成 empirical `g` 标签。synthetic teaching prior 只用于诊断排序，不能产生 empirical gain、net VoI、propensity 或在线选择权限。

## 下一批数据的准入合同

每场比赛必须在 decision cutoff 前冻结完整候选菜单，记录所有候选，而不是只记录 chosen。每个候选至少保留：

- `physical_match_id`、`kickoff_at`、`decision_cutoff_at`、`outcome_observed_at`
- `action_menu_frozen_at`、`candidate_action_id`、`candidate_set_hash`
- `selection_probability`（或受控随机探索的等价记录）
- `query_requested_at`、`source_public_at_or_available_at`、`response_captured_at`
- `response_raw_sha256`、`prediction_before`、`prediction_after`
- `query_cost`、`realized_signed_gain`

晋升前必须同时满足：至少 12 个独立 physical matches、至少 2 个决策日期、所有候选在选择前记录、查询和结果时钟可证明、不得把赛后正文或菜单放入赛前状态、原始 payload 可按 digest 重放，并按比赛分组留出验证。只有 rolling/group-held-out 的 empirical `g` 在预注册指标上优于冻结常数基线，且校准和不确定性报告可复现，才允许生成新的 empirical model manifest。

在这些条件满足前，runtime 保持 `UNKNOWN`/constant gate；不再把历史重建、单场正负转移或 synthetic prior 混入训练池。
