# Runtime 后验绑定

`autonomous_runtime_loop` 负责在开放信息环境中选择下一步动作，
`StateBoundContextValueEstimator` 负责估计候选信息的价值，二者都不应
伪造比赛结果概率。需要真实后验时，调用方通过两个可选回调接入预测器：

```python
run_information_loop(
    ...,
    prediction_adapter=services.predict,
    prediction_state_projector=projector,
)
```

`prediction_state_projector(actor_context, state_bundle)` 先把当前 actor-visible
状态显式投影为预测器能够理解的公开状态；runtime 不会把内部 `state_bundle`、
receipt 或 evaluator sidecar 自动传给预测器。`prediction_adapter(visible_state)`
必须返回包含 `match_id`、`prediction_cutoff` 和三分类 `probabilities` 的结果。
概率顺序固定为：

```text
[home_win, draw, away_win]
```

runtime 在 Agent 选择动作前生成 `prediction_before`。只有当 receipt 已完成、
证据通过公开状态准入、replan 生成新 actor state 且成本已结算时，才生成
`prediction_after`。这些字段放在 cycle 顶层，再由 `EpisodePersistence` 写入
journal，因此不会修改已经生成的 selection trace 或破坏 `trace_digest`。

适配器强制检查：

- 三个概率有限、非负且总和为 1；
- `match_id` 与当前比赛一致；
- `prediction_cutoff` 与 actor cutoff 精确一致；
- 如果返回 `state_digest`，它必须与当前 actor state 一致；
- before/after 必须使用同一比赛、同一 cutoff 和同一类别顺序。

以下情况只保留 `prediction_before`，`prediction_after` 设为 `null`：

- provider 异常或响应无法规范化；
- receipt 被拒绝、超时或晚于 cutoff；
- 公开状态 replan 失败；
- 预测器返回非法概率或不同 cutoff。

这类 episode 仍可审计，但 `EpisodePersistence.validate()` 会保持
`PARTIAL`，不会使用零向量、均匀分布或最终赛果补齐未知后验。

预测器返回的模型版本、数据集版本、artifact 和可见状态 lineage 只作为
runtime 的 prediction metadata 保存，不能替代 probability vector，也不能
把比赛结果、loss 或 evaluator 字段写入 actor-side journal。

该接线使一次自主信息动作具有可复盘的边界：

```text
actor-visible state
  -> prediction_before
  -> Agent/VoI 选择信息
  -> receipt 准入和公开状态更新
  -> prediction_after
  -> Episode Journal
```

`prediction_before` 和 `prediction_after` 描述 Agent 的认知变化；它们不是
VoI 数值本身。VoI 仍需结合预期决策损失、获取成本、延迟和信息可靠性，且
在没有独立结果与校准数据时不得宣称数值训练资格。
