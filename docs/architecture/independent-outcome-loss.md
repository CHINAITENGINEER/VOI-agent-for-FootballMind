# 独立赛果与逐动作损失 sidecar

`football_agent.independent_outcome_loss_registry` 负责在 Actor 运行之外保存
赛果和事后损失。它不把赛果写回 Actor 状态，也不允许用 Actor 使用过的来源
直接充当金标准。

## 接受条件

- sidecar 必须绑定 `episode_id`、`target_key` 和 `prediction_cutoff`；
- 结果来源、来源 ID 和内容摘要必须与 Actor 来源集合不同；
- `available_at` 必须晚于预测截止时间；
- `observed_at` 不得晚于 `available_at`，开球时间不得晚于观察时间；
- 标签必须有显式的 `label_mapping`；
- 记录带有内容摘要，篡改后拒绝计算损失；
- Actor binding 改变后不能复用原 sidecar。

逐动作损失只有在动作身份、请求身份、捕获时间、前后概率和标签映射都完整时
才计算。支持 `log_loss` 和 `brier`。任一动作缺证据时，注册表保持
`PARTIAL` 或 `UNKNOWN`，`numeric_voi_training_eligible=false`。

## 隔离规则

sidecar 的 public projection 只保留 episode、目标和截止时间等绑定信息，并明确
`outcome_visible_to_actor=false`、`loss_visible_to_actor=false`。比分、标签、结果
来源正文、损失和结果序列不能进入 Actor 的上下文或公共动作轨迹。

## 与 paired replay 的连接

`project_independent_registry_sidecar()` 是显式适配器。它只复制 sidecar 中已
存在且可校验的序列损失和动作证据；不会从赛果时间、最终损失或其他策略轨迹猜
测缺失的 `prediction_before`、`prediction_after`、`available_at` 或 `captured_at`。
适配失败时返回 `PARTIAL/UNKNOWN`，paired replay 保持不可比较。

