# 同初态 paired policy replay

`football_agent.paired_policy_replay.compare_aligned_policies()` 在同一公开初态、
候选菜单、预测截止时间和预算下回放四个策略：

- `autonomous`：由 interval/VoI 选择器决定下一条信息或停止；
- `fixed_order`：固定顺序控制；
- `full_query`：获取声明菜单中的全部信息；
- `stop`：零查询控制，只能使用同一 Episode 显式保存的初始预测。

每个策略输出动作轨迹和按 step 对齐的状态摘要。控制策略不会读取
`interval_provider`，也不会读取 evaluator sidecar。只有逐动作前后预测、来源时
钟、独立结果、基线损失和执行序列损失全部存在时，Episode 才能进入 paired
comparison；缺失项保持 `None`，不以零值替代。

适配器 `project_independent_registry_sidecar()` 将独立损失注册表转换为 replay
所需的 evaluator sidecar。它要求结果摘要、注册表状态、每条损失记录和显式动作
证据一一绑定；`PARTIAL` 注册表不能升级为可比较样本。

这套回放只能验证信息选择机制和评估边界。当前合成夹具没有真实来源请求、实时
模型调用或真实比赛金标准，因此不能据此宣称数值 VoI 已校准或足球预测性能已
提升。

