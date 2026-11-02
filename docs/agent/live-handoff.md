# Live 决策交接与恢复

live 刷新成功后，父运行保留原 cutoff、证据和预测。Agent 先校验新 cutoff 严格晚于原时点且早于开球，原始观测引用是非空字符串列表，并再次确认模型、特征读取配置与策略未变化。刷新期间配置发生变化时明确暂缓，已发生的请求、费用和原始引用仍进入审计。

交接分为三个可持久检查的步骤：

1. 父运行保存 `live_refresh_archived`，同时保存完整 `live_handoff.child_snapshot`、其哈希和固定 child ID。该快照冻结父模型/配置、原预算上限与累计消耗、旧预测和证据引用。
2. 以同一 child ID 创建子运行的 `initialized`。若记录已经存在，则校验首事件快照哈希并保留已有事件与计数；不覆盖运行。
3. 父运行保存 `STOPPED / LIVE_CUTOFF_ADVANCED`，`live_handoff.status=committed`。子运行随后获取新时点证据并预测。

父运行读取或执行时发现交接中断，会按同一 intent 补齐第 2、3 步，记录 `live_handoff_recovered`。恢复不调用刷新或模型工具。子运行先被读取时也会检查并完成父交接；恢复过程中再次退出，下一次继续使用原 child ID。没有产生第二个子运行或额外费用。

父与子操作使用现有 OS 锁。取得另一运行锁时采用非阻塞获取，竞争返回 `RUN_BUSY`，避免父持锁等子、子持锁等父的循环等待。子运行执行前验证父提交、比赛/模式/模型/策略身份、根运行 ID、预算继承和累计消耗，配置漂移返回明确暂缓。

`run.refresh_results` 和 `report.refresh_results` 直接显示每次刷新状态、错误、成本、来源请求数、延迟、输入输出引用、`observation_refs`、`decision_cutoff`、`content_changed` 和平台元信息。摘要与 `tool_returned` 在同一次日志存储中保存。固定历史 cutoff 排除新证据时，这些新原始观测引用仍可查看；超时、429、失败及来源报告实际成本/请求超限时也保留实际结果。

历史重建仅在明示 `basis` 和 `revision_risk` 后允许晚于预测 cutoff 的实际采集时间；`source_published_at` 与 `expires_at` 仍分别检查，未来公布或已过期的信息不进入预测。

验证入口：

```powershell
python -m pytest tests/agent/test_round2_live_review.py tests/agent/test_round2_live.py tests/agent/test_workflow.py -q
```

独立审查覆盖预算全部继承、请求上限、未变内容概率一致、刷新期间配置漂移、原始引用形状、3 个真实子进程退出位置、子运行先读、恢复时再次退出、父提交期间子运行争用、旧运行 ID 覆盖保护、刷新摘要以及重建发布时间/过期检查。构造工具均明确标记测试模式，没有网络请求。持久验收位于 `var/parallel/agent/round2/handoff-review/`。
