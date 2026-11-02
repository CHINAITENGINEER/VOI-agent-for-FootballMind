# Agent 第五轮

2026-09-26：`ROUND5_CALIBRATED_RISK_BRIDGE_AND_AUDIT_VERIFIED`。

新增合法参数比例、含重复/非法提案的调用比例、正数值价值动作 recall@1/3/5、候选遗漏和停止审计。Recall 参照数值估计器，未使用赛果或隐藏候选返回。纯数值策略的提案召回汇总为 null；缺 LLM 配置的运行保持零 LLM 调用。构造 proposer 的证据等级由环境和 provider 字段共同表明，不作为真实模型效果。

`ResearchServices.load` 支持配置 `value_estimator_kind=calibrated-risk-v2`，旧 empirical-v1 配置继续支持。新模型实际接线保留 dependency/cache 元数据，分别输出均值区间和单场预测区间。前者驱动保守门控，后者用于离线诊断。没有本地支持的回退值不伪装成受支持动作。

`football_agent.acquisition_evaluation` 汇总已存运行并检查审计链，按环境/策略/risk_mode 分组。真实费用没有观测时保持 null；预算预留与模拟费用不替代真实支出。运行态 stop_regret 保持 null，离线 continuation 计算在独立函数中进行；该函数只接受评估者数据，结果不进入 policy state。

正式验证：`var/parallel/agent/round5/integration-v2/acceptance.json`，七个时间折各选首场，mean/conservative 共14条冻结历史轨迹。与研究侧动作、概率、模拟费用一致，配置哈希和事件链可复核。它是桥接一致性验证，不是14场新数据。

启动审计汇总：

```powershell
python -m football_agent.acquisition_evaluation var/parallel/agent/round5/integration-v2/runs --output var/parallel/agent/round5/evaluation-rerun.json
```

测试包含 malformed proposal、非法参数、重复提案、显式 stop 仍校验提案、无 LLM 配置、负增量成本拒绝。最终跨 research/platform/agent/demo 回归共556项通过，明细见 `var/parallel/round5/final-tests.xml`。原有 UI 未重做，本轮复用既有 HTTP 回归；无本轮新浏览器截图或真实 LLM 效果声明。

整体结果、负结果和下一步研究瓶颈见 `docs/parallel/round5/README.md`。
