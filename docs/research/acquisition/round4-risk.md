# 第四轮：风险感知条件信息价值

> 2026-09-26 复核更正：下文保留为第四轮历史记录，不再作为当前算法验收结论。v1 用单场残差校准均值区间，混淆了两种不确定性；v1 二步选择要求第一步先为正，未覆盖“第一步不值、联合值得”。旧验证器验证哈希/概率/轨迹边界，未实际验证预算，原“概率/预算检查”表述过强。已就绪的 `risk_calibrated.py` / `lookahead_calibrated.py` 接续修复及逐决策复验见 [resume-audit.md](resume-audit.md)。新的验收文件仍写 round4，既有 round4/round5 冻结实验原样保存。

第四轮在第三轮冻结产物上解释 learned-net-gain-v1 的失败，并实现风险感知条件增益和有限二步前瞻。正式目录为 `var/parallel/baselines/round4/risk-aware-v4/`。第三轮 `active-information-v3`、账本和环境报告保持原样。

## 运行命令

```powershell
python -X utf8 -m football_research.acquisition round4 `
  --platform-export var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda `
  --round3 var/parallel/baselines/round3/active-information-v3 `
  --output var/parallel/baselines/round4/reproduce-new

python -X utf8 -m football_research.acquisition.verify_round4 `
  --run var/parallel/baselines/round4/risk-aware-v4 `
  --output var/parallel/baselines/round4/final-verification.json

python -m pytest tests/research/baselines tests/research/evaluation tests/research/acquisition -q `
  --junitxml=var/parallel/baselines/round4/tests.xml
```

`round4`读取第三轮的真实平台导出和 `oof-supervision.jsonl`，每个外层 fold 只使用训练截止时间之前的 OOF 样本。当前正式运行使用 380 场、290 场测试比赛、7 个时间 fold、16,240 个 episode、10 种策略。2025/26 的 290 场已在第三轮看过，仍属于探索性历史重建，不是独立确认集。

## 估计器

第三轮点估计主要按 mask/action 收缩。它的增益符号准确率为 49.14%，learned 策略比 all-available 的 log loss 高 0.015403。这支持进一步检查支持度、上下文辨识和停止校准；单场 Delta 的符号本身带噪声，这两项数字并不单独证明条件均值估计只在拟合噪声。

第四轮 `RiskAwareValueEstimator` 位于 `football_research/acquisition/risk.py`。训练监督仍是向前 OOF 的

```text
Delta(H,a) = log_loss(p(H), y) - log_loss(p(H ∪ result(a)), y)
```

运行时输入只有 VisibleState、mask、当前预测、ActionSpec 元数据和可见质量/时效状态。未揭示结果和标签不进入 `estimate_value`。估计器输出 `expected_gain`、`uncertainty`、`support`、`lower_gain`、`cost_vector`、`estimator_version`、`fallback_reason` 和 `quality_context`。

条件上下文保留并记录：

- visible mask 和 action group；
- 当前概率置信区间；
- `observed/source_missing/stale` 状态；
- 已知 `available_at` 的新鲜度分箱；
- `dependency_key`、`cache_policy` 和 cost scope；
- 未提供的质量参数保持 `unknown`，没有硬编码可靠性真值。

均值使用按比赛去重的局部均值向 action/confidence 父级收缩，残差标准误作为不确定性。内部训练尾段作为校准段选择风险乘子；本次七个 fold 乘子为 5.7281、6.6927、12.0201、14.2668、13.9464、7.9039、7.9812。乘子偏大本身是小样本风险的结果，报告没有把它包装成严格覆盖保证。

`risk_level='mean'` 使用条件均值，`risk_level='conservative'` 使用 `expected_gain - multiplier * uncertainty`。负增益保留。支持不足、无动作能力、未知成本、过期、source_missing 和预算耗尽分别返回停止原因。

## 有限二步前瞻

`football_research/acquisition/lookahead.py` 的 `select_two_step` 只对最多两个动作做 mask-transition lookahead。第一步的后续状态只把相关组标为 projected/observed，所有返回值仍为 null，预测器重新从公开状态计算；它不读取第一步真实返回或目标标签。实际获取后仍由正常可见状态重新估值。

这是一阶贪心局限的受控检验，不是强化学习或全局最优求解。二步计算时间单独计入 estimator latency，预算仍按原始 money/requests/latency_ms/tokens 分量约束。

## 严格策略对照

策略共享相同比赛、训练模型、初始状态、ActionSpec、模拟成本、预算和环境随机种子：

`base-only`、`all-available`、`random`、`fixed-order`、`entropy`、`disagreement`、第三轮 `learned-net-gain-v1`、`risk-mean`、`risk-conservative`、`two-step`。

预算 0.05、共同 290 场结果如下：

| 策略 | Log loss | Brier | 准确率 | 模拟费 | 平均请求 | 主要停止 |
|---|---:|---:|---:|---:|---:|---|
| base-only | 1.108492 | 0.673332 | 40.34% | 0 | 0 | base_only |
| all-available | 1.087116 | 0.650720 | 46.21% | .050000 | 2.000 | no_legal_action |
| random | 1.087116 | 0.650720 | 46.21% | .050000 | 2.000 | no_legal_action |
| fixed-order | 1.087116 | 0.650720 | 46.21% | .050000 | 2.000 | no_legal_action |
| entropy | 1.083401 | 0.650432 | 45.52% | .034483 | 1.483 | entropy threshold / no action |
| disagreement | 1.108492 | 0.673332 | 40.34% | 0 | 0 | disagreement threshold |
| learned-net-gain-v1 | 1.102519 | 0.667196 | 43.10% | .016379 | .628 | nonpositive / no action |
| risk-mean | 1.108492 | 0.673332 | 40.34% | 0 | 0 | insufficient support / nonpositive |
| risk-conservative | 1.108492 | 0.673332 | 40.34% | 0 | 0 | insufficient support / nonpositive |
| two-step | 1.108492 | 0.673332 | 40.34% | 0 | 0 | nonpositive two-step |

risk-mean 和 risk-conservative 在当前 fold 支持度下均选择停止。它们没有改善质量，但明确暴露了不确定性导致的保守退避；这比把弱支持动作伪装为可靠正价值更符合当前证据。二步策略在这组真实历史重建中没有找到可执行的正联合价值，不能据此推断真实场景不存在互补信息。

风险估计诊断覆盖 1,160 条转移、290 场：

- 总体符号准确率 48.45%；第三轮为 49.14%；
- 预计最佳动作与实际最佳动作一致率 47.93%；
- 实际负 Delta 比例 43.19%；
- 支持 `<12` 的 237 条转移符号准确率 51.48%；
- 支持 `>=12` 的 923 条转移符号准确率 47.67%。

估计分箱、支持度分层、失败案例和停止原因写入 `report.json`、`risk-diagnostics.jsonl`。图表为 `risk-cost-quality-frontier.png` 和 `risk-gain-calibration.png`。

按 27 个自然周进行共同比赛配对 bootstrap，risk-mean、risk-conservative 和 two-step 与 base 的损失差均为 0，因为它们在测试段均停止；all、entropy、第三轮 v1 的区间沿用同一环境重新计算，仍跨 0。第四轮没有把停止本身报告成预测改善。

## 机制和审计

新增测试覆盖：

- 风险均值/保守下界和负增益保留；
- OOF 时序、标签隔离和模型 hash；
- 来源质量、时效和 dependency/cache 元数据投影；
- 隐藏标签和未揭示字段不改变获取前估值；
- source_missing 不被当成 supported action；
- 二步状态只做 mask transition，不注入返回值；
- 保存模型重载和内容篡改检测。

`final-verification.json` 通过 14 个模型加载、16,240 个概率/预算检查、25,881 个可见轨迹事件和 9,641 次动作审计。第三轮 `data.py` hash 与冻结产物一致，没有修改前三轮模型或账本。

## 结论和后续

第四轮没有证明风险感知估计器在当前单赛季探索数据上优于第三轮或启发式。它证明了风险、支持不足、质量/新鲜度/依赖字段和二步限制已经进入可审计 API，并且弱支持状态会明确停止。下一轮应先预注册未看时间窗口或独立赛季，再重新校准风险覆盖；新增历史赛季只能说明迁移表现，不能把已看 290 场变成确认集。真实在线 API 费用、来源质量校准和 LLM 真实调用仍须独立计量。
