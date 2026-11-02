# Round201：无市场先验的官方 B → 查询收益 → g → 独立 C 闭环

日期：2026-10-07。主交付是让“用户给比赛，Agent自行选择下一步信息”在缺市场先验时也能经过实际预测、收益学习和执行回放；没有增加前端，也没有启动新的真实采集。

## 本轮解决的主问题

Round199/200 已有可选先验预测器和实际正文，但旧的 B/C 登记、运行及收益输入仍以市场先验为前提。仅允许模型加载 `None` 不能解决实际调用、训练收益和回放之间的断点。本轮将这些部分接在同一条正式执行链上。

1. `official_context_registration.py` / `official_context_collection.py` 增加官方身份下的 B/C 登记与公开采集入口。原档核验身份；模型实际拟合必须早于登记、登记早于执行；A、B、C 按物理比赛及别名隔离。B 使用冻结 f，C 使用同一个 f 及从独立 B 拟合的 g。声明一次执行后，换输出目录不能重新取得预算。
2. `OfficialContextVOIPolicy` 及显式 v6 runtime 在查询前运行实际 f，在实际返回时刻更新 f，g 按读前可见状态与候选标签估计有符号收益。候选仍由实际读到的父页动态发现，不提前读取待查正文，不人工安排“先伤病后天气”的主题权重。
3. `official_context_gain.py` 只从正式 B 原档、原模型及独立赛果重建收益。先完整零网络回放，再提取查询前和即时返回后的状态；不采用下一次决策前等待引起的状态老化作为本次查询收益。
4. `optional_context_query.py` / `optional_context_value.py` 将原始缺失先验、f 的经验基准、域支持和实际查询成本分开编码。复用旧数值内核，并保留上下文与候选的交互表示、独立物理样本和局部支持检查。旧模型固定时钟拟合字节保持相同。
5. 后台分析报告明确初始概率来自训练期独立结果分布，展示实际查询的条件收益与停止原因。原始市场先验仍为 null，没有把训练基准伪装为市场观测。

收益目标为同一个冻结预测器的实际 log-loss 改善：

`gain = log f(y | actual_after_return) - log f(y | actual_before_query)`。

这是一条有标签轨迹的实现收益，不是该查询的已知期望价值、因果效应或所有未查询信息的反事实。g 从独立 B 学习其条件期望；C 才能用于独立评估。支持不足返回 UNKNOWN；只有模型支持的实际零收益才保留零目标，负收益不会被截断。

## 已实际执行的构造机制

永久产物：`var/parallel/agent/round201/constructed-chain/`。所有来源、比赛和未来时钟均在 `scope.json` 标记为 `constructed_mechanism`，真实网络请求为 0。没有替换 f/g 的预测函数或人工填入 g 输出。

| 步骤 | 实际执行结果 |
|---|---|
| A | 12 场构造独立比赛、三类结果，实际拟合 f；缺先验基准为 0.50 / 0.25 / 0.25 |
| B | 12 个不同原生比赛身份、2 个开赛日期，正式注册后各实际执行 4 个构造来源返回 |
| B 赛果与收益 | 经独立官方赛程和完场原档格式核验，48 个实际查询前后对；本次永久产物为 13 正、23 零、12 负 |
| g | 从这 48 对实际派生目标拟合，绑定原 f 字节、物理 B 身份和实际拟合时刻 |
| C | 另一个独立身份正式登记；1 次 UNKNOWN 探索、3 次按计算正净收益选取；实际 f 更新 |
| 回放 | 12 条 B 在收益生成时完成原档精确回放，C 的 4 个返回也精确回放；独立验收重新计算收益和拟合 g |

这个样例的四项损失成本权重均预先设为 0；请求和预算上限仍执行。C 最终因 **physical_budget_exhausted** 停止，没有停止充分性证书。它证明正式代码组合后可执行，**不证明真实预测改善、成本收益优势、估值校准、合理停止或开放环境泛化**。C 概率的微小变化也不能当作预测质量提升。

## 独立检查与兼容性

除完整链路外，独立测试覆盖：缺结果不生成零目标；手填结果 JSON 不代替官方原档；更换合法 f、篡改实际预测或原始响应被拒；已核验赛果但缺域支持的目标仍排除；C 不得回填 B；重复 bundle 不增加独立比赛数量。执行测试覆盖失败费用、预算超支、未来正文、晚返回、倒退时钟和缺支持时的 UNKNOWN。

相关测试去重共 363 项通过。旧 20 条真实运行、16 场物理比赛、67 次返回用当前代码完成零网络精确回放。旧 1,832 份冻结文件和 119 份源代码快照保持原字节。另从保存原档重新生成 48 个目标，与原文件完全一致；固定原拟合时刻重训 g，保存文件逐字一致。源码和产物 SHA 见本轮 `acceptance.json`，不以测试数量代替研究有效性。

只读复验：

```powershell
python var/parallel/agent/round201/verify_round201.py
```

重新生成一份构造机制须使用全新目录，不能覆盖原产物：

```powershell
python var/parallel/agent/round201/run_constructed_chain.py --output var/parallel/agent/round201/constructed-chain-rerun
```

研究入口依次为：

```python
prepare_official_context_registration(..., purpose='calibration', predictor_path=f_path)
collect_official_context(registration_path, output=run_directory)
build_official_context_gain_from_runs(run_directories, result_archives, gain_directory,
                                     predictor_path=f_path, as_of=audit_time)
g = OptionalContextValueModel().fit([gain_directory], predictor_path=f_path,
                                  as_of=training_cutoff, dataset_version=version)
prepare_official_context_registration(..., purpose='confirmation',
                                      predictor_path=f_path, value_model_path=g_path)
collect_official_context(confirmation_registration, output=confirmation_run)
```

省略号仅表示入口文档摘要；完整必填设置、真实拟合和运行示例已在构造脚本及测试中执行。当前交付为正式研究 API，尚未自动把生产队列的证据采集任务转换为 B/C；真实模型未满足训练条件时不允许自动切换。

## 真实状态与下一最高收益工作

真实数据不变：20 run / 16 物理比赛 / 67 实际返回；6 个独立赛果（0 主胜、2 平、4 客胜）、35 个有标签前缀、76 个待标签前缀。真实 f 仍 `not_ready`，A16/B0/C0，真实 g 未拟合。本轮没有新 HTTP、没有新标签。

工程链路接通后，主要瓶颈转为真实、独立、可用的文本与赛果，而不是再增加 Agent 角色或接口。下一工作按原计划执行：

1. 到期核验已登记比赛的原始结果，并重新运行正式训练就绪检查。最早既定时点为北京时间 10 月 8 日 02:10；不提前轮询、不按时间自动制造赛果。
2. 为缺先验域补充预先限定、尚未曝光的连续比赛样本及比赛相关正文，记录重复文章、实际可见时刻和失败。该域目前只有 6 场有实际文本、3 场有实际正文，仍需至少 12 场独立已结算文本比赛及三类覆盖；这是支持下界，不是效果标准。不反复拉长时间窗口凑样本。
3. A 数据足够后在开发留出上检验 f 的增量与概率质量，再冻结 f；只有更晚新比赛可成为 B。从真实 B 学习 g，然后冻结配置进入独立 C。已有 A 轨迹不能因新接口完成而改称 B/C。
4. C 的主要目标是同一 f、初态与预算下的信息选择质量与成本，而非足球胜率刷分。明确区分“不查、固定顺序、随机、学习型选择”；没有共同响应面板就不能编造未查询返回。若无优势，报告负结果和适用边界。
