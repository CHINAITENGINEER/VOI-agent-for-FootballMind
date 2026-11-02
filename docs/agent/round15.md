# Agent 第十五轮：有限信息环境中的自主查询机制

冻结设计为 `docs/research/acquisition/round15-design.md`，SHA256 `aa70d9bd6ac92c04722935cdf021d29deaaa811e6f496f5d40fd420aab37a64b`。本轮区分两个问题：给定正确的条件概率，现有执行链是否真的会为后续查询机会购买信息、依返回选择不同工具并及时停止；学习得到的足球模型是否正确估计真实信息价值。这里直接检验前者，没有新增足球比赛、数据采集、拟合或 LLM 调用。

## 复用规划器和持久执行器

新增 `football_agent/finite_information_bridge.py`，复用冻结的 `JointRiskPlanner` 和 `AcquisitionWorkflow`。`FiniteInformationWorkflow` 只声明 finite/mean/numeric 模式及默认情景预算，不复制规划或执行算法。计划采用最多两步和固定 lambda=1；期望累计信息收益、期望累计费用与当前一次实际收费分别记录，选择分数直接使用 planner 的 12 位舍入结果。

`FiniteObservationEnvironment(categories=..., returns=...)` 只接收当前完整返回组合，不接收当前真实类别或 world ID。完整组合封存在环境对象中。`FiniteInformationBackend` 接收环境对象、已知总体概率模型与公共配置；它从来不读取完整组合，只在动作已经被选择后请求环境揭示该动作的一个类别。

实际交给模型的状态严格只有：

```json
{
  "observations": {"context": "0"},
  "action_history": [{"action_id": "context", "status": "observed"}]
}
```

为了兼容旧执行器，公开传输包把已获取类别放在 `groups[action_id].values.category`，并用 Boolean mask 标记。使用 `category` 是因为旧审计器会把 `token` 命名字段视为认证秘密而脱敏；旧脱敏器保持原样。模型边界重新投影到上述两键状态，不携带比赛身份、隐藏组合、实际类别或当前世界索引。

环境维护真正已经揭示的观测账本；每次预测、估值和获取前，backend 核对公开观测与该账本相符。伪造已经购买的类别、改变历史、异常类别、预算超过规格、实际消耗与账本不符均报错。每个动作只揭示一次，request ID 也不可重用。模型拒绝实际入口中的假设 envelope；只有规划器的显式数学接口可使用它。

## 配置、标识与类别语义

`make_finite_backend(config_path, environment)` 校验 model、scenario spec 和配置文件 SHA256。模型分布还必须属于该绑定规格声明的模型变体，动作目录、费用、请求上限和 lambda 必须一致。公开 case ID 使用随机 UUID，不编码类别、返回组合或 world ID；scenario ID 是允许公开的任务背景。

旧执行器内部概率传输字段仍为 home/draw/away，配置明确把它们映射为合成类别 0/1/2，并声明不代表足球赛果。新卡片将概率统一重标为 `"0"/"1"/"2"`，避免把有限变量解释成真实赛事预测。

模型拥有已知总体联合分布，包括所有可能类别和观测之间的关系；这与知道当前实例的真实类别不同。规格及模型文件绑定的是总体，当前世界的完整返回只在环境中，真实类别只在研究评分器与离线核验端。

## 五类固定机制

| 场景 | 正概率返回组合 | 正确模型下要验证的机制 |
| --- | ---: | --- |
| gated_stop | 6 | 先查 context，返回 0 再查 signal_a，返回 1 则停止 |
| gated_route | 18 | 先查同一个 context，依其内容选择 signal_a 或 signal_b |
| complementary | 9 | 单条信息不改变类别分布，两条合起来值得查询 |
| redundant | 3 | 查询较便宜的 signal_a 后，完全重复的 signal_b 不再值得购买 |
| irrelevant | 4 | 返回信息准确但与目标无关，初始即停止 |

总计 40 种组合全部覆盖，数量不是独立足球样本数。当前执行预算最多两次查询；期望费用从数学分支积分计算，每次真实收据仅扣当前动作费。一个 context 查询即使即时类别信息量为零、即时净值为负，也可以因后续条件选择价值而被执行。

gated_stop 同时涉及观察内容后的停止，固定工具顺序加动态 STOP 也可达到同样最优值；gated_route 才单独展示“下一工具身份随内容改变”的作用。互补场景展示规划深度，不混同为依内容换工具的优势。

## 40 组合的实际验收

`football_agent/verify_round15_finite.py` 从正式绑定的情景规格取得全部正质量返回组合，按观测字典排序，既不按真实类别也不按是否查询或收益筛选。对同一组合的所有研究世界，先核对其查询轨迹、概率和费用相同，然后仅执行一次构造工作流。

完整组合及匹配研究 world IDs 只放在验收文件的 `offline_reference_only` 区域，供根独立核验查证；这些信息不在运行配置、预测输入、估值输入、获取参数或工具回包中。实际审计快照逐事件检查：尚未收到的类别不进入公开 groups；每次真实返回之后都出现对应的后验更新。

查询卡复用冻结事件 builder，记录当时实际概率与 `acquired_tokens`、全部合法候选的累计期望收益/费用/净值和数学分支、实际选择、类别收据及实际后验。数学分支保留 `hypothetical=true`，不会计入动作次数。各卡片通过事件序号/哈希关联运行文件；`RunStore.verify` 另行核验全部事件链和工具 blob。

```powershell
python -m pytest tests/agent/test_round15_finite_information_bridge.py --import-mode=importlib -q
python -m football_agent.verify_round15_finite --research var/parallel/baselines/round15/finite-information-v1 --output var/parallel/agent/round15/finite-information-bridge-v1
```

专项 27 项通过，覆盖不同内容下的工具路线/停止、互补与冗余、实际费用、持久运行重新打开、伪造证据、未知返回、零联合支持返回、实际/假设边界、模型/规格绑定、重复动作/请求和请求预算；与冻结第十四轮专项合并为 61 项通过。零联合支持测试故意提供规格外互相矛盾的重复观测，显式报错而非产生默认后验，不计入正式 40 组合。持久性检查在首次真实揭示后，用新的 executor 对象打开同一运行日志并继续，保留同一个环境的私有观测账本；这不是完整进程重启后的环境恢复承诺。

## 正式 40 工作流结果

正式研究工件位于 `var/parallel/baselines/round15/finite-information-v1`。Agent 工件位于 `var/parallel/agent/round15/finite-information-bridge-v1`，`acceptance.json` 状态为 `ROUND15_FINITE_INFORMATION_40_PATTERN_AGENT_VERIFIED`。

| 场景 | 实际工作流结果 |
| --- | --- |
| gated_stop | 3 条 context → signal_a → STOP，3 条 context → STOP |
| gated_route | 9 条 context → signal_a → STOP，9 条 context → signal_b → STOP |
| complementary | 9 条 signal_a → signal_b → STOP |
| redundant | 3 条 signal_a → STOP |
| irrelevant | 4 条初始 STOP |

40 条均通过研究轨迹逐项核对，共检查 152 个候选、418 个模型数学分支、106 次初始/实际返回后概率、66 个真实构造收据和 1,328 个审计事件。40 份卡片文档包含 116 张查询/终止卡。40 工作流的实际模拟费用合计为 `6.96`，这是运行费用之和，不是以总体概率质量加权的期望费用。

每个返回组合在对应全部真实类别上的研究轨迹均相同；离线核验未依据哪一类得到更好结果来挑选代表。实际工具参数、模型状态及公开快照只含已经取得的证据。所有动作顺序、真实类别返回、每次后验和费用与研究一致，模型/spec/config 哈希及事件/blob 审计通过，源代码哈希复读一致。

验收文件 SHA256 为 `243af52a77bc01d8fc518d5887466a5b72ebfdafb29fb827505cb22a5d1ed672`。新增独立足球比赛、真实外部请求和 LLM 调用均为 0。40 工作流与专项次数不加入足球效果指标；总体已完全枚举，机制性能使用规格真实质量的精确加权结果，不使用把 40 个组合等权当随机样本的 bootstrap 区间。

本轮是已知构造模型的执行机制验证，复用既有数值规划器，不作新算法优越性或真实足球效用提升声明。足球中的晋级、轮换、替补能力和战术只是问题动机，未被赋予这些人工概率。真实来源、学习分布失配和前瞻检验仍属于后续工作。
