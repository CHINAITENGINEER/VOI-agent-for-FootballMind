# FootballMind 单场开放信息实验：目的与开赛前检查清单

**版本：** v1.0  
**制定日期：** 2026-10-18（北京时间）  
**适用对象：** FootballMind 单场“自主选择信息—f 更新—g 评估”实验  
**核心原则：** 用户只提供比赛，Agent 自主决定需要查什么、是否继续查询以及何时停止。人可以审阅证据和运行结果，但不能在运行前替算法指定信息主题、查询顺序或价值结论。

## 一、我们为什么做这个实验

FootballMind 研究的不是“把更多足球数据塞进预测模型”，也不是单纯比较谁的胜平负准确率更高。实验要验证的是一个更基础的问题：

> 在信息开放、来源质量不一、信息不断出现且获取有成本的环境中，Agent 能否根据当前已知状态，主动选择下一条最值得获取的信息，并在获得信息后重新判断是否继续。

一场比赛只是验证环境。实验要观察完整决策链：

```text
比赛身份与初始状态
        ↓
当前概率与已知证据
        ↓
候选信息动作
        ↓
Agent/数学模块选择动作或停止
        ↓
获取正文或结构化来源
        ↓
f 将合法证据转化为 prediction_after
        ↓
重新生成候选并继续/停止
        ↓
赛果公布后计算 loss 与 realized_gain
        ↓
只把资格完整的轨迹交给 g
```

这里的创新点是**自主信息选择**，而不是“使用了某一个大模型”或“增加了某一个足球特征”。如果 Agent 不知道何时值得查询、查完之后何时停止，那么它只是一个固定顺序的数据收集器。

本实验要回答四个问题：

1. Agent 是否能在查询前说明候选信息依赖和获取成本？
2. Agent 是否能选择边际价值最高的动作，而不是把所有来源全部读取？
3. 信息返回后，f 是否真的消费了正文并重新计算概率？
4. 赛果结算后，是否能用真实的 loss 差异训练或评估 g？

## 二、实验中几个必须分开的对象

### 1. f 是信息到预测的映射

f 接收当前市场或结构化基线，以及已经通过证据准入和语义校验的正文 claims，输出比赛概率：

```text
prediction_after = f(H_t + observed_information)
```

f 的职责是判断“已获得的信息如何改变当前预测”。它不负责决定下一步查什么。

### 2. g 是查询前的信息价值估计器

g 接收查询前可见的状态和候选动作，估计执行动作后对损失的期望改善，再扣除获取成本：

```text
g(H_t, a) = E[log f(H_t + o)[y] - log f(H_t)[y] | H_t, a]

net_gain(a) = g(H_t, a) - cost(a)
```

实际训练标签必须来自真实闭环：

```text
realized_gain = loss_before - loss_after
```

其中：

```text
loss_before = -log prediction_before[actual_outcome]
loss_after  = -log prediction_after[actual_outcome]
```

单场没有正文、f 没有运行、没有独立赛前 before、没有赛果，均不能生成 g 训练行。

### 3. shadow f 是受控实验模型，不等于另一套线上 f

当前 shadow f 的作用是：

- 固定模型版本和模型字节；
- 在实验期间保持输入输出规则不变；
- 允许离线回放和 before/after 对照；
- 防止正式 f 被实验过程偷偷改写。

它与 f 的关系是“同一 f 能力的冻结实验副本/受控运行实例”，不是“f 加 shadow f 共同投票”，也不是两个模型同时自由训练。除非实验明确声明模型版本和训练过程，否则不能把 shadow f 的结果称为正式 f 的新能力。

### 4. 三种看起来相同、含义完全不同的 after 状态

| 状态 | 含义 | 是否能作为信息负例 |
|---|---|---:|
| `after_not_run` | 正文没有准入、语义失败或 f 根本没有调用 | 否 |
| `after_unchanged` | 正文已进入 f，合法 claims 为空或没有本场可建模特征，f 合法返回但概率不变 | 可以作为有限诊断，需保留证据 |
| `after_changed` | 正文已进入 f，至少一条合法本场 claim 改变了特征或概率 | 可以进入信息转移分析，赛后才可能形成 g 标签 |

之前 Everton–Chelsea 的问题是把 `after_not_run` 误看成了 `after_unchanged`。这份清单将两者强制分开。

## 三、比赛开始前的硬检查

比赛进入正式实验前，必须逐项填写下表。不能用“前置检查完成”概括所有状态。

### Gate A：比赛身份

必须全部为 `PASS`：

- [ ] `physical_match_id` 已生成并唯一。
- [ ] source match ID 与主队、客队一一对应。
- [ ] 联赛、赛季、比赛日期一致。
- [ ] kickoff 已转换为 UTC，并保留原始时区时间。
- [ ] `prediction_cutoff` 已冻结，且严格早于 kickoff。
- [ ] 至少有一个独立身份来源可回放。
- [ ] 没有把相似球队、相邻轮次或截图文字直接当作 source ID。
- [ ] 当前比赛没有与旧轨迹混用；若复用历史材料，已记录其 `historical` 或 `replay` 身份。

最低输出字段：

```json
{
  "identity_ready": true,
  "physical_match_id": "...",
  "source_match_id": "...",
  "home_team": "...",
  "away_team": "...",
  "kickoff_utc": "...",
  "prediction_cutoff_utc": "...",
  "identity_evidence": ["..."],
  "identity_sha256": "..."
}
```

### Gate B：赛前市场基线

必须在 cutoff 前完成：

- [ ] 赔率原始响应已保存。
- [ ] 原始响应的 request、response、capture 时钟已保存。
- [ ] 报价确实早于 `prediction_cutoff`。
- [ ] 赔率转换为有限的 1X2 概率，并记录归一化方法。
- [ ] `prediction_before` 已冻结，不能在赛后用新赔率替换。
- [ ] prior model/version 和原始 payload SHA 已记录。
- [ ] 若没有真实市场先验，明确写 `prior_status=missing`，不能悄悄使用均匀分布代替。

最低输出字段：

```json
{
  "market_prior_ready": true,
  "quote_observed_at_utc": "...",
  "prediction_before": [0.0, 0.0, 0.0],
  "prior_model_version": "...",
  "source_payload_sha256": "...",
  "cutoff_compliant": true
}
```

### Gate C：来源通道可用性

在冻结本场正式运行前，至少要有一个来源通过一次小规模预检。预检不是让 Agent 先偷看所有文章，而是确认选择动作确实有可执行的来源。

- [ ] 允许域名和工具已配置。
- [ ] seed/index URL 可以访问。
- [ ] 返回内容不是登录页、目录 shell、导航页或纯 JSON-LD 导语。
- [ ] 至少一个候选 URL 能返回可保存的正文或明确的结构化来源。
- [ ] source identity 可以绑定到目标球队、球员或比赛。
- [ ] 文章的公开时间或可见性证据满足 cutoff；未知时标记 `available_at=unknown`，不伪造赛前资格。
- [ ] 失败、超时、NO_DATA、TLS 错误都有明确状态，不把 HTTP 200 直接当正文成功。

输出：

```json
{
  "source_route_ready": true,
  "verified_sources": [
    {
      "host": "...",
      "entrypoint": "...",
      "body_available": true,
      "source_identity_status": "verified",
      "time_status": "pre_cutoff_verified"
    }
  ],
  "route_blockers": []
}
```

如果 `source_route_ready=false`，可以继续做离线 f plumbing 或候选筛选，但不能启动正式的 g 轨迹。

### Gate D：正文准入能力

这一步检查“正文能否进入 f”，不是判断文章内容有没有价值。

- [ ] 正文可以抽取为稳定段落。
- [ ] 每段有可复现的 Unicode offset 或等价定位。
- [ ] 原始响应和正文投影的 SHA 一致可追溯。
- [ ] source assertion 可以建立。
- [ ] 文本不是标题、摘要、导航、照片说明或目录卡片替代品。
- [ ] 正文明确记录主体、赛事范围、时间范围和不确定性。
- [ ] 训练、名单、康复、赛季目标等弱证据不会自动被标成医学可用或本场首发。
- [ ] 语义 provider 使用实际可核验的模型回执；如果模型身份或调用时钟不可核验，单独标注限制。
- [ ] typed-v3 输出能通过 schema 校验；非法 claim 被拒绝，不进行部分挽救或强行映射。

这一步只决定：

```text
body_admission_ready
```

它不决定：

```text
information_is_useful
```

### Gate E：f 路径

正式实验前必须用一条已保存的测试正文完成一次本地接缝检查：

- [ ] 正文能被送入语义抽取。
- [ ] 合法 claims 有精确 `used_input_ids`。
- [ ] f 能收到这些 claims，而不是只收到标题或原始 URL。
- [ ] `prediction_after` 是 f 重新计算的结果，不是复制 `prediction_before`。
- [ ] 运行状态明确写出 `after_not_run`、`after_unchanged` 或 `after_changed`。
- [ ] f 的 model version、model SHA、representation SHA 已记录。
- [ ] 如果 claims 为空，after 可以不变，但必须写明“正文已消费、合法本场 claims 为 0”。
- [ ] 如果 semantic provider 失败，整次动作标为失败，不能把失败动作写成收益为 0。

最低运行字段：

```json
{
  "f_invoked": true,
  "body_consumed": true,
  "used_input_ids": ["..."],
  "modeled_claims": 0,
  "after_status": "unchanged",
  "prediction_before": [0.0, 0.0, 0.0],
  "prediction_after": [0.0, 0.0, 0.0],
  "model_version": "...",
  "model_sha256": "..."
}
```

### Gate F：Agent/g 运行资格

只有 A–E 都满足，才允许把本场作为完整开放信息轨迹启动。

- [ ] Agent 在查询前拿到比赛身份、市场 prior、截止时间和候选菜单。
- [ ] 候选动作由 Agent/数学模块选择，不由人工指定顺序。
- [ ] 每个动作有 action ID、information key、source、query、cost 和能力声明。
- [ ] 选择概率或受控随机探索 propensity 已记录。
- [ ] 查询前不读取候选正文、赛果或未来标签。
- [ ] 查询后保存 request/available/capture/handoff 时钟。
- [ ] 每次动作明确记录 selected 与 unselected。
- [ ] 预算和停止理由可复现。
- [ ] 同一事实组的重复来源不会被错误当成独立信息。

完整 g 资格还需要赛后追加：

```text
action
→ source cost
→ body consumed
→ f invoked
→ prediction_before
→ prediction_after
→ independent outcome
→ loss_before
→ loss_after
→ realized_gain
```

其中任何一个环节缺失，都只能保留为诊断轨迹，不能生成 `g_training_eligible=true`。

## 四、开赛前的 Go / No-Go 判定

### 可以启动正式实验

至少满足：

```text
identity_ready = true
market_prior_ready = true
source_route_ready = true
f_path_ready = true
prediction_cutoff > now
```

并且预计至少有一个可执行的信息动作。

### 可以只做 f 接缝测试，不能做 g 轨迹

以下情况允许做本地验证：

- 身份和市场基线已确认；
- f 模型和正文处理器可用；
- 但 source route 尚未通过，或者没有可验证的赛前正文。

这时应标记：

```text
f_plumbing_only=true
g_training_eligible=false
```

### 必须停止并记录阻断

- 比赛身份不唯一；
- cutoff 已过；
- 没有独立赛前市场基线；
- 来源只能返回目录或 shell；
- provider 或 semantic schema 未通过；
- f 入口未配置；
- 当前运行只能在赛后补采。

不能用“稍微晚一点”“先拿结果再补正文”把它改写成正式前瞻实验。晚采集可以作为结算或 plumbing 诊断，但不能产生同等资格的 g 标签。

## 五、运行中必须实时记录什么

每次查询都要保留：

```text
menu_frozen_at
selected_at
request_started_at
response_received_at
archive_saved_at
source_available_at
agent_handoff_at
prediction_update_at
```

每个动作还要保留：

```text
action_id
information_key
source
url/query
cost
selected
propensity
response_status
body_admitted
body_consumed
modeled_claims
used_input_ids
after_status
```

停止原因必须具体，例如：

- `no_positive_net_value`
- `budget_exhausted`
- `cutoff_reached`
- `no_feasible_candidate`
- `provider_failed_before_body_admission`
- `semantic_extraction_failed`
- `identity_unresolved`

“运行完成”不是停止原因；“预测没有变化”也不是停止原因。

## 六、赛后验收顺序

赛后只做一次独立结果读取，先核对身份再结算：

1. 确认最终 `footMatchState` 或等价终场状态。
2. 核对主队、客队、kickoff 和比分。
3. 冻结赛前 `prediction_before`，不能使用赛后赔率替换。
4. 使用实际 outcome index 计算 `loss_before`。
5. 如果存在合法 `prediction_after`，计算 `loss_after`。
6. 计算：

   ```text
   realized_gain = loss_before - loss_after
   ```

7. 检查动作是否真的完成了正文消费和 f 更新。
8. 只有资格字段完整时，才写入 g 训练表。
9. 对所有不合格轨迹写明具体缺口，不把缺口填成 0。

特别注意：

```text
body_consumed=false  → 不是“正文无价值”
f_invoked=false      → 不是“after 不变的有效负例”
no independent before → 不能计算严格前瞻信息收益
prediction_after=before → 先检查 after_status，再解释为无信息更新
```

## 七、实验结果的三类归档

每场结束后必须归入以下一种：

### A. 完整有效轨迹

```text
identity_ready=true
market_prior_ready=true
source_route_ready=true
body_consumed=true
f_invoked=true
outcome_observed=true
loss_before/loss_after/realized_gain 可计算
```

只有这类轨迹可以进入 empirical g。

### B. f 接缝或负例诊断

正文确实进入 f，但：

```text
modeled_claims=0
prediction_after=prediction_before
```

可以用于验证 f 能区分无关正文与本场证据，但需要保留原文、claims、used_input_ids 和 f 回执。它不自动等于“查询无价值”。

### C. 运行失败或不合格轨迹

例如：

```text
body_consumed=false
f_invoked=false
provider_failed
cutoff_missing
late_capture
identity_unresolved
```

这类轨迹用于发现工程缺陷，不能用于训练 g，也不能被解释为信息收益为 0。

## 八、已知反思与本清单的关系

本清单复用了以下已确认结论：

- `docs/reviews/everton-chelsea-preflight-failure-reflection-20261017.md`：前置检查必须分层；身份/赔率通过不等于正文/f/g 就绪；`after_not_run` 与 `after_unchanged` 必须分开；失败动作不能被写成零收益。
- `docs/research/footballmind-delivery-plan.md`：项目的主验收句是“用户给比赛，系统决定查什么、是否继续、何时停止”；历史重建、构造机制和真实前瞻必须分开。
- `docs/research/remaining-execution-plan-2026-10-07.md`：f 是信息到预测的映射，g 是查询前的条件收益估计器；实际 realized gain 必须来自固定 f 的 before/after 与独立赛果。

本文件不是把旧文档的所有研究沿革重新复制一遍，而是把它们压缩为一张**每场开赛前必须执行的操作清单**。如果旧文档与本清单发生范围冲突，以最新冻结的单场开放信息主线和本清单为准。

## 九、总控验收模板

每场正式运行前，总控必须给出下面这段可读摘要：

```text
比赛：<主队> vs <客队>
physical_match_id：<...>
kickoff：<UTC>
prediction_cutoff：<UTC>

Gate A 身份：PASS / BLOCKED
Gate B 市场 before：PASS / BLOCKED
Gate C 来源通道：PASS / BLOCKED
Gate D 正文准入：PASS / BLOCKED
Gate E f 路径：PASS / BLOCKED
Gate F Agent/g 资格：PASS / BLOCKED

可执行动作：<数量>
当前是否允许正式运行：YES / NO
如果 NO，具体阻断：<...>
```

赛后总控必须给出：

```text
正文是否消费：true / false
f 是否调用：true / false
after 状态：not_run / unchanged / changed
最终结果：<...>
loss_before：<...>
loss_after：<...>
realized_gain：<...>
g_training_eligible：true / false
不合格原因：<...>
```

## 十、最终目标

这份实验不是为了证明“任何文章都会让概率变好”，也不是为了强行制造正收益。实验成功的标准是：

```text
Agent 面对一场比赛时，
能够在开赛前根据当前可见状态提出有限候选，
自主选择是否获取，
将真正获得的证据交给 f，
按结果重新评估是否继续，
最后用可追溯的 before/after/loss/gain 判断这条信息动作是否值得。
```

即使最终某次查询没有改善预测，只要能够证明正文确实进入 f、after 确实由 f 生成、结果和损失计算正确，它也是有效的诊断结果。真正不能接受的是：采集失败、f 没有运行，却把 after 不变写成“信息无价值”。
