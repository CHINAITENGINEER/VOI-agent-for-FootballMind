# Everton–Chelsea 前置检查误判复盘

**复盘日期：** 2026-10-17  
**比赛：** Everton–Chelsea  
**比赛 ID：** `spftll:match:3003912`  
**physical_match_id：** `footballmind:premier-league:2026-27:2026-10-17:everton:chelsea`  
**复盘性质：** 总控判断错误、运行门禁设计不足、f/g 训练资格误判风险复盘

## 一、复盘结论

这次运行没有形成 FootballMind 所需要的有效 f/g 训练样本。

我之前向用户说“前置检查已经完成”，这个表述不准确。实际完成的只是：

- 比赛身份核验；
- 开球时间和预测截止时间核验；
- 市场赔率读取；
- `prediction_before` 冻结；
- 赛后独立读取比赛结果。

真正决定本项目核心目标的环节没有通过：

- 正文 provider 在正文准入前失败；
- `documents=0`；
- `body_consumed=false`；
- `f_invoked=false`；
- 没有由 f 重新计算得到的有效 `prediction_after`；
- 没有形成“信息动作—f 更新—赛果—realized gain”链路。

因此本场的 `realized_gain=0` 只能表示：

> 没有执行有效的 f 更新，所以 after 保持 before，两个 loss 相同。

它不能表示：

- g 判断查询信息没有价值；
- 正文信息不会改变足球预测；
- f 已经验证有效或无效；
- 当前 g 已经学会了停止策略。

## 二、实际发生的运行过程

比赛赛前已经完成了以下准备：

```text
官方/本地身份核验
        ↓
市场赔率读取
        ↓
冻结 prediction_before
        ↓
尝试执行信息采集
        ↓
provider 在正文准入前失败
        ↓
documents = 0
        ↓
f 没有调用
        ↓
prediction_after 保持初始值
```

冻结的市场先验为：

```text
主胜：0.29024751427967005
平局：0.25809181298921090
客胜：0.45166067273111915
```

赛后读取到的结果为：

```text
Everton 0–2 Chelsea
结果类别：away_win
```

因为客胜概率为 `0.45166067273111915`，赛前基线 log loss 为：

```text
loss_before = -log(0.45166067273111915)
            = 0.7948241051305114
```

由于 f 没有真正执行，结算文件中的 after 与 before 相同：

```text
prediction_after = prediction_before
loss_after       = loss_before
realized_gain    = 0
```

赛后独立结果读取本身是成功的，但它只能完成已有市场基线的结算，不能补救赛前没有执行的 f 转移。

## 三、当时我说错了什么

我当时使用了“前置检查已经完成”这样的笼统表述。这个表述把不同层级的准备工作合并成了一个状态，造成了错误含义。

我实际确认的是：

```text
identity_ready = true
market_prior_ready = true
```

但我没有确认：

```text
source_route_ready = true
f_path_ready = true
```

更没有确认：

```text
g_training_ready = true
```

所以“身份和赔率准备好”被我错误地说成了“整条赛前信息决策链路准备好”。

这是总控判断错误，不是用户理解错误。

## 四、证据如何证明这不是一个有效 f/g 样本

旧运行报告 `final-report.json` 已经明确记录：

```json
"real_http_requests": 4,
"documents": 0,
"f_invoked": false,
"g_training_eligible": false,
"reason": "provider_failed_before_body_admission"
```

这几个字段共同说明：

1. 运行确实发起了来源请求；
2. 请求没有转化为可用正文；
3. 没有正文进入 f；
4. 没有真实的模型后验更新；
5. 不能构造 empirical g 训练标签。

另外，初始化文件中还记录了：

```json
"response_model": null,
"value_scope": "unknown_until_response_model_is_trained",
"calibration_status": "not_established",
"numeric_voi_training_eligible": false
```

这说明当时市场基线可以使用，但 VOI 响应模型和 g 的经验训练仍然没有建立。

## 五、我把哪些概念混在了一起

### 1. 离线 f plumbing 测试与 live provider 链路

离线测试使用的是已经保存到本地的正文：

```text
本地正文
    ↓
离线语义抽取
    ↓
f
    ↓
prediction_after
```

这类测试可以证明 f 的代码能够接受合法输入，也能区分 modeled claim 和 unmodeled claim。

但比赛实跑要求的是：

```text
Agent选择动作
    ↓
provider请求
    ↓
正文可访问
    ↓
正文准入
    ↓
语义抽取
    ↓
断言校验
    ↓
f
    ↓
prediction_after
```

Everton–Chelsea 只完成了第一部分准备，没有完成第二条 live 链路。

### 2. provider 请求成功与正文可用

有 HTTP 请求返回，不等于正文已经可供 f 使用。

必须分别确认：

- HTTP 是否成功；
- 返回内容是否是目标文章；
- 是否包含可解析正文；
- 正文是否在 cutoff 前可见；
- 正文是否通过 source assertion；
- 语义输出是否符合 typed-v3 契约；
- 至少有一条合法结果进入 f。

Everton–Chelsea 在正文准入之前就失败了，因此不能把“请求发出”当成“f 输入准备完成”。

### 3. offline negative control 与真实负例

离线负例可以说明：

```text
正文已经进入 f，但没有支持本场的 modeled claim。
```

而本场 live 运行实际是：

```text
正文没有进入 f。
```

这两种情况最终都可能出现 `prediction_after = prediction_before`，但含义完全不同，不能用相同标签处理。

## 六、这次错误造成的影响

### 对 f 的影响

这场比赛没有证明 f 是否能处理真实 live 正文。

它没有增加以下任一证据：

- 真实本场伤停被 f 识别；
- 真实首发变化被 f 识别；
- 真实战术信息被 f 识别；
- f 在 live provider 场景下成功生成 after；
- f 产生有效或无效的概率更新。

### 对 g 的影响

这场比赛不能作为 g 的训练行。若把它写成：

```text
查询信息 → gain = 0
```

就会产生错误标签。正确的状态是：

```text
信息动作没有完成
f 没有运行
gain 不可用于评价信息价值
```

### 对项目进度的影响

项目在表面上完成了“采集、预测、赛后结算”，但核心的 f/g 联调没有发生。因此项目进度不能按完整闭环计算，只能按以下状态计算：

```text
数据身份与市场基线：已完成
正文 live 接入：失败
f live 联调：未完成
g empirical 训练：未开始
```

## 七、根因分析

### 根因一：门禁粒度过粗

系统存在很多局部检查，但没有把它们组织成一个严格的阶段状态机。于是“初始化成功”容易被解释为“运行就绪”。

### 根因二：没有把正文准入作为比赛执行前的硬门禁

在真正冻结比赛运行前，没有要求至少一个候选正文来源完成一次可用性验证。结果是比赛开始后才暴露 provider 失败。

### 根因三：没有区分 `after_not_run` 与 `after_unchanged`

这两个状态必须分开：

```text
after_not_run
```

表示 f 没有生成 after。

```text
after_unchanged
```

表示 f 已经运行，且合法输入没有改变预测。

之前的流程对这两个状态区分不够严格，容易让人误以为“after 等于 before”就是一次有效的 f 负例。

### 根因四：总控汇报没有同步展示硬证据

我汇报“前置完成”时，没有同时给出：

```text
documents
body_consumed
f_invoked
prediction_after_basis
g_training_eligible
```

如果当时直接展示这些字段，错误会立即暴露。

## 八、修正后的运行门禁

今后每场比赛必须按以下状态逐项报告，不能再用一个“前置完成”概括全部内容。

### Gate A：身份门禁

```text
identity_ready
```

必须确认：

- physical_match_id；
- source match ID；
- 主客队；
- 联赛和赛季；
- kickoff；
- prediction cutoff；
- 独立身份来源。

### Gate B：市场先验门禁

```text
market_prior_ready
```

必须确认：

- 赔率原始快照；
- quote 时间；
- 概率归一化；
- `prediction_before`；
- prior model version；
- market prior 的 SHA。

### Gate C：来源通道门禁

```text
source_route_ready
```

必须在正式比赛运行前确认至少一个来源满足：

- URL 可访问；
- 返回目标页面；
- 页面存在正文而不是目录或 shell；
- source identity 可绑定到目标比赛或目标球队；
- 文章时钟满足 cutoff；
- raw payload 可以保存并复现。

### Gate D：正文准入门禁

```text
body_admission_ready
```

必须确认：

- 正文可抽取；
- 正文段落有稳定 offset；
- source assertion 可以构建；
- available_at 或可见性证据满足赛前约束；
- 不是标题、导航、JSON-LD 导语或训练照片替代正文。

### Gate E：f 路径门禁

```text
f_path_ready
```

必须确认：

- 语义 provider 可运行；
- typed-v3 输出合法；
- 非法 claim 会被拒绝；
- 合法 claim 可进入 f；
- `used_input_ids` 能追溯正文；
- `after_status` 明确是 `generated`、`unchanged` 或 `not_run`。

### Gate F：g 训练门禁

```text
g_training_ready
```

只有同时满足以下条件，才允许进入 empirical g：

```text
action recorded
source cost recorded
body consumed
f invoked
prediction_before recorded
prediction_after recorded
independent outcome recorded
loss_before recorded
loss_after recorded
realized_gain computed
```

其中任意一项缺失，都只能标记为诊断或失败轨迹，不能生成 g 训练行。

## 九、下一场比赛的最低要求

下一场比赛不应只重复“身份+赔率+赛果”流程，而要优先完成一场**可验证的 f 正例或合法负例**。

至少需要形成下面两种状态之一：

### 合法负例

```text
body_consumed = true
modeled_claims = 0
f_invoked = true
prediction_after_status = unchanged
```

这可以证明正文进入了 f，但没有提供本场可建模信息。

### 合法正例

```text
body_consumed = true
modeled_claims > 0
f_invoked = true
prediction_after_status = generated
prediction_after != prediction_before
```

赛后再追加：

```text
outcome
loss_before
loss_after
realized_gain
```

只有这样，才能判断一次信息动作对预测风险的实际影响。

## 十、总控承诺

今后我不会再用“前置检查完成”表示整条链路已经可运行，而会逐项报告每个 gate 的状态。

对于任何 `documents=0`、`body_consumed=false` 或 `f_invoked=false` 的轨迹，我会明确标记：

```text
f transition not executed
g training row not eligible
```

对于 `prediction_after = prediction_before` 的轨迹，我会先检查：

```text
after 是否真正由 f 生成
```

只有确认 `f_invoked=true` 后，才能把它解释为“正文进入 f 但没有改变预测”。

这次 Everton–Chelsea 的准确归档结论是：

```text
赛前身份：通过
市场基线：通过
正文采集：失败
正文准入：失败
f调用：未发生
赛果结算：完成
有效f转移：没有
empirical g训练：没有新增样本
```

## 十一、相关证据文件

- 旧运行报告：`D:\a1-足彩\var\work\fm-g-complete-match-everton-chelsea-20261016-final\final-report.json`
- 旧身份与初始化：`D:\a1-足彩\var\work\fm-g-complete-match-everton-chelsea-20261016-final\identity-binding-live.json`
- 赛果结算：`D:\a1-足彩\var\work\fm-g-settlement-everton-chelsea-20261017-v1\settlement.json`
- 可读结算报告：`D:\a1-足彩\var\work\fm-g-settlement-everton-chelsea-20261017-v1\readable-report.md`

赛果结算包并没有改写原赛前包。它只完成了基线结果的独立结算，并保留了本次运行没有进入 f 的事实。
