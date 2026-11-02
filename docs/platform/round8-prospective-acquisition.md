# Round 8：真实前瞻信息获取记录与决策日志

本轮落实“先记录为什么查、再查、最后记录查到了什么”的执行边界。新增日志复用现有 `football_agent.audit.RunStore`、`RawArchive`、`RequestTransport` 和第七轮证据登记器；没有新建通用工作流框架。有限的四次真实请求已经结束，后续验证仅读本地文件。

验收结论是 `VERIFIED_PROSPECTIVE_CAPTURE_NOT_POLICY_GAIN_STUDY`。确认了一条来源当前列出的未来欧国联赛程记录，并形成一次针对该比赛的详情查询。尚未取得资格、排名目标、轮换、阵容或战术方面的结构化证据。此结果验证真实采集的时序和成本记录，不代表自主策略收益实验完成。

## 实际采集与范围

运行目录：`var/parallel/collectors/round8/prospective-v1`。运行 ID：`2a3f618d-2296-4c34-8ba1-6d897eda4565`。所有时间按 UTC 记录，四次请求均单次尝试，无自动重试。

| 动作 | 取得时间（UTC，2026-09-26） | HTTP / 语义结果 | 对研究的含义 |
| --- | --- | --- | --- |
| spftll 当前比赛列表 | 11:44:50.680697 | 200，当前列表 | 发现当前仍未来的候选赛程 |
| spftll 近期赛程 | 11:46:17.563892 | 200，列表 | 同时包含已结束或已开始比赛；这些比赛没有作为前瞻目标或收益标签 |
| 旧快照中未来德国杯 ID 3086946 的当前详情 | 11:47:02.724581 | 200，`code=0,data=null` | 当前详情未确认，记录 `NO_DATA`；实际请求仍计费 1 次 |
| 当前列表中英格兰—西班牙 ID 2981431 的详情 | 11:48:23.342359 | 200，有数据，`未开` | 当前来源的未来国际比赛记录；未取得目标上下文结论 |

目标比赛的来源原生标识是 `matchId=2981431`、`homeId=744`、`awayId=772`；详情返回 `leagueId=1864`。比赛列表与详情中的本地开赛时间均为 `2026-09-27T02:45:00`，依照现有 spftll normalizer 的 `Asia/Shanghai` 约定转换为 `2026-09-26T18:45:00Z`。四次响应、目标绑定、证据登记和停止事件都早于此时间。

列表最初只有赛事名“欧国联”，故不可变 fixture 使用该名称的 SHA256 作为临时来源内赛事身份。详情随后提供原生 `leagueId=1864`；验收报告保留这一对应关系，没有回写已登记 fixture。后续实体归一化需要在身份映射层处理，保持原始事件不变。

这是一条 spftll 当前响应支持的未来欧国联赛程，未取得独立官方足协的赛程交叉确认；其来源与时区约定均在报告里明示。它不是当前亚运会案例，也不是已确认的未来德国杯案例。三次发现/未确认查询和一次已绑定目标的详情查询，分别计数；没有把它们宣称为针对同一场比赛连续取得四类信息的研究轨迹。

## 时间和信息边界

- `decision_at` 是动作登记时刻；不可变 intent event 的时间也必须早于实际 `requested_at`。
- `observed_at` 来自 HTTP 传输观测；原始响应先写入 RawArchive，metadata 作为提交标志。
- `receipt_at` 是日志接受这次返回的时刻；回执关联 request ID、原始 observation ID、内容哈希和实际请求成本。
- `source_published_at` 没有来源时间锚点，本例保持 `null`。取得时间和来源发布时间没有混用。
- 本例文档于 `2026-09-26T11:54:12.651144Z` 登记，`operational_available_at` 等于实际登记可用时刻。投影以这个实际时间为门禁，使用 `operational` 模式。
- 目标响应只登记了原始比赛 ID 的字节锚点，`claim=null`，所以投影排除原因是 `no_structured_context_claim`。资格、两个排名目标、计划/实际轮换、阵容、战术和阶段均保持 `unknown`。
- 原始 body 和输入参数放在原始归档或内容寻址 blob；决策前态仅携带已取得 observation IDs、已发布 evidence IDs 和引用。未付费/未取得回执的 payload 不进入前态。
- 页面包含历史战绩不等于这些字段已成为模型输入。本例没有预测、结果标签或赛后补录，没有把阵容缺失、未知战术或未查工具的收益写为 0。

已购买且语义可用的响应才可经 `publish_evidence` 进入公开 competition context；登记文档本身并不等于它包含有效的结构化 claim。提前出线不会自动产生轮换、变弱或防守结论。

## 小型日志接口

代码：`football_platform/information/acquisition_journal.py`、`prospective_capture.py`。

```python
archive = RawArchive(root / "archive")
journal = AcquisitionDecisionJournal(root / "journal", archive)
run = journal.start(request_limit=4, fixture=known_future_fixture_or_none)

# capture_once 先调用 journal.begin 持久化 intent 和预算预留，再执行 HTTP。
result = capture_once(
    journal, run["run_id"], request_id="unique-request-1",
    action_id="source.context", source="source_name",
    endpoint="https://provider.example/context", params={"match_id": "native-id"},
    session=existing_provider_session,
)

# 解析有独立语义状态；HTTP 200/data:null 仍然花费了请求。
journal.record_processing(
    run["run_id"], request_id="unique-request-1", status="NO_DATA",
    extraction_sha256=actual_extraction_digest, missing_fields=["qualification"],
)
journal.stop(run["run_id"], "NO_CONFIGURED_MODEL")
```

这是接口示意；真实请求和精确参数已封存在本轮产物，示意中的主机不是网络测试目标。

`begin` 接收 `candidates`、`assessment`、`selection_basis`、`policy_version`、`selection_probability`。本轮真实记录全部为 `manual_predeclared`，选择概率是 `null`。模型缺失时固定 `NOT_CONFIGURED`，prediction 和 candidate_expected_gains 为 `null`，model_versions 为空。数值策略配置后才接受合法归一化概率、候选动作的实际模型输出、版本和调用方真正计算的选择概率；日志验证格式，不证明模型训练质量或期望收益正确性。人工动作不赋予虚构的 propensity，现有记录不用于逆倾向或其他要求可识别行为策略的离线政策收益估计。

同一 request ID 和相同意图重放不执行 HTTP，不重复扣费；同 ID 不同参数被拒绝。崩溃后存在唯一匹配的原始提交可恢复回执。若原始提交缺失或匹配不唯一，保留预留并返回 `NEEDS_RECONCILIATION`，不会凭空假设零成本、释放预算或重发。超时/HTTP 失败仍按实际尝试计请求；赛后才返回的结果记录真实成本但不发布赛前证据。

RunStore 的 `run.json` 是原子替换的物化头；事件链另存内容寻址不可变 blob。每次读取校验链和 blob 一致性。RawArchive 的实际 metadata 与传入 observation 必须完全一致，防止调用方仅改时间而沿用原始 body。这个本地完整性保证不是针对拥有全部文件修改权限的对手的外部时间戳证明。

## 复核与测试

离线命令：

```powershell
python -m football_platform.information.verify_round8_prospective var/parallel/collectors/round8/prospective-v1 --output var/parallel/collectors/round8/prospective-verification-new.json
python -m pytest tests/platform/information/test_acquisition_journal.py tests/platform/information/test_evidence_registry.py tests/platform/information/test_competition_context.py --import-mode=importlib -q
```

验证输出采用独占创建，不覆盖已有报告。首次正式结果为 `prospective-verification.json`：15 个事件与不可变 blob、4 个原始响应、4 次实际请求计数、全部决策/请求/返回/登记时序、原生比赛和球队身份、时区转换和上下文投影均通过。未结预算为 0，实际请求费用为 4，货币费用仍未知。验证本身新增 HTTP 请求为 0。

本轮新增 23 项构造测试，连同证据登记与上下文旧测试共 117 项通过。构造测试覆盖时序、持久化 intent 前请求拒绝、赛后隔离、未购买不读、伪造 metadata 拒绝、预算、失败付费、崩溃恢复、幂等、事件篡改、缺模型保持空值以及 `data:null` 语义分类。这些测试不计入真实策略样本。JUnit 文件为 `var/parallel/collectors/round8/tests.xml`。

## 下一轮最小所需输入

1. 一个可持续按当时可用时间取得的上下文动作：其 source-native match/team IDs、请求参数、原始返回、来源发布时间（若确实给出）、实际观测/登记时刻、字节锚点和明确的字段语义。例如资格的目标应区分 qualification 与 first_place，条件排名还需关联另一场 fixture ID 与对应证据 ID。
2. 为同一比赛建立真实初始 evidence IDs 和有限动作目录；阵容/轮换来源需要真实发布时间和当时获取记录。来源未返回字段时保持 unknown，而不是用报道空白证明没有轮换。
3. 目标域经过验证的概率模型与响应分布模型及各自版本，再启用数值策略和候选 expected gains。实际配置前继续按采集准备记录，不将英超训练模型概率包装成已验证杯赛概率。
4. 在多个独立比赛和多个真实决策时点预先登记策略、候选动作、预算、行为选择概率（只有真正计算时才记录）；比赛结束后单独登记结果供冻结方案评估。单场当前记录没有未选择动作的反事实回报，也没有 expected gain 真值。

本轮新增内容限定在独立信息模块、相关测试、本文与 round8 产物。第六轮契约、第七轮原始归档和证据登记器保持原样。总验收在 `var/parallel/collectors/round8/acceptance.json`，其中包含代码、依赖和产物的 SHA256 清单。
