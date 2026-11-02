# 第七轮：真实杯赛证据准备度与时间化登记

本轮结论是：本地已有部分真实杯赛赛前比赛列表，新增保存了日本足协预告的真实原始响应，并完成只追加的证据登记与购买后投影；目前仍没有足以估计杯赛主动获取策略收益的真实上下文轨迹。所有原始归档、旧导出和第六轮契约保持不变。

## 已有数据究竟有什么

机器清单为 `var/parallel/collectors/round7/local-archive-audit-v2.json`。审计明确范围为生产 `football_spider/platform_data/raw`、此前 collector 导出和一个 legacy match snapshot；排除 cookies、凭证、数据库文件、测试构造及本轮新增 HTTP 响应。逐项保留文件路径、SHA256、字节数、源观测时点、原始 JSON 路径和缺失原因。

| 检查 | 结果 |
| --- | ---: |
| 生产原始响应元数据 | 113 条 |
| 原始内容哈希复核通过 | 113 条 |
| 未提交的裸 payload | 0 |
| 既有 matches 导出 | 9 份，合计 1,972 行（含版本重复） |
| 去重后的 `(source, source_id)` | 779 个，仍非独立研究样本计数 |
| 规范化导出中杯赛候选行 | 0 |
| 原始响应中杯赛行出现次数 | 493 次、159 个源比赛 ID |
| 原始观测早于该行开球时间 | 103 次、34 个源比赛 ID |
| 被检索的标准上下文键候选 | 0 |
| 已核实杯赛上下文策略轨迹 | 0 |

杯赛候选通过联赛名和 JSON 结构寻找，并非语义验证。很多行来自历史交锋或嵌套赛程列表，重复抓取也被单独计数。`source_published_at` 在这些原始观测中没有核实字段；不能把上述 103 次解释为 103 场包含完整资格、阵容和策略动作的研究样本。

用户提到的两场比赛确实在既有 spftll 赛前列表内：

- 日本—泰国：源比赛 `3049653`，主客源队 ID `1498 / 5198`，`2026-09-23 18:30 Asia/Shanghai`，等同日本当地 19:30。原始观测为 `40d2b6a2-d5dd-4a12-b0ec-57577acb296c`，实际观测时点 `2026-09-22T09:16:58.504353Z`。
- 中国—阿联酋：源比赛 `3049659`，源队 ID `4934 / 5201`，`2026-09-23 13:30 Asia/Shanghai`。同一列表在开球前已抓取。

这些是赛程/比赛列表证据，不包含经验证的晋级规则、排名目标或阵容。案例绑定使用真实源 ID；联赛源名称因没有数值联赛 ID，使用名称的 UTF-8 SHA256 作为稳定命名空间。没有用比分决定实体对应。UTF-8 文件中的球队名称为“日本亚”“泰国亚”，旧终端显示乱码不代表原始文件乱码。

## 本轮真实官方响应

复用现有 `RequestTransport` 和 `RawArchive`，每次尝试均先归档，未添加通用爬虫。

1. [JFA 日本—泰国赛前预览](https://www.jfa.jp/national_team/u21_2026/asiangames_2026_men/preview_03.html)成功返回 HTTP 200，137,313 字节，实际 `observed_at=2026-09-26T11:13:02.744184Z`，内容 SHA256 为 `ff562f3d108b934fcaed51f5080c0946c644b2df1fdfdd49bdd6e1da28eefb7f`。文内支持日本已经晋级、打平即可确保第一的陈述，但当前 HTML 没有提取到明确发表时间戳。
2. [JFA 赛程与结果页](https://www.jfa.jp/national_team/u21_2026/asiangames_2026_men/schedule_result/)成功保存为另一原始观测。这是赛后取得的赛程/结果混合页面，按 `unknown` 文档范围进入 `evaluation-quarantine`，不进赛前决策。
3. 体育总局赛后报道发生一次 SSL 错误；新华社赛后报道发生一次连接错误。失败尝试各自留存，没有关闭 TLS 校验，也没有把错误页当成有效新闻证据。

总计 4 次 HTTP 尝试、2 次成功、2 次失败；没有付费 API 调用。原始响应位于 `var/parallel/collectors/round7/official-capture-v1/raw`。

JFA 预告的两条结构化映射均带精确原始字节区间及区间哈希，状态保持 `unverified`。区间验证证明原文存在，尚未建立可信语义验证器，故没有手填 `verified_fact`。以 9 月 23 日开球前一小时投影时，两条均因直到 9 月 26 日才实际取得和登记而排除。即使在专门的历史重建门禁测试中提供假设时间，仍因源发表时间未知被排除。该测试假设不进入数据集。

案例和门禁结果：`registered-case-v1/case.json`；真实混合结果页隔离：`registered-case-v1/quarantine-case.json`。原有 `case.json` 保留；新增隔离证据以新身份追加。

## 最小登记与投影接口

实现为 `football_platform/information/evidence_registry.py`，复用第六轮 `competition_context`：

```python
registry = CompetitionEvidenceRegistry(registry_root, existing_raw_archive)
record = registry.register(
    evidence_id="provider:claim:version1",
    fixture=fixture,
    observation=committed_raw_observation,
    document_scope="pre_match_preview",
    claim={"topic": "qualification", "value": "qualified", "team_id": team_id},
    anchor={"byte_start": start, "byte_end": end, "sha256": source_span_sha256},
)
projection = registry.project(
    fixture=fixture,
    prediction_cutoff=cutoff,
    acquired_evidence_ids=[record.evidence_id],
)
policy_context = projection.context
```

登记前重新读取 RawArchive 提交元数据与内容，检查调用方的观测 ID、来源、时间和内容哈希是否与归档一致。`registered_at` 使用真实时钟，`operational_available_at` 至少晚于实际观测与登记。调用方不能在 claim 内注入来源或可用时间。

记录写入前检查两个分区的同身份占用，现有 ID 一律拒绝覆盖，修订需新版本 ID。逻辑 ID 可以包含命名空间冒号，但磁盘文件名固定是该 ID 的 SHA256；不把 `sha256:...`、`CON`、`COM1` 等直接作为 Windows 文件名或 NTFS stream 名称。读取时重新校验记录和原始响应哈希。这提供应用层只追加与完整性检查，不声称具备独立可信时间戳、第三方签名或管理员防篡改存储。

未购买 ID 的记录不会读取。`post_match_report` 和 `unknown` 文档只能入评估隔离区，也不接受赛前上下文 claim。官方来源本身不等于语义验证完成，本入口只接收 `unverified` 或 `hypothesis`。

时间字段明确区分：

| 字段/模式 | 含义与门禁 |
| --- | --- |
| `source_published_at` | 可选源发表时间；若声明为明确时间，必须有原始字节区间，区间中的带时区 ISO 时间须与声明相等。仅页面日期、搜索索引日期不直接转成精确时间。 |
| `observed_at` | 已提交原始响应的实际采集时间。 |
| `registered_at` | 结构化候选首次登记的实际时间。 |
| `operational_available_at` | 本入口可操作使用时间，至少晚于观测与登记；超过预测 cutoff 就排除。 |
| `historical_reconstruction` | 必须逐条提供明确假设 ID，并有可核对源发表时间；假设时间不得早于发表时间。仍不等于当时实际取得，也不证明页面当年未修订。 |

`projection.context` 是可供策略读取的投影；`actual_observation_times`、排除原因及假设清单是审计侧元数据，不作为额外模型特征。重建模式中实际观测时间仍在审计侧保留，不把它伪装成历史观测时间。模式必须随外层投影保存，消费方不能只提取重建 context 后标记为 operational。

## 验证与下轮最小数据要求

注册模块 34 项新测试、既有上下文 60 项，共 **94 passed**。覆盖旧预告补抓排除、真实观测时间被伪造时拒绝、赛后隔离、购买门禁、原文锚点、明确发表时间锚点、追加防覆盖、Windows 文件名以及显式重建假设。测试只用构造源验证契约，不算真实政策实验。

```powershell
python -m football_platform.information.audit_round7 --output var/parallel/collectors/round7/NEW_AUDIT.json
python -m football_platform.information.prepare_round7_case --capture var/parallel/collectors/round7/official-capture-v1 --output var/parallel/collectors/round7/NEW_CASE_DIR
python -m pytest tests/platform/information/test_evidence_registry.py tests/platform/information/test_competition_context.py --import-mode=importlib -q
```

接下来真正缺少的是同一批真实杯赛、同一赛前 cutoff 下的一组可比较轨迹：身份及赛制；有来源的积分表和同分规则；按“晋级/第一”分别表达的条件；阵容与比较基准；每项信息的采集/处理/可用时点和原文；候选动作、成本与实际返回；动作前可见状态及预测；最后独立入库的结果标签及首次可用时间。战术仍需明确区分教练公开意图、赛前假设和赛中实际行为。当前没有给缺失项目填猜测，没有把单场 3:0 当成 VoI 真值。

最先可执行的研究不是增加网站数量，而是在已知合法 fixture 上前瞻冻结这些证据和动作。官方预告补抓只能补齐可追溯材料，不能补回过去没有记录的操作轨迹。
