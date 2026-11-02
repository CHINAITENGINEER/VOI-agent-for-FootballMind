# Round 9：独立核验比赛身份，不把来源响应当成默认真值

本轮针对第八轮 spftll `2981431` 英格兰—西班牙赛程进行官方独立核验，未扩展日常爬虫或收费数据源。结论分成两个层次：官方页面的阅读结果支持既有赛程，未发现冲突；本地四次直接抓取均失败，所以程序可重放的官方确认仍是 `UNCONFIRMED / official_raw_unavailable`。

这两个状态同时保留。官方赛程支持比赛身份，不代表支持“已出线”“需要轮换”或“采取防守”的推断。第八轮 fixture、源码和原始记录保持封存。

## 官方页面阅读结果

[英足协男子成年队赛程](https://www.englandfootball.com/england/mens-senior-team/fixtures-results) 将该场列为主场迎战西班牙，赛事为欧国联 A3 组，地点为温布利，时间为 2026 年 9 月 26 日 19:45 BST。明确的 BST 时区对应 UTC+01:00，因此换算结果为 `2026-09-26T18:45:00Z`，与第八轮一致。

[UEFA 官方比赛资料页](https://www.uefa.com/uefanationsleague/match/2048018--england-vs-spain/matchinfo/) 对应 England—Spain、温布利；其 URL 中的 UEFA 原生比赛标识是 `2048018`。这提供了待登记的跨来源比赛身份对应：`spftll:match:2981431` 与 `UEFA match 2048018`。对应关系来自人工核对对阵顺序、赛事、地点及英足协的准确开球时刻；没有根据相似队名自动合并实体。

网页工具能阅读这些页面。本地直连归档没有取得成功的官方正文，故网页工具摘录只作为单独的文献阅读笔记，`evidence_class=human_review_of_web_tool_output_not_provider_raw`。笔记保留来源链接、短摘录、记录时间和局限；原站发布时间和网页工具上游抓取时点未知。它没有 HTTP 200 伪造标记、没有原站正文哈希背书，也没有写入公开 CompetitionContext。

文献工具共进行了 3 次调用：2 个检索查询、2 个页面打开、1 个链接点击。其上游 HTTP 请求数未暴露；下文的“4 次”仅指项目自身通过 transport 明确计费和归档的真实采集尝试。

## 四次项目采集尝试

运行：`var/parallel/collectors/round9/official-corroboration-v1`，ID `dbfe802b-42ff-4a01-ba37-fa2f5edd7ad9`。全部复用第八轮 AcquisitionDecisionJournal、现有 RequestTransport/RawArchive；每个动作都先保存 intent 再执行一次网络请求，没有隐式重试。

| UTC 取得时点（2026-09-26） | 目标 | 实际结果 | 证据处理 |
| --- | --- | --- | --- |
| 12:19:45.713281 | UEFA 官方比赛页，标准 session | ConnectionError | 空返回及错误记录归档，计请求 1 |
| 12:20:04.630283 | 英足协赛程页，已有 AioSession | HTTP 403 | 错误正文 1,484 字节归档，计请求 1 |
| 12:20:29.178672 | UEFA 官方比赛页，已有 AioSession | Timeout | 空返回及错误记录归档，计请求 1 |
| 12:21:31.470484 | UEFA 公共 match 路由 `/v5/matches/2048018` | HTTP 403 | 错误正文 392 字节归档，计请求 1 |

UEFA 公共 match 路由是基于官方 match ID 的一次有界探测，403 未证明该路由能提供目标结构化数据。两种 session 都使用原有证书校验配置，没有关闭 TLS 校验。预算达到 4 后立即停止，没有继续扩大来源范围。

四份 observation metadata 与 body 哈希都可复验，包括空 body 的实际哈希。HTTP 403 的错误页面不作为赛程、阶段或阵容事实。日志共 10 个事件；离线验证校验物化头、不可变事件 blob、请求关联、原始提交 metadata、原始 body、实际时序和成本。最终 `charged_requests=4`、`reserved_requests=0`，没有未完成请求。

## 小型核验模块

新增 `football_platform/information/fixture_corroboration.py`，没有改动冻结模块。

`compare_assertion` 比较明确映射后的 source-native event 与目标 fixture 的主队、客队、赛事和带时区开球时间，结果为 `CONFIRMED`、`DISPUTED` 或 `UNCONFIRMED`。它不执行模糊名称匹配。以下边界是明确的：

- 没有显式的 source event → target match 对应关系，返回待确认。
- `documentary_only`、`unavailable` 输入先返回待确认，不读取其中的未取得字段。
- 已取得的结构化输入缺少实际可用时间、超过 cutoff 或在赛后，返回待确认。
- 同一明确映射事件的球队、赛事或 UTC 开球时刻不同，返回具体字段矛盾。
- 缺少字段为 unknown，不当成 false；资格、轮换和战术不在这个比较接口中。
- `CONFIRMED` 仅表示调用方提供的结构化身份字段全部一致；输出明确标注 `input_comparison_not_semantic_source_verification`。源内容抽取与语义验证仍是上游责任，这个函数本身不产生 verified_fact。

`audit_failed_official_capture` 对本轮真实失败记录做独立离线重放，拒绝把网页阅读笔记重新标为已获得的运营证据。正式结果位于 `var/parallel/collectors/round9/fixture-corroboration.json`。

复核命令：

```powershell
python -c "from pathlib import Path; from football_platform.information.fixture_corroboration import audit_failed_official_capture; print(audit_failed_official_capture(Path('var/parallel/collectors/round9/official-corroboration-v1'))['machine_corroboration_status'])"
python -m pytest tests/platform/information/test_fixture_corroboration.py --import-mode=importlib -q
```

19 项构造测试通过，覆盖时区等价、同一事件的身份/时间冲突、缺少映射、未知字段、赛后/未来时间门禁、未购字段遮蔽、名称不自动匹配、失败照实计费，以及文献笔记与运营证据的隔离。测试没有产生真实政策收益样本。

## 对研究的实际影响

本轮没有新增可运营的官方阶段或首发证据。资格、两个排名目标、计划/实际轮换、阵容与战术仍保持未知。所有实采都是 `manual_predeclared`；杯赛/国家队模型为 `NOT_CONFIGURED`，prediction、candidate_expected_gains 和 selection_probability 继续为 null。英超历史校准与本次国家队来源核验保持各自的模型适用范围。

本轮的直接贡献是防止两类错误：一是把来源列表中的赛程当成自动可信身份；二是把“查过但没有取得正文”当成已观察到某个足球状态。失败查询花费了成本，但并未带来该状态的知识。这正是未来信息价值估计需要包含的返回分支。

下一条必要证据是一份成功取得、实际时间明确的官方比赛响应，至少锚定官方 event ID、主客队、赛事阶段、带时区开球时刻，并保存具体字节位置与来源原文。取得后可建立可重放的来源间身份映射；在此之前，本轮机器状态继续待确认。该步骤无需新增通用采集框架，也无需借用英超模型生成国家队胜率。

旧归档审计扫描 production/round7/round8 共 121 份 observation metadata，没有找到已保存的 UEFA、英足协或西班牙足协对应响应。清单保留路径、metadata 哈希、body 哈希和来源时间。旧 round7 与 round8 的源码和产物哈希复验结果、最终代码清单及本轮产物清单见 `var/parallel/collectors/round9/acceptance.json`。
