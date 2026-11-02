# Round185：真实正文、来源语言关联与语料资格闭环

交付日期：2026-10-07，北京时间。核心仍是由 Agent 选择信息；本项完成数据处理包，
不拟合新模型，不改变旧开发集分数，也不把处理成功当成 VoI 收益成立。

## 已落地的处理链

1. `NewsDiscoveryTools(..., projection_version=2)` 新增显式版本的正文解析。
   默认 v1、旧传输配置及原始回放保持原行为。v2 在实际捕获的官方 jalgpall 新闻页面中
   校验来源路径、news 标记、唯一标题/日期/正文容器，提取实际段落与字符位置。
   无时区来源日期保留原文、日期和本地钟点，UTC 发布时间仍为空。
2. `local_text_scope` 将已核验的赛事身份与已归档本地新闻目录分别绑定。只在已观察
   官方来源和 et 语言范围中使用本地词形；它没有独立证明双语别名，输出明确标记
   `unverified_source_assertion`。本场对阵与正文中有依据的赛事日期共同出现才描述为
   match_specific；人物背景保留 team_context，两个类别都不表示信息价值。
3. `derived_text_corpus` 同时保存原实际轨迹、派生正文及独立监督 sidecar。重解析
   时间与原采集时间分开，原 actor 的准入、旧菜单、旧未知资格均不回填。研究面板中的
   两次响应不拼成真实序列，不附历史选择概率，不生成已有固定预测器的校准记录。
4. `collect_prepared_batch` 和 CLI 增加 `news_projection_version=2`，可供之后实际
   采集使用；CLI 同时暴露已有 `source_profile_version`。新 live 使用实际解析时钟，
   旧运行仍按原版本回放。默认部署配置和 Round182 发布包保持原样。

## 实际语料结果

| 统计项 | 本轮结果 |
|---|---:|
| 纳入的原始实际运行 | 9 |
| 独立比赛 | 6 |
| 实际响应 episodes | 26 |
| 原响应资格通过 / 拒绝 | 26 / 0 |
| 原冻结研究面板 | 5（含 4 个零响应面板）|
| 单独研究响应 | 2，均属于同一场比赛 |
| 新版本识别的正文候选 | 2 |
| 独立赛果标签 | 0 |
| 合格有标签训练记录 | 0 |
| 新增运行时 VoI 校准记录 | 0 |

两份正文来自原赛前归档，没有重新联网下载：

| 正文 | 新投影文本字符（含标题） | 正文段数 | 来源限定的关联描述 |
|---|---:|---:|---|
| 爱沙尼亚—冰岛赛前须知 | 3,402 | 11 | match_specific，来源断言 |
| Emil Atlason 人物文章 | 4,582 | 19 | team_context，来源断言 |

原捕获在北京时间 10 月 6 日 23:28；新解析在 10 月 7 日 00:44。两个时钟都明确保存。
新解析也发生在目标场次开球前，但它是研究处理结果，未进入原已结束的 actor 轨迹。
旧 Round184 的自动正文资格仍是 0，新派生语料候选是 2，两项统计不能相互覆盖。

f_training_candidate 表示“可进一步形成离线正文预测语料的候选”。标签缺失时
supervised_text_eligible=false；即使以后标签具备，也不自动成为在线 VoI 校准序列。
同场、转载、多个运行和不同投影版本必须按 match_id 一起划分训练/评估，不能把两篇
文章算成两场独立比赛。来源适用描述不作为有用性筛选或权重。

## 独立赛果核验与实际外部条件

有界核验截至北京时间 2026-10-07 00:29:12：

- Kazakhstan–Faroe Islands：官方原文确认 Finished，源终场时间为前一天 23:54:30。
  同一页面全部原生赛事对象及 JSON-LD 无数值比分，官方赛程目标链接仍显示 vs。
  因缺少可独立核验的终场比分，状态为 UNCONFIRMED。已经确认是当前原始响应缺少
  比分，不是现有解析器漏读现成字段；未降低网关条件来制造标签。
- Croatia–Spain、Scotland–Slovenia、Moldova–Slovakia、Luxembourg–Bulgaria、
  Estonia–Iceland 均在北京时间 10 月 7 日 02:45 开球，核验时为 NOT_DUE，未请求。

网络实际 3 次：比赛页 1 次成功；赛程页环境路由失败 1 次、预先允许的直连补救成功
1 次。所有失败保留，之后停止；正文重解析、语料导出及验证均零网络。没有循环轮询、
后台等待任务或新增搜索服务。旧搜索不对题问题仍未解决，不能声称来源覆盖已充分。

## 验收

- 最终相关回归 173 passed，覆盖新旧新闻解析、来源候选、局部语言、实际语料、
  面板、独立身份、赛果、采集入口和原始响应网关。
- 9 份原始真实运行、26 次响应在禁网下逐字回放通过。
- 全量新语料在禁网下重建校验通过；261 个原始文件字节未变。
- 独立审查发现并修复重复面板重新签名后可膨胀样本数的问题；输入路径、actor、
  记录 ID 唯一性和全部面板（包括空面板）原文件摘要已核验。
- 新旧正文、解析时间、身份和独立标签从原始字节重算；不能通过重写统计或手填
  outcome 获得监督资格。独立审查无剩余阻断。

## 可复用入口

未来采集可显式选择两个版本，默认保持原版本：

```python
collect_prepared_batch(
    plan_path,
    output=new_output,
    archive_root=market_archive,
    allowed_hosts=allowed_hosts,
    identity_bundle=identity_bundle,
    identity_archive=identity_archive,
    news_provider="bing_rss",
    source_profile_version=2,
    news_projection_version=2,
)
```

相应 CLI 参数为 `--source-profile-version 2 --news-projection-version 2`。
这是来源/解析能力选择，不为文章预设重要性或数值信息价值。

新语料模块支持 `build --run <原始目录> --panel <原面板文件> --result-archive <原始结果目录>
--output <新文件>`，run/panel 参数可重复。新结果或新解析必须写新文件。

```powershell
python -m football_research.data_adapters.derived_text_corpus verify var/parallel/agent/round185/corpus/corpus.json
python var/parallel/agent/round185/verify_round185.py
```

交付文件：

- `var/parallel/agent/round185/corpus/corpus.json`：完整原轨迹及派生语料。
- `var/parallel/agent/round185/corpus/qualification-report.json`：六场逐场资格和缺口。
- `var/parallel/agent/round185/settlement/verification-report.json`：独立赛果状态。
- `var/parallel/agent/round185/settlement/native-score-inspection.json`：比分缺口原文依据。
- `var/parallel/agent/round185/review/final-independent-review.md`：独立审查。
- `var/parallel/agent/round185/acceptance.json`：总验收。

后续优先获取已到期比赛的独立结果、形成真正有标签的语料，再用不同的后续比赛验证
正文预测器和信息价值校准。当前材料不足以训练并证明真实 VoI 优势；本项完成的是
数据处理和资格检查的可执行交付，足量真实监督及有效搜索覆盖仍是未完成的条件。
