# Understat 邻近赛季补录与时间方向计划

核验日期：2026-09-22。职责：第三轮平台的独立、有界来源补录；不训练研究模型、不修改消费者、不启动数据库。此前 EPL2025/26 和已看过的 290 场测试结果继续作为探索材料，原冻结文件全部保留。

## 已取得的真实证据

本轮以 **1 个逻辑请求、2 次网络尝试**取得 EPL2024/25 完整响应。首次 `SSLError` 和重试后的 HTTP 200 均先进入正式 RawArchive；未降低 TLS 校验。随后 `--offline` 重跑为 **0 网络请求**。

接口不是按惯例猜测：

1. 已有官方页面观测 `b7212198-7610-44d4-9f52-4a51dae38c05` 的联赛选择为 EPL，赛季选择项明确包含 `<option value="2024" >2024/2025</option>`。
2. 页面指向 `js/league.min.js?t=1765269520`；既有 JS 真实档案 `0b866180-92e1-4cc5-9c00-740097129626` 的原文明确 `url:"getLeagueData/"+league+"/"+season,type:"get",dataType:"json"`。
3. 工具执行前重新校验 HTML 和 JS 原始哈希以及这些内容，再请求 [官方 EPL2024 数据路径](https://understat.com/getLeagueData/EPL/2024)。

| 项目 | 真实值 |
| --- | --- |
| endpoint / method | `https://understat.com/getLeagueData/EPL/2024` / GET |
| headers | `Referer: https://understat.com/league/EPL/2024`; `X-Requested-With: XMLHttpRequest` |
| 请求参数 / 认证 | 无 query；未发送 API key |
| 成功 observation_id | `843eefe4-1c04-45c4-a3ae-b319bf6096e9` |
| 两次尝试共同 request_id | `607f90f0-75e1-4d8a-b164-c63b111301e3` |
| 成功 observed_at | `2026-09-22T09:50:12.613328+00:00` |
| 原文 SHA256 | `5c90d05b8d0bec0cfd848d6393a51e1194225cf772eab2b40408c222aa3987d4` |
| 原文长度 | 538,474 bytes |
| 失败 observation_id | `cbb9e70a-2989-4378-b2df-66c7fa51b2f4`，SSLError，HTTP status=null |
| 生产原始正文相对路径 | `raw/understat/2026-09-22/095012_613328_843eefe4-1c04-45c4-a3ae-b319bf6096e9.body.gz` |
| 初次真实运行证据 | `var/parallel/collectors/round3/adjacent/runs/175beab6-4159-4511-a225-d87031088694.json` |
| 零请求复用证据 | `var/parallel/collectors/round3/adjacent/runs/45e3ae28-def5-4c1d-98bb-72a9dacf6b98.json` |

正式 raw root 仍为 `football_spider/platform_data`。JSON 报告和原文均不是测试夹具。`report.json` 记录最近一次命令结果，`runs/<run_id>.json` 保留各次执行历史，因而 0 请求复用没有覆盖首次请求证据。

## 完整率与字段检查

成功正文包含 `dates`、`teams`、`players`。直接复用既有 `audit_season`、`normalize_understat` 做只读预检：

- 380 个唯一比赛 ID，380 对不同主客关系，没有重复比赛 ID；20 队每队恰好 38 场，符合双循环结构。
- 380 场 `isResult=true`；比分整数非负，0:0 保持合法；归一化预检得到 380 个比赛记录、1,520 条 Understat xG/xGA 统计、0 个质量 issue。
- `dates[].goals/xG` 与每队 `history[].scored/missed/xG/xGA` 共 3,040 项校验一致。此为同一供应商内部一致性，不等同外部独立正确率。
- 开球原始范围 `2024-08-16 19:00:00` 至 `2025-05-25 15:00:00`，均处于来源声明的 2024/25 赛季。
- `players` 有 562 条赛季聚合记录；它们不是逐场阵容/当时可见的球员状态，不作为赛前信息动作输入。
- `forecast` 和最终赛季汇总排除于特征。xGA 明确为对方同场 xG，供应商/定义版本不与其他来源混合。

`context` 已包含 `league="EPL"`、`season=2024`、`kickoff_timezone="UTC"`、`timezone_policy` 和发现链。时区继续使用第二轮官方 calendar JS 的转换证据（观测 `69874328-a5d6-4e32-af33-f971a909eb2e`），属于明确记录的推断，原始无时区字符串保留。

该命令不打开数据库，报告 `ingested_at=null`、`ingestion_status=NOT_PERFORMED_BY_PROBE`。平台主任务可用上述真实 observation 交给现有 `Repository.process`，入库时间必须取实际提交时间，再单独冻结规范化赛季及逐场时点快照。此处的归一化预检不冒充数据库验收或 COMPLETE 导出。

## 时间与费用边界

`source_published_at` 没有可靠来源字段，保持 null；本次实际观察发生在 2026-09-22，不回填为 2024/25。

- `operational`：这些历史比赛在其当年赛前不曾被系统观察，故这次载荷不产生当年合法 operational 赛前快照。
- `historical_reconstruction`：若沿用第二轮定义，必须显式保留赛程提前 168 小时、赛果开球后 3 小时、xG 开球后 24 小时的假设，以及改期/赛后修订偏差。这些是实验可见性假设，不是本次 response 证明的原始发布时间。
- 每场 cutoff 下只使用该场以前、且按相应模式已可用的结果/xG；目标比赛赛果、xG、来源 forecast 与全季汇总不进入当场特征。

费用为未知：`money=null`、`currency=null`；无公开、已核验的供应商免费请求额度。本次实际请求数 2、重试数 1、耗时约 6.641 秒。transport 的 `credits_reserved=0` 仅表示未启用虚构的供应商 credit 扣减，不表示价格已确认 0。20 次/日、3 秒间隔是已有本地保守限额，不是供应商配额。工具外层请求预算 4，当前唯一逻辑请求最多 2 次尝试；本轮没有持续轮询。

## 独立数据计划：遵守前后时间方向

| 数据 | 时间角色 | 可做的研究 | 不应宣称 |
| --- | --- | --- | --- |
| EPL2024/25，本次补录 | 早于已看过的 2025/26；训练/内部向前验证候选 | 训练部分信息预测器与 gain 估计器；按比赛日期向前交叉拟合、内部验证；为 2025/26 提供合法更早历史热身 | 不是训练在 2025 后、再向 2024 回测的“未来未见测试” |
| EPL2025/26，原 380 场与已看过的 290 测试结果 | 已披露探索集 | 比较信息组、成本敏感性和停止规则；记录所有基于结果进行的调整 | 即使改名、换 seed、换成本表也不是独立确认集 |
| 后续预注册时间段 | 严格晚于预注册、训练截止点和冻结策略版本 | 预先冻结预测器/策略/成本/预算/主指标/排除条件；目标赛前封存预测和获取轨迹，再收集赛果评分 | 预注册前已结束或已看过结果的比赛不是前瞻确认样本 |

建议研究所有者在正式确认开始前，写出唯一注册文件：UTC 注册时间、允许的联赛/开球日期窗口、样本停止条件、训练数据版本/截止时间、策略和部分预测器哈希、成本向量与权重、信息动作目录版本、随机种子、主损失与比较、按比赛/时间块的统计计划、缺失及隔离规则。未来窗口可以来自已运行来源的后续赛程，不以再接入新网站为前置条件；具体日期与规模由研究所有者预先冻结。

2024/25 的新增标签已经进入平台持有范围，本轮只报告覆盖与质量，不用它做确认性效果选择。训练与内部验证由研究所有者实施；平台不改其训练代码或评分结果。

## 可重复命令

在 `D:\a1-足彩` 执行：

```powershell
.\football_spider\.venv\Scripts\python.exe football_spider/tools/round3_adjacent_probe.py --offline
```

去掉 `--offline` 时仍先复用已有符合 endpoint/context/status 且哈希通过的真实档案；只有不存在时才执行有界请求。工具文件为 `football_spider/tools/round3_adjacent_probe.py`，不依赖新 collector、CLI、DB 服务或 API 密钥。父平台任务负责随后入库、独立 COMPLETE 冻结以及消费者交接。
