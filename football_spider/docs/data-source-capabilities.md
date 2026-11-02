# 数据来源能力与验收矩阵

来源研究2026-09-20，平台验收更新2026-09-22，显示时区Asia/Shanghai。文档能力、实现、样本、联网、入库分别标记。第二轮已接通Understat完整EPL2025/26；最新逐源说明见[来源索引](../../docs/sources/README.md)，实际证据见工作区var/parallel/collectors/round2/acceptance.json。football-data.org与football-data.co.uk仍严格区分，未购买订阅。

| 来源 | 官方资料 / 获取 / 认证 | 覆盖、时间、额度边界 | 当前实现及真实证据 |
| --- | --- | --- | --- |
| spftll | [页面](https://xx.spftll.cn/)，来源 JS / Network 证据见 site-analysis.md、endpoints.json；GET JSON，无账户 key；沿用已取证浏览器头 | 足彩入选近期比赛、详情带各 20 条左右的历史；不是完整联赛赛程。源时间按原页面北京时间解释。赔率未提供可信首次公开时间或 bookmaker。更新 SLA、请求配额未知；本地低频、有限重试 | 旧采集和新适配均真实通过。英超、德甲真实详情、赛果历史、1x2、竞彩三项让球、准确进球数；两次观测追加历史成功。所有观测报告 / 载荷保存在 platform_data。原消息事件保留 raw，不推断未验证的 dynamicKind 牌色和球员归属 |
| football-data.org | [v4 文档](https://docs.football-data.org/general/v4/index.html)、[Quickstart](https://www.football-data.org/documentation/quickstart)、[价格](https://www.football-data.org/pricing)；HTTPS GET；`X-Auth-Token` | 免费 12 项赛事、延迟比分/赛程、10 calls/min。价格页 ML Pack Light 列出 10 赛季历史，不能推定免费有同样深度。试点 PL / BL1，账号季节权限待测。utcDate 为 UTC；lastUpdated 不自动当作最早公开时间。v4 dateTo 排他，UTC 日窗口与本地日范围不同 | collector、纯归一化、比分阶段及请求参数夹具测试通过。`FOOTBALL_DATA_API_TOKEN` 未配：NEEDS_CREDENTIALS；0 个认证联网请求，0 个真实第二来源入库。实际两源同场对齐待凭据，未用夹具冒充 |
| The Odds API | [v4](https://the-odds-api.com/liveapi/guides/v4/)、[历史](https://the-odds-api.com/historical-odds-data/)、[市场](https://the-odds-api.com/sports-odds-data/betting-markets.html)、[更新间隔](https://the-odds-api.com/sports-odds-data/update-intervals.html)；GET；query `apiKey` | 运动/博彩公司/市场分别覆盖；spreads、totals 不是每个足球赛事都保证有。主市场历史从 2020-06 起，早期 10 分钟快照、2022-09 起 5 分钟；付费计划才有历史。历史每 region×market 10 credits。当前主市场通常赛前约 60 秒更新，不等于要每分钟抓取 | 当前及历史端点、h2h/spreads/totals、bookmaker 与供应商分离、快照时点、负盘口、额度与追加存储已实现 / 夹具测试通过。`THE_ODDS_API_KEY` 未配：NEEDS_CREDENTIALS。尚无真实报价入库或付费历史验收 |
| API-Football | [v3 文档](https://www.api-football.com/documentation-v3)、[官方入门说明](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide)、[定价](https://www.api-football.com/pricing)；`v3.football.api-sports.io`；`x-apisports-key` | 比赛、球队、统计、积分榜、阵容、伤停与事件按 league/season coverage 分项。免费每日额度有限，当前价格页为 100/day；历史赛季权限需账号实测。bookmakers 是目录，不是赔率；真实 odds 是另一接口。分页、剩余额度及更新周期依端点 | 仅研究，未创建空 collector。`API_FOOTBALL_KEY` 未配，没有凭据优势因此保持第三来源优先 The Odds API。伤停/阵容等仓库契约存在不代表数据来源已接入 |
| ClubElo | [官网](https://clubelo.com/)，历史公开 CSV 地址 `http://api.clubelo.com/YYYY-MM-DD` 需当期验证 | 日期评级的生效/有效区间、球队覆盖、历史可见性、访问配额均尚待当前端点确认；日级日期不代表当天任意赛前时刻可用 | 实测 2026-09-19 CSV 请求 HTTP 502 / 空正文，已归档。官网可浏览，但不把旧端点视为已通过。只保留研究记录，无 collector |
| Understat | [官网](https://understat.com/)、[EPL2025](https://understat.com/league/EPL/2025)；官方页面league.min.js给出GET getLeagueData/EPL/2025 | 20队380场，xG/xGA独立provider/definition；公开时间未知。UTC按官方calendar脚本及跨源时点证据解释。源没有历史公开快照，历史重建需延迟假设 | 已有两次真实响应→公共RawArchive→每次380比赛版本/1520统计→760逐场特征→基线消费。10队17场跨源确认；原研究observed_at保留，实际ingested_at不倒填。球员聚合只存raw，阵容/逐射门未启用 |
| FBref | [英超页面](https://fbref.com/en/comps/9/Premier-League-Stats)、[2026-01-20 官方公告](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/) | 官方公告说明供应商高级足球数据已移除；当前射门/控球/传球/防守/xG 每字段都要重新验收，旧清单不沿用。当前页面访问限制和完整度未验证 | 本机页请求 HTTP 403，已保存 HTML；挑战页字符串出现 xG 不算字段存在。只有重新验收记录，无 collector、无虚构高级数据 |

定价与政策页偶有套餐限额差异，以账号响应 / 当前套餐为准。本实现 football-data.org 按免费层 6.1 秒间隔保守执行。The Odds API 每次请求预留 credits，跨进程保存在 UTC 日预算文件，未知失败不自动退还预留预算。

## 已实现请求契约

```json
[
  {"url":"https://api.football-data.org/v4/competitions/{code}/matches","method":"GET","headers":{"X-Auth-Token":"[REDACTED]"},"params":{"season":"起始年，或 dateFrom/dateTo UTC 范围；dateTo 排他"},"response_example":{"matches":"按官方 schema 的数组；非本地真实样本"}},
  {"url":"https://api.football-data.org/v4/matches/{id}","method":"GET","headers":{"X-Auth-Token":"[REDACTED]"},"params":{},"response_example":{"id":"来源 ID","score":{"duration":"REGULAR / EXTRA_TIME / PENALTY_SHOOTOUT"}}},
  {"url":"https://api.the-odds-api.com/v4/sports/{sport}/odds","method":"GET","headers":{},"params":{"apiKey":"[REDACTED]","regions":"uk","markets":"h2h,spreads,totals","oddsFormat":"decimal","dateFormat":"iso"},"response_example":"事件数组 → bookmakers → markets → outcomes；仅 schema 描述"},
  {"url":"https://api.the-odds-api.com/v4/historical/sports/{sport}/odds","method":"GET","headers":{},"params":{"apiKey":"[REDACTED]","date":"UTC ISO 时间","regions":"uk","markets":"h2h,spreads,totals","oddsFormat":"decimal"},"response_example":{"timestamp":"不晚于请求时间的快照","previous_timestamp":"上一个快照","next_timestamp":"下一个快照","data":"事件数组"}}
]
```

spftll 真实接口详见既有 endpoints.json，另已验证 `GET https://justpost.haoyun999.cn/api/match/GetList`，无参数，返回 `data[].matchList` 与原始 messageList；只代表最近数据，**不是任意日期历史接口**。

## 真实验收范围与缺口

- 最终运行明细以 `platform_data/reports/real-acceptance.json`、`source-probes.json` 和 UUID 命名的各运行报告为准，避免静态示例计数伪装运行。
- spftll和Understat已完成真实原文→归一化→PostgreSQL→特征/消费。详情往绩的首次观测仍是实际抓取时间。
- 完整赛季试点为EPL2025/26；BL1完整覆盖待验收。未知球队/赛季/杯赛比分阶段继续明确保留。
- FD/Odds认证历史深度、bookmaker/市场覆盖仍待key。Understat/spftll两源已有10队17场证据确认；单次名字近似不强制合并。
- 联网探测失败可能是本机连接条件或供应商限制，不推导为来源永久下线。
