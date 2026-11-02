# The Odds API

核查 2026-09-20。官方：[v4](https://the-odds-api.com/liveapi/guides/v4/)、[历史说明](https://the-odds-api.com/historical-odds-data/)、[市场](https://the-odds-api.com/sports-odds-data/betting-markets.html)、[套餐](https://the-odds-api.com/)、[更新间隔](https://the-odds-api.com/sports-odds-data/update-intervals.html)。历史文档本机 200；认证 key 未配置。

## 已实现的请求

```json
{
  "url": "https://api.the-odds-api.com/v4/sports/{sport}/odds",
  "method": "GET",
  "headers": {},
  "params": {"apiKey": "[REDACTED]", "regions": "uk", "markets": "h2h,spreads,totals", "oddsFormat": "decimal", "dateFormat": "iso"},
  "response_example": "事件数组 -> bookmakers[] -> markets[] -> outcomes[]；仅结构说明"
}
```

历史端点 `GET /v4/historical/sports/{sport}/odds` 另加 UTC ISO `date`；返回 timestamp/previous_timestamp/next_timestamp/data。时间是“不晚于请求时点的最近可用快照”，不是保证精确命中任意秒。原先起止比赛日期不等同历史赔率补录日期。

`GET /v4/sports?apiKey=...` 是运动目录，官方说明不扣使用 credits，但仍需 key；目录不是报价验收。E 后续启用时先检查 soccer_epl / soccer_germany_bundesliga 是否存在；不得默认所有来源覆盖同赛事。没有此类 odds 端点的页码参数；事件过滤 eventIds 和时间过滤不是翻页。

## 给 B 的报价字典

| 字段 | 规则 |
| --- | --- |
| id, sport_key, commence_time | 赛事来源 ID、运动/联赛键、开球 UTC；不使用球队名称替代 ID |
| home_team/away_team | 标签；队伍来源稳定 ID 可能未提供，待实体候选确认 |
| bookmakers[].key/title | 实际 bookmaker；provider 才是 odds_api |
| bookmakers[].last_update / markets[].last_update | 来源更新时点；保留粒度和来源路径，不假设为真实开盘时间 |
| markets[].key=h2h | 足球三项须检查双方与 Draw 完整、同 bookmaker/周期/快照 |
| spreads / outcomes[].point | 带盘口报价；负数合法，不同 selection 有各自 line；不可混成竞彩三项让球 |
| totals / Over / Under / point | 大小球及盘口；不是准确总进球数市场 |
| outcomes[].price | 请求明确 decimal，缺失/null/<=1 等交 B 的语义规则；不把美式负赔率当十进制 |
| timestamp | 历史快照时点，与本次 observed_at 分开；historical_reconstruction 需记录来源依据及假定延迟 |

结算阶段依实际 market key 与 bookmaker 规则核验。默认市场名并不自动证明杯赛是否包括加时；未证明的周期/结算规则保留未知，不与常规 90 分钟 1X2 混合。

## 额度、历史与状态

官网免费 Starter 为 500 credits/月且不含历史。历史主市场自 2020-06-06 起，早期 10 分钟、2022-09 起 5 分钟快照；具体足球联赛可能更晚出现。当前预算保守按 market×region，历史乘 10；实际计费仍看 x-requests-last/used/remaining。每次重试均应先预留预算，未知失败不自动退款；新 HTTP 组件已测此行为。旧 collector 对付费请求每次只试一次，避免隐性重试扣额。

没有 key：THE_ODDS_API_KEY / NEEDS_CREDENTIALS。既有请求和合成报价/阶段测试不等于真实联网；本轮无认证请求、无真实该来源报价。不为了覆盖数而创建伪 bookmaker 或伪变价样本。spreads/totals 在目标足球联赛的实际覆盖仍待账号实测。
