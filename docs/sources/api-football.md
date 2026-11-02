# API-Football

核查 2026-09-20。官方：[v3](https://www.api-football.com/documentation-v3)、[2026-03-13 官方入门指南](https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide)、[套餐](https://www.api-football.com/pricing)。本机指南/价格直连 403，网页工具能读取官方指南及价格；这不是数据 API 403 证据。

状态：研究与独立分页策略测试，无生产 collector；`API_FOOTBALL_KEY` 未配置，数据 API 请求 0 次。未创建空实现，没有依据把它提前到 The Odds API 之前。

| 请求 GET（base=https://v3.football.api-sports.io） | 用途 / 参数 / 验收重点 |
| --- | --- |
| /leagues | country、id、season、current 等；先核对 seasons 与 coverage，不只数联赛总量 |
| /fixtures | league+season / date / id 等；fixture.id/date/timestamp/status，goals 与 score 分阶段 |
| /fixtures/statistics / fixtures/players | fixture，球队/球员赛后统计，注明 available time |
| /fixtures/events / fixtures/lineups | fixture；事件/阵容，不能凭未来数据填早期快照 |
| /standings / teams/statistics | league/season（后者含 team）；统计窗口与赛季终榜不混入赛前 |
| /injuries | league/season、fixture/team 等实际支持组合；伤停原始状态不是影响分数 |
| /odds/bookmakers | bookmaker ID/名称目录，无价格，不算 odds 接入成功 |
| /odds | fixture、league/season、date、bookmaker、bet、page；真实赛前报价 |
| /odds/live | 当前现场赔率；与赛前报价分离 |

认证：直连官方服务 header `x-apisports-key: [REDACTED]`。RapidAPI 账户是另一认证方式，未用一个 key 盲试多个平台。返回 envelope 包含 get/parameters/errors/results/paging/response；HTTP 200 也先查 errors，再区分空 response 与错误。

官方定价免费 100 请求/日、所有端点但赛季范围受限；账户实际限制以响应为准。每日与每分钟额度分别读 `x-ratelimit-requests-remaining` / `X-Ratelimit-Remaining`。分页按 paging.current/total 前进，检测重复页并有上限；不要将相同规则套给 football-data.org。

官方指南描述赛前赔率保留最近 7 日、常见提前 1–14 日、约 3 小时更新，均不是多年逐次报价档案。应自行持续归档并逐赛事检验；未联网确认前只标为文档能力。fixture 的时间、比分阶段、阵容公布时间、事件修订与来源更新粒度仍需真响应确定。伤停/阵容更新计划最终由 E 根据 coverage 和额度配置。
