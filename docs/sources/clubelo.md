# ClubElo

核查 2026-09-20。[官网](https://clubelo.com/) 可浏览；本轮访问 /API 跳回主页，没有拿到当前明确 API 合同。旧公开日期 CSV 地址 `http://api.clubelo.com/2026-09-19` 的真实探测返回 HTTP 502、空正文；不要把此失败解释为永久关闭。

样本 `tests/fixtures/collectors/clubelo/public_probe.metadata.json` 保存请求、状态、时间 `2026-09-20T11:00:35.438845Z`、原始 observation ID 与空正文哈希。它是失败证据，不是 Elo 样本。未安装第三方抓取包，也未以非官方教程代替当期接口核验。

预期用途是有日期语义的球队 Elo；日期查询和球队历史获取方式、实际 CSV 列、球队覆盖、排名/国家信息、From/To 是否表示评级适用区间、时区、历史是否会追溯修订均列为待确认，而非已实现字段。日级日期不代表当日任意赛前时刻都已公开。event/valid date 与本次 observed_at 必须分离。

认证、免费/付费边界、请求额度、更新频率本轮未取得足够当前接口文档证据，保留 UNKNOWN。状态 RESEARCH_ONLY / NETWORK_HTTP_502；无生产 collector，无真实 Elo 入库。A 后续先验证官方 API 链接与一份 CSV，再向 B 提交字段字典。
