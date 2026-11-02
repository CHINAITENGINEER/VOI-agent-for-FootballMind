# football-data.org

核查 2026-09-20。官方资料：[v4](https://docs.football-data.org/general/v4/index.html)、[赛事资源](https://docs.football-data.org/general/v4/competition.html)、[价格](https://www.football-data.org/pricing)。与 football-data.co.uk 是不同服务。

## 请求与认证

```json
{
  "url": "https://api.football-data.org/v4/competitions/{code}/matches",
  "method": "GET",
  "headers": {"X-Auth-Token": "[REDACTED]"},
  "params": {"season": "赛季起始年；也可选择 dateFrom/dateTo UTC 日窗口"},
  "response_example": {"matches": "比赛数组；这段是结构说明，不是真实比赛样本"}
}
```

单场为 `GET /v4/matches/{id}`。已实现于旧 `football_spider/collectors/sources.py`；新路径由 E 统一装配。环境变量 `FOOTBALL_DATA_API_TOKEN` 缺失时返回 NEEDS_CREDENTIALS，不发送伪 token。

赛事 matches 文档列出 season/matchday/status/dateFrom/dateTo/stage/group；此端点未见 page/offset，不把其他资源的分页能力泛化。按时间拆分是客户端分区，不叫服务器分页。v4 dateTo 排他；Asia/Shanghai 某日凌晨需扩大到前一 UTC 日，新 `football_data_windows` 已测试 UTC、上海和夏令时边界，B 再按用户本地窗口筛选事件。

## 时间与比分字典

| 字段 | 语义与交接规则 |
| --- | --- |
| id / competition.id / season.id / homeTeam.id / awayTeam.id | 来源稳定实体 ID；名称/简称是标签，不作为跨源唯一 ID |
| utcDate | 带 UTC 时区的开球时间；修改后仍是同一 source match ID |
| status / stage / matchday | 状态、比赛阶段/轮次分别保留；延期取消不是完赛 |
| score.duration | REGULAR / EXTRA_TIME / PENALTY_SHOOTOUT 区分比分口径 |
| score.fullTime | duration=REGULAR 时可对应常规结果；加时/点球情况下不直接当常规比分 |
| score.regularTime / halfTime / extraTime / penalties | 使用来源显式给出的阶段值；不推算缺失；空值不补 0 |
| lastUpdated | 来源对象更新时间，不是每个字段首次公开时间 |
| odds | 非本期保证能力，比赛接口不替代专业历史赔率源 |

今天请求的旧赛果 observed_at 仍是今天。即使有 lastUpdated，也不自动倒填 operational 可用时间。深度数据和历史赛季由套餐、赛事、具体响应决定。

## 范围、额度与真实证据

本轮价格页直连 200：免费层标示 12 赛事、延迟比分/赛程、10 calls/min；ML Pack Light 列出 10 赛季历史。免费层历史深度不按付费说明推定。旧实现以 6.1 秒间隔保守请求；未购买订阅。

真实公开请求 `GET https://api.football-data.org/v4/competitions` 于 `2026-09-20T11:53:57.498724+00:00` 返回 200，count=190。样本 `tests/fixtures/collectors/football_data/public_competitions.json`：PL id=2021、BL1 id=2002，当前赛季均为 2026/27。`numberOfAvailableSeasons` 是目录信息，不是账户获准季数，也不证明每个赛季完整。

验证分层：目录真响应通过；日期/认证头/缺凭据回归通过；认证比赛请求 0 次，真实比赛入库 0，本轮没有两来源同场验收。优先验收 PL 2025 与 2026，BL1 随后按实际权限扩展。
