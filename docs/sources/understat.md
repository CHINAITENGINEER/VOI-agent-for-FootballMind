# Understat：新 JS 数据入口已真实核验

来源核查2026-09-20；生产接线/隔离仓库验收2026-09-22。官方页面：[EPL 2025/26](https://understat.com/league/EPL/2025)。实际XHR、归一化、版本统计、特征与冻结导出已接通，非猜测API。

## 发现链与请求

1. 既有页面观测 `b7212198-7610-44d4-9f52-4a51dae38c05` 的 `<base href="https://understat.com/">` 与 script src 指向 `js/league.min.js?t=1765269520`。
2. 本轮获取该 JS：HTTP 200、23,961 bytes；证据 `0b866180-92e1-4cc5-9c00-740097129626`。
3. JS 明确 `$.ajax({url:"getLeagueData/"+league+"/"+season,type:"get",dataType:"json",...})`；成功时读取 data.dates / data.teams / data.players。
4. 原页面选项明确 EPL 与 2025。请求实际组合路径，不扩展未证实的 endpoint。

```json
{
  "url": "https://understat.com/getLeagueData/EPL/2025",
  "method": "GET",
  "headers": {"Referer": "https://understat.com/league/EPL/2025", "X-Requested-With": "XMLHttpRequest"},
  "params": {},
  "response_example": {"teams": "按来源 team ID 索引的对象", "players": "赛季聚合球员数组", "dates": "比赛数组"}
}
```

真实响应时间 `2026-09-20T11:59:41.602689+00:00`，HTTP 200，529,950 bytes，原始 SHA256 `d401b18a3386d8168f03727b43b1ab5dc9092e0469e77a1561136baf25151f42`。Content-Type 为 `text/javascript;charset=UTF-8`，正文实际是 JSON：按正文解析，不仅按 MIME 排除。无需 key，本次未发送凭据。

## 实际字段与完整率

`tests/fixtures/collectors/understat/epl_2025.json` 为脱敏重序列化副本，原文与时间证据在 `var/parallel/collectors/understat-research/`。

| 路径 | 源类型 / 含义 / 注意 |
| --- | --- |
| dates[].id；h/a.id/title/short_title | 字符串 ID 与标签；球队 ID 与 teams key 对照；不丢历史简称 |
| dates[].isResult | 布尔结果状态，不映射成复杂的延期/取消状态全集 |
| dates[].goals.h/a | 数字字符串；0:0合法；统一normalizer验证整数非负，未知保持null |
| dates[].xG.h/a | 数字字符串；供应商 Understat，不是实进球/赔率；不可与其他 xG 默认平均 |
| dates[].datetime；teams.*.history[].date | 无时区字符串原样保存并解释为UTC；依据官方calendar.min.js的timezoneOffset转换及跨源开球时间核对；显式标记为有证据的推断 |
| dates[].forecast | 来源派生概率，存在赛后 xG 解释的风险；本阶段不作为赛前概率或赔率输入 |
| teams.*.id/title/history | 每队按场记录：h_a、xG/xGA、npxG/npxGA、scored/missed、ppda/ppda_allowed、deep/deep_allowed、xpts 等 |
| players[] | id/player_name、games/time、goals/xG、assists/xA、shots/key_passes、yellow_cards/red_cards、position/team_title、npxG 等赛季聚合 |

实测 380 个唯一 match ID、20 队每队 38 条 history，380 场均 isResult=true 且双方 goals/xG 有数值，球员数组 537 条。日期覆盖 2025-08-15 至 2026-05-24。这是结构完整率，尚非逐场与第二来源核对的正确率；球员聚合不等同每场出场阵容或转会关系。

该入口无页码参数，一次取得页面选定联赛赛季。页面可见其他联赛/赛季选项不算全部联网验收。随后使用新 RequestTransport 再请求一次，于 `2026-09-20T12:16:13.448854+00:00` 返回 200，仍为 380 场且原始 SHA256 完全相同；第二观测 ID=`5c90a971-37e4-40a1-bd3f-f83bb76d1f29`，证据见 `var/parallel/collectors/transport-live/report.json`。两个短时成功请求不是长期稳定 SLA；公开请求额度仍未知，采用低频显式核验，不启用高频轮询。

## 时间边界与交接

观测时间仍为2026-09-20两次真实时间，不回填到2025/26；数据库ingested_at为2026-09-22实际提交时间。源没有可靠字段级published_at或不可变历史快照说明，历史operational特征全部排除。历史重建单独声明3h赛果、24h xG、168h赛程假设及事后修订偏差。

原始研究档案已显式登记到统一RawArchive，沿用两个原UUID和原始SHA；测试副本未入库。`football_spider/collectors/understat.py` → `normalization/understat.py` → `Repository.process` → `team_statistics` 已实现。同次重放幂等，两次相同载荷保留不同观测/版本；每次380场、1520项xG/xGA，合计760版本/3040统计。源球队history与dates的比分/xG/xGA共3040项核对无差异，结构完整不等于外部逐条正确率。

时区证据 `var/parallel/collectors/round2/timezone-evidence.json`，calendar JS观测 `69874328-a5d6-4e32-af33-f971a909eb2e`。真实数据库/备份恢复/特征消费证据 `round2/acceptance.json`。跨源10队/17场确认依据多日期、多对手、比分/主客/时点，不靠名称近似。

日请求上限20、间隔3秒为本地保守策略，不是供应商承诺配额。每次显式请求至多2次尝试。EPL2025之外仍未验收；537条球员赛季聚合只存原文，阵容和逐射门未启用。不把首次采样当原始发布时间。
