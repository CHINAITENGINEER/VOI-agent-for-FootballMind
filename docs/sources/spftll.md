# spftll：真实 XHR 与字段交接

核查 2026-09-20。来源：[页面](https://xx.spftll.cn/)。已存在的 Chrome Network / JS 取证保留于 `football_spider/docs/site-analysis.md`、`endpoints.json`、`docs/evidence/`，本轮不重复开发旧采集器。

## 请求

| GET 端点 | 参数 | 数据容器 / 真实样本 |
| --- | --- | --- |
| `https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll` | dateFormat="", notSingle="false", mKind=0, reqtype=0 | data[].list；spftll/list.json |
| `https://justpost.haoyun999.cn/api/match/GetList` | 无 | data[].matchList，附带 messageList；spftll/recent.json |
| `https://justpost.haoyun999.cn/api/match/GetInfo` | gameKind=0, matchId=来源 ID | data；spftll/detail.json |
| `https://slb.haoyun999.cn/api/Game/GetMoreSpInfo` | mKind=0, matchId=来源 ID | data.footballMoreSpInfo；spftll/odds.json |

响应 JSON，根 `code=0` 才是业务成功。headers 沿用取证结果：Accept、Referer=https://xx.spftll.cn/、X-Requested-With=XMLHttpRequest、XMIME-TYPE=3、AppType=0、DeviceKind=1、随机浏览器 User-Agent；旧 token/rkey 为字符串 null，不当作登录凭据。不存在新增密钥要求。完整实际请求 params 在每份 metadata 的 observation 中。

## 给 B 的字段字典

| 路径/字段 | 源类型与含义 | 解析注意 |
| --- | --- | --- |
| matchId；homeId/awayId；leagueId | 整数稳定来源 ID | 内部 ID 由 B/C 映射；列表可能缺 leagueId，不用名称强合并 |
| matchTime / matchtime | 不含 offset 的 ISO 字符串 | 按既有页面北京时间解析，保留原文/Asia/Shanghai；不是本系统首次可见时间 |
| startTime、lastUpdateTime | 字符串/null | 不等同已证实的开球时间/公开时间；样本 startTime 有明显不同值，先保留原始 |
| homeChs/awayChs/leagueChs；home/away/league | 来源中文标签，不同端点命名不同 | 含青年队等后缀；不去掉类别后强合并 |
| footMatchState、finishedTitle | 数字/文字；实测 0 未开、-1 完场 | 其他枚举保留未知；取消、延期、加时阶段需逐例核验 |
| homeScore/awayScore、homeHalfScore/awayHalfScore | 整数/null | 0:0 合法；null 不补 0；杯赛 full/regulation 尚缺完整阶段证据 |
| battleList/homeBattleList/awayBattleList | 历史比赛数组 | 是今天看到的历史，不是当时可获得的版本；非完整联赛样本 |
| homeScheduleList/awayScheduleList | 近期后续赛程数组 | 不保证固定未来 7 日或全部比赛 |
| homeRank/awayRank；homerank/awayrank | 排名字符串 | 时变事实，不是 team 静态属性；缺少来源公布时间 |
| spfWinFoot/spfEqualFoot/spfLoseFoot；spf_win/spf_draw/spf_lost | 当前十进制 1X2 报价 | provider=spftll；bookmaker 未提供，必须 unknown/null |
| goalFoot / rfspf_goal 与 rspf*Foot / rfspf_* | 竞彩三项让球与对应报价，line 可为负 | 不是亚洲二项让球；home/draw/away 三选项 |
| t0…t7 | 准确总进球数玩法（末档应保留来源标签） | 不是大小球 over/under，无大小球盘口线 |
| sw*/sd*/sl*；ht* | 比分 / 半全场玩法报价 | 按源标签解释，不造二项市场或跨周期拼接 |
| messageList | 近期事件原文 | 原始可保留；dynamicKind 与牌色/球员关系尚未完成逐项验证 |

请求频率/SLA 没有公开承诺；旧平台采用低频、请求上限与有限重试。dateFormat 曾实测不同日期返回同批在售比赛，因此不是历史日期查询 API。不存在可确认的服务端分页参数，不添加 page。

## 重复观测和失败

`repeated_detail_1/2` 是同一 source matchId=3003895 两次真实请求：11:44:09.294583Z 与 11:44:17.635710Z，原始 SHA256 相同但 observation ID 不同。样本没有变价，不宣称真实价格变动。`transport_failure` 保存旧请求 SSLError 及空正文；该网络条件下旧 aiohttp 传输成功，未关闭 TLS 验证。

第二轮公共RawArchive接线、领域RefreshService、PG追加历史、原始重放和旧入口回归已通过，完整测试见round2/platform-tests.xml。18场有界赛果请求13场有效完赛，5场真实半场比分超过全场而隔离；前瞻18场名单为12有效+6pending。另有Agent适配器对3012999单请求成功，真实观测3b3ff4de-5361-4b6a-92c3-8109b18be48b。

code=0,data=null表示NO_DATA，非零code为PROVIDER_ERROR；失败/空载荷都先归档。原空数据调用报告保留，后续可零请求重放重新分类，不篡改请求发生记录。阵容、天气、完整球员数据、完整历史赔率仍未通过；字段预留不算接入。
