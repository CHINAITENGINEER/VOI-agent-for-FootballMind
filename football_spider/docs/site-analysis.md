# xx.spftll.cn 网站与接口分析

分析日期：2026-09-20（Asia/Shanghai）。本报告同时使用了 Chrome DevTools
Network 事件和生产 JS 源码，接口没有依据路径命名进行猜测。

## 1. 页面结构

- 首页 `https://xx.spftll.cn/` 是部署在阿里云 OSS/CDN 上的 Vue SPA。
- HTML 入口加载 `/js/runtime.25c74f2c.js`、`/js/app.16e80cd9.js` 等资源。
- 路由使用 hash 模式；竞足页是 `/#/jc/dpfas1`，详情页是
  `/#/jc/ssd?matchId=MATCH_ID&gameKind=0`。直接请求 `/jc/dpfas1` 会得到 OSS
  404，不能把前端路由当成后端接口。
- 运行时加载竞足分包 `/js/jc-dpfas1.12a5d65b.js`，详情分包为
  `/js/chunk-2a3c62e8.d959ffe1.js`。

## 2. JS 请求层证据

`app.16e80cd9.js` 中的生产配置是：

```text
baseURL        = https://justpost.haoyun999.cn/api
dataserviceURL = https://dataservice.haoyun666.cn
slbbaseURL     = https://slb.haoyun999.cn/api
```

请求层把逻辑前缀 `/slbapi` 替换为空，再拼到 `slbbaseURL`；把
`/dataapi` 替换为 `/api` 后拼到 `dataserviceURL`。源码中的相关函数为：

```text
GET /slbapi/Game/GetSimpleMatchsAll
GET /slbapi/Game/GetMoreSpInfo
GET /match/GetInfo
```

源码还存在 `GET /match/GetList`，但目标竞足列表页本次没有调用它，所以生产
爬虫没有把它当作列表来源。

后续单场追溯验证确认，不带参数请求
`https://justpost.haoyun999.cn/api/match/GetList` 会返回近期日期分组、完整赛果及
`messageList` 比赛事件。它适合用已知比赛 ID 回溯单场，数据量约数百 KB；每日
增量列表仍使用更小的 `GetSimpleMatchsAll`，以降低请求和解析开销。

## 3. Chrome Network 实测

打开 `/#/jc/dpfas1`，XHR/Fetch 中出现：

```text
GET https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll
    ?dateFormat=&notSingle=false&mKind=0&reqtype=0
200 application/json
```

打开 `/#/jc/ssd?matchId=2929733&gameKind=0`，XHR/Fetch 中出现：

```text
GET https://justpost.haoyun999.cn/api/match/GetInfo
    ?gameKind=0&matchId=2929733
200 application/json
```

Chrome 实际渲染的详情页留档见
[`evidence/network-probe.png`](evidence/network-probe.png)，其中的交锋历史与
`match/GetInfo.data.battleList` 一一对应。

`GetMoreSpInfo` 由竞足分包在展开附加玩法时调用，参数是
`matchId` 和 `mKind=0`。三个接口的完整请求形状及精简响应样例见
[`endpoints.json`](endpoints.json)。

## 4. 返回数据与解析理由

列表接口的 `data` 是日期分组数组，每组的 `list` 才是比赛数组，因此解析时
需要先展开日期分组，再以 `matchId` 去重。主要字段映射如下：

| 源字段 | 统一字段 | 说明 |
|---|---|---|
| `matchId` | `id` | 稳定主键，SQLite 使用 TEXT 以避免源站类型变化 |
| `matchTime` | `time` | ISO 本地时间 |
| `leagueChs` | `league` | 联赛中文名 |
| `homeChs` / `awayChs` | `home_team` / `away_team` | 球队名称 |
| `footMatchState` | `status` | 只命名实测状态 0/-1，未知值原样保留为 `state_N` |
| `spfWin/Equal/LoseFoot` | `odds.home/draw/away` | 胜平负赔率 |
| `goalFoot` | `handicap.line` | 竞彩让球，不与亚洲盘混用 |
| `homeRank` / `awayRank` | `team_rank` | 页面显示的双方排名 |

详情接口直接提供当前比分、半场比分以及三类历史数组：`battleList`、
`homeBattleList`、`awayBattleList`。补充赔率接口给出胜平负、让球胜平负、
准确比分、准确总进球数、半全场赔率。

经实测，接口没有给出亚洲大小球的 `line/over/under`，准确总进球赔率不能冒充
大小球，所以 `over_under` 保持空值，准确总进球数据保存在
`extras.total_goals_exact_odds`。本次响应中也没有红黄牌事件、阵容、球员、天气
或赔率时间序列；模型预留了相应字段，源站将来出现时可以无损扩展。

## 5. 日期范围事实

空 `dateFormat` 是生产页面的真实参数。实测传入 `2026-09-20`、
`2026-09-22`、`2026-09-27`，服务端均返回同一批“当前在售”分组：
2026-09-20 共 27 场，2026-09-21 共 1 场。这说明参数当前被后端忽略或仅作为
兼容字段使用。采集器会对返回结果执行“今天至未来 7 天”的严格过滤；源站未
发布的未来比赛不会被补造。
