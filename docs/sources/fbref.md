# FBref：重新验收，不沿用旧高级字段清单

核查 2026-09-20。[英超页](https://fbref.com/en/comps/9/Premier-League-Stats) 既有本机请求返回 403 挑战页。样本 `tests/fixtures/collectors/fbref/public_probe.txt` 只保留可见提示，脚本、属性与挑战 token 已排除，原始响应仍保留。403 页面中的文本不作为足球字段证据。

[2026-01-20 官方公告](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/) 本机直连本轮同样 403；通过[官方 General 分类归档](https://www.sports-reference.com/blog/category/general/) 核查到相同公告，说明原供应商高级足球数据已被移除。文档可核查与数据页直连状态分别报告。

| 候选字段 | 本轮实际验证状态 |
| --- | --- |
| 基础赛程/比分/球队 | 当前数据页未取得，待重新实测 |
| 射门、射正 | 当前表格/列未验证 |
| 控球、传球 | 当前表格/列未验证 |
| 防守动作 | 当前表格/列未验证 |
| xG/xGA、高级逐射门质量 | 原供应商移除后的可用性未验证，禁止沿用旧列表宣称存在 |

现状是研究/失败审计，没有生产 collector。Stathead 订阅不自动代表数据 API 权限。认证、收费边界、字段覆盖、更新周期、请求额度及历史版本可得性需要按当前实际页面与条款分别核验。旧比赛页面今天下载也只能取得今天的 observed_at；没有公开时间/快照依据时不倒填历史可用时间。

403 不触发无限重试，不处理挑战绕过。本来源暂不作为赛前 xG 依赖，Understat 已取到的真实字段也不会冒充 FBref 字段。
