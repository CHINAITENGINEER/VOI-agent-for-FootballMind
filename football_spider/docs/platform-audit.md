# Step 1 — 2026-09-20 现状审计

实际项目位于 `D:\a1-足彩\football_spider`，不是 Git 工作树。未整体移动目录，旧代码和 SQLite 保留。

旧数据流：`main.py → match_list / match_detail → requests / aiohttp → MatchParser → SQLite upsert + crawl_progress`。
代码含 config、spiders/base.py、parser/match_parser.py、models/match.py、storage/database.py、两个既有测试及 fetch_match_snapshot.py。

数据位置：`data/matches.db` 29 行；`data/match_3049657.json` 是已保存伊朗亚—中国亚比赛聚合结果；`data/cookies.json`；`logs/spider.log`。旧 SQLite 用 ON CONFLICT 更新，details_json 是最新快照，没有字段可见时间或报价历史。创建时间不等于旧赔率的公开时间。

既有测试：Python 3.11.14 环境运行 `python -m pytest -q`，2 passed。旧解析器保留空比分、竞彩让球三项及准确总进球语义；预留 players/lineups/weather 等字段不是已接入能力。

本机 PostgreSQL 16 服务正在运行；未配置 DATABASE_URL，postgres 登录要求密码。未发现应用 pgpass。Docker Desktop 安装但 daemon 未启动。三个供应商密钥环境变量均未配置。不会改已有服务认证或安装另一套生产数据库；集成测试可复用已有 PostgreSQL 二进制在独立临时集群执行，明确区别测试服务与生产就绪。

配置试点：英超 PL、德甲 BL1。spftll SQLite 中仅有英超 4 场、德甲 3 场，不能据此验收完整赛季覆盖。football-data.org 的完整赛程与账号历史深度待凭据验收。

增量顺序：原始档案与旧 HTTP 兼容挂钩 → 类型契约、质量、纯归一化 → football-data.org 适配 → PostgreSQL 历史层与 SQLite 导入 → The Odds API → 时点特征、CLI、锁和 Windows 调度。真实联网与夹具测试分别列证据；不引入模型或网页。
