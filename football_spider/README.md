# FootballMind / football_spider

第二轮已接通Understat EPL2025/26真实第二来源、逐场xG/xGA历史和380比赛/760赛前重建快照，保留真实观测/入库时间。公共接口 `football_platform.api` 可从工作区根目录导入；有界指定比赛刷新和冻结预测赛果导出已实现。精确运行命令、验收、数据版本与消费限制见 [第二轮运行手册](../docs/parallel/round2/platform-runbook.md) 和 [公共接口](../docs/parallel/round2/platform-interfaces.md)。生产数据库/API key状态与隔离验证分开报告。

在原采集器基础上新增了原始响应档案、来源适配器、PostgreSQL 版本仓库、质量/实体映射、时点离线特征和 Windows 调度入口。**旧 `python main.py` 与 SQLite 保留兼容；新平台使用 `python platform_cli.py`。**

- [运行与恢复](docs/platform-runbook.md)
- [实际架构与时间契约](docs/platform-architecture.md)
- [来源能力与真实验收边界](docs/data-source-capabilities.md)
- [现状审计](docs/platform-audit.md)
- [阶段验收记录](docs/platform-acceptance.md)

生产 PostgreSQL 连接与供应商密钥请在本地 `.env` 配置。缺配置会如实返回 NEEDS_DATABASE / NEEDS_CREDENTIALS，不会自动切回 SQLite 冒充平台已上线。真实端到端验证报告由 `tools/verify_platform.py` 生成到 `platform_data/reports/real-acceptance.json`。

## 兼容采集器说明

一个只负责采集和持久化的 Python 3.11 足球数据项目，不包含预测模型。接口由
Chrome DevTools Network 与生产 JS 源码共同验证，取证过程和字段映射见
[`docs/site-analysis.md`](docs/site-analysis.md)，机器可读的接口清单见
[`docs/endpoints.json`](docs/endpoints.json)。

## 项目结构

```text
football_spider/
├── main.py
├── config.py
├── requirements.txt
├── spiders/
│   ├── base.py
│   ├── match_list.py
│   └── match_detail.py
├── parser/
│   └── match_parser.py
├── storage/
│   └── database.py
├── models/
│   └── match.py
├── docs/
│   ├── site-analysis.md
│   └── endpoints.json
└── tests/
```

## 安装与运行

```powershell
cd football_spider
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

默认窗口是当天至未来第 7 天（共 8 个自然日）。常用选项：

```powershell
python main.py --list-only
python main.py --limit 2
python main.py --refresh
python main.py --date 2026-09-20 --future-days 7
```

- `--list-only` 只获取列表并写 SQLite。
- `--limit` 用于小规模联调；正式每日任务不要传。
- 同一日期窗口会读取 `crawl_progress` 跳过已完成详情，实现断点续爬。
- `--refresh` 忽略断点，适合主动刷新赔率和状态。

已知站内比赛 ID 时，可以直接保存单场完整快照（通用赛程、详情、赔率、历史
和比赛事件），并同步 upsert 到 SQLite：

```powershell
python tools/fetch_match_snapshot.py 3049657
```

默认输出为 `data/match_3049657.json`。

数据默认写入 `data/matches.db`，日志写入 `logs/spider.log`。路径、并发、超时和
退避参数都可用 `FOOTBALL_*` 环境变量覆盖，具体名称见 `config.py`。

## 稳定性设计

- `requests.Session` 用于列表请求，`aiohttp.ClientSession` 用于并发详情请求；
  两者都在一次任务内复用连接。
- User-Agent 从配置池随机选择，请求前加入随机间隔。
- Cookie 保存至 `data/cookies.json`，下次运行自动恢复。
- 403、429、500、502、503、504 和网络超时执行带抖动的指数退避；尊重
  `Retry-After`。
- 每场详情完成后立即 upsert 并写断点，单场失败不影响其余比赛。
- `matches.id` 是主键，使用 `ON CONFLICT DO UPDATE` 去重。
- SQLite 开启 WAL、busy timeout 和索引。

## SQLite

`matches` 包含需求中的字段：`id`、`match_time`、`league`、`home_team`、
`away_team`、`home_score`、`away_score`、`odds_home`、`odds_draw`、
`odds_away`、`created_time`。另外保存状态、半场比分、让球、更新时间和
`details_json`；后者保留历史、排名及扩展赔率等结构化数据。

每天由任务计划程序运行时，工作目录设置为本项目目录，命令使用：

```text
python main.py
```
