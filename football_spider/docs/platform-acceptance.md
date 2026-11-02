# 分阶段实施与真实验收

日期 2026-09-20；本机 Windows；Python 3.11.14。没有概率预测模型、RAG、Agent 协商或大屏。工作区不是 Git repository，因此没有伪造 commit / PR。

总体状态：**基础管道已实现并通过单源真实闭环；多来源/生产自动运行验收 PARTIAL**。缺 `DATABASE_URL`、`FOOTBALL_DATA_API_TOKEN`、`THE_ODDS_API_KEY`；API_FOOTBALL_KEY 也未配置。下面把代码门槛、实测门槛分开，不将未完成来源算作通过。

| 阶段 | 实际修改与理由 | 验证及结果 |
| --- | --- | --- |
| Step 1 | platform-audit.md、data-source-capabilities.md、.env.example、platform_config.py；核对当前文件、29 行 SQLite、已运行 PG16、缺密钥；选择配置 PL/BL1 | 旧测试初始 2 passed；现有 PG 密码认证未改；七来源官方资料与三个公开端点探测有证据。试点完整联赛覆盖尚未验收 |
| Step 2 | storage/raw_archive.py；spiders/base.py 最小归档挂钩；collectors/http.py、aio_transport.py、sources.py、quota.py；保留旧解析与数据库入口 | `main.py --limit 1 --refresh` 真实列表 24 场 / 详情 1 成功 / 0 失败，SQLite 29 行不重复。3 个响应先归档。requests 在详情 host 遇 TLS EOF，新适配使用旧已验证 aiohttp transport；未降低证书校验 |
| Step 3 | models/contracts.py、normalization/sources.py、identity/resolver.py、quality/rules.py；统一时间、身份、比分/盘口语义，硬错误隔离 | null、合法 0:0、非整数/负比分、负盘口、半场冲突、延期/取消、同队、别名类别冲突等通过。未知阶段保留 unknown；xG 保持缺失 |
| Step 4 | football-data.org 真正请求端点、认证、UTC 排他日期窗口、季节与单场；纯 normalizer | 请求/响应契约夹具通过；账号无 key，NEEDS_CREDENTIALS，真实第二源联网、完整赛季、两源同场对齐 **待验收** |
| Step 5 | storage/migrations/001_platform.sql、storage/postgres.py、commands/sqlite_import.py；fm 历史表、约束/索引、事务和只读导入 | 使用已安装 PostgreSQL16 的独立临时集群，migration 重跑成功。真实 SQLite 29→29、唯一数 29、关键字段差异 0，二次 skipped。事务故障夹具回滚及重跑通过。pg_dump→另一隔离 DB 恢复计数和 feature replay 一致。生产 URL 未配置 |
| Step 6 | The Odds API 当前/历史 adapter 与 normalizer；bookmaker、source timestamp、可信 snapshot、追加报价、日 credits/pacing | 第三来源契约与历史假定延迟测试通过，真实联网 **待 key**。已用真实 spftll 相同比赛两次观测证明报价追加与无变化保留；真实报价未变，不声称发生变盘；变化场景只用测试夹具 |
| Step 7 | commands/pipeline.py、platform_cli.py、features/store.py/odds_series.py、scheduler/jobs.py/register-task.ps1；PIT、输入清单、Parquet、CLI、锁/补跑 | 单源真实全链通过；完整配置源 run-once 为 PARTIAL（两源缺 key），非伪成功。原始重放无重复，早时点不受后续消息影响。计划配置生成 / 手动 tick 通过，Windows 永久任务 **未注册** |

## 本次真实端到端证据

权威文件：`platform_data/reports/real-acceptance.json`，每次重跑会更新。共享版真实运行清单：`docs/evidence/platform-real-run.json`（认证字段已省略，仅 configured 布尔值）。

2026-09-20 19:44 最终复验选择的真实比赛：**曼城—桑德兰**，spftll `3003895`，内部 ID `69bb2f11-d1c7-598e-b3ac-0ad84744b65d`，页面开球时间北京时间 21:00。这些是来源字段，非平台预测。

两次详情观测：

- `ee1f85ab-db2f-40ad-841a-3cf2ac0cd7f0`，UTC 11:44:09.294583。
- `eb943dc9-c55c-49fc-baf0-964a574fbb1c`，UTC 11:44:17.635710。

标准记录内容哈希相同；两个原始观测都保留。每次还各请求一份报价，1x2 两次相同：主 1.19 / 平 5.45 / 客 9.5。没有把这次无变动说成变盘；没有把聚合平台编成 bookmaker。

第一次观测后生成特征，再写第二次观测，重算较早 cutoff，feature values 与 input_manifest_hash 均一致。快照 `cb5f2a5c-2f34-404b-a9cd-42cedd6c1183`，manifest hash `be18eb99c529fecbdaa4fbde80230dcbb268bfa1e807771214703c297899f60c`；原清单回放与恢复数据库后的回放均一致。4 个观测重跑：新增事实 0、报价 0、跳过 4。

## 一次完整真实运行

run_id：`5166ef07-629b-4e4d-9c30-c874db3a66bc`；耗时 21.719 秒；显示时区 Asia/Shanghai；机器清单明确 environment=validation。

| 本轮指标 | 实际结果 |
| --- | ---: |
| 外站请求 / 原始观测 | 6 / 6 |
| 新比赛观测版本 | 327 |
| 新报价 | 205 |
| 隔离 / 处理失败 | 0 / 0 |
| 新特征快照 | 36 |
| spftll | OK |
| football-data.org / The Odds API | NEEDS_CREDENTIALS / NEEDS_CREDENTIALS |
| 整轮状态 | PARTIAL |

327 是版本条数，**不是 327 场不同比赛**。本次详情数配置为 2，是为小规模真实闭环验收；默认设置是 10。各 source 的完整 counts 与 scope 位于共享版 JSON。

截至这一隔离验证库结束的查询计数（包括旧数据导入与之前真实档案重放，不是都由上述单轮产生）：715 个比赛身份、2,551 条比赛版本、1,409 条报价、38 份 feature snapshot、66 个原始观测索引、1,072 个版本标签。905 个待确认实体候选；2,573 条质量问题均为警告，其中主要是历史不完整，0 个硬隔离记录。今日比赛 32；计数来自库查询而非示例常量。

## 验证命令和边界

```powershell
.venv/Scripts/python.exe -m pytest -q --junitxml docs/evidence/platform-tests.xml
.venv/Scripts/python.exe -m compileall -q collectors commands features identity models normalization quality scheduler storage tests tools platform_cli.py platform_config.py
.venv/Scripts/python.exe tools/verify_platform.py
powershell -NoProfile -ExecutionPolicy Bypass -File scheduler/register-task.ps1
```

最终回归 **29 passed in 21.14s**，机器结果见 platform-tests.xml。测试覆盖原爬虫回归、原始校验和/脱敏、403/429/500 与 timeout、Retry-After、解析失败、事务失败、幂等/历史修订、SQLite 对账、历史快照与 operational 区分、清单确定性、并发锁、来源局部失败、改期计划、来源 ID 主客关系冲突、真实未变盘口和合成变盘测试分离。

对自动任务的真实核验：`Get-ScheduledTask -TaskName FootballMind-DataPlatform` 无注册项。手动 tick 是隔离 DB 下无到期待执行任务的入口验证，不能据此声称已长期自动运行或已经自动刷新阵容。现有 postgresql-x64-16 服务未改认证；本次临时验证服务均已停止，cluster_path / dump 可从验收 JSON 找到。

## 尚未完成的验收门槛

1. 生产 PostgreSQL 连接凭据与目标数据库，需要用户填入 DATABASE_URL；测试库真实通过不能替代该项。
2. 第二来源真实完整赛程及季节权限；第三来源真实 bookmaker/market/历史快照权限。未得到真正两源同场对齐记录。
3. PL/BL1 的完整历史覆盖、球队映射确认；当前来源样本包含多年前往绩但不构成完整赛季。
4. API-Football 事件/阵容/伤停、Understat xG、ClubElo 外部评级、FBref 当前高级字段尚未实测接入；无伪 collector、无伪生产值。
5. 未注册生产定时任务；默认脚本当前用户登录时才执行，注销后运行需要选定服务账号。

旧 SQLite / 原命令保留。恢复与回滚、配置及所有 CLI 命令见 platform-runbook.md。
