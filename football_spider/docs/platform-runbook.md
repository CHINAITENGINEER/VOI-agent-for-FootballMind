# FootballMind 运行与恢复

## 安装及配置

PowerShell，固定工作目录，Python 3.11：

```powershell
Set-Location 'D:\a1-足彩\football_spider'
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.lock pytest
Copy-Item .env.example .env
```

已有 `.venv` 已安装依赖；不要重复覆盖已有 `.env`。本次实际验证 Python 3.11.14。用本地编辑器填写 `.env`，不要把密钥贴到共享日志：

- `DATABASE_URL`：现有 PostgreSQL 上为平台准备的数据库连接串，账号需在该库创建 fm schema / 表 / 索引。已有服务运行不是凭据已经通过。
- `FOOTBALL_DATA_API_TOKEN`：能访问 PL/BL1 matches 的账号；历史补录需要账号拥有目标赛季权限。
- `THE_ODDS_API_KEY`：当前 odds 权限；历史快照额外需要付费历史能力。日预算默认 30 credits，历史两联赛三市场需要更多额度时由用户明确调整。
- `API_FOOTBALL_KEY`：仅配置状态检查；本期没有启用 API-Football collector。

`.env.example` 中 keys 为空，没有内置账号。免费/付费选择见 data-source-capabilities.md。直接已有连接即可，不需要安装第二套生产 PostgreSQL。Docker 未运行，当前不依赖 Docker。

```powershell
.venv/Scripts/python.exe platform_cli.py health
.venv/Scripts/python.exe platform_cli.py migrate
.venv/Scripts/python.exe platform_cli.py import-sqlite --path data/matches.db
.venv/Scripts/python.exe platform_cli.py import-sqlite --path data/matches.db
```

health 没有 URL 时返回 NEEDS_DATABASE、退出码 2；URL 有效则应用幂等 migration 并返回状态。旧库以只读方式打开，导入报告含所有旧 ID 到内部 UUID、行数 / 唯一数 / 关键字段差异。重复导入同内容应 reused observation、skipped，不创建新事实。

## 单次执行、补录与重放

```powershell
# 新平台：默认包含配置来源，未来本地今天到 +7 天；缺配置逐源报告
.venv/Scripts/python.exe platform_cli.py
.venv/Scripts/python.exe platform_cli.py run-once

# 旧入口仍保持 requests/aiohttp + SQLite 兼容，并新增原始响应归档
.venv/Scripts/python.exe main.py
.venv/Scripts/python.exe main.py --refresh --limit 1

# 仅 spftll；没有 PostgreSQL 时仍收集并归档，状态不冒充入库成功
.venv/Scripts/python.exe platform_cli.py run-once --sources spftll
.venv/Scripts/python.exe platform_cli.py refresh --sources spftll --match-ids 3003894

# 已配置第二来源后的目标赛季/日期补录
.venv/Scripts/python.exe platform_cli.py backfill --sources football_data --season 2025
.venv/Scripts/python.exe platform_cli.py backfill --sources football_data --start 2026-09-01 --end 2026-09-20

# 第三来源当前报价与真正的历史快照（需要相应账号权限）
.venv/Scripts/python.exe platform_cli.py run-once --sources odds_api
.venv/Scripts/python.exe platform_cli.py backfill --sources odds_api --start 2025-08-01 --historical-at 2025-08-01T10:00:00+00:00

# 原始重放，不请求外站；选源或 observation ID 限定范围
.venv/Scripts/python.exe platform_cli.py replay --sources spftll --no-features
.venv/Scripts/python.exe platform_cli.py replay --sources spftll --observation-ids OBSERVATION_UUID --no-features
```

真实比赛 ID 会随赛程变化，示例是本次已见来源 ID，不是永久测试数据。`--match-ids` 必须配恰好一个来源；跨来源 ID 不混用。spftll 不支持任意季节历史请求，超出范围显示 PARTIAL / COVERAGE_UNSUPPORTED，不用近期响应冒充完整补录。

football-data.org 按 UTC 排他 dateTo 分段；本地日期边缘向外扩展到整 UTC 日，避免漏掉凌晨比赛，仓库可能包含边缘额外数小时。保存完整响应而非截断原始载荷。The Odds API 日期参数只有显式历史快照意义，普通 `--start/--end` 不代表供应商提供任意日期历史。

## 特征与报价序列

```powershell
.venv/Scripts/python.exe platform_cli.py features --match-id INTERNAL_MATCH_UUID --as-of 2026-09-20T11:22:00+00:00
.venv/Scripts/python.exe platform_cli.py features --competition-id INTERNAL_COMPETITION_UUID --season-id INTERNAL_SEASON_UUID --lead-hours 24
.venv/Scripts/python.exe platform_cli.py features --match-id INTERNAL_MATCH_UUID --as-of 2025-08-01T10:01:00+00:00 --mode historical_reconstruction --assumed-delay-seconds 60
.venv/Scripts/python.exe platform_cli.py features --snapshot-id SNAPSHOT_UUID
.venv/Scripts/python.exe platform_cli.py odds-history --match-id INTERNAL_MATCH_UUID --as-of 2026-09-20T11:22:00+00:00
```

所有 cutoff 要带时区。今天导入的数据不会出现在去年 operational 特征中，这是正确结果，不是补录失败。historical_reconstruction 只接受可信快照，不拿 event_time 当公开时间。未知 xG 保持缺失；稀疏历史 Elo 只用于接口和管道验收，不等于已具备可靠训练样本。

## 查询血缘、映射和质量

```powershell
.venv/Scripts/python.exe platform_cli.py status
.venv/Scripts/python.exe platform_cli.py status --source spftll
.venv/Scripts/python.exe platform_cli.py status --run-id RUN_UUID
.venv/Scripts/python.exe platform_cli.py mapping-pending
.venv/Scripts/python.exe platform_cli.py mapping-confirm --source football_data --entity-type team --source-id SOURCE_TEAM_ID --target-id CANONICAL_TEAM_UUID --reason '人工核实国家、球队类别与对阵证据'
.venv/Scripts/python.exe platform_cli.py quality-resolve --issue-id 1 --note '复核原因与处理记录'
```

实体类型 team / competition / season / player / match。不同国家/类别有显式冲突时需人工确认，不静默合并。纠错也是一次新的 mapping-confirm，历史映射和审计保留，不追溯污染旧特征。

常用 PostgreSQL SQL（连接目标库后执行）：

```sql
SELECT * FROM fm.latest_matches WHERE source_id = '3003895';
SELECT id, source, observed_at, metadata FROM fm.raw_observations ORDER BY observed_at DESC LIMIT 10;
SELECT * FROM fm.processing WHERE status IN ('PARTIAL','FAILED');
SELECT rule,severity,count(*) FROM fm.quality_issues GROUP BY rule,severity;
SELECT * FROM fm.mapping_candidates WHERE status='pending';
SELECT match_id,prediction_cutoff,feature_version,input_manifest_hash,manifest FROM fm.feature_snapshots;
SELECT match_id,provider,bookmaker,market_type,selection,line,decimal_price,observed_at FROM fm.odds_history ORDER BY observed_at;
SELECT * FROM fm.match_version_intervals WHERE source_id='3003895' ORDER BY observed_at;
```

## 调度状态与启用

调度由 Windows Task Scheduler 每 15 分钟唤醒 `schedule-tick`，实际抓取任务来自 PostgreSQL 持久表，而不是隐藏在交互终端里的循环。每日 07:30 回补近 2 日赛果并同步未来赛程；赛前 180/60 分钟按最新开球时间规划。spftll 为近期回补，不保证完整；阵容/伤停源尚未接入，不声称已刷新。

锁：Task Scheduler IgnoreNew、进程文件锁、DB 调度 advisory lock。改期以 kickoff 参与 job key，旧未执行任务 superseded；重复新观测不重复建同一任务。日任务可补跑 48 小时；赛前任务有配置宽限且开球后过期。失败重试最多 5 次；源内 Retry-After >60 秒延后到下次运行，不提前突破供应商等待要求。The Odds API 有跨运行 UTC 日 credits 预算。

```powershell
.venv/Scripts/python.exe platform_cli.py schedule-plan
.venv/Scripts/python.exe platform_cli.py schedule-tick
powershell -NoProfile -ExecutionPolicy Bypass -File scheduler/register-task.ps1

# 配置 PostgreSQL、run-once 验收后再启用
powershell -NoProfile -ExecutionPolicy Bypass -File scheduler/register-task.ps1 -Register
Get-ScheduledTask -TaskName FootballMind-DataPlatform
Get-ScheduledTaskInfo -TaskName FootballMind-DataPlatform
Start-ScheduledTask -TaskName FootballMind-DataPlatform

# 仅移除自动任务，不删数据
powershell -NoProfile -ExecutionPolicy Bypass -File scheduler/register-task.ps1 -Unregister
```

本次：**配置生成通过、隔离 PG 下手动 tick 通过（无到期待执行任务）、未注册长期自动任务**。原因是生产 DATABASE_URL 与供应商 key 缺失。默认注册脚本采用当前用户 Interactive principal，仅用户登录时运行；需要注销后运行时，请运营者选定服务账号与凭据，不暗中保存 Windows 密码。注册成功会导出 XML 并查询状态。

## 测试、真实验收和恢复

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe tools/verify_platform.py
```

`verify_platform.py` 复用 `D:\PostgreSQL\bin`（可用 PG_BIN 调整）建立只监听 127.0.0.1 的独立临时验证集群，在随机空闲端口运行；不读取或修改已有 PostgreSQL 的账号和 pg_hba.conf。测试夹具只存在 pytest 临时目录；verify_platform 使用真实 archive 和 SQLite。验证后关闭服务，保留集群、PG dump、raw、report、Parquet 以便检查。

真正运行证据：`platform_data/reports/real-acceptance.json`；`validation/footballmind-real-validation.dump`；其引用的各 UUID report。生产未配置与验证集群通过是两件事。

生产恢复操作先连接**新建的恢复数据库**：

```powershell
# 在已配置好账号的终端，用 PGHOST/PGUSER/PGDATABASE/PGPASSFILE 提供连接；不要把密码放日志
& 'D:/PostgreSQL/bin/pg_dump.exe' -Fc -f 'platform_data/validation/production-backup.dump'
& 'D:/PostgreSQL/bin/pg_restore.exe' --dbname footballmind_restore --no-owner 'platform_data/validation/footballmind-real-validation.dump'
```

恢复 dump 的 DB 引用本地 raw 路径；换机器时也复制整个 platform_data（Cookie 不作共享发布）。无 dump 时先 migrate → import-sqlite → replay 原始载荷。这样可以恢复事实内容，但新库 ingested_at 是重放当天，**不能假装恢复旧 operational 可见时点**；需要精确复原旧时点应恢复保留原 ingested_at 的 PG 备份。

回滚采用新增与切换：停用平台计划任务、保留 fm schema/raw、继续 `main.py` 与旧 SQLite。没有 destructive down migration，未删旧数据。部分导入/处理失败重新运行即可；历史 schema 若修改，应追加 migration，不改已经应用的 checksum。
