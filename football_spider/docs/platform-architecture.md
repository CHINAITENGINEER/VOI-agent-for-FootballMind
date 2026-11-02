# FootballMind 平台结构与数据契约

## 实际新增结构

```text
football_spider/
  main.py                         旧入口，SQLite 兼容保留
  platform_cli.py                  新平台入口，无参数等价 run-once
  platform_config.py / .env.example
  collectors/                     HTTP + aiohttp transport / 来源调度 / 日额度
  normalization/sources.py         纯转换，异常行进入 quality
  models/contracts.py              Pydantic 统一模型 v1.0
  identity/resolver.py             来源 ID、别名候选、人工确认审计
  quality/rules.py                 规则 v1.0
  storage/raw_archive.py           gzip 正文 + 原子提交 observation metadata
  storage/postgres.py              PostgreSQL 事务 / 历史 / 运行状态
  storage/migrations/001_platform.sql
  commands/pipeline.py / sqlite_import.py
  features/store.py / odds_series.py
  scheduler/jobs.py / register-task.ps1
  tools/pg_test_cluster.py / verify_platform.py / probe_sources.py
  tests/                          旧测试 + 契约、数据库、恢复测试
  docs/                           来源、审计、运行、验收说明
  data/                           原 SQLite、cookies、单场 JSON 不迁走
  platform_data/                  原始观测、报告、Parquet、预算、cookies、日志
  .runtime/                       隔离验证 PG 文件（不是生产服务）
```

数据流：Collector → 原子 raw 正文和 metadata → pure normalizer → 质量校验与来源实体解析 → fm 历史层 → PIT 查询 → FeatureStore PostgreSQL + Parquet。数据库离线时仍归档响应并生成 NEEDS_DATABASE 报告；没有自动用 SQLite 冒充 PostgreSQL。

## 原始层与恢复

`platform_data/raw/<source>/<UTC-date>/<UTC-time>_<UUID>.body.gz` 保存还原正文；同名 `.observation.json` 是完成提交标记。保存正文后才保存 metadata，metadata 完成前中断可产生孤立正文，但不会产生指向尚未完成正文的已提交记录。每次真实重试独立 observation，共享逻辑 request_id。checksum 在重放时重新核验。响应不是 JSON 或网络异常也会留档，异常无正文时为空 payload。

参数递归脱敏，端点删除 query/userinfo，不保存请求认证头；供应商回显当前密钥时仅替换该值，并标记 credential_echo_redacted。Cookie 是私密运行数据，不能与 raw/report 一起公开分享。其他非认证业务字段不擅自改写。

档案按 UUID 不覆盖。DB 的 raw_observations 是索引；processing 按 observation + parser_version 保存 OK/PARTIAL/QUARANTINED/FAILED 及次数。事务失败后事实回滚，原始档案保留。重放只读本地，无重新请求；未知 quote match 待比赛身份入库后重试。硬错误被隔离，人工 quality-resolve 只追加处理记录，不覆盖原始事实。

## PostgreSQL 模型

独立 `fm` schema。schema_migrations 用文件名 + SHA256 校验，advisory lock 保证并发 migration；已有 migration 不就地改写。后续变化新增 migration。

- entities 是稳定 UUID 主表；teams/competitions/seasons/players 为按类型过滤视图；matches 为比赛身份表。来源实体拥有不同命名空间。赛程开球修改只增加版本，不改变 match UUID。
- entity_source_map 保留带有效区间的确认版本；team_aliases、mapping_candidates、mapping_audit 记录别名冲突与人工决策。同名仅提供候选。不详球队类别保持 unknown；未知源 ID 采用 provisional / match_scoped 身份，不冒充供应商稳定 ID。
- match_versions 每 observation + parser_version + source_id 唯一，包括阶段比分与时间字段。latest_matches 是每源最新观测；跨源选取由特征显式优先级决定。match_version_intervals 用窗口函数给出 effective_valid_to，不覆盖旧版本；supersedes_id 链表示入库时已存在的前版。乱序回放以 observed_at 选版本，不能让较晚入库的旧观测覆盖新观测。
- odds_history 每次观测追加，包括没有变化的报价；idempotency 覆盖 observation/parser/market/bookmaker/selection/line/price。provider 与 bookmaker 分列，spftll bookmaker 为 null。缺失报价只记缺失问题，不造零。
- team_statistics、team_state_history、player_team_memberships、availability_events、match_events、lineup_versions 定义时间及证据契约；本期仅 team_state_history 写入派生 Elo / form 状态。其余尚无已验证来源写入，不能将表存在当作能力完成。
- quality_issues / quarantined_records / quality_resolutions 可查询硬错误、警告、可恢复异常与处理；pipeline_runs / collector_runs 保留真实计数；feature_snapshots、match_labels 明确分开。

## 时间与标签

所有标准时间使用 aware datetime / PostgreSQL timestamptz，连接返回 UTC；显示调度 Asia/Shanghai，可配置。spftll 原始时间字符串与来源时区保留，未证明来源公开时间时 source_published_at 为 null。比赛时间只是 event_time，不是可见时间。

operational 要求 observed_at 与 ingested_at 同时不晚于 cutoff；防止今天重放旧档案或新解析器悄悄改变系统过去已知事实。historical_reconstruction 只使用可信供应商历史 snapshot_time，加显式 assumed_delay_seconds；football-data.org lastUpdated、当前历史网页、SQLite created_time 都不够作为历史可见依据。快照不足时返回缺失，不降级成另一个模式。

SQLite 导入先归档完整只读快照，按内容哈希复用同一次 import observation，重跑幂等。其事实来源标记 spftll_legacy，低于实时采集优先级；可用时间为导入观测，不向旧比赛日期倒填。原 created/updated 保留为不确定元数据。

标签仅选 finished + regulation + 完整比分；常规 90 分钟与补时，排除加时 / 点球。spftll 试点国内联赛接受 regulation；未验证的杯赛阶段保留 unknown，不产生标签。football-data.org duration 非 REGULAR 时，只接受明确 regularTime，不能直接用 fullTime。revealed_by 是本系统观察到结果的时刻上界，不声称精确终场时间。

## 特征版本 prematch-v1.1

`match_id + prediction_cutoff + feature_version + information_mode + input_manifest_hash` 为唯一键。原始输入版本与 quote ID、raw observation ID、算法参数、模式和假定延迟进入清单；排序后 SHA256。持久化以后不可原地改特征；新的算法使用新版本。开发期 prematch-v1 产物不属于 v1.1 回放实现。

首版计算：固定参数 Elo（初值 1500 / K20 / 主场 65）、最近 5 场积分和进失球、对应主/客场窗口、距最后已观察到的该联赛完赛的天数、同一原始观测/供应商/博彩公司/结算规则下完整 1x2 逆赔率归一概率与陈旧度。后者是市场换算，不是预测模型。目标比赛排除在历史输入之外。

实际旧比赛覆盖稀疏，因此 Elo、form 均带 incomplete 标记和每队已观察场数。休息天数不是跨全部杯赛赛程的完整休息天数。xG / xGA 没有接入来源时为 null + xg_provider_not_integrated，不平均供应商、不造数据。

每份 Parquet 是稳定主键、清单和 feature JSON 列的离线记录，可直接用 pyarrow 读取。SQL 是权威索引；Parquet 导出中断后通过 features --snapshot-id 重导出并验证。特征默认只针对开赛前，取消/延期/已开赛状态被拒绝。原始事件、目标结果、未来赔率未混入更早特征。

`odds-history` 仅推导 first/latest captured 与价格差；true_opening_price、true_closing_price 为 null。开球后才把最后采到的赛前值标注 last_captured_pre_kickoff_price，仍不保证是真实收盘。

## 已知设计限制

当前人工映射决策只影响后续入库，旧版本不被追溯改写以保护历史可见性。错误映射的纠正通过再次确认保留审计；必要时新 parser 版本离线重放，而不是删除过去输入。跨来源对齐还未获得真实第二来源数据验收。

第一个版本采用单机、源内串行限流、观测级事务；不承诺分布式调度。缺 API-Football 真实接入意味着阵容/伤停刷新尚无数据可更新。源端接口无历史查询时（spftll），backfill 只能解析近期响应自带历史，并不会取得任意指定季节的完整比赛。
