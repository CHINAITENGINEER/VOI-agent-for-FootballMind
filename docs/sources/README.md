# 任务 A：数据来源能力与验收索引

来源研究核查2026-09-20，接线验收更新2026-09-22。平台角色合并后，公共协议/RawArchive/仓库/装配已实际完成；不再等待B/C/E。测试副本仅测试，Understat正式导入来自保留原观测时间与哈希的真实研究档案。

## 当前结果

| 来源 | 代码/样本状态 | 真实联网范围 | 尚待完成 |
| --- | --- | --- | --- |
| [spftll](spftll.md) | 旧能力保留；公共归档/领域刷新已入库与幂等验收 | 列表/详情/当前竞彩赔率；18场有界赛果刷新13有效、5比分矛盾隔离 | 完整赛季、阵容、天气仍未验证；非完整历史赔率服务 |
| [football-data.org](football-data-org.md) | 既有比赛请求实现；日期/认证头/缺凭据行为回归通过 | 本轮公开 competitions 目录 200；190 项目录，含 PL/BL1。不是 190 项账户权限 | FOOTBALL_DATA_API_TOKEN；真实比赛、历史深度、两源同场对齐 |
| [The Odds API](the-odds-api.md) | 既有当前/历史请求实现；新传输预算和错误测试通过 | 文档核查；本轮 0 次认证请求 | THE_ODDS_API_KEY；实际足球市场/bookmaker 覆盖，历史需相应套餐 |
| [API-Football](api-football.md) | 文档研究 + 独立分页策略测试；未放空 collector | 官方网站直连 403；官方指南和价格可通过检索核查；0 次数据 API 请求 | API_FOOTBALL_KEY；coverage、赔率、阵容/伤停/事件逐项联网 |
| [Understat](understat.md) | collector/normalizer/质量/版本统计/时点特征/冻结导出已通过 | EPL2025/26两次响应380场、20队；JS与跨源证据解释UTC，3040项history字段对照一致 | 逐射门/阵容/其他赛季未启用；历史重建有延迟假设和事后修订风险 |
| [ClubElo](clubelo.md) | 研究/真实失败样本；无空 collector | 既有日期 CSV 请求 502，正文为空；官网可浏览 | 当前 API 文档和 CSV 日期/有效区间重新核验 |
| [FBref](fbref.md) | 重新验收记录；无空 collector | 既有英超页面 403；本轮公告直连 403，官方分类归档可核查公告 | 射门/控球/传球/防守/xG 逐字段复验；不沿用旧高级字段清单 |

主接入顺序仍为 spftll → football-data.org → The Odds API。API-Football 没有已配凭据优势，不调整顺序。Understat 是缺凭据期间推进的独立公开来源核验，不是替代第三来源的赔率能力。

试点优先 EPL：Understat 已提供结构完整的 2025/26 历史切片；spftll 有英超近期记录；football-data.org 的 PL 目录存在但比赛权限待测。BL1 保留第二候选，其完整赛季尚未验收。不同来源赛季、时区、球队类别不能仅按显示名称自动对齐。

## 证据与代码

- 样本目录：`tests/fixtures/collectors/`，总清单 `manifest.json`。
- 样本元数据包含原始观测/研究证据 ID、实际 observed_at、原正文与脱敏副本分别计算的 SHA256。重复内容仍保留两个真实 ID 和两个时间。
- 本轮原始研究证据：`var/parallel/collectors/public-verification/` 和 `understat-research/`；不属于生产 RawArchive。失败 HTML、空正文和错误类型均保留。
- 新组件：`football_platform/collectors/http_transport.py`、`policy.py`、`envelopes.py`。HTTP 层只返回 bytes 和调用采集事实挂钩，不定义第二套 RawObservation、实体或标准比赛模型。
- 生产适配器继续在 `football_spider/collectors/`，公共类型 `football_spider.models.contracts.Observation`，门面 `football_platform.api`；未复制另一平台。
- `on_attempt(**facts)` 是内部HTTP接线点，Understat与RefreshService已绑定公共RawArchive，先归档后解析；无挂钩会拒绝构造。
- 有限重试、退避、Retry-After、会话/Cookie、敏感头与正文回显脱敏、每次尝试请求与 credits 预算、原始先落盘后验包行为均有测试。持久跨进程预算通过 `reserve` 注入；旧实现已具备原子预算文件，新组件不另建持久配额库。
- 原始正文若包含请求密钥回显，会定点脱敏并标记 `credential_echo_redacted`；其他响应内容不修改。共享 HTML 样本只保留可见文本，原文仍在本机证据区。

## 可执行验证

从工作区根目录运行：

```powershell
football_spider/.venv/Scripts/python.exe -m pytest tests/platform/collectors --confcutdir tests/platform/collectors -q --junitxml=var/parallel/collectors/tests.xml
football_spider/.venv/Scripts/python.exe tests/platform/collectors/export_evidence.py
football_spider/.venv/Scripts/python.exe tests/platform/collectors/audit_samples.py
```

导出/审计没有外部请求、数据库写入。Understat 联网研究是显式操作，每次以下命令只请求一个资源；不要将其注册为日常调度：

```powershell
football_spider/.venv/Scripts/python.exe tests/platform/collectors/inspect_understat.py
football_spider/.venv/Scripts/python.exe tests/platform/collectors/inspect_understat.py --verify-data
```

第一条从既有页面档案解析真实 script src；第二条要求已存 JS 明确出现调用模板，才请求页面选定的联赛/赛季。未知结构会退出，避免猜测端点。

## 外部条件与后续接口

三个API key仍未配置，仅报告布尔状态。未购买订阅。复用已有PG16二进制启动隔离验证库，迁移/恢复核验后停止；未改原服务认证，未注册新调度。

001/002为上一轮交接归档；当前接口以 `docs/parallel/round2/platform-interfaces.md` 为准。真实数据库/特征/消费证据在 `var/parallel/collectors/round2/acceptance.json`，请求与样本测试分别记录。
