# 第二轮真实历史输入与研究投影

研究侧提供 `football_research.data_adapters.platform_history`，只读取平台已完成的规范化导出。Understat 原始响应解析、统一实体映射、Elo 更新及逐场特征计算仍由平台负责。

## 真实原文与尚待发布的规范化导出

真实来源观测的现有证据是 `var/parallel/collectors/understat-research/data-manifest.json`：`getLeagueData/EPL/2025`，观测时间 `2026-09-20T11:59:41.602689+00:00`，观测 ID `a9447ae0-3252-4207-a437-63bfc700f3a5`，载荷 SHA256 `d401b18a3386d8168f03727b43b1ab5dc9092e0469e77a1561136baf25151f42`。平台已有的来源审计报告记录 380 个唯一比赛、20 队各 38 场 history、完整比分和双方 xG。此处的数量引用来源审计，研究适配器不再直接解析原始响应。

正式研究输入应来自平台的新版本导出目录，保留原始观测、规范化版本和特征 manifest。`tests/fixtures` 里的真实复制件以及本适配器的构造测试均不用于正式历史评分。接口请求见 `docs/parallel/requests/baselines/003-round2-platform.md`。真实导出消费计数与实际评分结果由本轮总报告记录。

## 完成标记和数据完整性

导出版本目录含 `manifest.json`、`matches.jsonl`，可含 `features.jsonl`；最后由生产者原子写 `COMPLETE.json`。完成标记保存 `manifest_sha256`，manifest 列出所有文件相对路径、SHA256 和可选 `row_count`。声明文件须位于该版本目录内。消费者验证完整标记、每个文件的字节哈希、行数、时区、标识、赛果周期、信息模式、源证据与声明的时间假设。验证后的字节保留在 `VerifiedHistoryExport.payloads`，后续解析不重新打开原文件。

最小 manifest 形状：

```json
{
  "schema": "footballmind.platform.history-export.v1",
  "dataset_version": "producer-assigned-immutable-version",
  "source": "understat",
  "source_observed_at": "2026-09-20T11:59:41.602689+00:00",
  "information_mode": "historical_reconstruction",
  "source_evidence": [{"observation_id": "SOURCE_OBSERVATION", "sha256": "SOURCE_SHA256"}],
  "assumptions": {
    "result_delay_hours": 3,
    "xg_delay_hours": 24,
    "fixture_lead_hours": 168,
    "kickoff_timezone_policy": "producer-declared-timezone-assumption"
  },
  "files": {
    "matches": {"path": "matches.jsonl", "sha256": "MATCHES_SHA256", "row_count": 380},
    "features": {"path": "features.jsonl", "sha256": "FEATURES_SHA256", "row_count": 760}
  }
}
```

示例中的延迟和提前量仅展示 schema；真实重建必须使用平台发布的明确假设。`result_delay_hours` 和 `xg_delay_hours` 均从 kickoff 起算，包含比赛本身的时间；赛果假设至少 2 小时，xG 假设不早于赛果假设。明确给出的逐场 availability 要与 manifest 一致。来源无 offset 的原始时区推定由平台记录为 `kickoff_timezone_policy`。

比赛行是规范化平面字段：`match_id`、`competition_id`、`season`、`home_team_id`、`away_team_id`、带 offset 的 `kickoff`、`status`、`score_period`、`home_goals`、`away_goals`，可带 names、source、input_manifest、fixture_available_at、result_available_at。所有标识保持平台提供的字符串，研究侧不按球队名称猜测跨源 ID。

## 时间语义和模型输入边界

- `operational` 默认可见时间是该版本的实际 `max(observed_at, ingested_at)`；较早赛季的最终比分不会倒填成过去已可见。任何更早可见声明需要平台发布对应历史版本，本适配器拒绝对当前观测回填。
- `historical_reconstruction` 的比赛可见时间是 kickoff 减 fixture lead，赛果可见时间是 kickoff 加 result delay；这些都是实验假设。今天获取的比分、xG 和赛程可能包含历史修订。原始历史发布版本缺失，完整赛季重建结果不作为真实前瞻成绩。
- `ResearchMatch` 不含目标单场 xG、来源 forecast、players 及最终赛季汇总；这些字段即使出现在导出中也被排除，并在审计里列出。平台历史特征须分别按赛果/xG 延迟过滤，更早比赛的数据才进入后续比赛。
- 逐场特征通过另一批量适配器按 `match_id + prediction_cutoff + information_mode + feature_version` 连接。预测提前量须落在假设的 fixture lead 以内。
- 此来源未提供经验证的历史赛前报价，适配器输出零条 odds；市场基线覆盖单独报告，不由 forecast 或 xG 代替。

## 可执行入口与验证

```powershell
python -X utf8 -m football_research.data_adapters.platform_history --export PLATFORM_COMPLETED_VERSION_DIRECTORY --output var/parallel/baselines/round2/understat/dataset.jsonl
python -X utf8 -m pytest tests/research/baselines/test_platform_history.py -q
```

Python 接口：

```python
from football_research.data_adapters.platform_history import (
    load_completed_history_export, verify_completed_export,
)
dataset = load_completed_history_export(export_directory)
verified = verify_completed_export(export_directory)
feature_rows = verified.rows("features")
```

构造测试覆盖完成标记缺失、manifest/文件改动、路径越界、行数不符、重复实体、未知时间/状态/周期、信息模式和来源冲突、赛果可见时间回填、显式延迟不一致、目标 xG/forecast 隔离、JSONL 往返与只读确定性。
