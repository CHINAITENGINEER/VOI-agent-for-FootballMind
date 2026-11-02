"""Read-only data coverage and point-in-time evidence audit."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .spftll import load_spftll_archive, read_archive
from .types import ResearchDataset


def audit_sqlite(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        return {'status': 'missing', 'path': str(path)}
    # mode=ro prevents accidental creation or updates; no WAL/checkpoint writes.
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute('SELECT * FROM matches ORDER BY id')]
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    histories, invalid = [], []
    for row in rows:
        try:
            detail = json.loads(row.get('details_json') or '{}')
            for key, items in detail.get('history', {}).items():
                if isinstance(items, list):
                    histories.extend(items)
        except (ValueError, TypeError, AttributeError):
            invalid.append(str(row.get('id')))
    groups = defaultdict(list)
    for history in histories:
        groups[str(history.get('id'))].append(history)
    conflicts = [key for key, records in groups.items() if len({json.dumps(
        {k: r.get(k) for k in ('time', 'home_team', 'away_team', 'score')}, sort_keys=True)
        for r in records}) > 1]
    history_unique = [values[0] for values in groups.values()]
    aliases = defaultdict(set)
    for row in rows:
        aliases[(row['match_time'], row['home_team'], row['away_team'])].add(row['id'])
    row_digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'status': 'read_only_success', 'path': str(path), 'logical_snapshot_sha256': row_digest,
            'tables': tables, 'matches': len(rows),
            'leagues': dict(sorted(Counter(row['league'] for row in rows).items())),
            'statuses': dict(sorted(Counter(row['status'] for row in rows).items())),
            'score_pairs_present': sum(row.get('home_score') is not None and row.get('away_score') is not None for row in rows),
            'complete_1x2_rows': sum(all(row.get(k) is not None for k in ('odds_home', 'odds_draw', 'odds_away')) for row in rows),
            'kickoff_range_source_local': [min((r['match_time'] for r in rows), default=None), max((r['match_time'] for r in rows), default=None)],
            'same_fixture_multiple_ids': [sorted(ids) for ids in aliases.values() if len(ids) > 1],
            'history_rows': len(histories), 'history_unique_matches': len(groups),
            'history_duplicate_rows': len(histories)-len(groups), 'history_conflicting_match_ids': conflicts,
            'history_leagues': dict(sorted(Counter(r.get('league', '') for r in history_unique).items())),
            'history_years_not_verified_seasons': dict(sorted(Counter(str(r.get('time', ''))[:4] for r in history_unique).items())),
            'invalid_details_ids': invalid,
            'temporal_verdict': 'created_time_and_updated_time_are_row_lifecycle_not_field_first_visibility',
            'label_verdict': 'current_score_count_reported_separately;history_scores_require_period_and_visibility_validation',
            'odds_verdict': 'latest_prices_are_not_historical_quote_series',
            'mapping_verdict': 'history_names_have_no_team_IDs_in_legacy_projection;do_not_fuzzy_merge',
            'missing_matches': 'unknown: selected-match feed has no complete season denominator',
            'postponements': 'latest-row table does not retain kickoff revision history'}


def audit_dataset(dataset: ResearchDataset) -> dict:
    leagues = defaultdict(list)
    team_names = defaultdict(set)
    for match in dataset.matches:
        leagues[match.competition_id].append(match)
        team_names[match.home_team_id].add(match.home_team_name)
        team_names[match.away_team_id].add(match.away_team_name)
    coverage = []
    for cid, matches in sorted(leagues.items()):
        labelled = [m for m in matches if m.outcome is not None]
        coverage.append({'competition_id': cid, 'source_names': sorted({m.competition_name for m in matches}),
                         'matches': len(matches), 'regulation_final_labels': len(labelled),
                         'kickoff_min_utc': min(m.kickoff for m in matches).isoformat(),
                         'kickoff_max_utc': max(m.kickoff for m in matches).isoformat(),
                         'source_seasons': sorted({m.season for m in matches if m.season is not None}),
                         'years_not_verified_seasons': dict(sorted(Counter(str(m.kickoff.year) for m in matches).items())),
                         'team_count': len({t for m in matches for t in (m.home_team_id, m.away_team_id)})})
    labelled = [m for m in dataset.matches if m.outcome is not None]
    historical_eligible = [m for m in labelled if m.fixture_available_at is not None and m.fixture_available_at < m.kickoff]
    observation_times = [m.result_available_at for m in dataset.matches if m.result_available_at is not None]
    observation_times += [o.observed_at for o in dataset.odds]
    latest = max(observation_times) if observation_times else None
    future = [m for m in dataset.matches if latest is not None and m.kickoff > latest and m.fixture_available_at is not None
              and m.fixture_available_at <= latest and m.status == 'scheduled' and m.score_period == 'regulation']
    return {**dataset.audit, 'dataset_version': dataset.dataset_version, 'input_manifest': list(dataset.input_manifest),
            'coverage_by_competition': coverage, 'regulation_labels': len(labelled),
            'final_matches_with_fixture_seen_before_kickoff': len(historical_eligible),
            'strict_operational_historical_backtest_eligible': len(historical_eligible) if dataset.audit.get('information_mode') == 'operational' else None,
            'latest_input_time_utc': latest.isoformat() if latest else None,
            'future_regulation_fixtures_after_latest_input': len(future),
            'future_fixture_ids': [m.match_id for m in future],
            'team_source_ids_with_multiple_display_names': {key: sorted(names) for key, names in sorted(team_names.items()) if len(names) > 1},
            'unverified_season_matches': sum(m.season is None for m in dataset.matches),
            'missing_matches': 'unknown; selected team history and lottery coverage has no season denominator',
            'postponements': 'changed kickoff/identity quarantined; no assertion that unseen postponements are absent',
            'statistics_visibility': 'rank/xG/postmatch statistics are not model inputs; historical timestamps unverified',
            'result_definition': 'regulation including stoppage; extra time and penalties excluded; uncertain competitions excluded',
            'performance_claim': 'functional_verification_only; operational future predictions currently unscored'}


def audit_workspace(workspace: str | Path, dataset: ResearchDataset | None = None) -> dict:
    workspace = Path(workspace).resolve()
    dataset = dataset or load_spftll_archive(workspace / 'football_spider/platform_data/raw')
    return {'audit_schema': 'footballmind.research.audit.v1',
            'audited_at': datetime.now(timezone.utc).isoformat(), 'workspace': str(workspace),
            'sqlite': audit_sqlite(workspace / 'football_spider/data/matches.db'),
            'archive_dataset': audit_dataset(dataset),
            'standalone_match_snapshot': {'path': str(workspace / 'football_spider/data/match_3049657.json'),
                'use': 'audit_only; current cup score period and historical fixture visibility not verified'},
            'other_sources': {'understat_probe': 'HTML capability probe; no verified match payload exported',
                              'clubelo_probe': 'failed HTTP probe, no usable rating dataset',
                              'fbref_probe': 'failed HTTP probe, no usable dataset'}}


def render_markdown(report: dict) -> str:
    sql, data = report['sqlite'], report['archive_dataset']
    lines = ['# FootballMind 真实数据覆盖与时间语义审计', '',
             f"生成时间（UTC）：{report['audited_at']}", '',
             f"冻结数据版本：`{data['dataset_version']}`", '',
             '结论：现有档案支持已观测历史赛果训练与后续赛前预测；历史可见性完整的常规时间正式回测样本见下表。',
             '历史重建仅用于功能验证，结果公布延迟和赛程提前可见性是显式假设，不代表已核实的历史系统状态。', '',
             '| 项目 | 实际数量 |', '|---|---:|',
             f"| SQLite 当前比赛 | {sql.get('matches', 0)} |",
             f"| SQLite 当前完整比分 | {sql.get('score_pairs_present', 0)} |",
             f"| SQLite 历史行 / 去重比赛 | {sql.get('history_rows', 0)} / {sql.get('history_unique_matches', 0)} |",
             f"| 验证哈希的原始观测 | {data['verified_observations']} |",
             f"| 导出比赛 / 常规时间标签 | {data['exported_matches']} / {data['regulation_labels']} |",
             f"| 完整胜平负赔率快照 | {data['odds_snapshots']} |",
             f"| 严格 operational 历史回测合格比赛 | {data['strict_operational_historical_backtest_eligible']} |",
             f"| 最新输入时点后已知五大联赛赛程 | {data['future_regulation_fixtures_after_latest_input']} |", '',
             '## 联赛和年份覆盖', '',
             '来源没有赛季字段，因此年份分布只表示开赛年份。没有完整赛程分母，缺赛数量保持未知。', '',
             '| 来源赛事 ID | 来源名称 | 比赛 | 常规时间赛果 | UTC 首场 | UTC 末场 |',
             '|---|---|---:|---:|---|---|']
    for row in data['coverage_by_competition']:
        lines.append(f"| {row['competition_id']} | {' / '.join(row['source_names'])} | {row['matches']} | {row['regulation_final_labels']} | {row['kickoff_min_utc'][:10]} | {row['kickoff_max_utc'][:10]} |")
    lines += ['', '## 时间、缺失、重复和实体语义', '',
        '- 时间统一为 UTC；来源无时区的比赛时间按既有来源契约使用 Asia/Shanghai。',
        '- 最终比分、排名、统计与赔率只在真实观测后可用；SQLite created_time/updated_time 不作为字段首次可见时间。',
        '- 历史交锋和近况是对当前目标球队有选择的样本，具有选择偏差；重复历史条目按来源比赛 ID 去重。',
        '- 同一 ID 的开赛时间、球队、赛事或最终比分冲突整场隔离，保留审计原因；延期旧版本不被静默覆盖用于回放。',
        '- 球队使用来源 ID；同 ID 多显示名作为别名证据输出，不做模糊名称合并，不混入女子/青年/杯赛未知阶段赛果。',
        '- 当前英超、德甲、西甲、意甲、法甲常规时间规则复用既有 normalizer 的接受范围；任何非空加时字段再排除。',
        '- 赔率完整三项必须来自同一观测；源更新时间没有证据时留空，不把收盘或当前价格回填到较早时点。',
        '- 每个输入保留内容 SHA-256、观测 ID 和观测时间；冻结 JSONL 与版本可用于重复实验。', '',
        '## 排除与外部条件', '',
        f"身份/时间/赛果冲突隔离：`{json.dumps(data['quarantined_conflicts'], ensure_ascii=False)}`", '',
        f"解析或报价缺失问题数量：{len(data['issues'])}。完整细节见同名 JSON 审计。", '',
        'Understat 现有文件为 HTML 能力探测；ClubElo、FBref 探测没有可用数据集。尚缺历史赛程版本、来源公布时间、完整联赛赛季和历史赔率序列。', '',
        '真实数据预测暂未有相应最终赛果供评分。若运行显式重建回测，报告应称为真实记录上的功能验证，不能据此宣称稳定预测能力。', '']
    return '\n'.join(lines)
