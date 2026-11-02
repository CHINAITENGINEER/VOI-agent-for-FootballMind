"""Strict offline projection of existing spftll observations, without requests.

The public MatchRecord, Quote and Observation schemas are reused. The five
domestic-league regulation policy matches the existing source normalizer; cups,
youth matches and unknown periods are never promoted to regulation labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from ._contracts import EntityRef, MatchRecord, Observation, Quote, Score
from .types import ResearchDataset, ResearchMatch, ResearchOddsSnapshot, utc

ADAPTER_VERSION = 'spftll-research-1.0.0'
REGULATION_LEAGUES = frozenset({'英超', '德甲', '西甲', '意甲', '法甲'})
REGULATION_SETTLEMENT = '90_minutes_plus_stoppage_excludes_extra_time_penalties'
# The source's current fixture endpoint omits ``leagueId`` on some rows even
# though the league name is explicit.  Keep this small reviewed mapping in the
# normalizer so identity adapters do not fall back to the weaker
# ``competition:name:*`` namespace for a known competition.
KNOWN_LEAGUE_IDS = {
    '英超': '36', '英格兰超级联赛': '36', 'Premier League': '36', 'PL': '36',
    '德甲': '34', '西甲': '31', '意甲': '33', '法甲': '35',
}


def _stamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=ZoneInfo('Asia/Shanghai'))
    return stamp.astimezone(timezone.utc)


def _integer(value):
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value:
        raise ValueError('Noninteger score')
    return int(value)


def read_archive(archive_root: str | Path):
    """Yield verified (Observation, bytes, observation path), and audit issues.

    archive_root is the platform_data/raw directory. Hashes are checked against
    uncompressed original bytes. Failed payloads are preserved and counted.
    """
    archive_root = Path(archive_root).resolve()
    records, issues = [], []
    for observation_path in sorted(archive_root.rglob('*.observation.json')):
        try:
            observation = Observation.model_validate_json(observation_path.read_text(encoding='utf-8'))
            # Archived paths use Windows separators even on portable exports.
            parts = observation.payload_path.replace('\\', '/').split('/')
            candidate = Path(observation.payload_path)
            payload_path = candidate if candidate.is_absolute() else archive_root.parent.joinpath(*parts)
            payload_path = payload_path.resolve()
            if not payload_path.is_relative_to(archive_root):
                raise ValueError('Payload outside the selected archive root')
            content = payload_path.read_bytes()
            if payload_path.suffix == '.gz':
                content = gzip.decompress(content)
            if hashlib.sha256(content).hexdigest() != observation.content_hash:
                raise ValueError('Payload hash mismatch')
            records.append((observation, content, observation_path))
        except (ValueError, OSError, TypeError) as exc:
            issues.append({'observation': str(observation_path), 'reason': str(exc)[:180]})
    return records, issues


def _extract(observation: Observation, payload: dict):
    matches, prices = [], []
    if observation.context.get('legacy_import'):
        matches = [row['source_record'] for row in payload.get('records', []) if 'source_record' in row]
        prices = [(row, False) for row in matches]
    elif observation.endpoint.endswith('GetSimpleMatchsAll'):
        matches = [row for group in payload.get('data') or [] for row in group.get('list', [])]
        prices = [(row, False) for row in matches]
    elif observation.endpoint.endswith('GetInfo'):
        data = payload.get('data') or {}
        matches = [data] + [row for key in ('battleList', 'homeBattleList', 'awayBattleList') for row in data.get(key) or []]
    elif observation.endpoint.endswith('GetList'):
        matches = [row for group in payload.get('data') or [] for row in group.get('matchList', [])]
    elif observation.endpoint.endswith('GetMoreSpInfo'):
        data = (payload.get('data') or {}).get('footballMoreSpInfo')
        if data:
            prices = [(data, True)]
    return matches, prices


def _contract(row: dict, league_ids: dict[str, str]) -> MatchRecord:
    league = str(row.get('leagueChs') or row.get('league') or '')
    league_id = row.get('leagueId')
    if league_id is None:
        league_id = league_ids.get(league) or KNOWN_LEAGUE_IDS.get(league)
    title, state = str(row.get('finishedTitle') or ''), row.get('footMatchState')
    status = {'未开': 'scheduled', '完场': 'finished', '推迟': 'postponed', '延期': 'postponed',
              '取消': 'cancelled', '中断': 'suspended'}.get(title)
    status = status or {-1: 'finished', 0: 'scheduled'}.get(state, 'unknown')
    mid = str(row['matchId'])
    regulation = league in REGULATION_LEAGUES
    # A non-null extra-time field is a phase conflict even in an accepted league.
    if any(row.get(key) is not None for key in ('homeOT1', 'homeOT2', 'homeOT3', 'awayOT1', 'awayOT2', 'awayOT3')):
        regulation = False
    def team(side):
        sid = row.get(f'{side}Id')
        return EntityRef(source_id=str(sid) if sid is not None else f'match:{mid}:{side}',
                         name=str(row.get(f'{side}Chs') or row.get(side) or ''),
                         category='men_senior' if regulation else 'unknown',
                         id_quality='stable' if sid is not None else 'match_scoped')
    home, away = _integer(row.get('homeScore')), _integer(row.get('awayScore'))
    if home is None or away is None:
        home = away = None
    return MatchRecord(source_id=mid,
        competition=EntityRef(source_id=str(league_id) if league_id is not None else f'name:{league}',
                              name=league, id_quality='stable' if league_id is not None else 'source_name'),
        home=team('home'), away=team('away'), kickoff=_stamp(row['matchTime']),
        original_time=row['matchTime'], source_timezone='Asia/Shanghai', status=status,
        score_period='regulation' if regulation else 'unknown', score=Score(home=home, away=away),
        missing={'season': 'source_not_supplied', 'source_published_at': 'not_verified'},
        coverage='selected_matches_incomplete')


def load_spftll_archive(archive_root: str | Path, *, information_mode: str = 'operational',
                       result_delay_hours: float = 24.0,
                       assumed_fixture_lead_hours: float | None = None,
                       as_of: datetime | str | None = None) -> ResearchDataset:
    """Return hashed real archive data with explicit, conservative visibility.

    Operational availability is the first verified observation of each fact.
    Historical reconstruction is functional-only: labels become available after
    an explicitly assumed delay. Fixture visibility remains operational unless
    assumed_fixture_lead_hours is explicitly provided. No current quote is ever
    moved to an earlier timestamp, under either mode. Pass ``as_of`` whenever
    replaying a historical decision cutoff: its observation gate runs before
    source-ID mapping, conflict quarantine, aggregation and input manifests.
    """
    if information_mode not in {'operational', 'historical_reconstruction'}:
        raise ValueError('Expected operational or historical_reconstruction')
    if not isinstance(result_delay_hours, (float, int)) or not 0 < result_delay_hours < 8760:
        raise ValueError('result_delay_hours must be positive and finite')
    if assumed_fixture_lead_hours is not None and (information_mode != 'historical_reconstruction' or not 0 < assumed_fixture_lead_hours < 8760):
        raise ValueError('Fixture assumptions require reconstruction and a positive finite lead')
    cutoff = utc(datetime.fromisoformat(as_of.replace('Z', '+00:00')) if isinstance(as_of, str) else as_of) if as_of is not None else None
    archives, issues = read_archive(archive_root)
    verified_total = len(archives)
    # Gate BEFORE identity inference, duplicate handling and manifests. Future
    # corrections/aliases/conflicts must never affect an earlier projection.
    if cutoff is not None:
        archives = [entry for entry in archives if entry[0].observed_at <= cutoff]
    excluded_future = verified_total - len(archives)
    candidates, price_candidates, manifests = [], [], set()
    sources = Counter()
    for observation, content, path in archives:
        sources[observation.source] += 1
        if observation.source != 'spftll' or observation.http_status != 200 or observation.error_class:
            continue
        manifest = f'sha256:{observation.content_hash}|observation:{observation.id}|at:{observation.observed_at.isoformat()}'
        try:
            payload = json.loads(content)
            if not observation.context.get('legacy_import') and payload.get('code') != 0:
                raise ValueError('Non-success source business code')
            rows, prices = _extract(observation, payload)
            candidates.extend((row, observation, manifest) for row in rows)
            price_candidates.extend((row, more, observation, manifest) for row, more in prices)
            if rows or prices:
                manifests.add(manifest)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({'observation': str(path), 'reason': str(exc)[:180]})
    name_ids = defaultdict(set)
    for row, _, _ in candidates:
        if row.get('leagueId') is not None:
            name_ids[str(row.get('leagueChs') or row.get('league') or '')].add(str(row['leagueId']))
    league_ids = {name: next(iter(ids)) for name, ids in name_ids.items() if len(ids) == 1}
    versions = defaultdict(list)
    for row, observation, manifest in candidates:
        try:
            record = _contract(row, league_ids)
            versions[record.source_id].append((record, observation, manifest))
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({'match_id': str(row.get('matchId')), 'reason': str(exc)[:180]})
    matches, conflicts = [], []
    for source_id, values in sorted(versions.items()):
        values.sort(key=lambda v: v[1].observed_at)
        identity = {(r.kickoff, r.home.source_id, r.away.source_id, r.competition.source_id) for r, _, _ in values}
        scores = {(r.score.home, r.score.away) for r, _, _ in values if r.status == 'finished' and r.score.home is not None}
        if len(identity) != 1 or len(scores) > 1:
            conflicts.append({'match_id': source_id, 'identity_versions': len(identity), 'result_versions': len(scores),
                              'reason': 'Quarantined: replay needs per-version history reader'})
            continue
        finished = [(r, o, m) for r, o, m in values if r.status == 'finished' and r.score.home is not None]
        record, obs, _ = finished[0] if finished else values[-1]
        result_available = obs.observed_at if finished and record.score_period == 'regulation' else None
        if result_available is not None and result_available < record.kickoff:
            conflicts.append({'match_id': source_id, 'reason': 'Final score observed before kickoff'})
            continue
        fixture_available = values[0][1].observed_at
        time_basis = 'verified_observation_at;Asia/Shanghai_source_clock;season_unknown'
        if information_mode == 'historical_reconstruction' and result_available is not None:
            result_available = record.kickoff + timedelta(hours=result_delay_hours)
            time_basis += f';ASSUMED_result_delay_hours={result_delay_hours};FUNCTIONAL_ONLY'
            if assumed_fixture_lead_hours is not None:
                fixture_available = record.kickoff - timedelta(hours=assumed_fixture_lead_hours)
                time_basis += f';ASSUMED_fixture_lead_hours={assumed_fixture_lead_hours}'
        quality = 'source_id' if record.home.id_quality == record.away.id_quality == 'stable' else 'match_scoped'
        matches.append(ResearchMatch(
            match_id=f'spftll:match:{source_id}', competition_id=f'spftll:competition:{record.competition.source_id}',
            season=record.season, home_team_id=f'spftll:team:{record.home.source_id}',
            away_team_id=f'spftll:team:{record.away.source_id}', kickoff=record.kickoff,
            home_goals=record.score.home if finished else None, away_goals=record.score.away if finished else None,
            result_available_at=result_available, fixture_available_at=fixture_available,
            information_mode=information_mode, source='spftll', input_manifest=tuple(sorted({m for _, _, m in values})),
            score_period=record.score_period, status=record.status, time_basis=time_basis,
            entity_quality=quality, competition_name=record.competition.name,
            home_team_name=record.home.name, away_team_name=record.away.name))
    odds = []
    for row, more, observation, manifest in price_candidates:
        try:
            keys = ('spf_win', 'spf_draw', 'spf_lost') if more else ('spfWinFoot', 'spfEqualFoot', 'spfLoseFoot')
            quotes = [Quote(source_match_id=str(row['matchId']), provider='spftll', market_type='1x2',
                            period='regulation', selection=selection, decimal_price=row.get(key),
                            settlement_rule=REGULATION_SETTLEMENT)
                      for selection, key in zip(('home', 'draw', 'away'), keys)]
            odds.append(ResearchOddsSnapshot(match_id=f'spftll:match:{row["matchId"]}', observed_at=observation.observed_at,
                home_price=quotes[0].decimal_price, draw_price=quotes[1].decimal_price, away_price=quotes[2].decimal_price,
                input_manifest=(manifest,)))
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({'match_id': str(row.get('matchId')), 'reason': 'incomplete_1x2_snapshot: ' + str(exc)[:100]})
    valid_ids = {match.match_id for match in matches}
    odds = [row for row in odds if row.match_id in valid_ids]
    policy = {'adapter_version': ADAPTER_VERSION, 'information_mode': information_mode,
              'projection_as_of': cutoff.isoformat() if cutoff else None,
              'result_delay_hours': result_delay_hours if information_mode == 'historical_reconstruction' else None,
              'assumed_fixture_lead_hours': assumed_fixture_lead_hours}
    version_hash = hashlib.sha256(json.dumps({'inputs': sorted(manifests), **policy}, sort_keys=True).encode()).hexdigest()
    audit = {**policy, 'verified_observations': len(archives), 'verified_observations_total': verified_total,
             'excluded_future_observations': excluded_future, 'observation_sources': dict(sources),
             'match_rows_before_deduplication': len(candidates), 'unique_source_match_ids': len(versions),
             'duplicate_match_rows': len(candidates) - len(versions), 'exported_matches': len(matches),
             'regulation_labels': sum(m.outcome is not None for m in matches), 'odds_snapshots': len(odds),
             'quarantined_conflicts': conflicts, 'issues': issues,
             'source_seasons_known': sum(m.season is not None for m in matches),
             'coverage': 'selected_fixture_and_team_history_samples_not_complete_leagues',
             'score_policy': 'five_domestic_leagues_existing_normalizer_policy;explicit_OT_excluded',
             'historical_reconstruction_status': 'functional_only_with_unverified_visibility_assumptions' if information_mode != 'operational' else 'not_requested',
             'missing_features': ['verified_historical_publication_times', 'complete_league_seasons',
                                  'historical_odds_snapshots', 'independent_fixture_schedule_versions']}
    return ResearchDataset(tuple(sorted(matches, key=lambda m: (m.kickoff, m.match_id))),
                           tuple(sorted(odds, key=lambda o: (o.observed_at, o.match_id))),
                           f'{ADAPTER_VERSION}:{version_hash}', tuple(sorted(manifests)), audit)
