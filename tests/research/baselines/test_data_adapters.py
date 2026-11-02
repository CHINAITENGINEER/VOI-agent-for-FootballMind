"""Constructed fixtures test mechanics; these are never real evaluation data."""
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from football_research.data_adapters import load_spftll_archive, load_jsonl, save_jsonl
from football_research.data_adapters.audit import audit_dataset, audit_sqlite
from football_research.data_adapters.spftll import read_archive
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot


def archive(tmp_path, ident, time, payload, endpoint='GetInfo', corrupt=False):
    raw = tmp_path / 'raw' / 'spftll'
    raw.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False).encode()
    (raw / f'{ident}.body.gz').write_bytes(gzip.compress(content))
    observation = {'id': ident, 'source': 'spftll', 'endpoint': f'https://example.test/{endpoint}',
                   'observed_at': time, 'http_status': 200, 'payload_path': f'raw/spftll/{ident}.body.gz',
                   'content_hash': 'invalid' if corrupt else hashlib.sha256(content).hexdigest(), 'payload_format': 'json'}
    (raw / f'{ident}.observation.json').write_text(json.dumps(observation), encoding='utf-8')
    return tmp_path / 'raw'


def match(mid=1, **changes):
    row = {'matchId': mid, 'matchTime': '2026-01-01T20:00:00', 'leagueId': 36, 'league': '英超',
           'homeId': 1, 'home': 'Home', 'awayId': 2, 'away': 'Away', 'footMatchState': -1,
           'homeScore': 0, 'awayScore': 0}
    row.update(changes)
    return row


def test_current_final_record_keeps_real_observation_time(tmp_path):
    raw = archive(tmp_path, 'late', '2026-09-20T10:00:00Z', {'code': 0, 'data': match()})
    data = load_spftll_archive(raw)
    row = data.matches[0]
    assert row.kickoff.isoformat() == '2026-01-01T12:00:00+00:00'
    assert row.result_available_at == row.fixture_available_at == datetime(2026, 9, 20, 10, tzinfo=timezone.utc)
    assert row.outcome == 1 and row.home_goals == 0
    assert audit_dataset(data)['strict_operational_historical_backtest_eligible'] == 0


def test_fixture_first_seen_is_distinct_from_result_first_seen(tmp_path):
    row = match(footMatchState=0, homeScore=None, awayScore=None)
    raw = archive(tmp_path, 'fixture', '2026-01-01T10:00:00Z', {'code': 0, 'data': row})
    archive(tmp_path, 'result', '2026-01-01T15:00:00Z', {'code': 0, 'data': match(homeScore=2)})
    data = load_spftll_archive(raw)
    assert len(data.matches) == 1
    row = data.matches[0]
    assert row.fixture_available_at.hour == 10 and row.result_available_at.hour == 15
    assert row.outcome == 0
    assert len(row.input_manifest) == 2


def test_reconstruction_requires_explicit_fixture_visibility_assumption(tmp_path):
    raw = archive(tmp_path, 'late', '2026-09-20T10:00:00Z', {'code': 0, 'data': match()})
    plain = load_spftll_archive(raw, information_mode='historical_reconstruction')
    assert plain.matches[0].fixture_available_at > plain.matches[0].kickoff
    rebuilt = load_spftll_archive(raw, information_mode='historical_reconstruction', assumed_fixture_lead_hours=48)
    row = rebuilt.matches[0]
    assert (row.result_available_at-row.kickoff).total_seconds() == 86400
    assert (row.kickoff-row.fixture_available_at).total_seconds() == 172800
    assert 'FUNCTIONAL_ONLY' in row.time_basis
    assert plain.dataset_version != rebuilt.dataset_version
    with pytest.raises(ValueError):
        load_spftll_archive(raw, assumed_fixture_lead_hours=48)


@pytest.mark.parametrize('change', [{'league': '英足总杯'}, {'homeOT1': 0}, {'league': '女超'}])
def test_unverified_periods_never_become_regulation_labels(tmp_path, change):
    raw = archive(tmp_path, 'period', '2026-01-01T15:00:00Z', {'code': 0, 'data': match(**change)})
    row = load_spftll_archive(raw).matches[0]
    assert row.score_period == 'unknown' and row.outcome is None and row.result_available_at is None


def test_duplicate_history_and_changed_fixture_quarantined(tmp_path):
    raw = archive(tmp_path, 'one', '2026-01-01T15:00:00Z', {'code': 0, 'data': {**match(), 'battleList': [match()]}})
    assert len(load_spftll_archive(raw).matches) == 1
    archive(tmp_path, 'two', '2026-01-01T16:00:00Z', {'code': 0, 'data': match(matchTime='2026-01-02T20:00:00')})
    data = load_spftll_archive(raw)
    assert data.matches == () and len(data.audit['quarantined_conflicts']) == 1


def test_latest_postponement_is_not_a_scheduled_prediction_candidate(tmp_path):
    row = match(footMatchState=0, homeScore=None, awayScore=None)
    raw = archive(tmp_path, 'scheduled', '2026-01-01T10:00:00Z', {'code': 0, 'data': row})
    archive(tmp_path, 'postponed', '2026-01-01T11:00:00Z', {'code': 0, 'data': {**row, 'finishedTitle': '推迟'}})
    assert load_spftll_archive(raw).matches[0].status == 'postponed'


def test_odds_same_observation_and_no_backdating_in_reconstruction(tmp_path):
    row = match(footMatchState=0, homeScore=None, awayScore=None,
                spfWinFoot=2.0, spfEqualFoot=3.5, spfLoseFoot=4.0)
    raw = archive(tmp_path, 'first', '2025-12-31T23:00:00Z', {'code': 0, 'data': [{'list': [row]}]}, 'GetSimpleMatchsAll')
    archive(tmp_path, 'missing', '2026-01-01T01:00:00Z',
            {'code': 0, 'data': [{'list': [{**row, 'spfEqualFoot': None}]}]}, 'GetSimpleMatchsAll')
    data = load_spftll_archive(raw, information_mode='historical_reconstruction', assumed_fixture_lead_hours=48)
    assert len(data.odds) == 1
    assert data.odds[0].available_at == datetime(2025, 12, 31, 23, tzinfo=timezone.utc)
    assert data.odds[0].information_mode == 'operational'
    assert data.odds[0].prices == (2.0, 3.5, 4.0)


def test_hash_mismatch_is_rejected(tmp_path):
    raw = archive(tmp_path, 'bad', '2026-01-01T15:00:00Z', {'code': 0, 'data': match()}, corrupt=True)
    data = load_spftll_archive(raw)
    assert not data.matches
    assert 'hash mismatch' in data.audit['issues'][0]['reason']


def test_future_conflict_never_changes_earlier_projection_or_lineage(tmp_path):
    raw = archive(tmp_path, 'known', '2026-01-01T15:00:00Z', {'code': 0, 'data': match()})
    before = load_spftll_archive(raw, as_of='2026-01-01T16:00:00Z')
    archive(tmp_path, 'future_conflict', '2026-01-01T17:00:00Z', {'code': 0, 'data': match(homeId=99)})
    after = load_spftll_archive(raw, as_of='2026-01-01T16:00:00Z')
    assert before.matches == after.matches
    assert before.dataset_version == after.dataset_version
    assert before.input_manifest == after.input_manifest
    assert after.audit['excluded_future_observations'] == 1
    assert load_spftll_archive(raw).matches == ()


def test_jsonl_roundtrip_and_deterministic_input_version(tmp_path):
    raw = archive(tmp_path, 'ok', '2026-01-01T15:00:00Z', {'code': 0, 'data': match()})
    data = load_spftll_archive(raw)
    assert data.dataset_version == load_spftll_archive(raw).dataset_version
    output = tmp_path / 'export.jsonl'
    save_jsonl(data, output)
    assert load_jsonl(output) == data


def test_bad_timezone_scores_and_prices_fail_closed():
    with pytest.raises(ValueError):
        ResearchMatch('1', 'epl', None, 'h', 'a', datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        ResearchOddsSnapshot('1', datetime.now(timezone.utc), float('nan'), 3.0, 4.0)


def test_sqlite_audit_is_read_only_and_does_not_create_missing_file(tmp_path):
    missing = tmp_path / 'missing.db'
    assert audit_sqlite(missing)['status'] == 'missing'
    assert not missing.exists()
    path = tmp_path / 'existing.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE matches(id TEXT, match_time TEXT, league TEXT, status TEXT, home_team TEXT, away_team TEXT, details_json TEXT)')
        connection.execute("INSERT INTO matches VALUES('1','2026-01-01','TEST','finished','H','A','{}')")
    before = path.read_bytes()
    assert audit_sqlite(path)['matches'] == 1
    assert path.read_bytes() == before


def feature_file(tmp_path, *, corrupt_hash=False, mismatched_cutoff=False):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifest = {'match_id': 'canonical-1', 'prediction_cutoff': cutoff.isoformat(),
                'feature_version': 'prematch-v1.1', 'information_mode': 'operational',
                'availability_basis': 'max(observed_at,ingested_at)',
                'history_version_ids': ['history-1'], 'observation_ids': ['observation-1'], 'parameters': {'k': 20}}
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    row = {key: manifest[key] for key in ('match_id', 'feature_version', 'information_mode')}
    row.update(prediction_cutoff=cutoff.replace(hour=1) if mismatched_cutoff else cutoff,
               input_manifest_hash='invalid' if corrupt_hash else hashlib.sha256(encoded).hexdigest(),
               manifest_json=encoded.decode(), features_json=json.dumps({'elo': {'home': 1505., 'away': 1470.,
                   'input_version_ids': ['history-1'], 'parameters': {'k': 20}, 'coverage': 'incomplete'}}))
    path = tmp_path / 'feature.parquet'
    pq.write_table(pa.Table.from_pylist([row]), path)
    return path


def test_feature_export_preserves_canonical_ids_and_source_ratings(tmp_path):
    from football_research.data_adapters import load_feature_snapshot
    data = load_feature_snapshot(feature_file(tmp_path))
    assert data['match_id'] == 'canonical-1'
    assert data['home_elo'] == 1505. and data['away_elo'] == 1470.
    assert data['features_available_at'] == data['prediction_cutoff']
    assert len(data['parquet_sha256']) == 64


@pytest.mark.parametrize('kwargs', [{'corrupt_hash': True}, {'mismatched_cutoff': True}])
def test_feature_export_hash_and_cutoff_mismatch_fail_closed(tmp_path, kwargs):
    from football_research.data_adapters import load_feature_snapshot
    with pytest.raises(ValueError):
        load_feature_snapshot(feature_file(tmp_path, **kwargs))
