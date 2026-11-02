"""Constructed export tests, deliberately separate from real season evidence."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

import pytest

from football_research.data_adapters.platform_history import (
    EXPORT_SCHEMA, load_completed_history_export, verify_completed_export,
)
from football_research.data_adapters import load_jsonl, save_jsonl


def exported(tmp_path, *, mode='historical_reconstruction', row_changes=None,
             manifest_changes=None, rows=None):
    row = {
        'match_id': 'synthetic:match:1', 'competition_id': 'synthetic:competition:1',
        'season': '2025', 'home_team_id': 'synthetic:team:1', 'away_team_id': 'synthetic:team:2',
        'kickoff': '2025-08-15T19:00:00Z', 'status': 'finished', 'score_period': 'regulation',
        'home_goals': 2, 'away_goals': 1,
        'input_manifest': ['normalized_version:test-1'],
        'forecast': {'w': 0.9}, 'home_xg': 2.1, 'away_xg': 0.9,
    }
    row.update(row_changes or {})
    encoded = ''.join(json.dumps(item) + '\n' for item in (rows or [row])).encode()
    (tmp_path / 'matches.jsonl').write_bytes(encoded)
    manifest = {
        'schema': EXPORT_SCHEMA, 'dataset_version': 'synthetic-export-v1',
        'source': 'constructed_test', 'source_observed_at': '2026-09-20T12:00:00Z',
        'information_mode': mode,
        'source_evidence': [{'observation_id': 'constructed-test-observation', 'sha256': 'a' * 64}],
        'assumptions': {'result_delay_hours': 3, 'xg_delay_hours': 24, 'fixture_lead_hours': 168,
                        'kickoff_timezone_policy': 'synthetic_UTC'},
        'files': {'matches': {'path': 'matches.jsonl', 'sha256': hashlib.sha256(encoded).hexdigest(), 'row_count': len(rows or [row])}},
    }
    manifest.update(manifest_changes or {})
    reseal(tmp_path, manifest)
    return tmp_path, manifest


def reseal(path, manifest):
    encoded = json.dumps(manifest).encode()
    (path / 'manifest.json').write_bytes(encoded)
    (path / 'COMPLETE.json').write_text(json.dumps({'manifest_sha256': hashlib.sha256(encoded).hexdigest()}))


def test_reconstruction_retains_real_observation_and_explicit_availability_assumptions(tmp_path):
    path, _ = exported(tmp_path)
    dataset = load_completed_history_export(path)
    row = dataset.matches[0]
    assert row.result_available_at == datetime(2025, 8, 15, 22, tzinfo=timezone.utc)
    assert row.fixture_available_at == datetime(2025, 8, 8, 19, tzinfo=timezone.utc)
    assert row.information_mode == 'historical_reconstruction'
    assert row.outcome == 0
    assert dataset.audit['source_observed_at'] == '2026-09-20T12:00:00+00:00'
    assert dataset.audit['ignored_target_or_unverified_fields'] == ['away_xg', 'forecast', 'home_xg']
    assert dataset.odds == ()
    assert not hasattr(row, 'home_xg') and not hasattr(row, 'forecast')
    assert 'revised_snapshot_bias' in row.time_basis
    save_jsonl(dataset, tmp_path / 'research.jsonl')
    assert load_jsonl(tmp_path / 'research.jsonl') == dataset


def test_operational_snapshot_is_not_backdated_to_season(tmp_path):
    path, _ = exported(tmp_path, mode='operational')
    row = load_completed_history_export(path).matches[0]
    assert row.result_available_at == row.fixture_available_at == datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def test_operational_ingestion_upper_bound_is_preserved(tmp_path):
    path, _ = exported(tmp_path, mode='operational', row_changes={'ingested_at': '2026-09-20T13:00:00Z'})
    assert load_completed_history_export(path).matches[0].result_available_at.hour == 13


def test_per_match_receipt_commit_and_publication_survive_projection(tmp_path):
    path, _ = exported(tmp_path, mode='operational', row_changes={
        'observed_at': '2026-09-20T14:00:00Z', 'ingested_at': '2026-09-20T14:05:00Z',
        'source_published_at': '2026-09-20T13:59:00Z',
        'settlement_status': 'final', 'version_id': 'test-only-version'})
    dataset = load_completed_history_export(path)
    match = dataset.matches[0]
    assert match.observed_at == datetime(2026, 9, 20, 14, tzinfo=timezone.utc)
    assert match.ingested_at == match.result_available_at == match.operational_available_at
    assert match.ingested_at.minute == 5
    assert match.source_published_at.minute == 59
    assert dataset.audit['match_settlement_metadata'][match.match_id]['version_id'] == 'test-only-version'
    save_jsonl(dataset, tmp_path/'clocks.jsonl')
    assert load_jsonl(tmp_path/'clocks.jsonl') == dataset


def test_quarantine_pending_reason_survives_projection(tmp_path):
    path, _ = exported(tmp_path, mode='operational', row_changes={
        'status': 'scheduled', 'home_goals': None, 'away_goals': None,
        'settlement_status': 'pending', 'pending_reason': 'newer_observation_quarantined',
        'blocked_observation_id': 'test-bad-observation'})
    dataset = load_completed_history_export(path)
    from football_research.data_adapters.settlement import export_results
    rows, report = export_results(dataset)
    assert rows == []
    assert report['skipped'][0]['platform_settlement']['blocked_observation_id'] == 'test-bad-observation'


@pytest.mark.parametrize('row_changes', [
    {'fixture_available_at': '2025-08-01T12:00:00Z'},
    {'result_available_at': '2025-08-16T12:00:00Z'},
])
def test_operational_backdating_rejected(tmp_path, row_changes):
    path, _ = exported(tmp_path, mode='operational', row_changes=row_changes)
    with pytest.raises(ValueError, match='precedes observed/ingested'):
        load_completed_history_export(path)


def test_incomplete_export_rejected_before_any_rows_are_consumed(tmp_path):
    path, _ = exported(tmp_path)
    (path / 'COMPLETE.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        load_completed_history_export(path)


@pytest.mark.parametrize('target', ['manifest.json', 'matches.jsonl'])
def test_completed_export_tamper_detected(tmp_path, target):
    path, _ = exported(tmp_path)
    with (path / target).open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_completed_history_export(path)


def test_verified_payload_does_not_reopen_changed_file(tmp_path):
    path, _ = exported(tmp_path)
    verified = verify_completed_export(path)
    (path / 'matches.jsonl').write_text('{}\n')
    assert verified.rows('matches')[0]['match_id'] == 'synthetic:match:1'


def test_path_escape_rejected(tmp_path):
    path, manifest = exported(tmp_path)
    manifest['files']['matches']['path'] = '../outside.jsonl'
    reseal(path, manifest)
    with pytest.raises(ValueError, match='inside its version directory'):
        load_completed_history_export(path)


def test_declared_feature_file_hash_is_checked_even_for_goal_only_models(tmp_path):
    path, manifest = exported(tmp_path)
    (path / 'features.jsonl').write_text('{}\n')
    manifest['files']['features'] = {'path': 'features.jsonl', 'sha256': 'a' * 64}
    reseal(path, manifest)
    with pytest.raises(ValueError, match='file hash mismatch: features'):
        load_completed_history_export(path)


@pytest.mark.parametrize('changes', [
    {'assumptions': {}},
    {'assumptions': {'result_delay_hours': 1, 'xg_delay_hours': 24, 'fixture_lead_hours': 168, 'kickoff_timezone_policy': 'UTC'}},
    {'assumptions': {'result_delay_hours': 3, 'xg_delay_hours': 2, 'fixture_lead_hours': 168, 'kickoff_timezone_policy': 'UTC'}},
    {'source_evidence': []},
    {'source_evidence': [{'observation_id': 'obs', 'sha256': 'x' * 64}]},
])
def test_incomplete_assumptions_or_evidence_rejected(tmp_path, changes):
    path, _ = exported(tmp_path, manifest_changes=changes)
    with pytest.raises(ValueError):
        load_completed_history_export(path)


@pytest.mark.parametrize('changes', [
    {'information_mode': 'operational'}, {'source': 'other_source'},
    {'kickoff': '2025-08-15T19:00:00'},
    {'result_available_at': '2025-08-15T21:00:00Z'},
    {'input_manifest': 'not_a_list'},
    {'match_id': None}, {'home_team_id': ''}, {'status': 'mistyped'},
])
def test_row_semantic_conflicts_rejected(tmp_path, changes):
    path, _ = exported(tmp_path, row_changes=changes)
    with pytest.raises(ValueError):
        load_completed_history_export(path)


def test_unverified_period_does_not_supply_label(tmp_path):
    path, _ = exported(tmp_path, row_changes={'score_period': 'unknown'})
    row = load_completed_history_export(path).matches[0]
    assert row.outcome is None and row.result_available_at is None


def test_duplicate_match_ids_rejected(tmp_path):
    path, _ = exported(tmp_path)
    row = verify_completed_export(path).rows('matches')[0]
    exported(tmp_path, rows=[row, deepcopy(row)])
    with pytest.raises(ValueError, match='one current projection'):
        load_completed_history_export(path)


def test_row_count_is_enforced(tmp_path):
    path, manifest = exported(tmp_path)
    manifest['files']['matches']['row_count'] = 2
    reseal(path, manifest)
    with pytest.raises(ValueError, match='row count'):
        load_completed_history_export(path)


def test_repeated_read_is_deterministic_and_read_only(tmp_path):
    path, _ = exported(tmp_path)
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    assert load_completed_history_export(path) == load_completed_history_export(path)
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}
