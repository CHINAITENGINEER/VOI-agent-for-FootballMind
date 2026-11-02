"""Constructed separate-clock regressions; real scores live only in artifacts."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import pytest

from football_research.data_adapters.jsonl import load_jsonl, save_jsonl
from football_research.data_adapters.settlement import export_results
from football_research.data_adapters.types import ResearchDataset, ResearchMatch
from football_research.evaluation import ledger as module
from football_research.evaluation.ledger import ForecastLedger

OBSERVED = datetime(2025, 1, 1, 14, tzinfo=timezone.utc)
INGESTED = OBSERVED + timedelta(minutes=5)
NOW = INGESTED + timedelta(minutes=1)


def dataset(**changes):
    row = ResearchMatch(match_id='clock:1', competition_id='league', season='2025',
        home_team_id='home', away_team_id='away', kickoff=OBSERVED-timedelta(hours=3),
        home_goals=2, away_goals=1, result_available_at=INGESTED,
        information_mode='operational', source='test', input_manifest=('test:evidence',),
        score_period='regulation', status='finished', observed_at=OBSERVED,
        ingested_at=INGESTED, operational_available_at=INGESTED,
        source_published_at=OBSERVED-timedelta(minutes=1))
    return ResearchDataset((replace(row, **changes),), (), 'clock-test', ('test',),
                           {'source_observed_at': '2025-01-02T00:00:00Z'})


def test_receipt_at_1400_commit_at_1405_roundtrip_and_idempotence(tmp_path, monkeypatch):
    source = dataset()
    save_jsonl(source, tmp_path/'input.jsonl')
    assert load_jsonl(tmp_path/'input.jsonl') == source
    rows, report = export_results(source)
    assert len(rows) == 1
    row = rows[0]
    assert row['observed_at'] == OBSERVED.isoformat()  # ignores later manifest envelope
    assert row['available_at'] == row['ingested_at'] == INGESTED.isoformat()
    assert row['source_published_at'] < row['observed_at']
    clock = [NOW]
    monkeypatch.setattr(module, '_now', lambda: clock[0])
    ledger = ForecastLedger(tmp_path/'ledger')
    initial = ledger.settle_result(row)
    assert initial['record']['recorded_at'] == NOW.isoformat()
    clock[0] += timedelta(minutes=1)
    assert ledger.settle_result(row)['status'] == 'duplicate'
    assert ledger.score(as_of=INGESTED)['n_result_versions'] == 0  # recording is another gate
    assert ledger.score()['n_result_versions'] == 1


@pytest.mark.parametrize('field', ['available_at','observed_at','ingested_at','operational_available_at'])
def test_future_clock_is_rejected(tmp_path, monkeypatch, field):
    rows, _ = export_results(dataset())
    rows[0][field] = (NOW + timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(module, '_now', lambda: NOW)
    with pytest.raises(ValueError, match='current clock'):
        ForecastLedger(tmp_path).settle_result(rows[0])


@pytest.mark.parametrize('field', ['available_at', 'operational_available_at'])
def test_operational_backdating_is_rejected(tmp_path, field):
    rows, _ = export_results(dataset())
    rows[0][field] = OBSERVED.isoformat()
    with pytest.raises(ValueError, match='precedes'):
        ForecastLedger(tmp_path).settle_result(rows[0])


def test_historical_assumed_time_stays_distinct_from_receipt(tmp_path):
    source = dataset(information_mode='historical_reconstruction',
                     result_available_at=OBSERVED-timedelta(hours=1),
                     time_basis='assumed_result_delay')
    rows, _ = export_results(source)
    assert rows[0]['available_at'] < rows[0]['observed_at'] < rows[0]['ingested_at']
    assert ForecastLedger(tmp_path).settle_result(rows[0])['status'] == 'appended'


def test_legacy_payload_remains_byte_identical_on_duplicate(tmp_path):
    rows, _ = export_results(dataset(observed_at=None, ingested_at=None,
        operational_available_at=None, source_published_at=None,
        result_available_at=OBSERVED, input_manifest=(f'test|at:{OBSERVED.isoformat()}',)))
    assert 'time_contract_version' not in rows[0]
    ledger = ForecastLedger(tmp_path)
    first = ledger.settle_result(rows[0])
    before = next((tmp_path/'results').glob('*.json')).read_bytes()
    assert ledger.settle_result(rows[0])['record'] == first['record']
    assert next((tmp_path/'results').glob('*.json')).read_bytes() == before


def test_adapter_rejects_explicit_result_time_before_commit():
    rows, report = export_results(dataset(result_available_at=OBSERVED))
    assert not rows
    assert report['skip_reason_counts'] == {'operational_availability_before_evidence': 1}
