"""Explicit 7/14-day preparation, using only constructed or saved raw bytes."""
from copy import deepcopy
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import socket
from zoneinfo import ZoneInfo

import pytest

from football_research.data_adapters.official_fixture import (
    _digest, prepare_official_fixture, verify_official_fixture,
)
from football_spider.storage.raw_archive import RawArchive
from test_official_fixture import NOW, EVENT, Transport, prepare, read_binding
from test_finland_stage_fixture import schedule


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Official preparation horizon tests must not perform HTTP')
    for obj, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                      (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        monkeypatch.setattr(obj, name, blocked)


def source(lead):
    local = (NOW+lead).astimezone(ZoneInfo('Europe/Helsinki'))
    return Transport(schedule(date=local.strftime('%Y%m%d'), clock=local.strftime('%H:%M')))


@pytest.mark.parametrize('lead,expected', [(timedelta(minutes=29), 'not_prematch'),
    (timedelta(minutes=30), 'ready'), (timedelta(days=7, minutes=1), 'ready'),
    (timedelta(days=10), 'ready'), (timedelta(days=11), 'ready'),
    (timedelta(days=14), 'ready'), (timedelta(days=14, minutes=1), 'not_prematch')])
def test_explicit_fourteen_day_window_has_unchanged_lower_and_finite_upper_bounds(tmp_path, lead, expected):
    transport = source(lead)
    result = prepare(tmp_path, transport, horizon_days=14)
    assert result['status'] == expected and len(transport.calls) == 1
    plan = json.loads((tmp_path/'prepare/preparation-plan.json').read_text())
    assert plan['horizon_days'] == 14 and plan['horizon_hours'] == 336
    assert plan['minimum_lead_minutes'] == 30
    assert transport.calls[0]['preparation_horizon_days'] == 14
    if expected == 'ready':
        binding = read_binding(result)
        assert binding['preparation_horizon_days'] == 14
        assert binding['canonical']['kickoff'] == (NOW+lead).isoformat()
        assert verify_official_fixture(binding, archive_root=result['identity_archive'], as_of=NOW) == binding
    else:
        assert not Path(result['identity_path']).exists()


@pytest.mark.parametrize('query', [EVENT, {'home':'HJK', 'away':'VPS'},
    {'home':'HJK', 'away':'VPS', 'date':'2026-10-17', 'timezone':'UTC'}])
def test_native_named_and_named_dated_queries_use_the_same_precommitted_window(tmp_path, query):
    old = prepare(tmp_path/'old', source(timedelta(days=10)), query=query)
    assert old['status'] in {'not_found', 'not_prematch'}
    new = prepare(tmp_path/'new', source(timedelta(days=10)), query=query, horizon_days=14)
    assert new['status'] == 'ready'


@pytest.mark.parametrize('value', [True, False, 7.0, 14.0, '14', None, 0, 8, 15, 365])
def test_other_windows_are_rejected_before_plan_or_request_creation(tmp_path, value):
    transport = Transport()
    with pytest.raises(ValueError, match='horizon_days'):
        prepare(tmp_path, transport, horizon_days=value)
    assert not transport.calls and not (tmp_path/'prepare').exists()


@pytest.mark.parametrize('before,after', [(7, 14), (14, 7)])
def test_resume_cannot_change_its_window_even_if_fixture_fits_both(tmp_path, before, after):
    transport = Transport()
    first = prepare(tmp_path, transport, horizon_days=before)
    snapshot = {p:p.read_bytes() for p in (tmp_path/'prepare').rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='resume.*horizon'):
        prepare(tmp_path, transport, horizon_days=after, resume=True)
    assert len(transport.calls) == 1
    assert all(p.read_bytes() == value for p,value in snapshot.items())
    second = prepare_official_fixture(EVENT, output=tmp_path/'prepare', horizon_days=before, resume=True,
        transport=transport, clock=lambda:NOW+timedelta(days=20))
    assert first == second and len(transport.calls) == 1


@pytest.mark.parametrize('change', ['binding_only', 'strip_binding', 'intent_and_binding'])
def test_rehashed_horizon_cannot_replace_the_actual_request_commitment(tmp_path, change):
    result = prepare(tmp_path, horizon_days=14 if change == 'strip_binding' else 7)
    binding = read_binding(result)
    if change == 'strip_binding':
        binding.pop('preparation_horizon_days')
    else:
        binding['preparation_horizon_days'] = 14
    binding['binding_sha256'] = _digest({k:v for k,v in binding.items() if k != 'binding_sha256'})
    if change == 'intent_and_binding':
        path = Path(result['request_intent_path'])
        intent = json.loads(path.read_text())
        intent['preparation_horizon_days'] = 14
        intent['request_digest'] = _digest({k:v for k,v in intent.items() if k != 'request_digest'})
        path.write_text(json.dumps(intent))
    with pytest.raises(ValueError):
        verify_official_fixture(binding, archive_root=result['identity_archive'], as_of=NOW)


def test_fourteen_day_raw_after_interruption_is_recovered_once_with_its_original_window(tmp_path):
    transport = source(timedelta(days=10))
    transport.interrupt = 'after_raw'
    with pytest.raises(KeyboardInterrupt):
        prepare(tmp_path, transport, horizon_days=14)
    with pytest.raises(ValueError, match='resume.*horizon'):
        prepare(tmp_path, transport, resume=True)
    result = prepare(tmp_path, transport, horizon_days=14, resume=True)
    assert result['status'] == 'ready' and len(transport.calls) == 1
    assert read_binding(result)['preparation_horizon_days'] == 14


def test_seven_day_binding_and_saved_old_identity_rebuild_without_new_fields(tmp_path):
    root = Path(__file__).resolve().parents[3]
    backup = root/'var/parallel/agent/round200/source-before/football_research/data_adapters/official_fixture.py'
    if not backup.is_file():
        pytest.skip('Original pre-window source snapshot is unavailable')
    spec = importlib.util.spec_from_file_location('football_research.data_adapters._pre_r200_official', backup)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    result = prepare(tmp_path, horizon_days=7)
    binding = read_binding(result)
    archive = RawArchive(Path(result['identity_archive']))
    obs = archive.observations()[0]
    assert old._binding(binding['query'], obs, archive.read_body(obs), at=NOW)['binding'] == binding
    assert old.verify_official_fixture(binding, archive_root=archive.root, as_of=NOW) == binding
    assert 'preparation_horizon_days' not in binding
    assert 'preparation_horizon_days' not in json.loads(Path(result['request_intent_path']).read_text())
    assert 'horizon_days' not in json.loads((tmp_path/'prepare/preparation-plan.json').read_text())
    actual = list((root/'var/parallel/agent/round198/live-cohort').glob('task-*/official-preparation/identity.json'))
    for path in actual:
        original = path.read_bytes()
        identity = json.loads(original)
        from football_research.acquisition.state import parse_time
        assert verify_official_fixture(identity, archive_root=path.parent/'identity-archive',
            as_of=parse_time(identity['verified_at'])) == identity
        assert path.read_bytes() == original and 'preparation_horizon_days' not in identity


def test_extended_identity_reaches_existing_independent_result_gateway_without_schema_changes(tmp_path):
    from test_official_fixture_results import Source, BEFORE, AS_OF
    from football_research.data_adapters.finland_result_capture import capture_finland_result
    source = Source()
    result = prepare_official_fixture('veikkausliiga:2026:match:'+source.native,
        output=tmp_path/'prepared', horizon_days=14, clock=lambda:BEFORE-timedelta(days=10), transport=source)
    assert result['status'] == 'ready'
    binding = read_binding(result)
    finished = capture_finland_result(binding, identity_archive_root=result['identity_archive'],
        output=tmp_path/'results', clock=lambda:AS_OF, transport=source)
    assert finished['status'] == 'VERIFIED' and len(finished['settlement_rows']) == 1
    assert [call['stage'] for call in source.calls] == ['official_schedule', 'schedule', 'detail']
