"""Constructed raw schedules only; planning must perform no HTTP or exposure."""
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path

import pytest

from football_research.data_adapters.evidence_collection import prepare_evidence_plan
from football_research.data_adapters.official_cohort import plan_official_cohort
from football_research.data_adapters.official_fixture import _binding, prepare_official_fixture
from football_spider.storage.raw_archive import RawArchive
from test_finland_stage_fixture import schedule
from test_official_fixture import Transport

NOW = datetime(2026, 10, 7, 9, tzinfo=timezone.utc)
PREFIX = 'veikkausliiga:2026:match:'
OLD = [
    ('IF Gnistan', 'FC Inter', '4353542', '20261007', '19:00'),
    ('HJK', 'VPS', '4353553', '20261008', '18:00'),
    ('KuPS', 'AC Oulu', '4353563', '20261008', '19:00'),
    ('FC Inter', 'KuPS', '4353554', '20261011', '15:00'),
    ('VPS', 'IF Gnistan', '4353550', '20261012', '18:00'),
    ('AC Oulu', 'HJK', '4353543', '20261012', '19:00'),
]
NEW = [
    ('HJK', 'IF Gnistan', '4353570', '20261017', '16:00'),
    ('FC Inter', 'AC Oulu', '4353567', '20261017', '18:00'),
    ('KuPS', 'VPS', '4353559', '20261018', '15:00'),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    import subprocess
    def denied(*args, **kwargs):
        pytest.fail('The official cohort planner cannot open sockets or run HTTP helpers')
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(subprocess, 'Popen', denied)


def page(rows):
    bodies = [schedule(*row) for row in rows]
    return bodies[0].replace(b'</table>', b''.join(
        body[body.index(b'<tr '):body.index(b'</tr>')+5] for body in bodies[1:]) + b'</table>')


def setup_source(root, *, rows=None, exposed=OLD):
    body = page(rows or [NEW[2], OLD[1], NEW[0], *OLD[2:], OLD[0], NEW[1],
                        ('HJK', 'KuPS', '4353561', '20261025', '15:00')])
    transport = Transport(body)
    prepared = prepare_official_fixture(PREFIX+'4353553', output=root/'source',
                                       clock=lambda: NOW, transport=transport)
    assert prepared['status'] == 'ready' and len(transport.calls) == 1
    archive = RawArchive(Path(prepared['identity_archive']))
    observation, = archive.observations()
    previous, paths = [], []
    for row in exposed:
        binding = _binding({'native_event_id': row[2]}, observation, body, at=NOW)['binding']
        plan = prepare_evidence_plan(binding, previous_plans=previous, created_at=NOW)
        path = root/'registrations'/f'{row[2]}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(plan), encoding='utf-8')
        previous.append(plan)
        paths.append(path)
    return {'identity_path': prepared['identity_path'], 'identity_archive': prepared['identity_archive'],
            'previous_plan_paths': paths}, transport


def test_default_seven_days_empty_and_explicit_fourteen_days_three_new_fixtures(tmp_path):
    kwargs, transport = setup_source(tmp_path)
    seven = plan_official_cohort(**kwargs, output=tmp_path/'seven.json', as_of=NOW)
    fourteen = plan_official_cohort(**kwargs, output=tmp_path/'fourteen.json', horizon_days=14, as_of=NOW)
    assert seven['queries'] == [] and seven['counts']['eligible_unexposed'] == 0
    assert fourteen['queries'] == [PREFIX+row[2] for row in NEW]
    assert fourteen['counts'] == dict(observed_native_ids=10, registered_physical_matches=6,
                                     eligible_unexposed=3, selected=3, rejected=7)
    assert len(transport.calls) == 1 and fourteen['http_requests'] == fourteen['new_registrations'] == 0
    old = next(r for r in fourteen['candidates'] if r['native_event_id']=='4353542')
    assert old['source_event_id']=='palloliitto:match:4353542'
    assert old['reason']=='already_registered_physical_fixture'
    assert fourteen['fresh_identity_revalidation_required_before_acquisition']
    encoded = json.dumps(fourteen)
    assert all(field not in encoded for field in ('home_score', 'away_score', 'score_present', 'probabilities'))
    for path, expected in fourteen['input_sha256'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == expected


def test_ordered_limit_does_not_reselect_after_missing_news_and_output_is_reproducible(tmp_path):
    kwargs, _ = setup_source(tmp_path)
    first, second = tmp_path/'first.json', tmp_path/'other'/'second.json'
    a = plan_official_cohort(**kwargs, output=first, horizon_days=14, max_matches=2, as_of=NOW)
    b = plan_official_cohort(**kwargs, output=second, horizon_days=14, max_matches=2, as_of=NOW)
    assert a == b and first.read_bytes() == second.read_bytes()
    assert a['queries'] == [PREFIX+NEW[0][2], PREFIX+NEW[1][2]]
    assert next(r for r in a['candidates'] if r['native_event_id']==NEW[2][2])['reason']=='cohort_limit_reached'
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        plan_official_cohort(**kwargs, output=first, as_of=NOW)
    assert first.read_bytes() == before


@pytest.mark.parametrize('argument,value', [
    ('max_matches', True), ('max_matches', 0), ('max_matches', 21),
    ('max_matches', 2.0), ('max_matches', '2'), ('horizon_days', True),
    ('horizon_days', 8), ('horizon_days', 14.0), ('horizon_days', '14'),
    ('previous_plan_paths', 'not-an-explicit-sequence'),
    ('as_of', '2026-10-07T09:00:00'),
])
def test_invalid_parameters_do_not_create_a_plan(tmp_path, argument, value):
    kwargs, _ = setup_source(tmp_path)
    kwargs.update(as_of=NOW)
    kwargs[argument] = value
    with pytest.raises(ValueError):
        plan_official_cohort(**kwargs, output=tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()


def test_future_clock_and_output_inside_original_archive_are_rejected(tmp_path):
    kwargs, _ = setup_source(tmp_path)
    with pytest.raises(ValueError, match='future'):
        plan_official_cohort(**kwargs, output=tmp_path/'future.json',
                             as_of=datetime.now(timezone.utc)+timedelta(hours=1))
    for target in (Path(kwargs['identity_archive'])/'plan.json', Path(kwargs['identity_path']).parent/'plan.json'):
        with pytest.raises(ValueError, match='outside'):
            plan_official_cohort(**kwargs, output=target, as_of=NOW)
        assert not target.exists()


@pytest.mark.parametrize('tamper', ['binding', 'body', 'observation_params', 'registration'])
def test_changed_identity_raw_params_or_exposure_plan_fail_closed(tmp_path, tamper):
    kwargs, _ = setup_source(tmp_path)
    archive = RawArchive(Path(kwargs['identity_archive']))
    observation, = archive.observations()
    if tamper == 'body':
        (archive.root/observation.payload_path).write_bytes(gzip.compress(b'<html>changed</html>'))
    else:
        path = (Path(kwargs['identity_path']) if tamper=='binding' else
                kwargs['previous_plan_paths'][0] if tamper=='registration' else
                next(archive.root.rglob('*.observation.json')))
        value = json.loads(path.read_text(encoding='utf-8'))
        if tamper=='binding':
            value['canonical']['away_name'] = 'Unknown'
        elif tamper=='registration':
            value['purpose'] = 'confirmation'
        else:
            value['params'] = {'group': '1'}
        path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        plan_official_cohort(**kwargs, output=tmp_path/'result.json', horizon_days=14, as_of=NOW)
    assert not (tmp_path/'result.json').exists()


def test_duplicate_unsupported_and_non_prematch_rows_keep_rejection_reasons(tmp_path):
    rows = [OLD[1], NEW[0], NEW[0], ('Unknown FC', 'VPS', '9999001', '20261018', '15:00'),
            ('KuPS', 'VPS', '9999002', '20261018', '15:00', '0 — 0')]
    kwargs, _ = setup_source(tmp_path, rows=rows, exposed=[OLD[1]])
    plan = plan_official_cohort(**kwargs, output=tmp_path/'result.json', horizon_days=14, as_of=NOW)
    decisions = {r['native_event_id']: r for r in plan['candidates']}
    assert plan['queries'] == []
    assert decisions[NEW[0][2]]['resolver_status']=='ambiguous'
    assert decisions['9999001']['reason']=='official_target_team_or_squad_not_registered'
    assert decisions['9999002']['reason']=='official_target_has_score_or_non_scheduled_state'
    assert plan['counts']['observed_native_ids']==4
