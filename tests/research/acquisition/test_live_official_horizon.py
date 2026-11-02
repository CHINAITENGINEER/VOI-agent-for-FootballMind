"""Explicit longer lead is frozen through the real job/collector/replay path."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import socket
from zoneinfo import ZoneInfo

import pytest

from football_agent import live_match_workflow as workflow
from football_agent import evidence_live_workflow as evidence
from football_research.data_adapters.official_fixture import prepare_official_fixture
from test_evidence_live_workflow import job, read, write
from test_finland_stage_fixture import schedule


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('Horizon integration uses constructed raw sources only')
    for owner, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'), (socket.socket, 'connect')):
        monkeypatch.setattr(owner, name, denied)


@pytest.mark.parametrize('days', [True, 8, 15, 14.0, '14'])
def test_invalid_horizon_fails_before_job_or_io(job, days):
    settings = read(job['settings']); settings['official_horizon_days'] = days
    write(job['settings'], settings)
    with pytest.raises(ValueError, match='horizon'):
        workflow.start_match(job['query'], job['output'], settings_path=job['settings'])
    assert not job['output'].exists() and job['calls'] == {'prepare': 0, 'actor': 0, 'result': 0}


def test_ten_day_fixture_is_collected_and_replayed_with_frozen_explicit_horizon(job, monkeypatch):
    kickoff = (datetime.now(timezone.utc) + timedelta(days=10)).replace(second=0, microsecond=0)
    local = kickoff.astimezone(ZoneInfo('Europe/Helsinki'))
    body = schedule(date=local.strftime('%Y%m%d'), clock=local.strftime('%H:%M'))
    def transport(spec, *, archive, clock):
        job['calls']['prepare'] += 1
        return archive.save(spec['source'], spec['url'], spec['params'], body, status=200,
            observed_at=clock(), context={'requested_at': spec['requested_at'],
                'preparation_request_digest': spec['request_digest']})
    def prepare(query, **kwargs):
        return prepare_official_fixture(query, transport=transport, **kwargs)
    monkeypatch.setattr(evidence, 'prepare_official_fixture', prepare)
    settings = read(job['settings'])
    settings.update(official_horizon_days=14, news_projection_version=6)
    write(job['settings'], settings)
    result = workflow.start_match(job['query'], job['output'], settings_path=job['settings'])
    assert result['status'] == 'awaiting_result_no_prior', result
    request = read(job['output'] / 'request.json')
    assert request['settings']['official_horizon_days'] == 14
    binding = read(job['output'] / 'official-preparation/identity.json')
    assert binding['preparation_horizon_days'] == 14
    assert read(job['output'] / 'official-preparation/preparation-plan.json')['horizon_hours'] == 336
    assert result['initial_probabilities'] is result['final_probabilities'] is None
    before = deepcopy(job['calls'])
    settings['official_horizon_days'] = 7; write(job['settings'], settings)
    assert workflow.resume_match(job['output']) == result and job['calls'] == before
    replay = workflow.replay_match(job['output'], job['root'] / 'replay-horizon')
    assert replay['identical_replay'] and replay['replay_network_requests'] == 0
    assert replay['raw_responses_reprojected'] == job['calls']['actor']
