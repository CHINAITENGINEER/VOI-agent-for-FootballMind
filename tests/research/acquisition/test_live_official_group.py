"""Group selection is a frozen backend setting, not an Actor research topic."""
from copy import deepcopy
from datetime import datetime, timezone
import pytest
from zoneinfo import ZoneInfo

from football_agent import live_match_workflow as workflow
from football_agent import evidence_live_workflow as evidence
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_research.data_adapters import system_tls_source
from football_spider.storage.raw_archive import RawArchive
from test_evidence_live_workflow import job, read, write
from test_official_fixture_group3 import schedule3
from test_system_tls_source import FakeCurl, install, spec


@pytest.mark.parametrize('group', [None, True, False, 2, 3, '1', '03', '', {}])
def test_invalid_group_rejected_before_job_and_io(job, group):
    settings = read(job['settings']); settings['official_schedule_group'] = group
    write(job['settings'], settings)
    with pytest.raises(ValueError, match='schedule group'):
        workflow.start_match('veikkausliiga:2026:match:4353581', job['output'], settings_path=job['settings'])
    assert not job['output'].exists() and not any(job['calls'].values())


def test_group3_flows_through_original_collector_resume_and_replay(job, monkeypatch):
    local = job['kickoff'].astimezone(ZoneInfo('Europe/Helsinki'))
    body = schedule3(date=local.strftime('%Y%m%d'), clock=local.strftime('%H:%M'))
    def transport(request, *, archive, clock):
        job['calls']['prepare'] += 1
        assert request['params'] == {'group': '3'}
        return archive.save(request['source'], request['url'], request['params'], body, status=200,
            observed_at=clock(), context={'requested_at': request['requested_at'],
                'preparation_request_digest': request['request_digest']})
    monkeypatch.setattr(evidence, 'prepare_official_fixture',
                        lambda query, **kwargs: prepare_official_fixture(query, transport=transport, **kwargs))
    settings = read(job['settings']); settings.update(official_schedule_group='3', news_projection_version=6)
    write(job['settings'], settings)
    result = workflow.start_match('veikkausliiga:2026:match:4353581', job['output'], settings_path=job['settings'])
    assert result['status'] == 'awaiting_result_no_prior', result
    identity = read(job['output']/'official-preparation/identity.json')
    assert identity['canonical']['stage'] == 'Karsintasarja' and identity['schedule_group'] == '3'
    assert result['initial_probabilities'] is result['final_probabilities'] is None
    saved = deepcopy(job['calls'])
    settings['official_schedule_group'] = '2'; write(job['settings'], settings)
    assert workflow.resume_match(job['output']) == result and job['calls'] == saved
    replay = workflow.replay_match(job['output'], job['root']/'replay-group3')
    assert replay['identical_replay'] and replay['replay_network_requests'] == 0
    assert job['calls'] == saved


def test_transport_accepts_only_registered_group3_request_without_tls_weakening(tmp_path, monkeypatch):
    fake = FakeCurl(body=b'<html>official lower-stage fixture table</html>')
    system_tls_source._client.cache_clear()
    install(monkeypatch, fake)
    url = 'https://www.veikkausliiga.com/tilastot/2026/veikkausliiga/ottelut/'
    observation = system_tls_source.capture_system_tls_page(spec(url, params={'group': '3'}),
        archive=RawArchive(tmp_path/'valid'), clock=lambda: datetime.now(timezone.utc))
    assert observation.params == {'group': '3'} and observation.http_status == 200
    assert observation.context['transport']['tls_verification'] is True
    count = len(fake.calls)
    for params in ({'group': '1'}, {'group': 3}, {'group': '3', 'extra': '1'}):
        with pytest.raises(ValueError):
            system_tls_source.capture_system_tls_page(spec(url, params=params),
                archive=RawArchive(tmp_path/'invalid'), clock=lambda: datetime.now(timezone.utc))
    assert len(fake.calls) == count
    system_tls_source._client.cache_clear()
