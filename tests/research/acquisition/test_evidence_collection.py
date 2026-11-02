"""Actual collector/replay with constructed HTTP bytes, never a live request."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import socket
from zoneinfo import ZoneInfo

import pytest

from football_agent.learned_voi_replay import replay_saved_run
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.data_adapters import evidence_collection as collection
from football_research.data_adapters.prospective_learning import _previous_inventory, import_training_exposure
from test_official_fixture import Transport, read_binding, EVENT
from test_finland_stage_fixture import schedule
from football_research.data_adapters.official_fixture import prepare_official_fixture

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = 'https://www.veikkausliiga.com/uutiset/2026/10/07/hjk-vps-squad-preview/'
NOW = datetime.now(timezone.utc)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('Evidence collection integration may only use constructed raw responses')
    for owner, name in ((socket, 'getaddrinfo'), (socket, 'create_connection'),
                        (socket.socket, 'connect'), (socket.socket, 'connect_ex')):
        monkeypatch.setattr(owner, name, blocked)


def inputs(root):
    kickoff = (NOW+timedelta(hours=6)).astimezone(ZoneInfo('Europe/Helsinki'))
    prepared = prepare_official_fixture(EVENT,output=root/'prepare',clock=lambda:NOW,
        transport=Transport(schedule(date=kickoff.strftime('%Y%m%d'),clock=kickoff.strftime('%H:%M'))))
    binding = read_binding(prepared)
    plan = collection.prepare_evidence_plan(binding, previous_plans=[], created_at=NOW)
    path = root/'learning-plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    return prepared, binding, plan, path


def transport(monkeypatch, *, fail=False, interrupt=False):
    calls = []
    def execute(self, prepared, *, run_id, request_id):
        calls.append(deepcopy(prepared))
        if interrupt:
            raise OSError('Constructed uncertain return; do not retry')
        if prepared['tool'] == 'search_web':
            body = b'<rss><channel></channel></rss>'
        elif prepared['endpoint'] == ARTICLE:
            body = ('<html><script type="application/ld+json">'+json.dumps({
                '@type':'NewsArticle', 'headline':'HJK VPS team news',
                'datePublished':(NOW-timedelta(minutes=2)).isoformat(),
                'articleBody':'UNREAD_BODY_SENTINEL HJK may rotate their starting lineup against VPS.'
            })+'</script></html>').encode()
        else:
            body = ('<html><main>Official competition news and schedule.'
                    f'<a href="{ARTICLE}">Observed HJK VPS match preview</a></main></html>').encode()
        now = datetime.now(timezone.utc)
        obs = self.archive.save(prepared['source'], prepared['endpoint'], prepared['params'], body,
            status=503 if fail else 200, observed_at=now,
            context={'requested_at':now.isoformat(), 'inquiry_run_id':run_id})
        return {'document':self.project(obs, prepared), 'observation':obs.model_dump(mode='json'),
            'http_attempts':1, 'money':None, 'money_status':'not_measured', 'latency_seconds':.01}
    monkeypatch.setattr(NewsDiscoveryTools, 'execute', execute)
    monkeypatch.setattr(collection, 'uuid4', lambda:SimpleNamespace(hex='constructed-evidence-198'))
    return calls


def collect(root, prepared, plan_path, **kwargs):
    return collection.collect_official_evidence(prepared['identity_path'],
        identity_archive=prepared['identity_archive'], learning_plan_path=plan_path,
        previous_plan_paths=[], output=root/'run', **kwargs)


def test_official_identity_enters_real_executor_without_a_probability_model_and_replays(tmp_path, monkeypatch):
    prepared, binding, plan, path = inputs(tmp_path)
    calls = transport(monkeypatch)
    report = collect(tmp_path, prepared, path)
    run, config = read(tmp_path/'run/run.json'), read(tmp_path/'run/configuration.json')
    assert report['http_requests'] == len(calls) == 4
    assert report['initial_probabilities'] is report['final_probabilities'] is None
    assert config['schema'] == 'learned-voi-open-execution-v5'
    assert config['supervisor_metadata']['generator']['version'] == 'evidence-document-candidates-v11'
    assert collection.verify_evidence_run(tmp_path/'run', config) == binding
    assert not (tmp_path/'run/prior-archive').exists()
    assert not list((tmp_path/'run').glob('*model.json'))
    assert plan['numeric_training_eligible'] is False
    assert ARTICLE not in json.dumps(run['cycles'][0]['generated_actions'])
    assert any(ARTICLE in json.dumps(c['generated_actions']) for c in run['cycles'][1:])
    for cycle in run['cycles']:
        assert cycle['prediction_before'] is cycle['prediction_after'] is None
        if cycle['selected_action']:
            assert cycle['selection_basis'] == 'bounded_unknown_information_exploration'
            assert cycle['public_after']['prediction']['probabilities'] is None
            if cycle['selected_action'].get('url') == ARTICLE:
                assert 'UNREAD_BODY_SENTINEL' not in json.dumps(cycle['public_before'])
    checked = replay_saved_run(tmp_path/'run', output=tmp_path/'replay', project_root=ROOT)
    assert checked['identical_replay'] and checked['raw_responses_reprojected'] == 4
    assert checked['replay_network_requests'] == 0 and len(calls) == 4
    assert (tmp_path/'run/run.json').read_bytes() == (tmp_path/'replay/run.json').read_bytes()
    assert all(r['numeric_voi_training_eligible'] is False and r['estimated_net_voi'] is None
               for r in read(tmp_path/'replay/learning-records.json'))
    with pytest.raises(FileExistsError):
        collect(tmp_path, prepared, path)
    with pytest.raises(FileExistsError):
        collect(tmp_path/'another-output', prepared, path)
    assert len(calls) == 4


def test_failed_http_is_still_billed_and_replayable_without_an_invented_prediction(tmp_path, monkeypatch):
    prepared, _, _, path = inputs(tmp_path)
    calls = transport(monkeypatch, fail=True)
    report = collect(tmp_path, prepared, path, requests_per_match=2)
    run = read(tmp_path/'run/run.json')
    assert report['http_requests'] == len(calls) == 2
    assert all(row['status'] == 'FAILED' for row in run['attempts'])
    assert run['documents'] == [] and run['final_probabilities'] is None
    assert replay_saved_run(tmp_path/'run', output=tmp_path/'replay', project_root=ROOT)['returns'] == 2


def test_unreconciled_return_is_not_reissued_or_claimed_as_a_complete_replay(tmp_path, monkeypatch):
    prepared, _, _, path = inputs(tmp_path)
    calls = transport(monkeypatch, interrupt=True)
    report = collect(tmp_path, prepared, path)
    assert len(calls) == 1
    assert report['terminal_selection']['reason'] == 'execution_reconciliation_required'
    with pytest.raises(FileExistsError):
        collect(tmp_path/'retry', prepared, path)
    with pytest.raises(ValueError, match='reconciled'):
        replay_saved_run(tmp_path/'run', output=tmp_path/'replay', project_root=ROOT)
    assert len(calls) == 1


def test_prior_unavailable_exposure_blocks_same_physical_fixture_in_later_cohorts(tmp_path):
    _, binding, plan, _ = inputs(tmp_path)
    inventory = _previous_inventory([plan], NOW+timedelta(seconds=1))
    assert inventory[0]['source_event_id'] == binding['match_id']
    assert inventory[0]['purpose'] == 'f_train' and inventory[0]['model_bindings'] == {}
    with pytest.raises(ValueError, match='already registered'):
        collection.prepare_evidence_plan(binding, previous_plans=[plan], created_at=NOW+timedelta(seconds=1))


@pytest.mark.parametrize('part', ['identity', 'settings', 'registration', 'capabilities'])
def test_replay_cannot_change_identity_budget_registration_or_tools(tmp_path, monkeypatch, part):
    prepared, _, _, path = inputs(tmp_path)
    transport(monkeypatch)
    collect(tmp_path, prepared, path, requests_per_match=1)
    config = read(tmp_path/'run/configuration.json')
    meta = config['supervisor_metadata']
    if part == 'identity':
        meta['verified_fixture_identity']['canonical']['home_name'] = 'Another club'
    elif part == 'settings':
        meta['evidence_collection']['settings']['requests_per_match'] = 4
    elif part == 'registration':
        meta['evidence_collection']['learning_plan']['purpose'] = 'confirmation'
    else:
        meta['generator']['allowed_hosts'].append('example.org')
    with pytest.raises(ValueError):
        collection.verify_evidence_run(tmp_path/'run', config)


def test_evidence_run_cannot_be_reimported_as_market_based_numeric_training_exposure(tmp_path, monkeypatch):
    prepared, _, _, path = inputs(tmp_path)
    transport(monkeypatch)
    collect(tmp_path, prepared, path, requests_per_match=1)
    with pytest.raises(ValueError, match='Evidence-only'):
        import_training_exposure([tmp_path/'run'], created_at=datetime.now(timezone.utc))
