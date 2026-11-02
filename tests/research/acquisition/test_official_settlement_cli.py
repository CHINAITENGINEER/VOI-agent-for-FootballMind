"""Production capture entrypoint tests using archived synthetic transport returns."""
from copy import deepcopy
from datetime import datetime, timedelta
import json

import pytest

from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_research.data_adapters import verified_fixtures as gateway
from football_spider.storage.raw_archive import RawArchive
from test_verified_fixtures import (
    NOW, KICKOFF, AFTER, URL, identity, manifest, source_objects, body, save)
from test_official_settlement import native_result, API


def freeze_clock(monkeypatch, at):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at
    monkeypatch.setattr(gateway,'datetime',Clock)


def bundle_file(tmp_path, bindings):
    path = tmp_path/'bundle.json'
    path.write_text(json.dumps({'bindings':{b['match_id']:b for b in bindings}}),encoding='utf-8')
    return path


def second_binding(archive, *, kickoff):
    market = deepcopy(manifest()['market_initialization'])
    market['fixture'].update(match_id='spftll:match:m2',kickoff=kickoff.isoformat())
    market['quote']['match_id'] = 'spftll:match:m2'
    initial = MarketPriorPolicy(market['fixture'],market['quote'],frozen_at=market['frozen_at'],
        prediction_cutoff=kickoff-timedelta(minutes=1)).manifest
    event,match = source_objects(kickoff=kickoff)
    event['url'] = URL.replace('2047946','2047999')
    match['id'] = '2047999'
    obs = save(archive,body((event,match)),endpoint=event['url'])
    return gateway.verify_fixture_identity(initial,archive_root=archive.root,observation_ids=[obs.id],
        aliases={'home':['克罗地亚','Croatia'],'away':['西班牙','Spain']},as_of=NOW+timedelta(minutes=2))


def mock_collector(monkeypatch, *, at=AFTER, count=1, clock_delta=timedelta(), http_status=200):
    calls,constructors = [],[]
    class Collector:
        def __init__(self,archive,hosts,**kwargs):
            self.archive = archive
            constructors.append(hosts)
        def prepare(self,action,state):
            assert action['url'] in state['task']['seed_urls']
            return {'tool':'read_url','endpoint':action['url'],'params':{},'source':'open_page'}
        def execute(self,prepared,**kwargs):
            calls.append(prepared['endpoint'])
            data = native_result()
            raw = json.dumps(data).encode()
            obs = self.archive.save('open_page',prepared['endpoint'],{},raw,status=http_status,
                observed_at=at,context={'requested_at':(at+clock_delta).isoformat()})
            return {'observation':obs.model_dump(mode='json'),'http_attempts':count}
    monkeypatch.setattr('football_agent.inquiry_transport.RoutedOpenWebTools',Collector)
    def no_network(*args,**kwargs):
        raise AssertionError('CLI test attempted real HTTP')
    monkeypatch.setattr('requests.sessions.Session.request',no_network)
    return calls,constructors


def run_capture(bundle, archive, output, *, extra=()):
    return gateway.main(['capture-results','--identity-bundle',str(bundle),
        '--identity-archive',str(archive.root),'--output',str(output),*extra])


def test_official_capture_one_due_target_uses_canonical_id_and_replays_raw(tmp_path,monkeypatch):
    archive,binding = identity(tmp_path)
    bundle = bundle_file(tmp_path,[binding])
    freeze_clock(monkeypatch,AFTER)
    calls,constructors = mock_collector(monkeypatch)
    output = tmp_path/'capture'
    assert run_capture(bundle,archive,output,extra=['--result-source','official-api','--max-http-requests','1']) == 0
    assert calls == [API] and constructors == [['match.uefa.com']]
    plan = json.loads((output/'capture-plan.json').read_text(encoding='utf-8'))
    assert plan['schema'] == 'independent-results-capture-v2'
    assert plan['http_requests_planned']==1 and plan['cases'][0]['status']=='DUE'
    receipts = json.loads((output/'capture-receipts.json').read_text(encoding='utf-8'))
    assert receipts['http_requests_actual']==1
    expected = json.loads((output/'verification.json').read_text(encoding='utf-8'))
    replay = tmp_path/'replayed.json'
    gateway.main(['verify-results','--identity-bundle',str(bundle),'--identity-archive',str(archive.root),
        '--result-archive',str(output),'--output',str(replay)])
    assert json.loads(replay.read_text(encoding='utf-8')) == expected
    assert len(expected['settlement_rows'])==1 and calls==[API]


def test_official_capture_skips_not_due_target_without_constructing_transport(tmp_path,monkeypatch):
    archive,binding = identity(tmp_path)
    freeze_clock(monkeypatch,KICKOFF+timedelta(minutes=89))
    calls,constructors = mock_collector(monkeypatch)
    output = tmp_path/'capture'
    run_capture(bundle_file(tmp_path,[binding]),archive,output,extra=['--result-source','official-api'])
    plan = json.loads((output/'capture-plan.json').read_text(encoding='utf-8'))
    assert plan['cases'][0]['status']=='NOT_DUE'
    assert plan['cases'][0]['reason']=='kickoff_plus_90_minutes_not_reached'
    assert plan['urls']==[] and plan['http_requests_planned']==0
    assert calls==constructors==[] and RawArchive(output).observations()==[]
    assert json.loads((output/'verification.json').read_text(encoding='utf-8'))['settlement_rows']==[]


def test_mixed_cohort_only_requests_due_event_and_preserves_skipped_case(tmp_path,monkeypatch):
    archive,binding = identity(tmp_path)
    other = second_binding(archive,kickoff=AFTER+timedelta(hours=1))
    assert other['status']=='CONFIRMED'
    freeze_clock(monkeypatch,AFTER)
    calls,_ = mock_collector(monkeypatch)
    output = tmp_path/'capture'
    run_capture(bundle_file(tmp_path,[binding,other]),archive,output,
        extra=['--result-source','official-api','--max-http-requests','1'])
    plan = json.loads((output/'capture-plan.json').read_text(encoding='utf-8'))
    assert calls==[API] and [case['status'] for case in plan['cases']]==['DUE','NOT_DUE']


def test_official_capture_rejects_too_small_budget_before_directory_or_io(tmp_path,monkeypatch):
    archive,binding = identity(tmp_path)
    other = second_binding(archive,kickoff=KICKOFF)
    freeze_clock(monkeypatch,AFTER)
    calls,constructors = mock_collector(monkeypatch)
    output = tmp_path/'capture'
    with pytest.raises(ValueError,match='budget'):
        run_capture(bundle_file(tmp_path,[binding,other]),archive,output,
            extra=['--result-source','official-api','--max-http-requests','1'])
    assert not output.exists() and calls==constructors==[]


@pytest.mark.parametrize('attack',['attempts','clock'])
def test_official_capture_retains_raw_but_rejects_false_attempt_or_clock_receipt(tmp_path,monkeypatch,attack):
    archive,binding = identity(tmp_path)
    freeze_clock(monkeypatch,AFTER)
    calls,_ = mock_collector(monkeypatch,count=2 if attack=='attempts' else 1,
        clock_delta=timedelta(seconds=1) if attack=='clock' else timedelta())
    output = tmp_path/'capture'
    with pytest.raises(ValueError,match='request count, identity or actual receipt clock'):
        run_capture(bundle_file(tmp_path,[binding]),archive,output,extra=['--result-source','official-api'])
    assert calls==[API] and len(RawArchive(output).observations())==1
    assert (output/'response-0.json').exists() and not (output/'verification.json').exists()


def test_official_capture_http_failure_is_archived_once_and_not_retried(tmp_path,monkeypatch):
    archive,binding = identity(tmp_path)
    freeze_clock(monkeypatch,AFTER)
    calls,_ = mock_collector(monkeypatch,http_status=503)
    output = tmp_path/'capture'
    run_capture(bundle_file(tmp_path,[binding]),archive,output,extra=['--result-source','official-api'])
    assert calls==[API] and RawArchive(output).observations()[0].http_status==503
    assert json.loads((output/'verification.json').read_text(encoding='utf-8'))['settlement_rows']==[]


def test_default_capture_keeps_legacy_html_urls_plan_and_hosts(tmp_path,monkeypatch):
    archive,binding = identity(tmp_path)
    freeze_clock(monkeypatch,AFTER)
    calls,constructors = mock_collector(monkeypatch,http_status=503)
    output = tmp_path/'capture'
    run_capture(bundle_file(tmp_path,[binding]),archive,output)
    plan = json.loads((output/'capture-plan.json').read_text(encoding='utf-8'))
    assert calls==[gateway.SCHEDULE_URL,URL] and constructors==[['www.uefa.com']]
    assert plan['schema']=='independent-results-capture-v1'
    assert 'cases' not in plan and 'result_source' not in plan
    assert not (output/'capture-receipts.json').exists()
