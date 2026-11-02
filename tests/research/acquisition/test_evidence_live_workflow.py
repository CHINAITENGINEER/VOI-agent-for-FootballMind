"""Actual local official prepare/collector/replay chain over constructed bytes."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import socket
from zoneinfo import ZoneInfo

import pytest

from football_agent import live_match_workflow as w
from football_agent import evidence_live_workflow as e
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_spider.storage.raw_archive import RawArchive
from test_finland_stage_fixture import schedule


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*a,**k):raise AssertionError('The workflow mechanism uses no live network')
    monkeypatch.setattr(socket,'getaddrinfo',deny)
    monkeypatch.setattr(socket,'create_connection',deny)
    monkeypatch.setattr(socket.socket,'connect',deny)


@pytest.fixture
def job(tmp_path,monkeypatch):
    now=datetime.now(timezone.utc)
    kickoff=(now+timedelta(hours=6)).replace(second=0,microsecond=0)
    local=kickoff.astimezone(ZoneInfo('Europe/Helsinki'))
    body=schedule(date=local.strftime('%Y%m%d'),clock=local.strftime('%H:%M'))
    settings={'schema':w.SETTINGS_VERSION,'purpose':'f_train','preparation_http_budget':3,
        'actor_requests_per_match':2,'seed':198,'route':'direct','settlement_delay_minutes':130,
        'settlement_retry_minutes':60,'settlement_max_attempts':2,'task_root':str(tmp_path/'jobs'),
        'previous_learning_plans':[],'official_evidence_fallback':True}
    settings_path=tmp_path/'settings.json';write(settings_path,settings)
    calls={'prepare':0,'actor':0,'result':0}
    def transport(spec,*,archive,clock):
        calls['prepare']+=1
        return archive.save(spec['source'],spec['url'],spec['params'],body,status=200,observed_at=clock(),
            context={'requested_at':spec['requested_at'],'preparation_request_digest':spec['request_digest']})
    def prepare(query,**kwargs):return prepare_official_fixture(query,transport=transport,**kwargs)
    monkeypatch.setattr(e,'prepare_official_fixture',prepare)
    def execute(self,prepared,*,run_id,request_id):
        calls['actor']+=1
        observed=datetime.now(timezone.utc)
        raw=b'<rss><channel></channel></rss>' if prepared['tool']=='search_web' else b'<html><main>Official schedule navigation</main></html>'
        observation=self.archive.save(prepared['source'],prepared['endpoint'],prepared['params'],raw,
            status=200,observed_at=observed,context={'requested_at':observed.isoformat()})
        return {'document':self.project(observation,prepared),'observation':observation.model_dump(mode='json'),
                'http_attempts':1,'money':None,'money_status':'not_measured','latency_seconds':.01}
    monkeypatch.setattr(NewsDiscoveryTools,'execute',execute)
    return {'root':tmp_path,'output':tmp_path/'jobs/case','settings':settings_path,'calls':calls,
            'kickoff':kickoff,'body':body,'query':'veikkausliiga:2026:match:4353553'}


def start(job,query=None):
    return w.start_match(query or job['query'],job['output'],settings_path=job['settings'])


def test_official_start_registers_evidence_and_replays_without_prior_or_labels(job):
    result=start(job)
    assert result['status']=='awaiting_result_no_prior',result
    assert result['preparation_requests']==1 and result['actor_requests']==job['calls']['actor']<=2
    assert result['numeric_voi_training_eligible'] is False and result['qualified_response_rows']==0
    assert result['numeric_voi_status']=='prior_unavailable'
    assert 'learning-plan.json.acquisition.json' in result['actor_sha256']
    assert result['initial_probabilities'] is result['final_probabilities'] is None
    folder=job['output']/result['run_directory']
    assert not (folder/'prior-bindings.json').exists() and not (folder/'prior-archive').exists()
    config=read(folder/'configuration.json')
    assert config['schema']=='learned-voi-open-execution-v5'
    assert config['supervisor_metadata']['evidence_collection']['learning_plan']==read(job['output']/'learning-plan.json')
    before=deepcopy(job['calls'])
    assert w.resume_match(job['output'],settle=True)==result
    assert w.tick_tasks(job['root']/'jobs')['tasks_advanced']==0 and job['calls']==before
    replay=w.replay_match(job['output'],job['root']/'replay')
    assert replay['identical_replay'] and replay['replay_network_requests']==0
    assert replay['raw_responses_reprojected']==job['calls']['actor']


@pytest.mark.parametrize('damage',[None,'receipt','claim'])
def test_complete_actor_adopts_without_retry_but_missing_receipt_reconciles(job,monkeypatch,damage):
    real=e.adopt_collection
    def stopped(*a,**k):raise RuntimeError('lost actor completion marker')
    monkeypatch.setattr(e,'adopt_collection',stopped)
    with pytest.raises(RuntimeError,match='marker'):start(job)
    state=read(job['output']/'state.json');assert state['status']=='collecting_evidence'
    if damage=='receipt':(job['output']/state['run_directory_pending']/'receipt-000.json').unlink()
    elif damage=='claim':
        path=job['output']/'learning-plan.json.acquisition.json'
        claim=read(path);claim['run_id']='changed-claim-run';write(path,claim)
    monkeypatch.setattr(e,'adopt_collection',real)
    before=deepcopy(job['calls'])
    result=w.resume_match(job['output'])
    assert result['status']==('reconciliation_required' if damage else 'awaiting_result_no_prior'),result
    assert job['calls']==before


def test_partial_actor_is_not_reissued_and_prior_exposure_blocks_another_job(job,monkeypatch):
    def interrupted(*a,output,**k):
        Path(output).mkdir(parents=True)
        raise RuntimeError('unknown external request state')
    monkeypatch.setattr(e,'collect_official_evidence',interrupted)
    result=start(job)
    assert result['status']=='reconciliation_required'
    assert w.resume_match(job['output'])['status']=='reconciliation_required'
    duplicate=w.start_match(job['query'],job['root']/'jobs/again',settings_path=job['settings'])
    assert duplicate['status']=='duplicate_fixture' and job['calls']['actor']==0


@pytest.mark.parametrize('budget,unresolved,enabled,expected',[
    (2,False,True,'awaiting_result_no_prior'),(1,False,True,'unavailable'),
    (2,True,True,'unavailable'),(2,False,False,'unavailable')])
def test_named_market_failure_falls_back_only_after_complete_accounted_requests(job,monkeypatch,budget,unresolved,enabled,expected):
    import football_research.data_adapters.live_fixture_preparation as preparation
    settings=read(job['settings']);settings['preparation_http_budget']=budget
    if enabled:settings['official_evidence_fallback']=True
    else:settings.pop('official_evidence_fallback')
    write(job['settings'],settings)
    def failed(query,*,output,**kwargs):
        output=Path(output);output.mkdir(parents=True)
        write(output/'requests/fixture.intent.json',{'constructed':True})
        if not unresolved:
            write(output/'requests/fixture.receipt.json',{'constructed':True})
            RawArchive(output/'market-archive').save('spftll','https://fixture.invalid/',{},b'',status=503,
                observed_at=datetime.now(timezone.utc))
        result={'status':'unavailable','reason':'constructed_market_unavailable','request_intents':1,
                'archived_http_responses':0 if unresolved else 1}
        write(output/'prepare-result.json',result)
        return result
    monkeypatch.setattr(preparation,'prepare_live_fixture',failed)
    result=start(job,'HJK vs VPS')
    assert result['status']==expected,result
    assert job['calls']['prepare']==int(expected=='awaiting_result_no_prior')
    if expected=='awaiting_result_no_prior':assert result['preparation_requests']==2
    else:
        before=deepcopy(job['calls']);w.resume_match(job['output']);assert job['calls']==before


def test_expired_official_snapshot_does_not_start_an_actor(job,monkeypatch):
    real=e._register_and_collect
    def expire(output,request,state):
        monkeypatch.setattr(w,'_now',lambda:job['kickoff'])
        return real(output,request,state)
    monkeypatch.setattr(e,'_register_and_collect',expire)
    result=start(job)
    assert result['status']=='expired' and job['calls']['actor']==0


@pytest.mark.parametrize('damage',[False,True])
def test_independent_official_result_is_retained_without_numeric_training_join(job,monkeypatch,damage):
    from test_official_fixture_results import Source
    from football_research.data_adapters.finland_result_capture import capture_finland_result
    import football_research.data_adapters.settlement as settlement_module
    result=start(job);assert result['status']=='awaiting_result_no_prior',result
    due=parse_time(result['result_due_at'])+timedelta(seconds=1)
    monkeypatch.setattr(w,'_now',lambda:due)
    class FutureClock(datetime):
        @classmethod
        def now(cls,tz=None):return due.astimezone(tz) if tz else due.replace(tzinfo=None)
    monkeypatch.setattr(settlement_module,'datetime',FutureClock)
    source=Source();source.native='4353553'
    local=job['kickoff'].astimezone(ZoneInfo('Europe/Helsinki'))
    source.schedule=lambda score='-':schedule(date=local.strftime('%Y%m%d'),clock=local.strftime('%H:%M'),score=score)
    old=source.terminal
    source.terminal=lambda:old().replace(b'05.10.2026',local.strftime('%d.%m.%Y').encode())
    def capture(binding,**kwargs):
        job['calls']['result']+=1
        return capture_finland_result(binding,transport=source,**kwargs)
    monkeypatch.setattr(e,'capture_finland_result',capture)
    real=e.adopt_settlement
    def stopped(*a,**k):raise RuntimeError('lost result completion marker')
    monkeypatch.setattr(e,'adopt_settlement',stopped)
    with pytest.raises(RuntimeError,match='marker'):w.resume_match(job['output'],settle=True)
    state=read(job['output']/'state.json');assert state['status']=='settling_evidence'
    if damage:
        path=job['output']/state['settlement_path']/'capture-receipts.json'
        receipt=read(path);receipt['receipts'][0]['observation_id']='missing';write(path,receipt)
    monkeypatch.setattr(e,'adopt_settlement',real)
    before=deepcopy(job['calls'])
    finished=w.resume_match(job['output'],settle=True)
    assert finished['status']==('reconciliation_required' if damage else 'evidence_settled_prior_unavailable'),finished
    assert job['calls']==before and finished['qualified_response_rows']==0
    if not damage:
        assert finished['independent_labels']==1 and finished['result_requests']==2
        qualification=read(job['output']/finished['qualification_path'])
        assert qualification['independent_result_verification']['status']=='VERIFIED'
        assert qualification['numeric_training_eligible'] is False and qualification['training_records']==[]


from football_research.acquisition.state import parse_time
