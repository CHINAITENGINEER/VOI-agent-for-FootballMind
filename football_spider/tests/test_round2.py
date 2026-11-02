"""Round2 contract/failure/PIT tests; all generated records are test fixtures."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
import pytest

from football_spider.storage.raw_archive import RawArchive,json_bytes
from football_spider.normalization.understat import normalize_understat,audit_season
from football_spider.features.store import FeatureStore
from football_spider.features.season import VERSION,DEFAULT_ASSUMPTIONS
from football_spider.identity.resolver import stable_id
from football_spider.platform_config import PlatformSettings
from football_spider.services.refresh import RefreshService
from football_spider.services.export import export_season
from football_spider.services.reader import PlatformReader
from football_platform.collectors.http_transport import RequestTransport,TransportError
from test_platform_contracts import fixture_row,save_detail

def payload():
    # Small synthetic season, deliberately incomplete and not a production archive.
    teams={'1':{'title':'Alpha','history':[]},'2':{'title':'Beta','history':[]}}
    rows=[]
    for n in range(3):
        rows.append({'id':str(100+n),'isResult':True,'h':{'id':'1','title':'Alpha'},'a':{'id':'2','title':'Beta'},
            'datetime':f'2025-08-{10+n*7:02d} 15:00:00','goals':{'h':str(n),'a':'0'},'xG':{'h':'1.2','a':'0.4'},'forecast':{'w':.99}})
    return {'dates':rows,'teams':teams,'players':[]}

def save(a,p=None):
    return a.save('understat','https://fixture.invalid/getLeagueData/EPL/2025',{},json_bytes(p or payload()),status=200,
        context={'league':'EPL','season':2025,'kickoff_timezone':'UTC','test_fixture':True})

def test_understat_contract_zero_score_and_no_forecast(tmp_path):
    a=RawArchive(tmp_path);obs=save(a);batch=normalize_understat(obs,payload())
    assert len(batch.matches)==3 and len(batch.statistics)==12
    assert batch.matches[0].score.home==0 and batch.matches[0].half_score.home is None
    assert batch.matches[0].kickoff.tzinfo==timezone.utc
    assert batch.matches[0].coverage=='incomplete_season'
    assert 'forecast' not in batch.matches[0].model_dump()
    assert {s.provider for s in batch.statistics}=={'understat'}

@pytest.mark.parametrize('field,value',[('goals',{'h':'-1','a':'0'}),('goals',{'h':'0.5','a':'0'}),
    ('xG',{'h':'NaN','a':'0'}),('datetime','2025-08-10T15:00:00+01:00'),('isResult','true')])
def test_understat_bad_rows_quarantinable(tmp_path,field,value):
    p=payload();p['dates'][0][field]=value
    batch=normalize_understat(save(RawArchive(tmp_path),p),p)
    assert len(batch.matches)==2
    assert any(i.severity=='error' and i.record_key=='100' for i in batch.issues)

def test_understat_duplicates_and_missing_xg(tmp_path):
    p=payload();p['dates'].append(deepcopy(p['dates'][0]));del p['dates'][1]['xG']
    batch=normalize_understat(save(RawArchive(tmp_path),p),p)
    assert len(batch.matches)==2 and len(batch.statistics)==4
    assert any(i.rule=='MISSING_XG' for i in batch.issues)

def test_understat_pg_replay_history_and_features(repo,tmp_path):
    a=RawArchive(tmp_path);first=save(a)
    assert repo.process(first,a)['statistics']==12
    before=repo.conn.execute('SELECT ingested_at FROM fm.match_versions LIMIT 1').fetchone()['ingested_at']
    assert repo.process(first,a)['skipped']==1
    versions=[str(r['id']) for r in repo.conn.execute('SELECT id FROM fm.match_versions').fetchall()]
    mid=stable_id('understat','match','102');cutoff=datetime(2025,8,23,15,tzinfo=timezone.utc)
    store=FeatureStore(repo,tmp_path/'features')
    initial=store.get_features(mid,cutoff,VERSION,'historical_reconstruction',version_ids=versions,assumptions=DEFAULT_ASSUMPTIONS)
    assert initial['features']['home_rolling_xg_for']==1.2
    assert initial['features']['home_form']['matches']==2
    assert mid not in initial['manifest']['history_match_ids']
    assert len(initial['manifest']['statistic_ids'])==8
    assert store.replay_snapshot(initial['snapshot_id'])['consistent']
    changed=payload();changed['dates'][0]['goals']['h']='8'
    assert repo.process(save(a,changed),a)['statistics']==12
    repeated=store.get_features(mid,cutoff,VERSION,'historical_reconstruction',version_ids=versions,assumptions=DEFAULT_ASSUMPTIONS)
    assert initial['manifest_hash']==repeated['manifest_hash'] and initial['features']==repeated['features']
    assert repo.conn.execute('SELECT count(*) n FROM fm.team_statistics').fetchone()['n']==24
    with pytest.raises(ValueError,match='No target'):
        store.get_features(mid,cutoff,VERSION,'operational')
    with pytest.raises(ValueError,match='pinned'):
        store.get_features(mid,cutoff,VERSION,'historical_reconstruction')
    result=export_season(repo,a,tmp_path/'exports',observation_id=first.id,lead_hours=(24,))
    assert result['manifest']['files']['matches']['row_count']==3
    assert export_season(repo,a,tmp_path/'exports',observation_id=first.id,lead_hours=(24,))['reused']

def test_season_operational_future_ingestion_does_not_change_cutoff(repo,tmp_path):
    a=RawArchive(tmp_path)
    for row in (fixture_row(mid=1,days=2),fixture_row(mid=2,days=-2,status=-1,hs=0,aws=0)):
        repo.process(save_detail(a,row),a)
    store=FeatureStore(repo,tmp_path/'features');cutoff=datetime.now(timezone.utc)
    mid=stable_id('spftll','match','1');first=store.get_features(mid,cutoff,VERSION)
    repo.process(save_detail(a,fixture_row(mid=2,days=-2,status=-1,hs=9,aws=1)),a)
    repeated=store.get_features(mid,cutoff,VERSION)
    assert first['manifest_hash']==repeated['manifest_hash']
    assert first['features']==repeated['features']
    assert store.replay_snapshot(first['snapshot_id'])['consistent']

def test_refresh_archive_first_and_repeated_operation_no_request(tmp_path):
    a=RawArchive(tmp_path)
    body=json_bytes({'code':0,'data':{'matchInfo':fixture_row()}})
    fake=SimpleNamespace(get=lambda *a,**k:SimpleNamespace(content=body,status_code=200,headers={},close=lambda:None),close=lambda:None)
    service=RefreshService(PlatformSettings(data_dir=tmp_path,database_url=''))
    with patch('football_spider.services.refresh.AioSession',return_value=fake),patch('football_spider.services.refresh.request_slot',return_value=0):
        op=str(uuid4());first=service.refresh('1',request_id=op,max_requests=1)
        assert first['requests_this_call']==1 and len(first['observations'])==1
        assert first['cost'] is None and first['cost_status']=='UNKNOWN'
        assert first['database']['status']=='NEEDS_DATABASE'
        assert service.refresh('1',request_id=op,max_requests=1)['requests_this_call']==0
        with pytest.raises(ValueError,match='different'):
            service.refresh('2',request_id=op,max_requests=1)
    assert a.read_body(a.observations()[0])==body

def test_deadline_never_shortens_retry_after(tmp_path):
    attempts=[];clock=[0.0]
    response=SimpleNamespace(content=b'{}',status_code=429,headers={'retry-after':'60'},close=lambda:None)
    fake=SimpleNamespace(get=lambda *a,**k:response,close=lambda:None)
    with RequestTransport('test',on_attempt=lambda **f:attempts.append(f),session=fake,max_attempts=3,deadline=10,
                          monotonic=lambda:clock[0],sleep=lambda x:clock.__setitem__(0,clock[0]+x)) as client:
        with pytest.raises(TransportError):client.get('https://fixture.invalid/data')
        assert client.stats['attempts']==1
    assert len(attempts)==1 and attempts[0]['http_status']==429

def test_raw_observation_id_global_collision_and_utc(tmp_path):
    archive=RawArchive(tmp_path);stamp=datetime(2026,9,22,16,tzinfo=timezone(timedelta(hours=8)))
    request=str(uuid4());oid=str(uuid4())
    args=dict(status=200,request_id=request,observation_id=oid,observed_at=stamp)
    first=archive.save('test','https://fixture.invalid/data',{},b'{}',**args)
    assert first.observed_at.hour==8
    assert archive.save('test','https://fixture.invalid/data',{},b'{}',**args).id==oid
    args['observed_at']=stamp+timedelta(days=1)
    with pytest.raises(ValueError,match='collision'):
        archive.save('test','https://fixture.invalid/data',{},b'{}',**args)
    assert len(archive.observations())==1

def test_result_quality_pending_without_mutating_fact_history(repo,tmp_path):
    from football_spider.services.results import export_results
    archive=RawArchive(tmp_path)
    good=fixture_row(mid=1,days=-2,status=-1,hs=2,aws=0)
    repo.process(save_detail(archive,good),archive)
    bad=deepcopy(good);bad['awayHalfScore']=1
    assert repo.process(save_detail(archive,bad),archive)['quarantined']==1
    bundle=export_results(repo,archive,['spftll:match:1'],tmp_path/'exports')
    root=Path(bundle['directory'])
    current=json.loads((root/'matches.jsonl').read_text())
    fact=json.loads((root/'result_versions.jsonl').read_text())
    assert current['settlement_status']=='pending' and current['status']=='disputed'
    assert current['home_goals'] is None and current['result_available_at'] is None
    assert fact['status']=='finished' and fact['home_goals']==2
    assert bundle['manifest']['files']['quality']['row_count']==1
    assert len(bundle['manifest']['source_evidence'])==2

def test_corrupt_aio_cookies_recover_without_overwrite(tmp_path):
    from football_spider.collectors.aio_transport import AioSession
    path=tmp_path/'cookies.json';path.write_text('{broken',encoding='utf-8')
    client=AioSession(path)
    try:assert client.cookie_errors==1
    finally:client.close()
    assert path.read_text()=='{broken' and path.with_suffix('.recovered.json').exists()

def test_successful_null_detail_is_no_data_not_parser_failure(repo,tmp_path):
    archive=RawArchive(tmp_path)
    obs=archive.save('spftll','https://fixture.invalid/GetInfo',{},b'{"code":0,"data":null}',status=200,context={'test_fixture':True})
    result=repo.process(obs,archive)
    assert result['status']=='NO_DATA' and result['failed']==0 and result['valid']==0

def test_refresh_wrong_target_preserves_raw_but_reports_mismatch(tmp_path):
    body=json_bytes({'code':0,'data':fixture_row(mid=999)})
    fake=SimpleNamespace(get=lambda *a,**k:SimpleNamespace(content=body,status_code=200,headers={},close=lambda:None),close=lambda:None)
    service=RefreshService(PlatformSettings(data_dir=tmp_path,database_url=''))
    with patch('football_spider.services.refresh.AioSession',return_value=fake),patch('football_spider.services.refresh.request_slot',return_value=0):
        result=service.refresh('1',request_id=str(uuid4()),max_requests=1)
    assert result['status']=='PARTIAL' and result['errors']==[{'category':'TARGET_ID_MISMATCH'}]
    assert len(result['observations'])==1 and result['requests_this_call']==1
