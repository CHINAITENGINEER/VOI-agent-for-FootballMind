"""Constructed official fixtures without market quotes exercise settlement only."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import gzip
import json
import socket

import pytest

from football_research.data_adapters.finland_stage_fixture import REGISTERED_TEAMS,RESULTS_URL,RESULTS_PARAMS
from football_research.data_adapters.finland_results import (
    copy_rule_observations,match_followup_url,verify_finland_settlement,
)
from football_research.data_adapters.finland_result_capture import (
    capture_finland_result,detail_url,verify_capture_receipts,
)
from football_research.data_adapters.official_fixture import prepare_official_fixture,verify_official_fixture
from football_research.data_adapters.verified_fixtures import verify_independent_settlement
from football_spider.storage.raw_archive import RawArchive

UTC=timezone.utc
BEFORE=datetime(2026,10,5,10,tzinfo=UTC)
KICKOFF=datetime(2026,10,5,16,tzinfo=UTC)
AFTER=KICKOFF+timedelta(hours=2,minutes=10)
AS_OF=datetime.now(UTC)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args,**kwargs):
        pytest.fail('Official result tests use only constructed responses')
    monkeypatch.setattr(socket,'getaddrinfo',deny)
    monkeypatch.setattr(socket,'create_connection',deny)
    monkeypatch.setattr(socket.socket,'connect',deny)


class Source:
    def __init__(self,*,legacy=False):
        self.native='4353542' if legacy else '9876543'
        self.home,self.away=('IF Gnistan','FC Inter') if legacy else ('HJK','VPS')
        self.calls=[]
        self.score='2 — 0'
        self.finished=True
        self.extra=False
        self.fail_stage=None

    def schedule(self,score='-'):
        teams=''.join(f'<option value="{t["native_team_id"]}">{t["name"]}</option>' for t in REGISTERED_TEAMS.values())
        path='/tilastot/2026/veikkausliiga/ottelut/'+self.native+'/'
        return f'''<html><title>Otteluohjelma | Veikkausliiga</title>
        <select id="season"><option value="2026" selected>2026</option></select>
        <select id="category"><option value="veikkausliiga" selected>Veikkausliiga</option></select>
        <select name="group"><option value="2" selected>Mestaruussarja</option></select>
        <select name="team">{teams}</select><a href="/">Logo</a><a href="/uutiset/">Uutiset</a>
        <table><tr data-time="20261005"><td>1</td><td>Ma 5.10.2026</td><td>19:00</td><td>5.10.2026</td>
        <td><a href="{path}">{self.home} - {self.away}</a></td>
        <td><a href="{path}seuranta/">Seuranta</a></td><td>{score}</td><td>-</td></tr></table></html>'''.encode()

    def terminal(self):
        return f'''<html><meta property="og:url" content="{RESULTS_URL+self.native+'/seuranta/'}">
        <main><section class="game-header"><div class="head {'finished' if self.finished else 'live'}">
        <div class="home"><div class="team-name"><a>{self.home}</a></div></div>
        <div class="away"><div class="team-name"><a>{self.away}</a></div></div>
        <span id="home-goals">2</span><span id="away-goals">0</span>
        <div class="game-state">Pelattu 05.10.2026</div></div></section>
        <div class="game-events"><p class="info center">Ottelu päättyi</p>
        <p class="info center">Puoliaika päättyi</p>
        <div class="regular-event time">{'105' if self.extra else '90'}'</div>
        <p class="info center">Puoliaika alkoi</p><p class="info center">Puoliaika päättyi</p>
        <p class="info center">Puoliaika alkoi</p><p class="info center">Ottelu alkoi</p></div></main></html>'''.encode()

    def __call__(self,spec,*,archive,clock):
        self.calls.append(deepcopy(spec))
        assert (archive.root/'requests'/(spec['stage']+'.intent.json')).exists()
        body=(self.schedule() if spec['stage']=='official_schedule' else
              self.schedule(self.score) if spec['stage']=='schedule' else self.terminal())
        return archive.save(spec['source'],spec['url'],spec['params'],body,
            status=503 if spec['stage']==self.fail_stage else 200,observed_at=clock(),
            context={'requested_at':spec['requested_at'],'preparation_request_digest':spec['request_digest']})


def prepared(root,*,legacy=False):
    source=Source(legacy=legacy)
    query=('palloliitto:match:' if legacy else 'veikkausliiga:2026:match:')+source.native
    result=prepare_official_fixture(query,output=root/'prepared',clock=lambda:BEFORE,transport=source)
    assert result['status']=='ready',result
    initial=json.loads((root/'prepared/identity.json').read_text(encoding='utf-8'))
    assert 'initialization_manifest' not in initial
    return source,initial,root/'prepared/identity-archive'


@pytest.mark.parametrize('legacy',[False,True])
def test_official_identity_can_reach_independent_finished_result_without_any_market_material(tmp_path,legacy):
    source,binding,identity_archive=prepared(tmp_path,legacy=legacy)
    assert verify_official_fixture(binding,archive_root=identity_archive,as_of=AS_OF)==binding
    assert detail_url(binding)==match_followup_url(binding)==RESULTS_URL+source.native+'/seuranta/'
    result=capture_finland_result(binding,identity_archive_root=identity_archive,output=tmp_path/'result',
                                  clock=lambda:AS_OF,transport=source)
    assert [c['stage'] for c in source.calls]==['official_schedule','schedule','detail']
    assert result['status']=='VERIFIED' and len(result['settlement_rows'])==1
    row=result['settlement_rows'][0]
    assert row['match_id']==binding['canonical']['source_event_id']
    assert row['status']=='final' and (row['home_goals'],row['away_goals'])==(2,0)
    assert row['score_period']=='regulation' and len(row['input_manifest'])==5
    assert row['available_at']==AS_OF.isoformat()
    assert row['identity_binding_sha256']==binding['binding_sha256']
    assert 'p0' not in json.dumps(result) and 'market_initialization' not in json.dumps(result)
    receipts=verify_capture_receipts(binding,tmp_path/'result',as_of=AS_OF)
    assert receipts['http_requests_actual']==2 and len(receipts['cached_rule_observation_ids'])==3
    archive=RawArchive(tmp_path/'result')
    replay=verify_independent_settlement(binding,identity_archive_root=identity_archive,
        result_archive_root=archive.root,observation_ids=[o.id for o in archive.observations()],as_of=AS_OF)
    assert replay==result


def test_official_not_due_capture_spends_zero_requests_and_remains_pending(tmp_path):
    source,binding,identity_archive=prepared(tmp_path)
    result=capture_finland_result(binding,identity_archive_root=identity_archive,output=tmp_path/'not-due',
                                  clock=lambda:BEFORE,transport=source)
    assert [c['stage'] for c in source.calls]==['official_schedule']
    assert result['status']=='PENDING' and result['settlement_rows']==[]
    assert verify_capture_receipts(binding,tmp_path/'not-due',as_of=BEFORE)['http_requests_actual']==0


def test_official_live_score_is_not_a_result_until_finished_two_halves_are_observed(tmp_path):
    source,binding,identity_archive=prepared(tmp_path)
    source.finished=False
    source.score='-'
    result=capture_finland_result(binding,identity_archive_root=identity_archive,output=tmp_path/'live',
                                  clock=lambda:AS_OF,transport=source)
    assert result['status']=='PENDING' and result['settlement_rows']==[]
    assert len(source.calls)==3


def test_official_extra_time_or_wrong_native_score_still_rejected(tmp_path):
    source,binding,identity_archive=prepared(tmp_path)
    source.extra=True
    result=capture_finland_result(binding,identity_archive_root=identity_archive,output=tmp_path/'extra',
                                  clock=lambda:AS_OF,transport=source)
    assert result['status']=='DISPUTED' and result['settlement_rows']==[]


def test_official_schedule_failure_is_single_result_request_without_detail_retry(tmp_path):
    source,binding,identity_archive=prepared(tmp_path)
    source.fail_stage='schedule'
    result=capture_finland_result(binding,identity_archive_root=identity_archive,output=tmp_path/'failed',
                                  clock=lambda:AS_OF,transport=source)
    assert [c['stage'] for c in source.calls]==['official_schedule','schedule']
    assert result['status']=='PENDING' and result['settlement_rows']==[]
    assert verify_capture_receipts(binding,tmp_path/'failed',as_of=AS_OF)['http_requests_actual']==1


def test_raw_official_identity_and_result_checksums_remain_enforced(tmp_path):
    source,binding,identity_archive=prepared(tmp_path)
    report=capture_finland_result(binding,identity_archive_root=identity_archive,output=tmp_path/'result',
                                  clock=lambda:AS_OF,transport=source)
    assert report['status']=='VERIFIED'
    archive=RawArchive(tmp_path/'result')
    obs=next(o for o in archive.observations() if o.endpoint.endswith('/seuranta/'))
    (archive.root/obs.payload_path).write_bytes(gzip.compress(b'changed',mtime=0))
    with pytest.raises(ValueError,match='checksum'):
        verify_finland_settlement(binding,identity_archive_root=identity_archive,result_archive_root=archive.root,
            observation_ids=[o.id for o in archive.observations()],as_of=AS_OF)
    identity=RawArchive(identity_archive)
    obs=identity.observations()[0]
    (identity.root/obs.payload_path).write_bytes(gzip.compress(b'changed identity',mtime=0))
    with pytest.raises(ValueError,match='checksum'):
        verify_finland_settlement(binding,identity_archive_root=identity_archive,result_archive_root=archive.root,
            observation_ids=[],as_of=AS_OF)


@pytest.mark.parametrize('field,value',[
    ('match_id','veikkausliiga:2026:match:123'),
    ('canonical.category','women_club'),('canonical.stage','Relegation'),
    ('canonical.competition','Other League'),('canonical.season','2025'),
])
def test_followup_url_rejects_wrong_official_identity_scope(tmp_path,field,value):
    _,binding,_=prepared(tmp_path)
    target=binding
    parts=field.split('.')
    for part in parts[:-1]:target=target[part]
    target[parts[-1]]=value
    with pytest.raises(ValueError,match='scope|registered 2026'):
        detail_url(binding)
