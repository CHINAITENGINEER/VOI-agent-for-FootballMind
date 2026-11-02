from datetime import datetime,timedelta,timezone
from html import escape
from pathlib import Path
import gzip
import json

import pytest

from football_research.data_adapters.live_fixture_preparation import prepare_live_fixture
from football_research.data_adapters.prospective_text import verify_prior_archive
from football_research.data_adapters.verified_fixtures import verify_identity_binding,SCHEDULE_URL
from football_spider.storage.raw_archive import RawArchive

NOW=datetime(2026,11,12,10,tzinfo=timezone.utc)
KICKOFF=NOW+timedelta(hours=6)
URL='https://www.uefa.com/uefanationsleague/match/2047999--croatia-vs-spain/'


def row(**changes):
    return {'matchId':3999999,'matchTime':'2026-11-13T00:00:00','homeId':768,'home':'克罗地亚',
        'awayId':772,'away':'西班牙','leagueId':1864,'league':'欧国联','footMatchState':0,
        'homeScore':None,'awayScore':None,**changes}


def official(**changes):
    event={'@type':'SportsEvent','url':URL,'name':'Croatia vs Spain',
        'description':'UEFA Nations League 2027 League phase: Croatia-Spain','startDate':KICKOFF.isoformat(),
        'eventStatus':'EventScheduled','homeTeam':{'@type':'SportsTeam','name':'Croatia',
            'sameAs':'https://www.uefa.com/uefanationsleague/teams/56370--croatia/'},
        'awayTeam':{'@type':'SportsTeam','name':'Spain',
            'sameAs':'https://www.uefa.com/uefanationsleague/teams/122--spain/'}}
    embedded={'id':'2047999','status':'Upcoming','seasonYear':'2027',
        'homeTeam':{'id':'56370','internationalName':'Croatia'},
        'awayTeam':{'id':'122','internationalName':'Spain'},
        'competition':{'id':'2014','displayName':'UEFA Nations League','age':'Adult','sex':'Male',
            'sportsType':'Football','teamCategory':'National'},
        'round':{'displayName':'League phase','mode':'Group'},'kickOffTime':{'dateTime':KICKOFF.isoformat()}}
    if changes.get('wrong_kickoff'):
        event['startDate']=(KICKOFF+timedelta(hours=1)).isoformat()
        embedded['kickOffTime']['dateTime']=event['startDate']
    if changes.get('wrong_id'):
        event['homeTeam']['sameAs']='https://www.uefa.com/uefanationsleague/teams/999--croatia/'
        embedded['homeTeam']['id']='999'
    return ('<html><script type="application/ld+json">'+json.dumps(event)+'</script><div data-options="'
        +escape(json.dumps({'match':embedded}),quote=True)+'"></div></html>').encode()


class Transport:
    def __init__(self, *, fixture=None, schedule=None, official_body=None, failure=None,
                 interrupt=None, interrupt_after_raw=False):
        self.fixture=fixture or row()
        self.schedule=schedule if schedule is not None else [self.fixture]
        self.official_body=official_body or official()
        self.failure,self.interrupt=failure,interrupt
        self.interrupt_after_raw=interrupt_after_raw
        self.calls=[]

    def __call__(self,spec,*,archive,clock):
        self.calls.append(spec)
        stage=spec['stage']
        if stage==self.interrupt and not self.interrupt_after_raw:
            raise KeyboardInterrupt('test interruption before archive')
        if stage=='fixture':body=json.dumps({'code':0,'data':self.fixture}).encode()
        elif stage=='schedule':body=json.dumps({'code':0,'data':[{'matchList':self.schedule}]}).encode()
        elif stage=='quote':body=json.dumps({'code':0,'data':{'footballMoreSpInfo':{
            'matchId':self.fixture['matchId'],'spf_win':2.5,'spf_draw':3.2,'spf_lost':2.8}}}).encode()
        elif stage=='official_schedule':body=('<html><a href="'+URL+'">Croatia vs Spain</a></html>').encode()
        else:body=self.official_body
        obs=archive.save(spec['source'],spec['url'],spec['params'],body,
            status=503 if stage==self.failure else 200,observed_at=clock(),
            context={'preparation_request_digest':spec['request_digest'],'requested_at':spec['requested_at']})
        if stage==self.interrupt:raise KeyboardInterrupt('test interruption after original raw commit')
        return obs


def prepare(tmp_path,transport=None,query='3999999',**kwargs):
    return prepare_live_fixture(query,output=tmp_path/'prepare',clock=lambda:NOW,
        transport=transport or Transport(),**kwargs)


def test_provider_id_to_ready_uses_existing_raw_market_and_official_identity(tmp_path):
    transport=Transport()
    result=prepare(tmp_path,transport)
    assert result['status']=='ready' and result['archived_http_responses']==4
    assert [x['stage'] for x in transport.calls]==['fixture','quote','official_schedule','official_match']
    assert result['collection']['candidate_discovery_version']==8
    assert result['collection']['news_projection_version']==5
    assert result['collection']['seed_urls']==[URL]
    initial=json.loads(Path(result['initializations_path']).read_text(encoding='utf-8'))['selected'][0]
    binding=json.loads(Path(result['identity_bundle_path']).read_text(encoding='utf-8'))['bindings'][result['match_id']]
    assert verify_prior_archive(initial,result['market_archive'])['status']=='verified'
    assert verify_identity_binding(binding,archive_root=result['identity_archive'],as_of=NOW)==binding
    assert binding['canonical']['home_id']=='uefa:team:56370'
    assert result['labels']==[] and not result['numeric_voi_training_eligible']


@pytest.mark.parametrize('home,away',[('克罗地亚','西班牙'),('Croatia','Spain')])
def test_names_and_default_beijing_date_resolve_exact_observed_fixture(tmp_path,home,away):
    result=prepare(tmp_path,query={'home':home,'away':away,'date':'2026-11-13'})
    assert result['status']=='ready' and result['query']['timezone']=='Asia/Shanghai'


def test_explicit_utc_date_and_wrong_calendar_day(tmp_path):
    q={'home':'Croatia','away':'Spain','date':'2026-11-12','timezone':'UTC'}
    assert prepare(tmp_path/'utc',query=q)['status']=='ready'
    q.pop('timezone')
    assert prepare(tmp_path/'beijing',query=q)['status']=='not_found'


def test_named_ambiguity_retains_all_candidates_without_guessing(tmp_path):
    transport=Transport(schedule=[row(),row(matchId=3999998)])
    result=prepare(tmp_path,transport,query={'home':'克罗地亚','away':'西班牙','date':'2026-11-13'})
    assert result['status']=='ambiguous' and len(result['candidates'])==2
    assert len(transport.calls)==1


def test_optional_date_resolves_only_unique_future_fixture(tmp_path):
    q={'home':'Croatia','away':'Spain','date':None}
    assert prepare(tmp_path/'unique',query=q)['status']=='ready'
    transport=Transport(schedule=[row(),row(matchId=3999998,matchTime='2026-11-14T00:00:00')])
    assert prepare(tmp_path/'ambiguous',transport,query=q)['status']=='ambiguous'


def test_omitted_date_excludes_old_and_near_kickoff_matches(tmp_path):
    transport=Transport(schedule=[row(),row(matchId=3999998,matchTime='2026-11-12T18:20:00')])
    assert prepare(tmp_path,transport,query={'home':'Croatia','away':'Spain'})['status']=='ready'


def test_unknown_competition_stops_after_fixture_without_quote_or_official_calls(tmp_path):
    transport=Transport(fixture=row(league='芬超',leagueId=99,home='赫尔火花',homeId=2429,
        away='国际图尔',awayId=386))
    result=prepare(tmp_path,transport)
    assert result['status']=='unsupported' and result['archived_http_responses']==1
    assert not Path(result['initializations_path']).exists()
    assert not Path(result['identity_bundle_path']).exists()


def test_known_competition_unknown_alias_cannot_create_mapping(tmp_path):
    result=prepare(tmp_path,Transport(fixture=row(home='Croatian Women')))
    assert result['status']=='unsupported' and result['reason']=='official_team_mapping_not_registered'


@pytest.mark.parametrize('fixture',[row(footMatchState=-1,homeScore=1,awayScore=0),
    row(matchTime='2026-11-12T18:20:00'),row(matchTime='2026-11-25T00:00:00')])
def test_not_prematch_or_outside_bounded_horizon(tmp_path,fixture):
    transport=Transport(fixture=fixture)
    result=prepare(tmp_path,transport)
    assert result['status']=='not_prematch' and len(transport.calls)==1


@pytest.mark.parametrize('limit,calls',[(0,0),(1,1),(2,2),(3,3)])
def test_every_actual_attempt_has_a_bounded_precommitted_intent(tmp_path,limit,calls):
    transport=Transport()
    result=prepare(tmp_path,transport,max_http_requests=limit)
    assert result['status']=='budget_exhausted' and len(transport.calls)==calls
    assert result['request_intents']==calls


def test_failed_http_is_archived_and_resume_never_retries_it(tmp_path):
    transport=Transport(failure='quote')
    first=prepare(tmp_path,transport)
    assert first['status']=='unavailable' and len(transport.calls)==2
    second=prepare(tmp_path,transport,resume=True)
    assert second==first and len(transport.calls)==2


def test_interrupted_issued_request_without_raw_is_never_silently_reissued(tmp_path):
    transport=Transport(interrupt='quote')
    with pytest.raises(KeyboardInterrupt):prepare(tmp_path,transport)
    result=prepare(tmp_path,transport,resume=True)
    assert result['status']=='interrupted' and len(transport.calls)==2
    assert result['request_intents']==2 and result['archived_http_responses']==1


def test_interrupted_receipt_recovers_raw_then_only_sends_unissued_requests(tmp_path):
    transport=Transport(interrupt='quote',interrupt_after_raw=True)
    with pytest.raises(KeyboardInterrupt):prepare(tmp_path,transport)
    transport.interrupt=None
    result=prepare(tmp_path,transport,resume=True)
    assert result['status']=='ready'
    assert [r['stage'] for r in transport.calls].count('quote')==1
    assert len(transport.calls)==4


def test_interrupted_status_is_preserved_when_late_original_raw_allows_recovery(tmp_path):
    transport=Transport(interrupt='quote')
    with pytest.raises(KeyboardInterrupt):prepare(tmp_path,transport)
    paused=prepare(tmp_path,transport,resume=True)
    assert paused['status']=='interrupted'
    spec=transport.calls[-1]
    archive=RawArchive(tmp_path/'prepare/market-archive')
    body=json.dumps({'code':0,'data':{'footballMoreSpInfo':{
        'matchId':3999999,'spf_win':2.5,'spf_draw':3.2,'spf_lost':2.8}}}).encode()
    archive.save('spftll',spec['url'],spec['params'],body,status=200,observed_at=NOW,
        context={'preparation_request_digest':spec['request_digest'],'requested_at':spec['requested_at']})
    transport.interrupt=None
    complete=prepare(tmp_path,transport,resume=True)
    assert complete['status']=='ready' and len(transport.calls)==4
    history=list((tmp_path/'prepare/result-history').glob('*.json'))
    assert len(history)==1 and json.loads(history[0].read_text(encoding='utf-8'))==paused


def test_missing_derived_artifacts_rebuild_from_original_bytes_without_network(tmp_path):
    transport=Transport()
    result=prepare(tmp_path,transport)
    paths=[Path(result['initializations_path']),Path(result['identity_bundle_path'])]
    originals={p:p.read_bytes() for p in paths}
    for p in paths:p.unlink()
    rebuilt=prepare(tmp_path,transport,resume=True)
    assert rebuilt==result and len(transport.calls)==4
    assert all(p.read_bytes()==content for p,content in originals.items())


def test_mutated_raw_fails_resume_before_another_request(tmp_path):
    transport=Transport()
    result=prepare(tmp_path,transport)
    archive=RawArchive(Path(result['market_archive']))
    (archive.root/archive.observations()[0].payload_path).write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError,match='checksum'):prepare(tmp_path,transport,resume=True)
    assert len(transport.calls)==4


@pytest.mark.parametrize('changes',[{'wrong_kickoff':True},{'wrong_id':True}])
def test_official_clock_or_registered_identity_conflict_never_ready(tmp_path,changes):
    result=prepare(tmp_path,Transport(official_body=official(**changes)))
    assert result['status']=='unavailable' and not Path(result['identity_bundle_path']).exists()


@pytest.mark.parametrize('query',[None,'../3999999',True,{'provider_id':False},
    {'home':'Croatia'},{'home':'Croatia','away':'Croatia','date':'2026-11-13'}])
def test_invalid_query_fails_before_network_or_output(tmp_path,query):
    transport=Transport()
    with pytest.raises((ValueError,TypeError)):
        prepare(tmp_path,transport,query=query)
    assert not transport.calls and not (tmp_path/'prepare').exists()


def test_resuming_different_query_or_budget_is_rejected(tmp_path):
    transport=Transport()
    prepare(tmp_path,transport)
    with pytest.raises(ValueError,match='Resume'):prepare(tmp_path,transport,query='9999',resume=True)
    with pytest.raises(ValueError,match='Resume'):prepare(tmp_path,transport,max_http_requests=7,resume=True)
    assert len(transport.calls)==4
