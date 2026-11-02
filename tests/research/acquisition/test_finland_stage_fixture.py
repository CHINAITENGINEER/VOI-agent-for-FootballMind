"""Mechanism fixtures only: none of these synthetic responses are real samples."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip

import pytest

from football_research.data_adapters.finland_stage_fixture import (
    HOME_URL, NEWS_URL, RESULTS_URL, RESULTS_PARAMS, REGISTERED_TEAMS, VERSION,
    _digest, bind_finland_stage_fixture, verify_finland_stage_identity,
)
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot, json_record
from football_spider.storage.raw_archive import RawArchive

NOW = datetime(2026,10,7,7,tzinfo=timezone.utc)
OBSERVED = NOW+timedelta(minutes=1)
AUDIT = NOW+timedelta(minutes=2)
KICKOFF = datetime(2026,10,8,15,tzinfo=timezone.utc)


def manifest(**changes):
    row=json_record(ResearchMatch('spftll:match:3085211','spftll:competition:13',None,
        'spftll:team:383','spftll:team:2226',KICKOFF,source='spftll',status='scheduled',
        competition_name='芬超',home_team_name='赫尔辛基',away_team_name='瓦萨',
        fixture_available_at=NOW-timedelta(minutes=2),input_manifest=('observation:test-fixture',)))
    row.update(changes)
    quote=ResearchOddsSnapshot(row['match_id'],NOW-timedelta(minutes=1),1.43,4.2,5.2,
                              input_manifest=('observation:test-quote',))
    return MarketPriorPolicy(row,json_record(quote),frozen_at=NOW,
        prediction_cutoff=datetime.fromisoformat(row['kickoff'])-timedelta(minutes=1)).manifest


def schedule(home='HJK',away='VPS',native='4353553',date='20261008',clock='18:00',score='-'):
    teams=''.join('<option value="'+t['native_team_id']+'">'+t['name']+'</option>'
                  for t in REGISTERED_TEAMS.values())
    path='/tilastot/2026/veikkausliiga/ottelut/'+native+'/'
    return ('''<html><head><title>Otteluohjelma | Veikkausliiga</title></head><body>
      <select id="season"><option selected value="2026">2026</option></select>
      <select id="category"><option selected value="veikkausliiga">Veikkausliiga</option></select>
      <select name="group"><option selected value="2">Mestaruussarja</option></select>
      <select name="team">'''+teams+'''</select><a href="/">logo</a><a href="/uutiset/">Uutiset</a>
      <table><tr data-time="'''+date+'''"><td>11</td><td>To 8.10.2026</td><td>'''+clock+'''</td>
      <td>8.10.2026</td><td><a href="'''+path+'''">'''+home+' - '+away+'''</a></td>
      <td><a href="'''+path+'''seuranta/">Seuranta</a></td><td><span>'''+score+'''</span></td><td>-</td>
      </tr></table></body></html>''').encode()


def identity(tmp_path,*,initial=None,body=None,endpoint=RESULTS_URL,params=None,
             observed_at=OBSERVED,requested_at=None,status=200,duplicate=False,complete=True):
    archive=RawArchive(tmp_path/'identity')
    ids=[]
    for _ in range(2 if duplicate else 1):
        obs=archive.save('open_page',endpoint,RESULTS_PARAMS if params is None else params,
            schedule() if body is None else body,status=status,observed_at=observed_at,
            context={'requested_at':(requested_at or observed_at-timedelta(seconds=1)).isoformat(),
                     'response_complete':complete})
        ids.append(obs.id)
    binding=bind_finland_stage_fixture(initial or manifest(),archive_root=archive.root,
                                      observation_ids=ids,as_of=AUDIT)
    return archive,binding


def rehash(binding):
    binding['binding_sha256']=_digest({k:v for k,v in binding.items() if k!='binding_sha256'})
    return binding


def test_raw_stage_identity_roundtrip_and_observed_generic_sources(tmp_path):
    archive,binding=identity(tmp_path)
    assert binding['schema']==VERSION and binding['status']=='CONFIRMED'
    assert binding['canonical']['source_event_id']=='veikkausliiga:2026:match:4353553'
    assert binding['canonical']['home_id']=='veikkausliiga:team:60731'
    assert binding['canonical']['away_id']=='veikkausliiga:team:60901'
    assert binding['native_team_ids']=={'home':'60731','away':'60901'}
    assert binding['native_team_id_namespace']=='veikkausliiga'
    assert binding['canonical']['stage']=='Mestaruussarja'
    assert binding['canonical']['season']=='2026'
    assert binding['canonical']['category']=='senior_men_club'
    assert binding['canonical']['kickoff']==KICKOFF.isoformat()
    assert binding['canonical']['source_url']==RESULTS_URL+'4353553/'
    assert binding['observed_generic_entry_urls']==[HOME_URL,NEWS_URL]
    assert binding['aliases']['away']==['瓦萨','VPS瓦萨','VPS']
    assert binding['regulation_rule'] is None and binding['settlement_rows']==[]
    assert verify_finland_stage_identity(binding,archive_root=archive.root,as_of=AUDIT)==binding


@pytest.mark.parametrize('home,away,date,clock,kickoff',[
    ('1986','2228','20261008','19:00','2026-10-08T16:00:00+00:00'),
    ('2226','2429','20261012','18:00','2026-10-12T15:00:00+00:00'),
    ('2228','383','20261012','19:00','2026-10-12T16:00:00+00:00'),
    ('386','1986','20261011','15:00','2026-10-11T12:00:00+00:00'),
    ('2226','383','20261102','18:00','2026-11-02T16:00:00+00:00'),
])
def test_other_registered_pairs_and_new_native_ids_need_no_fixture_specific_patch(tmp_path,home,away,date,clock,kickoff):
    h,a=REGISTERED_TEAMS['spftll:team:'+home],REGISTERED_TEAMS['spftll:team:'+away]
    initial=manifest(match_id='spftll:match:999999',home_team_id='spftll:team:'+home,
        away_team_id='spftll:team:'+away,home_team_name=h['provider_names'][0],
        away_team_name=a['provider_names'][0],kickoff=kickoff)
    _,binding=identity(tmp_path,initial=initial,body=schedule(h['name'],a['name'],'9999888',date,clock))
    assert binding['status']=='CONFIRMED'
    assert binding['canonical']['source_event_id']=='veikkausliiga:2026:match:9999888'
    assert binding['canonical']['home_name']==h['name']
    assert binding['canonical']['home_id']=='veikkausliiga:team:'+h['native_team_id']


@pytest.mark.parametrize('old,new',[
    ('Otteluohjelma | Veikkausliiga','Youth League'),
    ('selected value="2026">2026','selected value="2025">2025'),
    ('selected value="veikkausliiga">Veikkausliiga','selected value="women">Women'),
    ('selected value="2">Mestaruussarja','selected value="1">Runkosarja'),
    ('value="60731">HJK','value="99999">HJK'),
    ('value="60901">VPS','value="60901">VPS U17'),
    ('<td>18:00</td>','<td>19:00</td>'),
    ('>HJK - VPS</a>','>VPS - HJK</a>'),
    ('>HJK - VPS</a>','>HJK Women - VPS</a>'),
    ('data-time="20261008"','data-time="20261009"'),
    ('<span>-</span>','<span>0 — 0</span>'),
    ('<span>-</span>','<span>Siirretty</span>'),
    ('4353553/seuranta/','999999/seuranta/'),
    ('href="/tilastot/2026/veikkausliiga/ottelut/4353553/"','href="https://evil.invalid/4353553/"'),
])
def test_wrong_scope_native_id_date_time_order_or_live_score_does_not_confirm(tmp_path,old,new):
    assert old.encode() in schedule()
    _,binding=identity(tmp_path,body=schedule().replace(old.encode(),new.encode()))
    assert binding['status']!='CONFIRMED' and binding['canonical'] is None
    assert binding['settlement_rows']==[]


def test_repeated_native_event_does_not_guess_one(tmp_path):
    original=schedule()
    row=original[original.index(b'<tr '):original.index(b'</tr>')+5]
    _,binding=identity(tmp_path,body=original.replace(b'</table>',row+b'</table>'))
    assert binding['status']!='CONFIRMED'


def test_legacy_gnistan_inter_pair_cannot_be_recounted_in_new_namespace(tmp_path):
    initial=manifest(match_id='spftll:match:3085210',home_team_id='spftll:team:2429',
        away_team_id='spftll:team:386',home_team_name='赫尔火花',away_team_name='国际图尔',
        kickoff='2026-10-07T16:00:00+00:00')
    _,binding=identity(tmp_path,initial=initial,
        body=schedule('IF Gnistan','FC Inter','4353542','20261007','19:00'))
    assert binding['status']=='UNCONFIRMED'
    assert binding['reason']=='require_existing_pair_identity_provider'
    assert binding['canonical'] is None


@pytest.mark.parametrize('change',[
    {'home_team_id':'spftll:team:999'}, {'home_team_name':'HJK U17'},
    {'away_team_name':'瓦萨女足'}, {'competition_id':'spftll:competition:226'},
    {'competition_name':'芬兰杯'}, {'season':'2025'},
])
def test_unknown_provider_entity_or_competition_not_inferred(tmp_path,change):
    _,binding=identity(tmp_path,initial=manifest(**change))
    assert binding['canonical'] is None and binding['status']!='CONFIRMED'


@pytest.mark.parametrize('params',[{}, {'group':'1'}, {'group':2}, {'group':'2','team':'60731'}])
def test_unregistered_query_parameters_fail(tmp_path,params):
    with pytest.raises(ValueError,match='registered official'):
        identity(tmp_path,params=params)


def test_wrong_source_endpoint_fails(tmp_path):
    with pytest.raises(ValueError,match='registered official'):
        identity(tmp_path,endpoint=RESULTS_URL.replace('www.',''))


def test_future_receipt_or_inverted_request_clock_fails(tmp_path):
    with pytest.raises(ValueError,match='unavailable'):
        identity(tmp_path/'future',observed_at=AUDIT+timedelta(seconds=1))
    with pytest.raises(ValueError,match='must follow'):
        identity(tmp_path/'inverted',requested_at=OBSERVED+timedelta(seconds=1))


def test_response_failure_incomplete_body_and_multiple_snapshots_remain_unconfirmed(tmp_path):
    for name,kwargs in [('failed',{'status':503}),('incomplete',{'complete':False}),('multiple',{'duplicate':True})]:
        _,binding=identity(tmp_path/name,**kwargs)
        assert binding['status']!='CONFIRMED' and binding['canonical'] is None


def test_rehashing_changed_canonical_or_manifest_does_not_reproduce_raw(tmp_path):
    archive,binding=identity(tmp_path)
    altered=deepcopy(binding)
    altered['canonical']['home_name']='KuPS'
    with pytest.raises(ValueError,match='does not reproduce'):
        verify_finland_stage_identity(rehash(altered),archive_root=archive.root,as_of=AUDIT)


def test_raw_body_tampering_is_detected(tmp_path):
    archive,binding=identity(tmp_path)
    observation=archive.observations()[0]
    (archive.root/observation.payload_path).write_bytes(gzip.compress(schedule()+b'changed',mtime=0))
    with pytest.raises(ValueError,match='checksum'):
        verify_finland_stage_identity(binding,archive_root=archive.root,as_of=AUDIT)


def test_absent_generic_news_link_not_invented(tmp_path):
    _,binding=identity(tmp_path,body=schedule().replace(b'<a href="/uutiset/">Uutiset</a>',b''))
    assert binding['status']=='CONFIRMED'
    assert binding['observed_generic_entry_urls']==[HOME_URL]
