"""Constructed opt-in lower-stage identities; no live HTTP or real labels."""
from copy import deepcopy
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import socket

import pytest

from football_research.data_adapters.official_fixture import (
    _binding, _digest, prepare_official_fixture, verify_official_fixture,
)
from football_research.data_adapters.official_cohort import plan_official_cohort
from football_research.data_adapters.official_stage_registry import GROUP3_TEAMS
from football_research.data_adapters.evidence_collection import prepare_evidence_plan
from football_spider.storage.raw_archive import RawArchive
from test_official_fixture import NOW, Transport, read_binding
from test_finland_stage_fixture import schedule

EVENT = 'veikkausliiga:2026:match:4353581'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail('Group3 identity tests use constructed raw responses only')
    for obj, key in ((socket,'getaddrinfo'),(socket,'create_connection'),
                     (socket.socket,'connect'),(socket.socket,'connect_ex')):
        monkeypatch.setattr(obj,key,deny)


def schedule3(home='SJK',away='FC Lahti',native='4353581',date='20261009',clock='18:00',score='-'):
    body=schedule(home,away,native,date,clock,score).replace(
        b'selected value="2">Mestaruussarja',b'selected value="3">Karsintasarja')
    start=body.index(b'<select name="team">')+len(b'<select name="team">')
    end=body.index(b'</select>',start)
    options=''.join('<option value="'+spec['native_id']+'">'+name+'</option>'
                    for name,spec in GROUP3_TEAMS.items()).encode()
    return body[:start]+options+body[end:]


def prepare(root, *, body=None, query=EVENT, **kwargs):
    transport=Transport(schedule3() if body is None else body)
    result=prepare_official_fixture(query,output=root/'prepare',schedule_group='3',
                                    clock=lambda:NOW,transport=transport,**kwargs)
    return result,transport


def test_opt_in_stage_binds_native_entities_original_params_and_receipt(tmp_path):
    result,transport=prepare(tmp_path)
    binding=read_binding(result)
    plan=json.loads((tmp_path/'prepare/preparation-plan.json').read_text())
    assert result['status']=='ready' and len(transport.calls)==1
    assert result['schedule_group']==binding['schedule_group']==plan['schedule_group']=='3'
    assert transport.calls[0]['params']=={'group':'3'} and transport.calls[0]['schedule_group']=='3'
    assert binding['canonical']['stage']=='Karsintasarja'
    assert binding['canonical']['home_id']=='veikkausliiga:team:60970'
    assert binding['canonical']['away_id']=='veikkausliiga:team:60969'
    assert binding['aliases']=={'home':['SJK'],'away':['FC Lahti']}
    assert 'spftll' not in json.dumps(binding)
    assert 'preparation_horizon_days' not in binding
    assert binding['regulation_rule'] is None and binding['settlement_rows']==[]
    assert verify_official_fixture(binding,archive_root=result['identity_archive'],as_of=NOW)==binding


@pytest.mark.parametrize('query', [EVENT,{'home':'SJK','away':'FC Lahti'},
                                  {'home':'SJK','away':'FC Lahti','date':'2026-10-09'}])
def test_native_and_official_name_queries_resolve_same_source(tmp_path,query):
    result,_=prepare(tmp_path,query=query)
    assert result['status']=='ready' and result['source_event_id']==EVENT


@pytest.mark.parametrize('value',[None,True,False,2,3,'1','03','all',''])
def test_group_must_be_an_explicit_registered_string_before_any_request(tmp_path,value):
    transport=Transport(schedule3())
    with pytest.raises(ValueError,match='schedule_group'):
        prepare_official_fixture(EVENT,output=tmp_path/'prepare',schedule_group=value,
                                  clock=lambda:NOW,transport=transport)
    assert not transport.calls and not (tmp_path/'prepare').exists()


@pytest.mark.parametrize('old,new',[
    ('selected value="3">Karsintasarja','selected value="2">Mestaruussarja'),
    ('selected value="3">Karsintasarja','selected value="3">Mestaruussarja'),
    ('selected value="2026">2026','selected value="2025">2025'),
    ('selected value="veikkausliiga">Veikkausliiga','selected value="liigacup">Liigacup'),
    ('value="60970">SJK','value="999">SJK'),
    ('value="60969">FC Lahti','value="60969">FC Lahti Women'),
    ('>SJK - FC Lahti</a>','>SJK U19 - FC Lahti</a>'),
    ('<span>-</span>','<span>0 — 0</span>'),
    ('4353581/seuranta/','4353000/seuranta/'),
])
def test_stage_squad_season_native_id_or_non_scheduled_state_is_not_admitted(tmp_path,old,new):
    result,_=prepare(tmp_path,body=schedule3().replace(old.encode(),new.encode()))
    assert result['status'] in {'unsupported','not_prematch'}
    assert not Path(result['identity_path']).exists()


def test_group3_and_default_group2_cannot_consume_each_others_response(tmp_path):
    other,_=prepare(tmp_path/'as3',body=schedule())
    old=prepare_official_fixture(EVENT,output=tmp_path/'as2',clock=lambda:NOW,transport=Transport(schedule3()))
    assert other['status']!='ready' and old['status']!='ready'
    future,_=prepare(tmp_path/'future',body=schedule3(date='20261017'))
    assert future['status']=='not_prematch'
    assert not Path(future['identity_path']).exists()


@pytest.mark.parametrize('change',['strip_binding_group','relabel_stage','mutate_request','mutate_raw_params'])
def test_rehashed_mutation_cannot_reassign_the_observed_group(tmp_path,change):
    result,_=prepare(tmp_path)
    binding=read_binding(result)
    if change=='strip_binding_group':
        binding.pop('schedule_group')
    elif change=='relabel_stage':
        binding['canonical']['stage']='Mestaruussarja'
    elif change=='mutate_request':
        path=Path(result['request_intent_path']);intent=json.loads(path.read_text())
        intent.update(schedule_group='2',params={'group':'2'})
        intent['request_digest']=_digest({k:v for k,v in intent.items() if k!='request_digest'})
        path.write_text(json.dumps(intent))
    else:
        path=next(Path(result['identity_archive']).rglob('*.observation.json'))
        observation=json.loads(path.read_text());observation['params']={'group':'2'}
        path.write_text(json.dumps(observation))
    binding['binding_sha256']=_digest({k:v for k,v in binding.items() if k!='binding_sha256'})
    with pytest.raises(ValueError):
        verify_official_fixture(binding,archive_root=result['identity_archive'],as_of=NOW)


def test_interruption_recovers_original_raw_without_reissuing_or_changing_group(tmp_path):
    source=Transport(schedule3(),interrupt='after_raw')
    options=dict(query=EVENT,output=tmp_path/'prepare',clock=lambda:NOW,transport=source)
    with pytest.raises(KeyboardInterrupt):
        prepare_official_fixture(**options,schedule_group='3')
    with pytest.raises(ValueError,match='resume'):
        prepare_official_fixture(**options,resume=True)
    result=prepare_official_fixture(**options,schedule_group='3',resume=True)
    assert result['status']=='ready' and len(source.calls)==1
    before=read_binding(result)
    result2=prepare_official_fixture(**options,schedule_group='3',resume=True)
    assert result2==result and before==read_binding(result2) and len(source.calls)==1


def six_fixture_body():
    rows=[('Ilves','SJK','4353585','20261017','14:00'),
        ('SJK','FC Lahti','4353581','20261009','18:00'),
        ('FF Jaro','IFK Mariehamn','4353586','20261017','14:00'),
        ('IFK Mariehamn','Ilves','4353583','20261010','19:00'),
        ('FC Lahti','TPS','4353584','20261017','14:00'),
        ('TPS','FF Jaro','4353582','20261010','17:00')]
    bodies=[schedule3(*row) for row in rows]
    return bodies[0].replace(b'</table>',b''.join(body[body.index(b'<tr '):body.index(b'</tr>')+5]
        for body in bodies[1:])+b'</table>')


def test_group3_cohort_is_continuous_three_in_seven_days_with_no_other_stage_or_repeated_exposure(tmp_path):
    result,_=prepare(tmp_path,body=six_fixture_body())
    args=dict(identity_path=result['identity_path'],identity_archive=result['identity_archive'],
              previous_plan_paths=[],schedule_group='3',as_of=NOW)
    plan=plan_official_cohort(**args,output=tmp_path/'cohort.json')
    assert plan['schedule_group']=='3' and plan['queries']==[
        'veikkausliiga:2026:match:'+str(native) for native in (4353581,4353582,4353583)]
    assert plan['counts']['observed_native_ids']==6 and plan['counts']['rejected']==3
    assert all(row['stage']=='Karsintasarja' for row in plan['selected'])
    with pytest.raises(ValueError,match='group'):
        plan_official_cohort(**(args|{'schedule_group':'2'}),output=tmp_path/'wrong-stage.json')
    exposure=prepare_evidence_plan(read_binding(result),previous_plans=[],created_at=NOW)
    old=tmp_path/'exposure.json';old.write_text(json.dumps(exposure))
    plan2=plan_official_cohort(**(args|{'previous_plan_paths':[old]}),output=tmp_path/'later-plan.json')
    assert plan2['queries']==plan['queries'][1:]


def test_default_group2_binding_has_exact_original_pre_group_extension_shape(tmp_path):
    root=Path(__file__).resolve().parents[3]
    oldpath=root/'var/parallel/agent/round201/source-before/football_research/data_adapters/official_fixture.py'
    spec=importlib.util.spec_from_file_location('football_research.data_adapters._pre_group3_official',oldpath)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    result=prepare_official_fixture('veikkausliiga:2026:match:4353553',output=tmp_path/'prepare',
                                    clock=lambda:NOW,transport=Transport(schedule()))
    binding=read_binding(result);archive=RawArchive(Path(result['identity_archive']));observation,=archive.observations()
    assert old._binding(binding['query'],observation,archive.read_body(observation),at=NOW)['binding']==binding
    assert old.verify_official_fixture(binding,archive_root=archive.root,as_of=NOW)==binding
    assert 'schedule_group' not in binding and 'schedule_group' not in result
