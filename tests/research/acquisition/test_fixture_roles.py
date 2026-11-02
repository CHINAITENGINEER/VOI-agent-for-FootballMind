"""Original-byte identity and visible-text roles; all test responses constructed."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import base64
import json
from pathlib import Path
import shutil
import socket

import pytest

from football_agent.evidence_only_policy import EvidenceOnlyPolicy
from football_research.acquisition.fixture_roles import (
    ROLES, _mentions, bind_fixture_roles, role_state, candidate_role_state,
    validate_role_state, validate_candidate_role_state,
)
from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.fixture_role_identity import (
    resolve_fixture_role_identity, verify_fixture_role_identity,
)
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_spider.storage.raw_archive import digest
from test_finland_stage_fixture import schedule
from test_official_fixture import Transport, NOW
from test_visible_state import document, parent, action


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args,**kwargs):
        pytest.fail('Fixture roles cannot perform network access')
    for name in ('connect','connect_ex'):
        monkeypatch.setattr(socket.socket,name,deny)
    monkeypatch.setattr(socket,'create_connection',deny)
    monkeypatch.setattr(socket,'getaddrinfo',deny)


def identity_inputs(tmp_path, *, home='HJK', away='VPS', native='4353553',
                    date='20261008', clock='18:00', at=NOW):
    """Reusable real-parser fixture; callers must mark every sample constructed."""
    transport=Transport(schedule(home,away,native,date,clock))
    result=prepare_official_fixture('veikkausliiga:2026:match:'+str(native),output=tmp_path/'identity',
        transport=transport,clock=lambda:at)
    assert result['status']=='ready',result
    binding=json.loads(Path(result['identity_path']).read_text(encoding='utf-8'))
    resolved=resolve_fixture_role_identity(binding,archive_root=result['identity_archive'],as_of=at)
    return {'binding':binding,'archive':Path(result['identity_archive']),'identity':resolved,'at':at}


def visible_for(inputs, *texts, labels=(), prior=None, decision_at=None):
    """Visible bodies and parent labels only, with the actual fixture scope."""
    at=decision_at or inputs['at']+timedelta(minutes=2)
    policy=EvidenceOnlyPolicy(inputs['binding'])
    docs=[]
    for index,text in enumerate(texts):
        doc=document(text,docid='body-'+str(index))
        doc['observed_at']=doc['available_at']=at.isoformat()
        for segment in doc['structured_news']['segments']:
            segment['source_published_at']=(at-timedelta(minutes=2)).isoformat()
            segment['source_modified_at']=(at-timedelta(minutes=1)).isoformat()
        docs.append(doc)
    if labels:
        doc=parent()
        doc['observed_at']=doc['available_at']=at.isoformat()
        doc['links'][0]['observed_anchors']=[{'anchor_index':i,'text':text,'text_truncated':False}
                                              for i,text in enumerate(labels)]
        docs.append(doc)
    public={'state':policy.initial_state,'decision_at':at.isoformat(),'documents':docs,'attempts':[],
        'prediction':{'probabilities':prior,'probability_basis':'observed_market_snapshot' if prior else 'unavailable_no_market_prior'},
        'budget':{'limits':{'requests':max(4,len(docs)),'money':1},'spent':{'requests':len(docs),'money':0}}}
    return encode_visible_state(public,[action()])


def bound_visible(inputs,*texts,**kwargs):
    return bind_fixture_roles(visible_for(inputs,*texts,**kwargs),inputs['identity'])


def test_role_partition_is_fixture_specific_with_unknown_and_observed_parent_labels(tmp_path):
    inputs=identity_inputs(tmp_path)
    value=visible_for(inputs,'HJK midfielder returns.','VPS missing a player.','HJK host VPS.',
        'Rain expected.','VPSB is a different token.',labels=['VPS pre-match update'])
    original=deepcopy(value)
    bound=bind_fixture_roles(value,inputs['identity'])
    state=role_state(bound)
    assert value==original
    assert bound['before']==value['before'] and bound['candidates']==value['candidates']
    assert state['role_texts']=={'home':['HJK midfielder returns.'],
        'away':['VPS missing a player.','VPS pre-match update'],'both':['HJK host VPS.'],
        'unattributed':['Rain expected.','VPSB is a different token.'],'ambiguous':[]}
    assert sorted(sum(state['role_texts'].values(),[]))==state['texts']
    assert state['identity_scope']['home_id']=='veikkausliiga:team:60731'
    assert state['identity_scope']['away_id']=='veikkausliiga:team:60901'
    assert state['physical_match_id']==inputs['binding']['match_id']
    assert state['role_counts']=={'home':1,'away':2,'both':1,'unattributed':2,'ambiguous':0}
    assert abs(sum(state['role_numeric'])-1)<1e-12
    assert validate_role_state(state)==state
    assert bind_fixture_roles(bound,inputs['identity'])==bound


def test_swapping_verified_order_changes_roles_without_changing_body(tmp_path):
    home=identity_inputs(tmp_path/'home')
    away=identity_inputs(tmp_path/'away',home='VPS',away='HJK',native='4353554')
    text='HJK key player will return.'
    first=role_state(bound_visible(home,text))
    second=role_state(bound_visible(away,text))
    assert first['role_texts']['home']==second['role_texts']['away']==[text]
    assert first['role_texts']['away']==second['role_texts']['home']==[]


@pytest.mark.parametrize('text,aliases,want',[
    ('hJk:n player',{'home':['HJK'],'away':['VPS']},'home'),
    ('HJK2 and preHJK',{'home':['HJK'],'away':['VPS']},'unattributed'),
    ('HJK Klubi player',{'home':['HJK'],'away':['HJK Klubi']},'away'),
    ('The CLUB statement',{'home':['Club'],'away':['CLUB']},'ambiguous'),
    ('A B C',{'home':['A B'],'away':['B C']},'ambiguous'),
    ('STRASSE and HÉLSINKI',{'home':['Straße'],'away':['Hélsinki']},'both'),
    ('weather is unclear',{'home':['HJK'],'away':['VPS']},'unattributed'),
])
def test_unicode_boundaries_longer_names_and_overlapping_mentions(text,aliases,want):
    assert _mentions(text,aliases)[0]==want


def test_only_canonical_names_are_allowed_not_unproved_registered_aliases(tmp_path):
    inputs=identity_inputs(tmp_path)
    assert '赫尔辛基' in inputs['binding']['aliases']['home']
    state=role_state(bound_visible(inputs,'赫尔辛基本场阵容'))
    assert state['role_texts']['unattributed']==['赫尔辛基本场阵容']
    assert inputs['identity']['scope']['aliases']['home']==['HJK']


def test_candidate_roles_use_actual_parent_labels_not_body_url_or_query(tmp_path):
    inputs=identity_inputs(tmp_path)
    value=bound_visible(inputs,'HJK already-read body',labels=['VPS observed headline'])
    aid=next(iter(value['candidates']))
    candidate=candidate_role_state(value,aid)
    assert candidate['texts']==candidate['role_texts']['away']==['VPS observed headline']
    assert candidate['role_texts']['home']==[]
    assert validate_candidate_role_state(candidate)==candidate
    empty=candidate_role_state(bound_visible(inputs,'HJK body only'),aid)
    assert not empty['texts'] and not any(empty['role_masks'].values())


@pytest.mark.parametrize('field,value',[
    ('match_id','other'),('home_team_id','veikkausliiga:team:60901'),
    ('away_team_id','veikkausliiga:team:60731'),('competition_id','Cup'),
    ('kickoff','2026-10-08T16:00:00+00:00'),('prediction_cutoff','2026-10-08T14:58:00+00:00'),
])
def test_ordered_scope_match_competition_and_clocks_are_exact(tmp_path,field,value):
    inputs=identity_inputs(tmp_path)
    visible=visible_for(inputs,'HJK text')
    target=visible['before'] if field in {'match_id','prediction_cutoff'} else visible['before']['fixture']
    target[field]=value
    with pytest.raises(ValueError,match='scope'):
        bind_fixture_roles(visible,inputs['identity'])


def test_identity_observed_after_old_decision_is_not_retroactive_input(tmp_path):
    inputs=identity_inputs(tmp_path)
    visible=visible_for(inputs,decision_at=inputs['at']-timedelta(seconds=1))
    with pytest.raises(ValueError,match='available|identity'):
        bind_fixture_roles(visible,inputs['identity'])
    # A later research derivation time is recorded separately, not substituted
    # for source availability or manufactured as an old source receipt.
    assert inputs['identity']['derivation']['actual_created_at']!=inputs['identity']['scope']['available_at']


@pytest.mark.parametrize('mutation',['role','numeric','text','scope','body','candidate'])
def test_sidecar_tampering_is_rejected_even_after_self_digest_update(tmp_path,mutation):
    inputs=identity_inputs(tmp_path)
    value=bound_visible(inputs,'HJK text',labels=['VPS headline'])
    side=value['fixture_roles']
    if mutation=='role':
        side['state']['role_texts']['away']+=side['state']['role_texts']['home']
    elif mutation=='numeric':
        side['state']['role_numeric'][0]=99
    elif mutation=='text':
        content=value['before']['contents'][0]
        content['text']='VPS text'
    elif mutation=='scope':
        side['identity_scope']['home_name']='VPS'
    elif mutation=='body':
        side['candidates'][next(iter(side['candidates']))]['role_texts']['home']=['HJK hidden body']
    else:
        value['candidates'][next(iter(value['candidates']))]['observed_link_refs']=['invented']
    side['projection_sha256']=digest({k:v for k,v in side.items() if k!='projection_sha256'})
    with pytest.raises(ValueError):
        role_state(value)


def test_state_and_candidate_vectors_cannot_be_detached_from_visible_proof(tmp_path):
    inputs=identity_inputs(tmp_path)
    value=bound_visible(inputs,'HJK text',labels=['VPS headline'])
    state=role_state(value)
    state['role_masks']['home']=False
    with pytest.raises(ValueError,match='proof'):
        validate_role_state(state)
    candidate=candidate_role_state(value,next(iter(value['candidates'])))
    candidate['role_texts']['home']=['HJK hidden body']
    with pytest.raises(ValueError,match='proof'):
        validate_candidate_role_state(candidate)


def test_portable_original_proof_survives_locator_change_and_has_no_model_wallclock(tmp_path):
    inputs=identity_inputs(tmp_path/'original')
    target=tmp_path/'copy'
    shutil.copytree(inputs['archive'],target)
    other=resolve_fixture_role_identity(inputs['binding'],archive_root=target,as_of=inputs['at'])
    assert other['derivation']['archive_root']!=inputs['identity']['derivation']['archive_root']
    before=visible_for(inputs,'HJK player')
    left=bind_fixture_roles(before,inputs['identity'])
    right=bind_fixture_roles(before,other)
    assert left==right and role_state(left)==role_state(right)
    encoded=json.dumps(role_state(left))
    assert 'archive_root' not in encoded and 'actual_created_at' not in encoded


@pytest.mark.parametrize('mutation',['bytes','path','binding','scope'])
def test_identity_capsule_rechecks_original_bytes_not_a_verified_flag(tmp_path,mutation):
    inputs=identity_inputs(tmp_path)
    identity=deepcopy(inputs['identity'])
    if mutation=='bytes':
        entry=next(row for row in identity['proof']['files'] if row['path'].endswith('.body.gz'))
        body=base64.b64decode(entry['bytes_base64'])+b'changed'
        entry.update(bytes_base64=base64.b64encode(body).decode(),sha256=sha256(body).hexdigest())
    elif mutation=='path':
        identity['proof']['files'][0]['path']='../outside.observation.json'
    elif mutation=='binding':
        identity['binding']['canonical']['home_name']='VPS'
        identity['binding']['binding_sha256']=digest({k:v for k,v in identity['binding'].items() if k!='binding_sha256'})
    else:
        identity['scope']['home_id']='other-team'
    identity['proof']['archive_sha256']=digest(identity['proof']['files'])
    identity['identity_sha256']=digest({k:v for k,v in identity.items() if k not in {'identity_sha256','derivation'}})
    with pytest.raises((ValueError,OSError,EOFError)):
        verify_fixture_role_identity(identity,as_of=inputs['at'])


def test_existing_market_identity_uses_exact_provider_scope_and_official_canonical_names(tmp_path):
    from test_finland_stage_fixture import identity as market_identity, AUDIT
    archive,binding=market_identity(tmp_path)
    capsule=resolve_fixture_role_identity(binding,archive_root=archive.root,as_of=AUDIT)
    assert capsule['scope']['match_id']=='spftll:match:3085211'
    assert capsule['scope']['physical_match_id']=='veikkausliiga:2026:match:4353553'
    assert capsule['scope']['runtime_fixture']['home_team_id']=='spftll:team:383'
    assert capsule['scope']['home_name']=='HJK'
    assert verify_fixture_role_identity(capsule,as_of=AUDIT)['scope']==capsule['scope']
