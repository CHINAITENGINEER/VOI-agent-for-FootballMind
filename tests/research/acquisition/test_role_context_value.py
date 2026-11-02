"""Constructed role-conditioned f/g mechanisms, never real B or efficacy.

Identity and text projections use archived constructed HTML. Outcomes are
explicit constructed design labels. Gain bundles are marked constructed and
do not claim the independent raw result/registration gateway tested elsewhere.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import pytest

from football_agent.learned_voi_web import _web_action
from football_agent.news_discovery import NewsDiscoveryTools
from football_research.acquisition.context_gain import _digest
from football_research.acquisition.context_value import PARAMETERS
from football_research.acquisition.fixture_roles import bind_fixture_roles, ROLES
from football_research.acquisition.optional_context_value import OptionalContextValueModel
from football_research.acquisition.role_context_predictor import RoleContextPredictor
from football_research.acquisition.role_context_predictor import VERSION as F_VERSION
from football_research.acquisition.role_context_query import GAIN_VERSION, query_input
from football_research.acquisition.role_context_value import RoleContextValueModel, _public, training_readiness
from football_research.acquisition.visible_state import encode_visible_state
from football_research.data_adapters.fixture_role_identity import resolve_fixture_role_identity
from football_research.data_adapters.official_fixture import prepare_official_fixture
from football_spider.storage.raw_archive import RawArchive
from test_official_context_gain import Clock, OfficialSource

UTC = timezone.utc
UPDATE = 'HJK striker is available after recovery and full squad training.'
BALANCED = 'HJK and VPS prepare balanced squads for their evenly matched meeting.'
LABELS = ('HJK VPS squad bulletin', 'HJK VPS fitness report')
F_CUTOFF = '2026-10-10T11:00:00+00:00'
G_CUTOFF = '2026-10-14T11:00:00+00:00'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding='utf-8')


def forbidden(*args, **kwargs):
    raise AssertionError('Role gain tests cannot use network')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def make_views(root, index, *, decision_day, kickoff_day, swap=False, text=UPDATE, candidates=True):
    """Same literal source texts, independently verified ordered fixture."""
    Clock.at = datetime(2026, 10, decision_day, 12, tzinfo=UTC)
    source = OfficialSource(3000+index, kickoff_day)
    if swap:
        source.home, source.away = source.away, source.home
    prepared = prepare_official_fixture('veikkausliiga:2026:match:'+source.native,
        output=root/f'id-{index}', clock=lambda: Clock.now(UTC), transport=source)
    assert prepared['status'] == 'ready', prepared
    binding = json.loads(Path(prepared['identity_path']).read_text(encoding='utf-8'))
    identity = resolve_fixture_role_identity(binding, archive_root=prepared['identity_archive'], as_of=Clock.now(UTC))
    archive = RawArchive(root/f'text-{index}')
    tools = NewsDiscoveryTools(archive, ['www.veikkausliiga.com'], projection_version=6)
    actions = [_web_action('read_url', f'https://www.veikkausliiga.com/uutiset/2026/10/{decision_day:02}/unread-{j}/',
                           cost={'money': .01, 'requests': 1.}) for j in range(2)]
    docs = []
    bodies = [text]
    for j, body in enumerate(bodies + [UPDATE, BALANCED]):
        url = f'https://www.veikkausliiga.com/uutiset/2026/10/{decision_day:02}/body-{j}/'
        html = ('<script type="application/ld+json">'+json.dumps({'@type': 'NewsArticle',
            'headline': 'Official squad bulletin', 'datePublished': f'2026-10-{decision_day:02}T11:00:00Z',
            'articleBody': body})+'</script>').encode()
        obs = archive.save('open_page', url, {}, html, status=200, observed_at=Clock.now(UTC))
        request = tools.prepare({'tool': 'read_url', 'url': url}, {'task': {'seed_urls': [url]}, 'documents': []})
        docs.append(tools.project(obs, request))
    parent = None
    if candidates:
        html = ('<main>'+''.join(f'<a href="{a["url"]}">{label}</a>' for a,label in zip(actions, LABELS))+'</main>').encode()
        obs = archive.save('open_page', 'https://www.veikkausliiga.com/', {}, html, status=200, observed_at=Clock.now(UTC))
        request = tools.prepare({'tool': 'read_url', 'url': obs.endpoint}, {'task': {'seed_urls': [obs.endpoint]}, 'documents': []})
        parent = tools.project(obs, request)
    public = {'state': {'match_id': binding['match_id'], 'prediction_cutoff': binding['prediction_cutoff'],
        'base': {'home_team_id': binding['canonical']['home_id'], 'away_team_id': binding['canonical']['away_id'],
                 'competition_id': binding['canonical']['competition'], 'kickoff': binding['canonical']['kickoff']},
        'action_history': []}, 'prediction': {'probabilities': None}, 'attempts': [],
        'decision_at': Clock.now(UTC).isoformat(), 'documents': [docs[0]] + ([parent] if parent else [])}
    def encode(documents):
        current = deepcopy(public)
        current['documents'] = documents
        encoded = encode_visible_state(current, actions if candidates else [])
        return bind_fixture_roles(encoded, identity)
    before = encode(public['documents'])
    after = [encode(public['documents']+[doc]) for doc in docs[1:]]
    return binding['canonical']['source_event_id'], before, after, actions


def gain_bundle(folder, f, fpath, *, count=24):
    folder.mkdir(parents=True)
    features, targets = [], []
    for i in range(count):
        physical, before, after, actions = make_views(folder/'raw', 100+i, decision_day=11,
            kickoff_day=12+i%2, swap=i>=count//2, text='HJK training update.')
        p = f.predict(before, physical_match_id=physical)
        for j, action in enumerate(actions):
            q = f.predict(after[j], physical_match_id=physical)
            ident = f'constructed-role:{i}:{j}'
            features.append({'sample_id': ident, 'inputs': query_input(before, action, p)})
            targets.append({'sample_id': ident, 'run_id': 'constructed-role:'+str(i),
                'physical_match_id': physical, 'before_prefix_id': 'before:'+str(j), 'after_prefix_id': 'after:'+str(j),
                'outcome': 0, 'result_available_at': '2026-10-14T10:00:00+00:00',
                'signed_gain': math.log(q['probabilities'][0])-math.log(p['probabilities'][0]),
                'prediction_before': p['probabilities'], 'prediction_after': q['probabilities'],
                'before_support': p['support_status'], 'after_support': q['support_status']})
    for name,value in [('features.json',features),('targets.json',targets),('calibration-registrations.json',[]),
        ('source-inputs.json',{'schema':'constructed-role-prefix-pairs-v1','input_bindings':{},
                               'scope':'constructed_design_labels_not_independent_raw_result_supervision'})]:
        save(folder/name,value)
    save(folder/'report.json',{'schema':GAIN_VERSION,'as_of':G_CUTOFF,'evidence_origin':'constructed_mechanism',
        'predictor_sha256':sha256(fpath.read_bytes()).hexdigest(),'predictor_model_version':f.model_version,
        'features_sha256':_digest(features),'targets_sha256':_digest(targets),
        'source_inputs_file_sha256':sha256((folder/'source-inputs.json').read_bytes()).hexdigest(),
        'registrations_file_sha256':sha256((folder/'calibration-registrations.json').read_bytes()).hexdigest()})
    return folder


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    root=tmp_path_factory.mktemp('constructed-role-g')
    with patch.object(socket,'getaddrinfo',forbidden), patch.object(socket.socket,'connect',forbidden), \
            patch('football_agent.inquiry_tools.datetime',Clock):
        records=[]
        for i in range(12):
            y=i//4
            physical,visible,_,_=make_views(root/'A',i,decision_day=7,kickoff_day=8+i%2,
                swap=y==2,text=BALANCED if y==1 else UPDATE,candidates=False)
            records.append({'physical_match_id':physical,'visible_state':visible,'outcome':y,
                            'result_available_at':'2026-10-10T10:00:00+00:00'})
        Clock.at=datetime(2026,10,10,12,tzinfo=UTC)
        with patch('football_research.acquisition.role_context_predictor.datetime',Clock):
            f=RoleContextPredictor().fit(records,as_of=F_CUTOFF,dataset_version='constructed-role-A',evidence_origin='constructed_mechanism')
        fpath=root/'f.json';f.save(fpath)
        folder=gain_bundle(root/'B',f,fpath)
        Clock.at=datetime(2026,10,14,12,tzinfo=UTC)
        with patch('football_research.acquisition.role_context_value.datetime',Clock):
            g=RoleContextValueModel().fit(folder,predictor_path=fpath,as_of=G_CUTOFF,dataset_version='constructed-role-B')
        queries=[]
        for i in range(2):
            physical,before,_,actions=make_views(root/'C',500+i,decision_day=15,kickoff_day=16,swap=bool(i),text='HJK training update.')
            p=f.predict(before,physical_match_id=physical)
            queries.append([query_input(before,action,p) for action in actions])
    return f,fpath,g,queries


def predict(g,query):
    return g.predict(query,physical_match_id=query['forecast']['physical_match_id'],predictor_sha256=g.metadata['predictor_sha256'])


def test_same_news_swapped_fixture_has_actual_role_gain_difference(models,record_property):
    f,path,g,queries=models
    a,b=[predict(g,group[0]) for group in queries]
    assert a['support_status']==b['support_status']=='supported'
    assert abs(a['expected_signed_gain']-b['expected_signed_gain'])>1e-10
    record_property('constructed_home_role_signed_gain',a['expected_signed_gain'])
    record_property('constructed_away_role_signed_gain',b['expected_signed_gain'])
    assert all(result['support']['physical_matches']>=12 and result['support']['kickoff_dates']>=2 for result in (a,b))
    assert a['cost_deducted'] is b['cost_deducted'] is False
    assert a['uncertainty'] is a['stop_certificate'] is None
    assert queries[0][0]['state']['prior'] is None and queries[1][0]['state']['prior'] is None
    targets=json.loads((path.parent/'B/targets.json').read_text(encoding='utf-8'))
    assert any(row['signed_gain']<0 for row in targets) and any(row['signed_gain']>0 for row in targets)


def test_role_by_text_and_state_by_candidate_are_numerical_interactions(models):
    _,_,g,queries=models
    vectors=[[g._vector(_public(query,predictor=g.predictor))[0] for query in group] for group in queries]
    width=len(g.encoder['vocabulary']); size=PARAMETERS['interaction_size']
    assert np.linalg.norm(vectors[0][0][1:1+5*width]-vectors[1][0][1:1+5*width])>0
    mixed=vectors[0][0][-size:]-vectors[0][1][-size:]-vectors[1][0][-size:]+vectors[1][1][-size:]
    assert np.linalg.norm(mixed)>1e-10
    assert g.encoder['numeric_size']==len(_public(queries[0][0])['numeric'])


def test_query_roles_cost_and_forecast_are_bound_to_original_visible_proof(models):
    _,_,g,queries=models
    for field in ('role','candidate','cost','forecast'):
        query=deepcopy(queries[0][0])
        if field=='role':query['state']['role_texts']['away'].append('UNREAD_BODY')
        elif field=='candidate':query['candidate_roles']['role_texts']['away'].append('UNREAD_BODY')
        elif field=='cost':query['candidate']['declared_cost']['money']=0
        else:query['forecast']['probabilities']=query['prediction_before']=[1/3]*3
        with pytest.raises(ValueError):predict(g,query)


def test_unknown_fields_and_unread_body_never_enter_numerical_features(models):
    _,_,g,queries=models
    query=deepcopy(queries[0][0]); expected=predict(g,query)
    query['future_response']='UNREAD_SECRET';query['outcome']=2;query['after']={'body':'UNREAD_SECRET'}
    assert predict(g,query)==expected
    query['candidate']['hidden_body']='UNREAD_SECRET'
    # The feature whitelist ignores unknown return fields, while the bound
    # candidate object rejects an attempted substitution of its exact menu.
    with pytest.raises(ValueError,match='original visible proof'):predict(g,query)


def test_role_g_roundtrip_and_old_schema_rejection(models,tmp_path):
    _,fpath,g,queries=models
    path=tmp_path/'g.json';g.save(path)
    restored=RoleContextValueModel.load(path,predictor_path=fpath)
    assert predict(restored,queries[0][0])==predict(g,queries[0][0])
    assert restored._payload()==g._payload()
    with pytest.raises(ValueError,match='schema'):OptionalContextValueModel.load(path,predictor_path=fpath)


def test_repeated_b_does_not_inflate_independent_support(models):
    _,fpath,g,_=models;folder=fpath.parent/'B'
    assert training_readiness([folder,folder],predictor_path=fpath,as_of=G_CUTOFF)==g.metadata['readiness']
    assert g.metadata['readiness']['physical_matches']==24 and g.metadata['readiness']['transitions']==48


def test_no_visible_role_neighborhood_stays_unknown_not_zero(models):
    f,_,g,queries=models
    original=queries[0][0]
    before=original['state']['role_proof']['visible_state']['before']
    public={'state':{'match_id':before['match_id'],'prediction_cutoff':before['prediction_cutoff'],
                     'base':deepcopy(before['fixture']),'action_history':[]},
            'decision_at':before['decision_at'],'documents':[],'attempts':[],
            'prediction':{'probabilities':None}}
    action=_web_action('read_url',original['candidate']['request']['url'],cost={'money':.01,'requests':1.})
    identity=original['state']['role_proof']['visible_state']['fixture_roles']['identity']
    visible=bind_fixture_roles(encode_visible_state(public,[action]),identity)
    forecast=f.predict(visible,physical_match_id=original['forecast']['physical_match_id'])
    result=predict(g,query_input(visible,action,forecast))
    assert result['support_status']=='UNKNOWN' and result['expected_signed_gain'] is None
    assert result['support']['physical_matches']==0


def test_resigned_constructed_bundle_cannot_claim_real_raw_B(models,tmp_path):
    f,fpath,_,_=models
    fake=deepcopy(f);fake.metadata['evidence_origin']='real_observational'
    fake.model_version=F_VERSION+':'+_digest(fake._payload())[:24]
    fake_path=tmp_path/'forged-f.json';fake.save(fake_path)
    folder=tmp_path/'forged-B';folder.mkdir()
    for name in ('features.json','targets.json','calibration-registrations.json','source-inputs.json','report.json'):
        value=json.loads((fpath.parent/'B'/name).read_text(encoding='utf-8'))
        if name=='report.json':
            value.update(evidence_origin='real_observational',predictor_model_version=fake.model_version,
                         predictor_sha256=sha256(fake_path.read_bytes()).hexdigest())
        save(folder/name,value)
    with pytest.raises(ValueError,match='reconstructable original official raw gateway'):
        training_readiness(folder,predictor_path=fake_path,as_of=G_CUTOFF)
