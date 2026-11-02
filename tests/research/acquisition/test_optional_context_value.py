"""Constructed optional-prior B/g/C contracts; no real model or effect claim."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import pytest

from football_research.acquisition.context_gain import _digest
from football_research.acquisition.context_value import ContextValueModel
from football_research.acquisition.optional_context_query import GAIN_VERSION, query_input
from football_research.acquisition.optional_context_value import OptionalContextValueModel, training_readiness, _public
from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.acquisition.optional_prior_predictor import VERSION as F_VERSION
from test_context_predictor import AS_OF, WORDS
from test_context_value import _fixture_part, gain_bundle as legacy_bundle
from test_optional_prior_predictor import rows, missing

G_AS_OF = '2027-02-01T00:00:00+00:00'


class FitClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2027,2,2,tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a,**k):
        raise AssertionError('Optional gain tests must remain offline')
    monkeypatch.setattr(socket,'getaddrinfo',denied)
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr('football_research.acquisition.optional_context_value.datetime',FitClock)


def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def part(f, index=0, *, context='positive', physical=None, failure=False, domain='missing', single_day=False):
    original = _fixture_part((f,Path('not-read'),'0'*64),index,context=context,
        physical=physical or f'B:{index}',failure=failure)
    prefixes,pairs,_=original
    if single_day:
        day=prefixes[0]['visible_state']['before']['fixture']['kickoff'][:10]
        prefixes,pairs=json.loads(json.dumps([prefixes,pairs]).replace(day,'2027-01-20'))
    if domain=='missing':
        for row in prefixes:
            row['visible_state']=missing(row['visible_state'])
    return prefixes,pairs


def bundle(f, fpath, folder, *, count=24, domain='missing', single_day=False, origin='constructed_mechanism'):
    folder.mkdir(parents=True)
    features,targets,all_prefixes,all_pairs=[],[],[],[]
    for index in range(count):
        prefixes,pairs=part(f,index,context='positive' if index<count//2 else 'negative',
            failure=index==count//2-1,domain=domain,single_day=single_day)
        before,after=prefixes;pair=pairs[0];physical=before['physical_match_id'];y=before['outcome']
        p=f.predict(before['visible_state'],physical_match_id=physical)
        q=f.predict(after['visible_state'],physical_match_id=physical)
        ident='sample:'+str(index)
        features.append({'sample_id':ident,'inputs':query_input(before['visible_state'],pair['action'],p)})
        targets.append({'sample_id':ident,'run_id':before['run_id'],'physical_match_id':physical,
            'before_prefix_id':before['prefix_id'],'after_prefix_id':after['prefix_id'],
            'outcome':y,'result_available_at':before['result_available_at'],
            'signed_gain':math.log(q['probabilities'][y])-math.log(p['probabilities'][y]),
            'prediction_before':p['probabilities'],'prediction_after':q['probabilities'],
            'before_support':p['support_status'],'after_support':q['support_status'],
            'response_status':pair['response_status'],'actual_cost':pair['actual_cost']})
        all_prefixes.extend(prefixes);all_pairs.extend(pairs)
    save(folder/'features.json',features);save(folder/'targets.json',targets)
    save(folder/'source-inputs.json',{'schema':'constructed-optional-prefix-pairs-v1','input_bindings':{},
        'prefixes':all_prefixes,'transitions':all_pairs,'scope':'constructed_mechanism_only'})
    save(folder/'calibration-registrations.json',[])
    report={'schema':GAIN_VERSION,'as_of':G_AS_OF,'evidence_origin':origin,
        'predictor_sha256':sha256(fpath.read_bytes()).hexdigest(),'predictor_model_version':f.model_version,
        'features_sha256':_digest(features),'targets_sha256':_digest(targets),
        'source_inputs_file_sha256':sha256((folder/'source-inputs.json').read_bytes()).hexdigest(),
        'registrations_file_sha256':sha256((folder/'calibration-registrations.json').read_bytes()).hexdigest()}
    save(folder/'report.json',report)
    return folder


def fitted(path, *, domain='missing'):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    f=OptionalPriorContextPredictor().fit(rows(domain),as_of=AS_OF,dataset_version='constructed-optional-f',
        evidence_origin='constructed_mechanism')
    fpath=path/'f.json';f.save(fpath)
    folder=bundle(f,fpath,path/'gains',domain=domain)
    with patch('football_research.acquisition.optional_context_value.datetime',FitClock):
        g=OptionalContextValueModel().fit(folder,predictor_path=fpath,as_of=G_AS_OF,dataset_version='constructed-optional-B')
    return f,fpath,g


def inputs(f, *, context='positive', domain='missing', physical='C:future'):
    prefixes,pairs=part(f,physical=physical,context=context,domain=domain)
    visible=json.loads(json.dumps(prefixes[0]['visible_state']).replace('2027-01-20','2027-02-10'))
    prediction=f.predict(visible,physical_match_id=physical)
    return query_input(visible,pairs[0]['action'],prediction)


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    return (*fitted(tmp_path_factory.mktemp('optional-g')),)


def test_missing_prior_remains_none_and_numeric_base_is_explicit(models):
    f,_,g=models
    query=inputs(f)
    assert query['state']['prior'] is None and query['prior_missing']
    assert query['forecast']['base_probabilities']==[7/12,3/12,2/12]
    assert query['forecast']['base_source']=={'training_as_of':AS_OF,'physical_matches':12}
    before=deepcopy(query);context=_public(query,predictor=f)
    assert query==before and context['mask'][-1] is True
    result=g.predict(query,physical_match_id='C:future',predictor_sha256=g.metadata['predictor_sha256'])
    assert result['support_status']=='supported' and result['prior_domain']=='missing_prior'
    assert result['support']['physical_matches']>=12 and result['support']['kickoff_dates']>=2


def test_signed_targets_and_actual_failed_costs_are_retained(models):
    f,path,g=models
    targets=json.loads((path.parent/'gains/targets.json').read_text(encoding='utf-8'))
    assert any(row['signed_gain']< -1e-12 for row in targets)
    assert any(row['signed_gain']>1e-12 for row in targets)
    assert any(row['response_status']=='FAILED' and row['actual_cost']['requests']==1 for row in targets)
    assert g.metadata['negative_targets']==sum(row['signed_gain']< -1e-12 for row in targets)
    values=[g.predict(inputs(f,context=c),physical_match_id='C:future',predictor_sha256=g.metadata['predictor_sha256'])
            for c in ('positive','negative')]
    assert abs(values[0]['expected_signed_gain']-values[1]['expected_signed_gain'])>1e-10
    assert all(row['uncertainty'] is None and row['stop_certificate'] is None for row in values)


def test_unsupported_other_prior_domain_is_unknown_not_zero(models):
    f,_,g=models
    query=inputs(f,domain='provided')
    result=g.predict(query,physical_match_id='C:future',predictor_sha256=g.metadata['predictor_sha256'])
    assert result['support_status']=='UNKNOWN'
    assert result['reason']=='predictor_prior_domain_context_unsupported'
    assert result['expected_gain'] is result['expected_signed_gain'] is None


def test_unsupported_domain_cannot_be_fitted_as_zero_gain(models,tmp_path):
    f,fpath,_=models
    folder=bundle(f,fpath,tmp_path/'invalid',domain='provided')
    with pytest.raises(ValueError,match='Unsupported prior domain'):
        training_readiness(folder,predictor_path=fpath,as_of=G_AS_OF)


def test_query_is_before_only_and_does_not_expose_unread_payload(models):
    f,_,_=models
    prefixes,pairs=part(f,physical='C:new')
    visible=json.loads(json.dumps(prefixes[0]['visible_state']).replace('2027-01-20','2027-02-10'))
    prediction=f.predict(visible,physical_match_id='C:new');action=pairs[0]['action']
    original=query_input(visible,action,prediction)
    visible['outcome']=2;visible['after']={'text':'SECRET_UNREAD'}
    visible['candidates'][action['action_id']]['future_response']='SECRET_UNREAD'
    visible['before']['current_probabilities']=[.001,.001,.998]
    action['signed_gain']=1234;prediction['future_outcome']='SECRET_UNREAD'
    assert query_input(visible,action,prediction)==original
    assert 'SECRET_UNREAD' not in json.dumps(original)


@pytest.mark.parametrize('field',['base','support','probabilities','model','actual_fit'])
def test_base_and_query_forecast_cannot_be_replaced_by_caller(models,field):
    f,_,g=models;query=inputs(f)
    if field=='base':query['forecast']['base_probabilities']=[1/3]*3
    elif field=='support':query['forecast']['prior_domain_context_supported']=False
    elif field=='probabilities':
        query['forecast']['probabilities']=query['prediction_before']=[1/3]*3
    elif field=='model':query['forecast']['model_version']='optional-prior-visible-context-v1:wrong'
    else:query['state']['decision_at']='2025-01-20T10:00:01+00:00'
    with pytest.raises(ValueError):
        g.predict(query,physical_match_id='C:future',predictor_sha256=g.metadata['predictor_sha256'])


@pytest.mark.parametrize('field',['candidate_host','parent_time','g_alias','physical_A','same_f'])
def test_unknown_source_and_identity_time_boundaries(models,field):
    f,_,g=models;query=inputs(f);physical='C:future';sha=g.metadata['predictor_sha256']
    if field=='candidate_host':
        query['candidate']['request']['url']='https://unseen.example/new'
        result=g.predict(query,physical_match_id=physical,predictor_sha256=sha)
        assert result['support_status']=='UNKNOWN' and result['expected_gain'] is None
        return
    if field=='parent_time':
        query['candidate']['observed_parents']=[{'parent':{'document_id':'future','raw_sha256':'f'*64,
            'observed_at':'2027-02-11T00:00:00Z','available_at':'2027-02-11T00:00:00Z'},'labels':[{'text':'future'}]}]
    elif field=='g_alias':
        query['forecast']['match_id']=query['state']['provider_match_id']=g.metadata['training_match_ids'][0]
    elif field=='physical_A':physical=f.metadata['training_physical_match_ids'][0]
    else:sha='0'*64
    with pytest.raises(ValueError):g.predict(query,physical_match_id=physical,predictor_sha256=sha)


@pytest.mark.parametrize('count,single_day,reason',[(6,False,'12_independent'),(24,True,'2_B_kickoff')])
def test_independent_match_and_date_gates_are_not_relaxed(models,tmp_path,count,single_day,reason):
    f,fpath,_=models
    folder=bundle(f,fpath,tmp_path/'short',count=count,single_day=single_day)
    ready=training_readiness(folder,predictor_path=fpath,as_of=G_AS_OF)
    assert not ready['ready_to_fit'] and any(reason in x for x in ready['reasons'])
    model=OptionalContextValueModel()
    with pytest.raises(ValueError,match='not_ready'):
        model.fit(folder,predictor_path=fpath,as_of=G_AS_OF,dataset_version='not-ready')
    assert not model.coefficients and not model.model_version


def test_repeated_bundle_does_not_inflate_support_or_reweight(models):
    _,fpath,g=models;folder=fpath.parent/'gains'
    duplicate=OptionalContextValueModel().fit([folder,folder],predictor_path=fpath,as_of=G_AS_OF,dataset_version='duplicate')
    assert duplicate.metadata['readiness']==g.metadata['readiness']
    assert duplicate.encoder==g.encoder and duplicate.support_records==g.support_records
    np.testing.assert_array_equal(duplicate.coefficients,g.coefficients)


def test_save_load_exact_forecast_and_schema_rejection(models,tmp_path):
    f,fpath,g=models;path=tmp_path/'g.json';g.save(path)
    restored=OptionalContextValueModel.load(path,predictor_path=fpath)
    assert restored.predict(inputs(f),physical_match_id='C:future',predictor_sha256=g.metadata['predictor_sha256'])==g.predict(
        inputs(f),physical_match_id='C:future',predictor_sha256=g.metadata['predictor_sha256'])
    with pytest.raises(ValueError,match='schema'):ContextValueModel.load(path,predictor_path=fpath)
    with pytest.raises(FileExistsError):g.save(path)
    payload=json.loads(path.read_text(encoding='utf-8'));payload['coefficients'][0]+=.1;save(tmp_path/'bad.json',payload)
    with pytest.raises(ValueError,match='version mismatch'):OptionalContextValueModel.load(tmp_path/'bad.json',predictor_path=fpath)


def test_constructed_bundle_cannot_be_relabelled_real(models,tmp_path):
    f,fpath,_=models
    folder=bundle(f,fpath,tmp_path/'fake',origin='real_observational')
    with pytest.raises(ValueError,match='origin'):
        training_readiness(folder,predictor_path=fpath,as_of=G_AS_OF)


def test_resigning_origin_and_all_hashes_still_cannot_replace_raw_gateway(models,tmp_path):
    # An adversarial artifact, explicitly not a real training experiment.
    f=deepcopy(models[0]);f.metadata['evidence_origin']='real_observational'
    f.model_version=F_VERSION+':'+_digest(f._payload())[:24]
    fpath=tmp_path/'forged-real-f.json';f.save(fpath)
    folder=bundle(f,fpath,tmp_path/'forged-real',origin='real_observational')
    with pytest.raises(ValueError,match='reconstructable original official raw gateway'):
        training_readiness(folder,predictor_path=fpath,as_of=G_AS_OF)


def test_b_neighborhood_never_borrows_other_prior_domain_even_when_f_supports_both(tmp_path):
    f=OptionalPriorContextPredictor().fit(rows()+rows('provided','market'),as_of=AS_OF,
        dataset_version='constructed-both-domain-f',evidence_origin='constructed_mechanism')
    fpath=tmp_path/'both.json';f.save(fpath)
    folder=bundle(f,fpath,tmp_path/'missing-B')
    g=OptionalContextValueModel().fit(folder,predictor_path=fpath,as_of=G_AS_OF,dataset_version='missing-B-only')
    query=inputs(f,domain='provided')
    assert query['forecast']['prior_domain_context_supported']
    result=g.predict(query,physical_match_id='C:future',predictor_sha256=g.metadata['predictor_sha256'])
    assert result['support_status']=='UNKNOWN' and result['expected_gain'] is None
    assert result['reason']=='insufficient_independent_local_matches' and result['support']['physical_matches']==0


def test_original_g_shared_kernel_refactor_preserves_entire_payload_bytes(tmp_path):
    from test_context_value import fitted as legacy_fitted, query_inputs
    path=Path(__file__).resolve().parents[3]/'var/parallel/agent/round201/source-before/football_research/acquisition/context_value.py'
    spec=importlib.util.spec_from_file_location('football_research.acquisition._r201_previous_context_value',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    with patch('football_research.acquisition.context_value.datetime',FitClock),patch.object(old,'datetime',FitClock):
        f,fpath,current=legacy_fitted(tmp_path/'legacy')
        previous=old.ContextValueModel().fit(fpath.parent/'gains',predictor_path=fpath,as_of=G_AS_OF,
            dataset_version='constructed-g-value-B')
    current_path,previous_path=tmp_path/'current.json',tmp_path/'previous.json'
    current.save(current_path);previous.save(previous_path)
    assert current_path.read_bytes()==previous_path.read_bytes()
    kwargs={'physical_match_id':'heldout:future','predictor_sha256':sha256(fpath.read_bytes()).hexdigest()}
    for context in ('positive','negative'):
        assert current.predict(query_inputs(f,context=context),**kwargs)==previous.predict(query_inputs(f,context=context),**kwargs)
