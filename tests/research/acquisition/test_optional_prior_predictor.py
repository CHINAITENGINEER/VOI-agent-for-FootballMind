"""Constructed optional-prior contracts; no real model or benefit claims."""
from copy import deepcopy
import json
import socket

import numpy as np
import pytest
from scipy.special import logsumexp

from football_research.acquisition.context_predictor import ContextPredictor, PARAMETERS, _state, _digest
from football_research.acquisition.optional_prior_predictor import (
    OptionalPriorContextPredictor, training_readiness, BASE_VERSION, VERSION,
)
from test_context_predictor import AS_OF, WORDS, records as provided_records, visible

LABELS = [0]*7+[1]*3+[2]*2
MISSING_BASIS = 'not_supplied_no_prior_inferred_from_current_prediction'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a,**k):raise AssertionError('Optional predictor tests must remain offline')
    monkeypatch.setattr(socket,'getaddrinfo',denied)
    monkeypatch.setattr(socket,'create_connection',denied)
    monkeypatch.setattr(socket.socket,'connect',denied)


def missing(value):
    result=deepcopy(value)
    result['before']['prior_probabilities']=None
    result['before']['prior_basis']=MISSING_BASIS
    # A supplied current posterior remains inert even when no original prior
    # exists. It cannot replace an observed snapshot or a learned base.
    result['before']['current_probabilities']=[.99,.005,.005]
    return result


def rows(domain='missing',prefix='missing'):
    data=provided_records()
    for i,row in enumerate(data):
        index=i//2
        row['physical_match_id']=f'{prefix}:official:{index}'
        row['visible_state']['before']['match_id']=f'{prefix}:provider:{index}'
        row['outcome']=LABELS[index]
        if domain=='missing':row['visible_state']=missing(row['visible_state'])
    return data


def fit(data,version='constructed-optional-prior-unit-v1'):
    return OptionalPriorContextPredictor().fit(data,as_of=AS_OF,dataset_version=version,
        evidence_origin='constructed_mechanism')


@pytest.fixture(scope='module')
def missing_model():return fit(rows())


@pytest.fixture(scope='module')
def market_model():return fit(rows('provided','market'))


def test_six_real_style_physical_matches_remain_not_ready_even_with_repeated_prefixes():
    data=rows()[:12]
    for i,row in enumerate(data):row['outcome']=[1,1,2,2,2,2][i//2]
    ready=training_readiness(data*20,as_of=AS_OF,evidence_origin='real_observational')
    assert not ready['ready_to_fit'] and ready['physical_matches']==6
    assert ready['class_counts']=={'0':0,'1':2,'2':4}
    assert ready['distinct_prefixes']==12 and ready['synthetic_mask_views']==0
    assert set(ready['reasons'])=={'fewer_than_12_independent_physical_matches','physical_matches_do_not_cover_all_three_outcomes'}
    model=OptionalPriorContextPredictor()
    with pytest.raises(ValueError,match='not_ready'):
        model.fit(data,as_of=AS_OF,dataset_version='not-a-real-fit',evidence_origin='real_observational')
    assert not model.model_version and not model.coefficients
    assert PARAMETERS['min_physical_matches']==12


def test_missing_prior_uses_only_cutoff_training_class_counts_with_no_uniform_imputation(missing_model):
    state=missing(visible(text=None))
    result=missing_model.predict(state,physical_match_id='future:official')
    assert result['prior_probabilities'] is None and state['before']['prior_probabilities'] is None
    assert result['base_probabilities']==result['probabilities']==[7/12,3/12,2/12]
    assert result['forecast_basis']==BASE_VERSION and result['prediction_kind']=='empirical_base_fallback'
    assert result['base_source']=={'training_as_of':AS_OF,'physical_matches':12}
    assert result['probabilities']!=state['before']['current_probabilities']
    assert result['numeric_information_value'] is None and result['information_value_status'].startswith('UNKNOWN')


def test_supported_missing_text_uses_the_same_offset_softmax_and_explicit_origin(missing_model):
    result=missing_model.predict(missing(visible()),physical_match_id='future:official')
    assert result['prior_domain_context_supported'] and result['fallback_reason'] is None
    expected=np.exp(np.log([7/12,3/12,2/12])+np.asarray(result['logit_residual']))
    np.testing.assert_allclose(result['probabilities'],expected/expected.sum(),atol=1e-15,rtol=0)
    assert min(result['probabilities'])>0 and abs(sum(result['probabilities'])-1)<1e-12
    assert result['evidence_origin']=='constructed_mechanism' and not result['production_ready']
    assert missing_model.encoder['version']=='visible-character-tfidf-context-v1'


def test_only_market_training_cannot_claim_missing_domain_context_support(market_model):
    result=market_model.predict(missing(visible()),physical_match_id='future:official')
    assert result['probabilities']==[7/12,3/12,2/12]
    assert result['fallback_reason']==result['support_status']=='missing_prior_context_unsupported'
    assert result['logit_residual']==[0.,0.,0.] and result['context_effect_status']=='UNKNOWN'
    assert result['numeric_information_value'] is None
    assert market_model.metadata['prior_domains']['missing_prior']['physical_matches']==0
    assert market_model.metadata['synthetic_mask_views']==0


def test_missing_only_training_preserves_actual_market_input_without_claiming_supported_correction(missing_model):
    state=visible()
    result=missing_model.predict(state,physical_match_id='future:official')
    assert result['probabilities']==[.45,.3,.25]
    assert result['forecast_basis']=='observed_market_snapshot'
    assert result['fallback_reason']=='provided_prior_context_unsupported'


def test_supplied_prior_fit_reuses_exact_legacy_encoder_optimizer_and_predictions(market_model):
    legacy=ContextPredictor().fit(rows('provided','market'),as_of=AS_OF,dataset_version='same-inputs',
                                evidence_origin='constructed_mechanism')
    assert legacy.encoder==market_model.encoder
    np.testing.assert_array_equal(legacy.coefficients,market_model.coefficients)
    assert legacy.metadata['optimizer']==market_model.metadata['optimizer']
    assert legacy.predict(visible(),physical_match_id='future:official')['probabilities']==market_model.predict(
        visible(),physical_match_id='future:official')['probabilities']
    with pytest.raises(ValueError,match='probabilities'):
        legacy.predict(missing(visible()),physical_match_id='future:official')


def test_actual_both_domains_share_one_f_but_are_counted_as_independent_physical_fixtures():
    data=rows()+rows('provided','market')
    model=fit(data)
    assert model.metadata['readiness']['physical_matches']==24
    assert model.metadata['empirical_base']['class_counts']=={'0':14,'1':6,'2':4}
    assert all(d['context_supported'] for d in model.metadata['prior_domains'].values())
    assert model.metadata['synthetic_mask_views']==0
    for state in (visible(),missing(visible())):
        result=model.predict(state,physical_match_id='future:official')
        assert result['prediction_kind']=='regularized_context_forecast'


def test_twelve_empty_missing_prefixes_do_not_borrow_supported_market_text():
    data=rows('provided','market')+[row for row in rows() if not row['visible_state']['before']['contents']]
    model=fit(data)
    support=model.metadata['prior_domains']['missing_prior']
    assert support['physical_matches']==12 and support['text_physical_matches']==0 and not support['context_supported']
    result=model.predict(missing(visible()),physical_match_id='future:official')
    assert result['support_status']=='missing_prior_context_unsupported' and result['numeric_information_value'] is None


def test_insufficient_missing_class_support_is_not_patched_with_market_classes():
    missing_data=[row for row in rows() if row['outcome']!=2]
    model=fit(rows('provided','market')+missing_data)
    support=model.metadata['prior_domains']['missing_prior']
    assert support['text_class_counts']['2']==0 and not support['context_supported']
    assert model.predict(missing(visible()),physical_match_id='future:official')['fallback_reason']=='missing_prior_context_unsupported'


def test_empty_and_out_of_vocabulary_keep_empirical_base_without_learning_from_current_response(missing_model):
    for text in (None,'龍龘麤靐龖龗'):
        result=missing_model.predict(missing(visible(text=text)),physical_match_id='future:official')
        assert result['probabilities']==[7/12,3/12,2/12]
        assert result['logit_residual']==[0.,0.,0.]


def test_visible_index_labels_are_valid_but_unread_body_current_posterior_and_labels_are_not(missing_model):
    state=missing(visible(headline=True))
    original=deepcopy(state)
    result=missing_model.predict(state,physical_match_id='future:official')
    assert result['known_character_features']>0 and state['before']['contents']==[]
    state['candidates']={'future':{'future_body':WORDS[2],'outcome':2,'expected_gain':100}}
    state['before']['current_probabilities']=[.001,.001,.998]
    state['before']['budget']={'limits':{'money':999}}
    state['outcome']=2
    assert missing_model.predict(state,physical_match_id='future:official')==result
    assert missing_model.metadata['empirical_base']['probabilities']==[7/12,3/12,2/12]
    assert original['before']['prior_probabilities'] is None


def test_duplicate_prefixes_and_provider_aliases_do_not_reweight_labels_or_f(missing_model):
    data=rows()
    original=deepcopy(data)
    alias=deepcopy(data[1]);alias['visible_state']['before']['match_id']='new-provider:same-physical'
    other=fit(data+[deepcopy(data[1]) for _ in range(35)]+[alias])
    assert data==original
    assert other.metadata['readiness']['physical_matches']==12 and other.metadata['readiness']['distinct_prefixes']==24
    assert other.metadata['empirical_base']==missing_model.metadata['empirical_base']
    assert other.encoder==missing_model.encoder
    np.testing.assert_array_equal(other.coefficients,missing_model.coefficients)


def test_fixture_with_both_actual_prior_states_counts_once_in_empirical_frequency():
    data=rows('provided','market')
    actual_missing=deepcopy(data[1]);actual_missing['visible_state']=missing(actual_missing['visible_state'])
    model=fit(data+[actual_missing])
    assert model.metadata['readiness']['physical_matches']==12
    assert model.metadata['empirical_base']['probabilities']==[7/12,3/12,2/12]
    assert model.metadata['prior_domains']['missing_prior']['physical_matches']==1


@pytest.mark.parametrize('mutation,reason',[
    (lambda r:r.update(purpose='confirmation'),'f_train'),
    (lambda r:r.update(result_available_at='2025-01-16T00:00:00Z'),'Settlement'),
    (lambda r:r['visible_state']['before'].update(decision_at='2025-01-01T20:00:00Z'),'prematch'),
    (lambda r:r['visible_state']['before']['documents'][0].update(available_at='2025-01-02T00:00:00Z'),'clock'),
    (lambda r:r['visible_state']['before'].update(prior_basis='observed_market_snapshot'),'no-prior'),
])
def test_actual_input_clock_and_origin_contract_is_not_bypassed_by_missing_prior(mutation,reason):
    data=rows();mutation(data[1])
    with pytest.raises(ValueError,match=reason):fit(data)


def test_physical_identity_and_statistical_holdout_time_are_required(missing_model):
    with pytest.raises(ValueError,match='Training physical'):
        missing_model.predict(missing(visible()),physical_match_id='missing:official:0')
    with pytest.raises(ValueError,match='Training physical'):
        missing_model.predict(missing(visible(mid='missing:provider:0')),physical_match_id='renamed-physical')
    with pytest.raises(ValueError,match='training cutoff'):
        missing_model.predict(missing(visible(day=13)),physical_match_id='future:official')


def test_new_schema_roundtrip_and_cross_loading_are_explicit(missing_model,market_model,tmp_path):
    path=tmp_path/'optional.json';missing_model.save(path)
    loaded=OptionalPriorContextPredictor.load(path)
    assert loaded.predict(missing(visible()),physical_match_id='future:official')==missing_model.predict(
        missing(visible()),physical_match_id='future:official')
    with pytest.raises(ValueError,match='schema'):ContextPredictor.load(path)
    legacy=ContextPredictor().fit(provided_records(),as_of=AS_OF,dataset_version='legacy',evidence_origin='constructed_mechanism')
    old_path=tmp_path/'old.json';legacy.save(old_path)
    with pytest.raises(ValueError,match='schema'):OptionalPriorContextPredictor.load(old_path)
    with pytest.raises(FileExistsError):missing_model.save(path)


@pytest.mark.parametrize('field',['empirical_base','coefficients','domain_support'])
def test_frozen_artifact_changes_do_not_silently_rebase_predictions(missing_model,tmp_path,field):
    path=tmp_path/'model.json';missing_model.save(path)
    payload=json.loads(path.read_text(encoding='utf-8'))
    if field=='empirical_base':payload['metadata']['empirical_base']['probabilities']=[1/3]*3
    elif field=='coefficients':payload['coefficients'][0][0]+=.1
    else:payload['metadata']['prior_domains']['provided_prior']['context_supported']=True
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError,match='version mismatch'):OptionalPriorContextPredictor.load(path)


@pytest.mark.parametrize('field',['empirical_base','domain_support'])
def test_recomputed_version_cannot_authorize_unlearned_uniform_base_or_fake_domain_support(missing_model,tmp_path,field):
    path=tmp_path/'model.json';missing_model.save(path)
    payload=json.loads(path.read_text(encoding='utf-8'))
    if field=='empirical_base':payload['metadata']['empirical_base']['probabilities']=[1/3]*3
    else:
        payload['metadata']['prior_domains']['provided_prior']['context_supported']=True
        payload['metadata']['readiness']['prior_domains']['provided_prior']['context_supported']=True
    payload['model_version']=VERSION+':'+_digest({k:v for k,v in payload.items() if k!='model_version'})[:24]
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError,match='Invalid'):OptionalPriorContextPredictor.load(path)


def test_objective_preserves_equal_fixture_weight_after_missing_offset_substitution():
    data=rows()
    extra=deepcopy(data[1]);extra['visible_state']=missing(visible('missing:provider:0',day=1,text=WORDS[0]+' Another real visible sentence.'))
    data.append(extra)
    model=fit(data)
    losses={}
    for row in data:
        state=_state(row['visible_state'],allow_missing_prior=True)
        vector=model._vector(state)[0]
        logits=np.log(model.metadata['empirical_base']['probabilities'])+vector@np.asarray(model.coefficients)
        losses.setdefault(row['physical_match_id'],[]).append(logsumexp(logits)-logits[row['outcome']])
    expected=np.mean([np.mean(x) for x in losses.values()])+PARAMETERS['regularization']/2*np.square(model.coefficients).sum()
    assert abs(expected-model.metadata['optimizer']['objective'])<1e-12
