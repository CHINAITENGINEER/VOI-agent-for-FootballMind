"""Opt-in source-statement context; synthetic history is never empirical gold."""
from copy import deepcopy
from pathlib import Path
import runpy
import json
import hashlib

import pytest

from football_agent.acquisition_boundary import CandidateBatch, policy_identity
from football_agent import runtime_source_claims as claims
from football_agent.runtime_voi_model import (
    RuntimeVoIModel, SOURCE_CLAIM_VERSION, SOURCE_CLAIM_SEMANTICS,
    SOURCE_CLAIM_CONTEXT_KEYS,
)
from football_agent.state_bound_runtime_policy import StateBoundVoISelectionPolicy


HELPERS = runpy.run_path(str(Path(__file__).with_name('test_state_bound_runtime_policy.py')))
HOME_QUALIFICATION = 'home_reported_qualification_state'
SOURCE_HELPERS = runpy.run_path(str(Path(__file__).with_name('test_runtime_source_claims.py')))


def claim_package(*, keys=None):
    config = HELPERS['package'](keys=keys or [HOME_QUALIFICATION])
    config.update(version=SOURCE_CLAIM_VERSION, context_semantics=SOURCE_CLAIM_SEMANTICS)
    for row in config['history']:
        for key in SOURCE_CLAIM_CONTEXT_KEYS.intersection(row['context']):
            row['context'][key] = 'already_qualified' if key.endswith('qualification_state') else 'defensive'
    return config


def test_v1_configuration_and_policy_fingerprint_remain_identical():
    model = RuntimeVoIModel(HELPERS['package']())
    selector = StateBoundVoISelectionPolicy(model, allow_synthetic=True)
    assert model.version == 'runtime-voi-model-v1'
    assert model.configuration()['model_hash'] == 'a3a5340bacf60e322760980ae36a07e7e931f99d0af6330881ee4d26e76b580d'
    assert 'context_semantics' not in model.configuration()
    assert selector.requires_source_claims is False
    assert 'source_claims_version' not in selector.configuration
    assert policy_identity(selector)['fingerprint'] == '03a2ab15adb61ff435ebc5b94181b047fa186da2f16a0242ee1f5733c6dd3c3e'


def test_v2_requires_explicit_subject_bound_source_claim_semantics():
    config = claim_package(keys=['operation_document_count', *sorted(SOURCE_CLAIM_CONTEXT_KEYS)])
    model = RuntimeVoIModel(config)
    selector = StateBoundVoISelectionPolicy(model, allow_synthetic=True)
    descriptor = model.configuration()
    assert model.version == descriptor['adapter_version'] == SOURCE_CLAIM_VERSION
    assert descriptor['context_semantics'] == SOURCE_CLAIM_SEMANTICS
    assert selector.requires_source_claims is True
    assert selector.version == 'state-bound-runtime-policy-v3'
    assert descriptor['evidence_basis_verified'] is False
    assert descriptor['numeric_voi_calibrated'] is False
    assert set(model.estimator.context_keys) == set(config['context_keys'])
    config['context_semantics'] = 'world_facts'
    descriptor['context_semantics'] = 'world_facts'
    assert model.context_semantics == SOURCE_CLAIM_SEMANTICS


@pytest.mark.parametrize('semantics', [None, '', 'world_facts', 'independent_gold',
                                      'actor_source_claims_not_world_facts_v2', [], {}])
def test_v2_rejects_missing_or_wrong_semantics(semantics):
    config = claim_package()
    config['context_semantics'] = semantics
    with pytest.raises(ValueError, match='context_semantics'):
        RuntimeVoIModel(config)
    del config['context_semantics']
    with pytest.raises(ValueError, match='exact fields'):
        RuntimeVoIModel(config)


@pytest.mark.parametrize('key', ['qualification_state', 'tactical_layout',
                               'rotation_risk', 'home_reported_rotation_risk', 'outcome'])
def test_v2_rejects_unscoped_or_undeclared_football_context(key):
    config = claim_package()
    config['context_keys'] = [key]
    for row in config['history']:
        row['context'] = {key: 'already_qualified'}
    with pytest.raises(ValueError, match='context_keys'):
        RuntimeVoIModel(config)


def test_v1_does_not_silently_accept_v2_claim_semantics_or_keys():
    config = HELPERS['package']()
    config['context_semantics'] = SOURCE_CLAIM_SEMANTICS
    with pytest.raises(ValueError, match='exact fields'):
        RuntimeVoIModel(config)
    config = claim_package()
    config['version'] = 'runtime-voi-model-v1'
    del config['context_semantics']
    with pytest.raises(ValueError, match='context_keys'):
        RuntimeVoIModel(config)


def test_v2_history_retains_strict_context_and_independent_gain_boundaries():
    config = claim_package()
    config['history'][0]['context']['qualification_state'] = 'already_qualified'
    with pytest.raises(ValueError, match='history context'):
        RuntimeVoIModel(config)
    config = claim_package()
    config['history'][0]['link']['observed_risk_reduction'] = None
    with pytest.raises(ValueError, match='measured usable evidence'):
        RuntimeVoIModel(config)


def test_v2_technical_context_still_requires_explicit_semantics():
    config = claim_package(keys=['operation_document_count'])
    model = RuntimeVoIModel(config)
    assert StateBoundVoISelectionPolicy(model).requires_source_claims is True
    legacy = deepcopy(config)
    legacy['version'] = 'runtime-voi-model-v1'
    del legacy['context_semantics']
    assert model.configuration()['model_hash'] != RuntimeVoIModel(legacy).configuration()['model_hash']


def conditional_package():
    config = HELPERS['package']((('a', 2, .9), ('b', 2, -.5),
                                  ('a', 2, -.5), ('b', 2, .9)),
                               keys=['operation_document_count', HOME_QUALIFICATION])
    config.update(version=SOURCE_CLAIM_VERSION, context_semantics=SOURCE_CLAIM_SEMANTICS)
    for row in config['history']:
        cell = int(row['link']['link_id'].split('-')[1])
        row['context'][HOME_QUALIFICATION] = 'already_qualified' if cell < 2 else 'must_win'
    return config


def claim_batch(public, items):
    responses = []
    for name in ('a', 'b'):
        item = HELPERS['match_decision'](public)
        item['query'] = public['match_input'] + ' source ' + name
        item[claims.FIELD] = deepcopy(items)
        responses.append(item)
    return CandidateBatch.build(public, responses, origin='synthetic_proposer', recommended_index=0)


def choose(selector, batch):
    choice = selector.select(batch.context, batch)
    return choice, json.loads(choice.diagnostics_json)['state_bound_voi']


def test_same_observed_state_current_claim_changes_conditional_information_choice(tmp_path):
    _, public, _ = SOURCE_HELPERS['source_public'](tmp_path)
    before = deepcopy(public)
    selector = StateBoundVoISelectionPolicy(RuntimeVoIModel(conditional_package()), allow_synthetic=True)
    qualified = claim_batch(public, [SOURCE_HELPERS['claim'](public)])
    must_win = claim_batch(public, [SOURCE_HELPERS['claim'](public, value='must_win')])
    first, first_diag = choose(selector, qualified)
    second, second_diag = choose(selector, must_win)
    assert qualified.context == must_win.context
    assert first.candidate_id == qualified.candidates[0].candidate_id
    assert second.candidate_id == must_win.candidates[1].candidate_id != must_win.recommended_id
    for diag, expected in ((first_diag, 'already_qualified'), (second_diag, 'must_win')):
        assert diag['blocked_reason'] is None
        assert diag['projection']['context'][HOME_QUALIFICATION] == expected
        assert diag['projection']['context']['qualification_state'] == 'unknown'
        assert diag['projection']['context']['tactical_layout'] == 'unknown'
        projected = diag['source_claim_projection']
        assert projected['context'][HOME_QUALIFICATION] == expected
        assert projected['state']['home']['context']['rotation_risk'] == 'unknown'
        assert projected['world_truth_verified'] is False
        assert projected['numeric_voi_training_eligible'] is False
        assert projected['subject_and_interval_entailment_verified'] is False
        assert diag['numeric_voi_calibrated'] is False
    assert public == before and public[claims.HISTORY_FIELD] == []


def test_v2_configuration_declares_distinct_projection_and_source_semantics():
    selector = StateBoundVoISelectionPolicy(RuntimeVoIModel(claim_package()), allow_synthetic=True)
    config = selector.configuration
    assert config['context_projection'] == 'runtime-observed-context-v2'
    assert config['source_claims_version'] == claims.VERSION
    assert config['context_semantics'] == claims.SEMANTICS
    assert config['unknown_policy'] == 'DEFER_NO_QUALITATIVE_FALLBACK'


def test_v2_missing_capability_defers_without_legacy_or_qualitative_fallback():
    public = HELPERS['public']()
    selector = StateBoundVoISelectionPolicy(RuntimeVoIModel(claim_package()), allow_synthetic=True)
    choice, diag = choose(selector, HELPERS['batch'](public))
    assert choice.terminal_action == 'defer' and choice.candidate_id is None
    assert diag['blocked_reason'] == 'source_claim_capability_required'
    assert diag['source_claim_projection'] is None and not diag['intervals']
    assert set(diag['projection']['context'][key] for key in SOURCE_CLAIM_CONTEXT_KEYS) == {'unknown'}


@pytest.mark.parametrize('kind', ['missing_interval', 'conflicting', 'no_claims'])
def test_unknown_source_context_never_borrows_positive_history_or_recommendation(tmp_path, kind):
    _, public, _ = SOURCE_HELPERS['source_public'](tmp_path)
    if kind == 'missing_interval':
        items = [SOURCE_HELPERS['claim'](public, validity=False)]
    elif kind == 'conflicting':
        items = [SOURCE_HELPERS['claim'](public), SOURCE_HELPERS['claim'](public, value='must_win')]
    else:
        items = []
    selector = StateBoundVoISelectionPolicy(RuntimeVoIModel(conditional_package()), allow_synthetic=True)
    choice, diag = choose(selector, claim_batch(public, items))
    assert choice.terminal_action == 'defer' and choice.candidate_id is None
    assert diag['projection']['context'][HOME_QUALIFICATION] == 'unknown'
    assert {row['support_status'] for row in diag['intervals'].values()} == {'context_incomplete'}


def test_v1_ignores_source_claims_and_keeps_technical_projection(tmp_path):
    _, public, _ = SOURCE_HELPERS['source_public'](tmp_path)
    selector = StateBoundVoISelectionPolicy(RuntimeVoIModel(HELPERS['package']((('a', 2, .9),
                                                                            ('b', 2, -.5)))),
                                           allow_synthetic=True)
    choice, diag = choose(selector, claim_batch(public, [SOURCE_HELPERS['claim'](public)]))
    assert choice.candidate_id is not None
    assert diag['projection']['version'] == 'runtime-observed-context-v1'
    assert 'source_claim_projection' not in diag
    assert not SOURCE_CLAIM_CONTEXT_KEYS.intersection(diag['projection']['context'])
    assert diag['projection']['actor_semantic_assessments_used'] is False


def test_distinct_source_reports_change_selection_with_identical_match_and_menu(tmp_path):
    archive, baseline, original_documents = SOURCE_HELPERS['source_public'](tmp_path)
    selector = StateBoundVoISelectionPolicy(RuntimeVoIModel(conditional_package()), allow_synthetic=True)
    outcomes = []
    for value in ('already_qualified', 'must_win'):
        public, originals = deepcopy(baseline), deepcopy(original_documents)
        item = SOURCE_HELPERS['claim'](public, value=value)
        document = public['documents'][-1]
        document['excerpt'] = item['quote'] + '\n' + item['validity']['quote']
        document['raw_sha256'] = hashlib.sha256(document['excerpt'].encode()).hexdigest()
        originals[-1] = deepcopy(document)
        SOURCE_HELPERS['rebind'](archive, public, originals)
        batch = claim_batch(public, [item])
        choice, diag = choose(selector, batch)
        outcomes.append(diag['route_bindings'][choice.candidate_id]['route_id'])
        assert diag['projection']['context']['operation_document_count'] == 2
        assert diag['projection']['context'][HOME_QUALIFICATION] == value
        assert public['match_input'] == baseline['match_input']
        assert [c.response['query'] for c in batch.candidates] == [
            public['match_input'] + ' source a', public['match_input'] + ' source b']
    assert outcomes == ['a', 'b']


def test_same_source_support_stops_buying_route_when_cost_exceeds_value_upper_bound(tmp_path):
    _, public, _ = SOURCE_HELPERS['source_public'](tmp_path)
    batch = claim_batch(public, [SOURCE_HELPERS['claim'](public)])
    config = conditional_package()
    low_cost = StateBoundVoISelectionPolicy(RuntimeVoIModel(config), allow_synthetic=True)
    selected, initial = choose(low_cost, batch)
    assert initial['route_bindings'][selected.candidate_id]['route_id'] == 'a'
    supported_upper_bound = initial['intervals'][selected.candidate_id]['upper_bound']
    next(route for route in config['routes'] if route['route_id'] == 'a')['cost'] = supported_upper_bound + 1
    expensive = StateBoundVoISelectionPolicy(RuntimeVoIModel(config), allow_synthetic=True)
    choice, diag = choose(expensive, batch)
    assert choice.candidate_id is None and choice.terminal_action == 'stop'
    assert diag['projection'] == initial['projection']
    assert diag['source_claim_projection'] == initial['source_claim_projection']
    route = diag['intervals'][selected.candidate_id]
    assert route['upper_bound'] < route['cost']


def test_v2_cold_start_unknown_football_features_defers_without_positive_fallback():
    from football_agent.runtime_fixture_bridge import prepare_fixture_identity
    public = HELPERS['public']()
    public[claims.CAPABILITY_FIELD] = claims.capability()
    public[claims.HISTORY_FIELD] = []
    public['fixture_identity'] = prepare_fixture_identity(None, public, [])
    selector = StateBoundVoISelectionPolicy(RuntimeVoIModel(conditional_package()), allow_synthetic=True)
    choice, diag = choose(selector, claim_batch(public, []))
    assert choice.candidate_id is None and choice.terminal_action == 'defer'
    assert diag['projection']['source_claims_used'] is True
    assert {diag['projection']['context'][key] for key in SOURCE_CLAIM_CONTEXT_KEYS} == {'unknown'}
    assert not diag['identity_prerequisite_selected']
    assert {row['support_status'] for row in diag['intervals'].values()} == {'context_incomplete'}
