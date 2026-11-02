from __future__ import annotations

import json

from football_agent.acquisition_boundary import CandidateBatch
from football_agent.query_value_contract import (
    BUDGET_EXHAUSTED,
    DEFER,
    MODELED_DECLARED_ASSUMPTIONS,
    MODEL_SCOPED_STOP,
    NO_FEASIBLE_REGISTERED_ACTION,
    QUERY,
    VALUE_UNMEASURED,
    WITHHELD_OPEN_WORLD_VALUE_UNKNOWN,
    QueryValueAssessment,
    StopCertificate,
)
from football_agent.value_selection_policy import ValueAwareSelectionPolicy


def make_batch(*actions, remaining=None, final=False):
    state = {
        'task_id': 'formal-contract',
        'remaining': remaining if remaining is not None else {'http_calls': 2, 'local_reads': 1},
        'final_decision_only': final,
    }
    responses = []
    for index, action in enumerate(actions):
        responses.append({
            'task_id': state['task_id'], 'assessment': {}, 'analysis_plan': [],
            'previous_return': 'none', 'match_identity': {}, 'return_assessment': None,
            'action': action,
            'url': 'https://example.test/' + str(index) if action in {'search_web', 'read_url'} else None,
            'query': 'candidate-' + str(index) if action == 'search_web' else None,
            'engine': 'fixture' if action == 'search_web' else None,
            'question_to_user': 'Which match?' if action == 'ask_user' else None,
            'reason': 'fixture',
        })
    return CandidateBatch.build(state, responses, origin='formal-contract-test')


def candidate_assessor(expected='positive'):
    def assess(_, candidate):
        action = candidate.response['action']
        if expected == 'unknown':
            return QueryValueAssessment.unmeasured(candidate.candidate_id, cost=.05)
        if expected == 'negative':
            return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=.05, cost=.05)
        if action == 'search_web':
            return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=.40, cost=.05)
        return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=.10, cost=.05)
    return assess


def diagnostics(choice):
    return json.loads(choice.diagnostics_json)


def test_positive_acquisition_exposes_formal_query_decision():
    batch = make_batch('search_web', 'query_acquired')
    choice = ValueAwareSelectionPolicy(candidate_assessor()).select(batch.context, batch)
    data = diagnostics(choice)
    formal = data['query_value_decision']
    assert choice.candidate_id == batch.candidates[0].candidate_id
    assert formal['action'] == QUERY
    assert formal['candidate_id'] == batch.candidates[0].candidate_id
    assert formal['stop_certificate'] is None
    assert {item['value_status'] for item in formal['assessments']} == {MODELED_DECLARED_ASSUMPTIONS}
    assert data['query_value_contract_scope'] == 'acquisition_actions_only'


def test_open_world_defer_preserves_unknown_certificate_and_resource_mapping():
    batch = make_batch('search_web', remaining={'http_calls': 2, 'local_reads': 1})
    choice = ValueAwareSelectionPolicy(candidate_assessor('unknown')).select(batch.context, batch)
    data = diagnostics(choice)
    formal = data['query_value_decision']
    assert choice.terminal_action == 'defer'
    assert formal['action'] == DEFER
    assert formal['stop_certificate']['kind'] == WITHHELD_OPEN_WORLD_VALUE_UNKNOWN
    assert formal['stop_certificate']['scope'] == 'open_world'
    assert formal['stop_certificate']['remaining_budget'] == {'http_calls': 2.0, 'local_reads': 1.0}
    assert formal['stop_certificate']['actionable_unknown_count'] == 1
    assert formal['assessments'][0]['value_status'] == VALUE_UNMEASURED


def test_operational_exhaustion_is_defer_with_budget_certificate():
    batch = make_batch('search_web', remaining={'http_calls': 0, 'local_reads': 0})
    choice = ValueAwareSelectionPolicy(candidate_assessor()).select(batch.context, batch)
    formal = diagnostics(choice)['query_value_decision']
    assert choice.terminal_action == 'defer'
    assert formal['action'] == DEFER
    assert formal['stop_certificate']['kind'] == BUDGET_EXHAUSTED
    assert formal['stop_certificate']['scope'] == 'operational_budget'
    assert formal['stop_certificate']['remaining_budget'] == {'http_calls': 0.0, 'local_reads': 0.0}


def test_model_scoped_stop_carries_maximum_value_and_no_global_claim():
    batch = make_batch('search_web')
    choice = ValueAwareSelectionPolicy(candidate_assessor('negative')).select(batch.context, batch)
    formal = diagnostics(choice)['query_value_decision']
    assert choice.terminal_action == 'stop'
    assert formal['stop_certificate']['kind'] == MODEL_SCOPED_STOP
    assert formal['stop_certificate']['scope'] == 'declared_finite_model'
    assert formal['stop_certificate']['maximum_modeled_net_value'] == 0.0
    assert formal['stop_certificate']['global_claim'] is False


def test_terminal_only_batch_is_defer_and_not_a_zero_value_assessment():
    batch = make_batch('stop', 'defer')
    choice = ValueAwareSelectionPolicy(lambda *_: (_ for _ in ()).throw(AssertionError('estimator called'))).select(
        batch.context, batch)
    formal = diagnostics(choice)['query_value_decision']
    assert choice.terminal_action == 'defer'
    assert formal['action'] == DEFER
    assert formal['stop_certificate']['kind'] == NO_FEASIBLE_REGISTERED_ACTION
    assert formal['assessments'] == []


def test_identity_clarification_is_explicitly_outside_query_contract():
    batch = make_batch('ask_user', final=True)
    choice = ValueAwareSelectionPolicy(candidate_assessor()).select(batch.context, batch)
    data = diagnostics(choice)
    assert choice.candidate_id == batch.candidates[0].candidate_id
    assert data['query_value_decision'] is None
    assert data['query_value_decision_omission_reason'] == 'identity_clarification_outside_acquisition_contract'


def test_stop_certificate_accepts_scalar_and_rejects_invalid_resource_maps():
    scalar = StopCertificate(MODEL_SCOPED_STOP, 'declared_finite_model', False,
                             'fixture', 0, ('candidate',), .05, 0.0)
    assert scalar.as_dict()['remaining_budget'] == .05
    mapped = StopCertificate(BUDGET_EXHAUSTED, 'operational_budget', False,
                             'fixture', 1, ('candidate',), {'http_calls': 1, 'local_reads': 0}, None)
    assert mapped.as_dict()['remaining_budget'] == {'http_calls': 1.0, 'local_reads': 0.0}

