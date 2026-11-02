import json

import pytest

from football_agent.acquisition_boundary import CandidateBatch
from football_agent.query_value_contract import MODELED_DECLARED_ASSUMPTIONS, VALUE_UNMEASURED
from football_agent.runtime_cli import main
from football_agent.value_model_adapter import VERSION, CandidateValueModel
from football_agent.value_selection_policy import ValueAwareSelectionPolicy


def response(query):
    return {
        'task_id': 'task-1', 'assessment': {}, 'previous_return': 'none',
        'analysis_plan': {}, 'match_identity': {}, 'return_assessment': None,
        'action': 'search_web', 'url': 'https://example.com/' + query,
        'query': query, 'engine': 'bing', 'question_to_user': None,
        'reason': 'need evidence',
    }


def batch(*queries):
    return CandidateBatch.build({'task_id': 'task-1', 'remaining': {'http_calls': 2}},
                                [response(query) for query in queries], origin='test')


def test_registered_rule_matches_query_and_returns_declared_value():
    model = CandidateValueModel({
        'version': VERSION,
        'rules': [{
            'rule_id': 'lineup', 'action': 'search_web', 'query_contains': '伤停',
            'expected_value': 0.40, 'cost': 0.05,
        }],
    })
    candidates = batch('伤停信息', '天气')
    value = model.assess({}, candidates.candidates[0])
    assert value.value_status == MODELED_DECLARED_ASSUMPTIONS
    assert value.expected_value == 0.40 and value.cost == 0.05


def test_unmatched_candidate_remains_explicitly_unmeasured():
    model = CandidateValueModel({'version': VERSION, 'rules': []})
    value = model.assess({}, batch('天气').candidates[0])
    assert value.value_status == VALUE_UNMEASURED
    assert value.net_value is None


def test_context_condition_changes_candidate_value():
    model = CandidateValueModel({
        'version': VERSION,
        'rules': [{
            'rule_id': 'qualification', 'action': 'search_web',
            'query_contains': '出线形势',
            'when': {'match_context': {'qualification_uncertainty': True}},
            'expected_value': 0.60, 'cost': 0.05,
        }],
    })
    candidate = batch('出线形势').candidates[0]
    active = model.assess({'match_context': {'qualification_uncertainty': True}}, candidate)
    inactive = model.assess({'match_context': {'qualification_uncertainty': False}}, candidate)
    assert active.value_status == MODELED_DECLARED_ASSUMPTIONS
    assert inactive.value_status == VALUE_UNMEASURED


def test_context_descriptor_declares_exact_recursive_matching():
    model = context_model({'match_identity': {'status': 'unresolved'}})
    assert model.configuration()['context_match_mode'] == 'exact_recursive'


def context_model(when):
    return CandidateValueModel({
        'version': VERSION,
        'rules': [{'rule_id': 'context', 'action': 'search_web',
                   'when': when, 'expected_value': 0.40, 'cost': 0.05}],
    })


@pytest.mark.parametrize('expected,state,matched', [
    ({'flag': None}, {}, False),
    ({'flag': None}, {'flag': None}, True),
    ({'flag': True}, {'flag': 1}, False),
    ({'flag': False}, {'flag': 0}, False),
    ({'flag': 1}, {'flag': True}, False),
    ({'flag': 1}, {'flag': 1.0}, True),
    ({'nested': {}}, {}, False),
    ({'nested': {}}, {'nested': None}, False),
    ({'nested': {}}, {'nested': {'extra': 'present'}}, True),
    ({'flags': [True]}, {'flags': [1]}, False),
    ({'flags': [True]}, {'flags': [True]}, True),
    ({'flags': [True]}, {'flags': [True, False]}, False),
    ({'nested': {'flag': None}}, {'nested': {}}, False),
    ({'nested': {'flag': True}}, {'nested': {'flag': True, 'other': 3}}, True),
])
def test_context_matches_only_present_typed_values(expected, state, matched):
    value = context_model(expected).assess(state, batch('context query').candidates[0])
    assert (value.value_status == MODELED_DECLARED_ASSUMPTIONS) is matched
    if not matched:
        assert value.net_value is None


@pytest.mark.parametrize('condition', [
    None, [], {'flag': float('nan')}, {'flag': float('inf')},
    {1: 'not a JSON object key'}, {'flag': (True,)}, {'flag': {1, 2}},
])
def test_invalid_context_condition_rejected_before_selection(condition):
    with pytest.raises(ValueError, match='VALUE_MODEL_WHEN'):
        context_model(condition)


def test_context_condition_depth_is_bounded():
    condition = {}
    for _ in range(18):
        condition = {'nested': condition}
    with pytest.raises(ValueError, match='VALUE_MODEL_WHEN_DEPTH'):
        context_model(condition)


def test_context_condition_is_copied_and_part_of_model_identity():
    condition = {'match_identity': {'status': 'unresolved'}}
    model = context_model(condition)
    original = model.configuration()
    condition['match_identity']['status'] = 'identified'
    assert model.configuration() == original
    assert context_model(condition).configuration()['model_hash'] != original['model_hash']
    value = model.assess({'match_identity': {'status': 'unresolved'}}, batch('query').candidates[0])
    assert value.value_status == MODELED_DECLARED_ASSUMPTIONS


def test_unconditional_model_preserves_round123_identity():
    rule = {'rule_id': 'lineup', 'action': 'search_web', 'query_contains': '伤停',
            'expected_value': 0.40, 'cost': 0.05}
    model = CandidateValueModel({'version': VERSION, 'rules': [rule]})
    assert model.configuration()['model_hash'] == '86b1943e14b59f0fd3549f841269c209b3e26d3adad13474fe387f86f941fe91'
    assert 'context_match_mode' not in model.configuration()
    empty = CandidateValueModel({'version': VERSION, 'rules': [{**rule, 'when': {}}]})
    assert empty.configuration() == model.configuration()


def test_invalid_model_is_rejected():
    with pytest.raises(ValueError, match='MODELED_RULE_NEEDS_VALUE_AND_COST'):
        CandidateValueModel({
            'version': VERSION,
            'rules': [{'rule_id': 'bad', 'action': 'search_web', 'expected_value': 0.2}],
        })


def test_policy_identity_binds_registered_model_descriptor():
    model = CandidateValueModel({'version': VERSION, 'rules': []})
    policy = ValueAwareSelectionPolicy(model.assess, model_descriptor=model.configuration())
    assert policy.configuration['method'] == 'LOCAL_VALUE_AWARE_SELECTION'
    assert policy.configuration['model_descriptor']['model_hash'] == model.configuration()['model_hash']


def test_cli_create_persists_value_aware_selection_policy(tmp_path, capsys):
    model_path = tmp_path / 'value-model.json'
    model_path.write_text(json.dumps({'version': VERSION, 'rules': []}), encoding='utf-8')
    main(['--root', str(tmp_path), '--selection-policy', 'value-aware',
          '--value-model', str(model_path), 'create', '--match', '甲队 vs 乙队'])
    state = json.loads(capsys.readouterr().out)
    assert state['selection_method'] == 'LOCAL_VALUE_AWARE_SELECTION'
    assert state['config']['selection_policy']['configuration']['method'] == 'LOCAL_VALUE_AWARE_SELECTION'
    assert state['config']['selection_policy']['configuration']['model_descriptor']['rule_count'] == 0
