from football_agent.acquisition_boundary import CandidateBatch
from football_agent.query_value_contract import QueryValueAssessment
from football_agent.value_selection_policy import VERSION, ValueAwareSelectionPolicy


def public(remaining):
    return {'task_id': 'task-1', 'remaining': remaining}


def response(query):
    return {'task_id': 'task-1', 'assessment': {}, 'previous_return': 'none',
            'analysis_plan': {}, 'match_identity': {}, 'return_assessment': None,
            'action': 'search_web', 'url': 'https://example.com/' + query,
            'query': query, 'engine': 'bing', 'question_to_user': None,
            'reason': 'need evidence'}


def batch(state, *queries):
    return CandidateBatch.build(state, [response(query) for query in queries], origin='test')


def test_value_policy_selects_highest_modeled_net_value():
    state = public({'http_calls': 2})
    candidates = batch(state, 'injury', 'weather')
    values = {'injury': 0.40, 'weather': 0.05}

    def assess(_, candidate):
        return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=values[candidate.response['query']], cost=0.01)

    policy = ValueAwareSelectionPolicy(assess)
    choice = policy.select(candidates.context, candidates)
    assert policy.version == VERSION
    assert choice.candidate_id == next(c.candidate_id for c in candidates.candidates if c.response['query'] == 'injury')
    assert choice.terminal_action is None


def test_value_policy_defers_when_open_world_value_is_unmeasured():
    state = public({'http_calls': 2})
    candidates = batch(state, 'injury')

    def assess(_, candidate):
        return QueryValueAssessment.unmeasured(candidate.candidate_id, cost=0.01)

    choice = ValueAwareSelectionPolicy(assess).select(candidates.context, candidates)
    assert choice.terminal_action == 'defer'
    assert 'open_world_value_withheld' in choice.diagnostics_json


def test_value_policy_stops_only_when_all_feasible_values_are_modeled_nonpositive():
    state = public({'http_calls': 2})
    candidates = batch(state, 'weather')

    def assess(_, candidate):
        return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=0.01, cost=0.01)

    choice = ValueAwareSelectionPolicy(assess).select(candidates.context, candidates)
    assert choice.terminal_action == 'stop'
    assert 'declared_finite_model' in choice.diagnostics_json


def test_value_policy_defers_when_resource_budget_is_exhausted():
    state = public({'http_calls': 0})
    candidates = batch(state, 'injury')

    def assess(_, candidate):
        return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=0.40, cost=0.01)

    choice = ValueAwareSelectionPolicy(assess).select(candidates.context, candidates)
    assert choice.terminal_action == 'defer'
    assert 'BUDGET_EXHAUSTED' in choice.diagnostics_json
