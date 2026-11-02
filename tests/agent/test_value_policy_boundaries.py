"""Stopping and resource constraints are not measured information utility."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from football_agent.acquisition_boundary import CandidateBatch, policy_identity
from football_agent.acquisition_replay import verify_run
from football_agent.audit import stored_digest
from football_agent.experience import ExperienceService
from football_agent.query_value_contract import QueryValueAssessment
from football_agent.value_selection_policy import ValueAwareSelectionPolicy


def make_batch(*actions, remaining=None, final=False):
    state = {'task_id': 'constructed', 'remaining': remaining if remaining is not None else {'http_calls': 2},
             'final_decision_only': final}
    responses = []
    for index, action in enumerate(actions):
        responses.append({'task_id': state['task_id'], 'assessment': {}, 'analysis_plan': [],
            'previous_return': 'none', 'match_identity': {}, 'return_assessment': None,
            'action': action, 'url': None, 'query': 'candidate-' + str(index) if action == 'search_web' else None,
            'engine': 'bing' if action == 'search_web' else None,
            'question_to_user': 'Which date?' if action == 'ask_user' else None, 'reason': 'fixture'})
    return CandidateBatch.build(state, responses, origin='constructed_scope_test', recommended_index=0)


def modeled(_, candidate):
    return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=0.4, cost=0.05)


def choose(batch, assessor=modeled, **options):
    return ValueAwareSelectionPolicy(assessor, **options).select(batch.context, batch)


@pytest.mark.parametrize('control', ['stop', 'defer'])
def test_terminal_recommendation_does_not_compete_for_information_value(control):
    batch = make_batch(control, 'search_web')
    seen = []

    def assess(state, candidate):
        seen.append(candidate.candidate_id)
        assert candidate.response['action'] == 'search_web'
        return modeled(state, candidate)

    choice = choose(batch, assess)
    assert choice.candidate_id == batch.candidates[1].candidate_id
    assert seen == [batch.candidates[1].candidate_id]
    diagnostics = choice.record()['diagnostics']
    assert diagnostics['ignored_terminal_candidate_ids'] == [batch.candidates[0].candidate_id]
    assert diagnostics['candidates'][0]['value'] is None
    assert diagnostics['candidates'][0]['evaluation_status'] == 'NOT_APPLICABLE'


def test_terminal_only_is_candidate_gap_not_low_information_value():
    batch = make_batch('stop', 'defer')
    choice = choose(batch, lambda *_: pytest.fail('Control action must not call the estimator'))
    diagnostics = choice.record()['diagnostics']
    assert choice.terminal_action == 'defer'
    assert diagnostics['stop_kind'] == 'NO_FEASIBLE_REGISTERED_ACTION'
    assert diagnostics['stop_scope'] == 'proposed_candidate_set'
    assert diagnostics['evaluated_candidate_ids'] == []
    assert not diagnostics['candidate_generation_complete'] and not diagnostics['global_optimum_claim']


def test_terminal_proposal_does_not_add_unmeasured_information_or_prevent_scoped_stop():
    batch = make_batch('search_web', 'stop')

    def assess(_, candidate):
        if candidate.response['action'] == 'stop':
            return QueryValueAssessment.unmeasured(candidate.candidate_id)
        return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=0.05, cost=0.05)

    choice = choose(batch, assess)
    diagnostics = choice.record()['diagnostics']
    assert choice.terminal_action == 'stop'
    assert diagnostics['stop_kind'] == 'MODEL_SCOPED_STOP'
    assert diagnostics['maximum_modeled_net_value'] == 0
    assert diagnostics['evaluated_candidate_ids'] == [batch.candidates[0].candidate_id]


def test_terminal_value_does_not_override_unmeasured_query():
    batch = make_batch('stop', 'search_web')

    def assess(_, candidate):
        if candidate.response['action'] == 'stop':
            return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=100, cost=0)
        return QueryValueAssessment.unmeasured(candidate.candidate_id)

    choice = choose(batch, assess)
    assert choice.terminal_action == 'defer'
    assert choice.record()['diagnostics']['stop_kind'] == 'WITHHELD_OPEN_WORLD_VALUE_UNKNOWN'


@pytest.mark.parametrize('remaining,final,exclusion', [
    ({'http_calls': 0}, False, 'http_calls_exhausted'),
    ({'http_calls': 2, 'llm_calls': 1}, True, 'final_decision_only'),
])
def test_blocked_acquisition_plus_terminal_is_operational_defer(remaining, final, exclusion):
    batch = make_batch('search_web', 'stop', remaining=remaining, final=final)
    choice = choose(batch)
    diagnostics = choice.record()['diagnostics']
    assert choice.terminal_action == 'defer'
    assert diagnostics['stop_kind'] == 'BUDGET_EXHAUSTED'
    assert diagnostics['stop_scope'] == 'operational_budget'
    assert exclusion in diagnostics['candidates'][0]['exclusion_reasons']


def test_local_read_still_competes_when_http_is_exhausted():
    batch = make_batch('search_web', 'query_acquired', 'stop',
                       remaining={'http_calls': 0, 'local_reads': 1})
    choice = choose(batch)
    assert choice.candidate_id == batch.candidates[1].candidate_id


def test_known_positive_choice_does_not_claim_to_outrank_unmeasured_candidates():
    batch = make_batch('search_web', 'read_url')

    def assess(state, candidate):
        if candidate.response['action'] == 'read_url':
            return QueryValueAssessment.unmeasured(candidate.candidate_id)
        return modeled(state, candidate)

    diagnostics = choose(batch, assess).record()['diagnostics']
    assert diagnostics['selection_scope'] == 'modeled_feasible_candidates'
    assert diagnostics['unmeasured_candidate_ids'] == [batch.candidates[1].candidate_id]
    assert not diagnostics['global_optimum_claim']


def test_explicitly_valued_identity_clarification_is_not_a_stop_proposal():
    # Full match/clarification admissibility is validated by the runtime first.
    batch = make_batch('ask_user', 'stop', final=True)
    choice = choose(batch)
    assert choice.candidate_id == batch.candidates[0].candidate_id
    assert choice.record()['diagnostics']['candidates'][0]['candidate_role'] == 'identity_clarification'


def test_assessor_must_preserve_the_candidate_binding():
    batch = make_batch('search_web')
    with pytest.raises(ValueError, match='VALUE_ASSESSMENT_CANDIDATE_MISMATCH'):
        choose(batch, lambda *_: QueryValueAssessment.modeled('another-candidate', expected_value=1, cost=0))


def test_unsupported_actions_are_not_implicitly_free_information():
    with pytest.raises(ValueError, match='VALUE_POLICY_UNSUPPORTED_ACTION'):
        choose(make_batch('unknown_tool'))


@pytest.mark.parametrize('cap', [float('nan'), float('inf'), -float('inf'), -1, True, '1'])
def test_cost_cap_must_be_finite_nonnegative_number(cap):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        ValueAwareSelectionPolicy(modeled, value_budget=cap)


def test_declared_cost_cap_is_recorded_and_changes_policy_identity():
    low = ValueAwareSelectionPolicy(modeled, value_budget=0.01)
    high = ValueAwareSelectionPolicy(modeled, value_budget=0.2)
    assert policy_identity(low) != policy_identity(high)
    assert high.configuration['value_budget_scope'] == 'PER_SELECTION_COST_CAP'
    assert high.configuration['value_budget'] == 0.2
    configuration = high.configuration
    configuration['value_budget'] = 0
    assert high.configuration['value_budget'] == 0.2
    batch = make_batch('search_web', 'stop')
    blocked = low.select(batch.context, batch).record()['diagnostics']
    assert blocked['stop_kind'] == 'BUDGET_EXHAUSTED'
    assert blocked['candidates'][0]['exclusion_reasons'] == ['declared_cost_exceeds_selection_cap']
    assert high.select(batch.context, batch).candidate_id == batch.candidates[0].candidate_id


def test_unknown_cost_under_a_cap_is_not_zero_or_a_supported_stop():
    batch = make_batch('search_web', 'stop')
    choice = choose(batch, lambda _, c: QueryValueAssessment.unmeasured(c.candidate_id), value_budget=0)
    diagnostics = choice.record()['diagnostics']
    assert choice.terminal_action == 'defer'
    assert diagnostics['stop_kind'] == 'WITHHELD_OPEN_WORLD_VALUE_UNKNOWN'
    assert diagnostics['candidates'][0]['value']['cost'] is None


def runtime_helpers():
    return runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))


def test_last_planning_call_defers_cleanly_and_replays_instead_of_invalid_acquisition(tmp_path):
    helpers = runtime_helpers()
    seen = []

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            request = helpers['match_decision'](public)
            request['query'] = 'constructed query ' + str(len(seen))
            stop = helpers['match_decision'](public, 'stop')
            return {'candidates': [stop, request], 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(tmp_path, proposer_factory=lambda f, e: Proposer(),
        selection_policy=ValueAwareSelectionPolicy(modeled),
        tools_factory=lambda f: helpers['MatchTools'](), mode='test_fixture')
    try:
        run = service.create({'match': '构造甲队 vs 构造乙队',
                              'limits': {'http_calls': 2, 'llm_calls': 2, 'seconds': 600}})['run_id']
        service._drive(run, True)
        state = service.get(run)
        assert state['status'] == 'deferred' and state['last_error'] is None
        assert state['spent'] == {'llm_calls': 2, 'http_calls': 1}
        assert [turn['action'] for turn in state['turns']] == ['search_web', 'defer']
        assert [public['final_decision_only'] for public in seen] == [False, True]
        assert state['question_to_user'] is None and state['user_context'] == []
        op = state['selection_trace'][-1]['operation_id']
        choice = json.loads((service.folder(run) / 'operations' / op / 'policy-output.json').read_text(encoding='utf-8'))
        assert choice['diagnostics']['stop_kind'] == 'BUDGET_EXHAUSTED'
        replay = verify_run(service.folder(run))
        assert replay['accepted_decisions_verified'] == 2 and replay['linked_http_receipts'] == 1
        assert replay['real_calls_made_by_replay'] == 0
    finally:
        service.close()


@pytest.mark.parametrize('change', ['cost_cap', 'legacy_v1'])
def test_resume_rejects_changed_selection_semantics_before_provider_call(tmp_path, change):
    def no_provider(*_):
        pytest.fail('Mismatched saved policy must be detected before model invocation')

    service = ExperienceService(tmp_path, proposer_factory=no_provider,
        selection_policy=ValueAwareSelectionPolicy(modeled, value_budget=0.1), mode='test_fixture')
    try:
        run = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        if change == 'cost_cap':
            service.selection_policy = ValueAwareSelectionPolicy(modeled, value_budget=0.2)
        else:
            legacy = {'version': 'value-aware-selection-v1', 'configuration': {
                'method': 'LOCAL_VALUE_AWARE_SELECTION', 'numeric_voi_calibrated': False,
                'unmeasured_policy': 'DEFER_OPEN_WORLD_VALUE_UNKNOWN'}}
            legacy['fingerprint'] = stored_digest(legacy)
            service.mutate(run, lambda s: s['config'].update(selection_policy=legacy))
        service.step(run)
        state = service.get(run)
        assert state['status'] == 'paused' and 'POLICY_OR_PROPOSER_CHANGED' in state['last_error']
        assert state['spent'] == {'llm_calls': 0, 'http_calls': 0} and not state['operations']
    finally:
        service.close()


def test_bad_estimator_binding_pauses_before_a_request_is_queued(tmp_path):
    helpers = runtime_helpers()

    class Proposer:
        def propose(self, public):
            return {'candidates': [helpers['match_decision'](public)], 'test_double': True}

    service = ExperienceService(tmp_path, proposer_factory=lambda f, e: Proposer(),
        selection_policy=ValueAwareSelectionPolicy(lambda *_:
            QueryValueAssessment.modeled('wrong-candidate', expected_value=1, cost=0)), mode='test_fixture')
    try:
        run = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        service.step(run)
        state = service.get(run)
        assert state['status'] == 'paused'
        assert 'VALUE_ASSESSMENT_CANDIDATE_MISMATCH' in state['last_error']
        assert state['spent'] == {'llm_calls': 1, 'http_calls': 0}
        assert state['queued_action'] is None and not state['turns']
        assert state['last_provider_outcome']['stage'] == 'policy_selection'
    finally:
        service.close()
