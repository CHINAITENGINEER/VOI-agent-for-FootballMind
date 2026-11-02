"""Current-return selection tests use constructed evidence and declared values."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from football_agent import match_return_assessment as review
from football_agent.acquisition_boundary import CandidateBatch
from football_agent.acquisition_replay import verify_run
from football_agent.experience import ExperienceService
from football_agent.query_value_contract import MODELED_DECLARED_ASSUMPTIONS, VALUE_UNMEASURED
from football_agent.value_model_adapter import VERSION, CandidateValueModel
from football_agent.value_selection_policy import ValueAwareSelectionPolicy


def evidence(outcome='partial'):
    question = {'id': 'lineup', 'question': 'Does this establish the starting lineup?',
                'decision_relevance': 'Assess rotation uncertainty', 'status': 'open'}
    selected = {'analysis_plan': [question], 'active_question_id': 'lineup',
                'unknown_to_resolve': 'Starting lineup', 'useful_if': 'Confirmed match-specific starters',
                'action': 'read_url'}
    document = {'document_id': 'body', 'acquisition_id': 'acquisition',
                'selection_operation_id': 'prior-selection', 'status': 'readable',
                'content_kind': 'body_text', 'excerpt': 'Squad list only; starters unconfirmed.',
                'return_observation': {'evidence_surface': 'body_text'}}
    context = review.bind_return(selected, {'operation_id': 'prior-selection',
        'decision_digest': 'fixture-decision'}, 'acquisition', {'document': document, 'requests': []})
    public = {'task_id': 'fixture', 'remaining': {'http_calls': 2}, 'documents': [document],
              review.CAPABILITY_FIELD: review.capability(), review.PENDING_FIELD: context,
              review.HISTORY_FIELD: [{'assessment': {'outcome': 'discovery_only'}}],
              'current_assessment': {'status': 'unresolved'}}
    result = {'task_id': 'fixture', 'assessment': {}, 'previous_return': 'partial',
              'analysis_plan': [question], 'match_identity': {}, 'action': 'search_web',
              'url': None, 'query': 'confirm lineup', 'engine': 'bing', 'question_to_user': None,
              'reason': 'constructed inquiry', review.FIELD: {
                  'acquisition_id': 'acquisition', 'question_id': 'lineup',
                  'context_digest': context['context_digest'], 'outcome': outcome,
                  'reason': 'Constructed provisional interpretation of the visible body',
                  'citations': [{'document_id': 'body', 'quote': 'Squad list only'}],
                  'remaining_unknowns': [] if outcome == 'resolved' else ['Confirmed starters']}}
    return public, result


def model_for(pattern):
    return CandidateValueModel({'version': VERSION, 'rules': [{
        'rule_id': 'return-rule', 'action': 'search_web', 'when_return': pattern,
        'expected_value': 0.4, 'cost': 0.05}]})


def assess(model, public, response):
    batch = CandidateBatch.build(public, [response], origin='constructed_return_test')
    return model.assess(batch.context.state, batch.candidates[0])


def test_current_review_is_available_before_history_commit_and_does_not_mutate_inputs():
    public, response = evidence()
    original_public, original_response = deepcopy(public), deepcopy(response)
    value = assess(model_for({'assessment': {'outcome': 'partial'}}), public, response)
    assert value.value_status == MODELED_DECLARED_ASSUMPTIONS
    assert public[review.HISTORY_FIELD][-1]['assessment']['outcome'] == 'discovery_only'
    assert public == original_public and response == original_response
    view = review.current_return_evidence(public, response)
    assert view['context']['question']['id'] == 'lineup'
    assert view['label_status'] == 'ACTOR_PROVISIONAL_NOT_GOLD'
    assert not view['world_truth_verified'] and not view['independent_semantic_validation']
    assert view['realized_decision_gain'] is None and not view['numeric_voi_training_eligible']
    view['assessment']['outcome'] = 'resolved'
    view['context']['question']['id'] = 'modified-view-only'
    assert public == original_public and response == original_response


@pytest.mark.parametrize('outcome,expected', [('partial', 'confirm lineup'), ('irrelevant', 'replace source')])
def test_latest_bound_review_changes_choice_in_the_same_planning_call(outcome, expected):
    public, first = evidence(outcome)
    second = deepcopy(first)
    second['query'] = 'replace source'
    model = CandidateValueModel({'version': VERSION, 'rules': [
        {'rule_id': 'partial', 'action': 'search_web', 'query_contains': 'confirm',
         'when_return': {'assessment': {'outcome': 'partial'}, 'context': {'question': {'id': 'lineup'}}},
         'expected_value': 0.4, 'cost': 0.05},
        {'rule_id': 'irrelevant', 'action': 'search_web', 'query_contains': 'replace',
         'when_return': {'assessment': {'outcome': 'irrelevant'}},
         'expected_value': 0.4, 'cost': 0.05},
    ]})
    batch = CandidateBatch.build(public, [first, second], recommended_index=0, origin='constructed_return_test')
    choice = ValueAwareSelectionPolicy(model.assess, model_descriptor=model.configuration()).select(batch.context, batch)
    selected = next(c.response['query'] for c in batch.candidates if c.candidate_id == choice.candidate_id)
    assert selected == expected
    assert public[review.HISTORY_FIELD][-1]['assessment']['outcome'] == 'discovery_only'


@pytest.mark.parametrize('kind', ['legacy', 'consumed_return', 'first_turn'])
def test_missing_current_review_does_not_reuse_history_or_match_empty_condition(kind):
    public, response = evidence()
    if kind == 'legacy':
        public.pop(review.CAPABILITY_FIELD)
    else:
        public[review.PENDING_FIELD] = None
        response[review.FIELD] = None
        if kind == 'first_turn':
            public['documents'] = []
            public[review.HISTORY_FIELD] = []
    value = assess(model_for({}), public, response)
    assert value.value_status == VALUE_UNMEASURED and value.net_value is None


@pytest.mark.parametrize('field,value,error', [
    ('acquisition_id', 'other-acquisition', 'CONTEXT_MISMATCH'),
    ('question_id', 'other-question', 'CONTEXT_MISMATCH'),
    ('context_digest', 'different-digest', 'CONTEXT_MISMATCH'),
    ('citations', [{'document_id': 'body', 'quote': 'hidden information'}], 'QUOTE_NOT_VISIBLE'),
    ('citations', [], 'REQUIRES_BODY'),
])
def test_unbound_or_uncited_current_review_is_rejected_before_valuation(field, value, error):
    public, response = evidence()
    response[review.FIELD][field] = value
    with pytest.raises(ValueError, match=error):
        assess(model_for({}), public, response)


@pytest.mark.parametrize('kind,error', [
    ('capability', 'CAPABILITY_CHANGED'), ('no_return', 'WITHOUT_ACTUAL_RETURN'),
    ('missing_field', 'CURRENT_RETURN_ASSESSMENT_MISSING'),
])
def test_current_review_contract_is_enforced(kind, error):
    public, response = evidence()
    if kind == 'capability':
        public[review.CAPABILITY_FIELD]['version'] = 'changed'
    elif kind == 'no_return':
        public[review.PENDING_FIELD] = None
    else:
        response.pop(review.FIELD)
    with pytest.raises(ValueError, match=error):
        assess(model_for({}), public, response)


def test_current_review_rejects_other_task_and_preserves_original_question():
    public, response = evidence()
    other = deepcopy(response)
    other['task_id'] = 'other-task'
    with pytest.raises(ValueError, match='TASK_MISMATCH'):
        review.current_return_evidence(public, other)
    response['analysis_plan'][0]['question'] = 'New question must not rewrite the old test'
    view = review.current_return_evidence(public, response)
    assert view['context']['question']['question'] == 'Does this establish the starting lineup?'
    value = assess(model_for({'world_truth_verified': True}), public, response)
    assert value.value_status == VALUE_UNMEASURED


@pytest.mark.parametrize('condition', [None, [], 'partial', {'flag': float('nan')}])
def test_invalid_current_return_condition_is_rejected(condition):
    with pytest.raises(ValueError, match='VALUE_MODEL_WHEN'):
        model_for(condition)


def test_current_return_descriptor_and_conditions_are_bound_to_model_identity():
    plain = CandidateValueModel({'version': VERSION, 'rules': []}).configuration()
    descriptor = model_for({'assessment': {'outcome': 'partial'}}).configuration()
    assert 'current_return_evidence_version' not in plain
    assert descriptor['current_return_evidence_version'] == review.SELECTION_EVIDENCE_VERSION
    assert descriptor['current_return_judgement'] == 'ACTOR_PROVISIONAL_NOT_GOLD'
    assert descriptor['model_hash'] != model_for({'assessment': {'outcome': 'irrelevant'}}).configuration()['model_hash']


@pytest.mark.parametrize('outcome,expected', [('partial', 'fixture confirm lineup'),
                                            ('irrelevant', 'fixture replace source')])
def test_runtime_uses_fresh_return_before_history_commit_without_extra_model_call(tmp_path, outcome, expected):
    helpers = runpy.run_path(str(Path(__file__).with_name('test_match_return_assessment.py')))
    match_decision, actual_review, tools = (helpers[k] for k in ('match_decision', 'actual_review', 'MatchTools'))
    model = CandidateValueModel({'version': VERSION, 'rules': [
        {'rule_id': 'initial', 'action': 'search_web', 'query_contains': 'fixture initial',
         'when': {'documents': []}, 'expected_value': 0.5, 'cost': 0.05},
        {'rule_id': 'read-body', 'action': 'read_url',
         'when_return': {'assessment': {'outcome': 'discovery_only'}},
         'expected_value': 0.4, 'cost': 0.05},
        {'rule_id': 'confirm', 'action': 'search_web', 'query_contains': 'fixture confirm',
         'when_return': {'assessment': {'outcome': 'partial'}},
         'expected_value': 0.3, 'cost': 0.05},
        {'rule_id': 'replace', 'action': 'search_web', 'query_contains': 'fixture replace',
         'when_return': {'assessment': {'outcome': 'irrelevant'}},
         'expected_value': 0.3, 'cost': 0.05},
    ]})
    seen = []

    class Proposer:
        def propose(self, public):
            seen.append(deepcopy(public))
            count = len(public['documents'])
            first = match_decision(public)
            first['query'] = 'fixture initial' if not count else 'fixture confirm lineup'
            second = match_decision(public, 'read_url') if count == 1 else deepcopy(first)
            if count != 1:
                second['query'] = 'fixture replace source'
            current = actual_review(public, outcome if count >= 2 else None)
            for candidate in (first, second):
                candidate[review.FIELD] = deepcopy(current)
            return {'candidates': [first, second], 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(tmp_path, proposer_factory=lambda f, e: Proposer(),
        selection_policy=ValueAwareSelectionPolicy(model.assess, model_descriptor=model.configuration()),
        tools_factory=lambda f: tools(), mode='test_fixture', enable_return_assessment=True)
    try:
        run = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        # Three planning calls, two completed acquisitions; the next selected
        # request stays queued, so no extra return is manufactured for this test.
        for _ in range(5):
            service.step(run)
        state = service.get(run)
        assert state['status'] == 'ready', state['last_error']
        assert state['queued_action']['query'] == expected
        assert state['spent'] == {'llm_calls': 3, 'http_calls': 2}
        assert len(seen) == 3 and not state['user_context'] and state['question_to_user'] is None
        assert seen[-1][review.HISTORY_FIELD][-1]['assessment']['outcome'] == 'discovery_only'
        assert [item['assessment']['outcome'] for item in state[review.HISTORY_FIELD]] == ['discovery_only', outcome]
        assert all(not item['numeric_voi_training_eligible'] for item in state[review.HISTORY_FIELD])
        audit = verify_run(service.folder(run))
        assert audit['accepted_decisions_verified'] == 3 and audit['linked_http_receipts'] == 2
        assert audit['question_return_assessments_verified'] == 2
        assert audit['real_calls_made_by_replay'] == 0 and state['numeric_voi'] is None
        op = state['selection_trace'][-1]['operation_id']
        record = json.loads((service.folder(run) / 'operations' / op / 'proposal.json').read_text(encoding='utf-8'))
        assert record['candidates'][0]['response'][review.FIELD]['outcome'] == outcome
    finally:
        service.close()
