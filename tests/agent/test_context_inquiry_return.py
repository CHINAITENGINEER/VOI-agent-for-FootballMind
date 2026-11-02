'''Integration checks for evidence-state feedback, not football truth or utility.'''
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from football_agent.audit import atomic_write
from football_agent.context_inquiry_return import project_handoff_context
from football_agent.context_query_readiness import ContextState, filter_context_routes
from football_agent.decision_risk_experiment import proposal, H1, TEXT_K
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate

HELPER = runpy.run_path(str(Path(__file__).with_name('test_context_inquiry_handoff.py')))
NOW, CUTOFF = HELPER['NOW'], HELPER['CUTOFF']


def scenario(tmp_path, *, answer='yes', relation=None, scope_key=None, provider_failure=False):
    bridge, tools, _, clock, freeze = HELPER['setup'](tmp_path)
    parent = HELPER['parent'](initial={'competition': 'fixture', 'qualification': None})
    calls = []
    class Provider:
        def configuration(self):
            return {'provider': 'constructed_handoff_provider', 'configured': True}
        def decide(self, public):
            calls.append(deepcopy(public))
            if provider_failure:
                raise TimeoutError('Constructed provider timeout; no external call')
            if not public['documents']:
                response = proposal(['verify_k'])
            else:
                response = proposal([], kind='stop')
                doc = public['documents'][0]
                if doc['status'] == 'OK':
                    response['claim_updates'] = [{
                        'claim_id': 'c-k', 'question_id': 'q1', 'text': doc['text'], 'status': 'source_claim',
                        'citations': [{'document_id': doc['document_id'], 'quote': doc['text']}],
                        'evidence': {'kind': 'official_lineup', 'subject': H1['subject'],
                                     'scope': 'match', 'scope_key': scope_key or H1['scope_key']},
                        'hypothesis_links': [{'hypothesis_id': 'h1',
                                              'relation': relation or ('supports' if answer == 'yes' else 'challenges')}],
                    }]
                    if answer == 'mixed':
                        template = response['claim_updates'][0]
                        response['claim_updates'] = []
                        for token, direction in [('yes', 'supports'), ('no', 'challenges')]:
                            item = deepcopy(template)
                            item.update(claim_id='c-' + token, text=TEXT_K[token])
                            item['citations'][0]['quote'] = TEXT_K[token]
                            item['hypothesis_links'][0]['relation'] = direction
                            response['claim_updates'].append(item)
            clock[0] += timedelta(seconds=1)
            return {'response': response, 'receipt': {'real_provider_call': False}}
    bridge.workflow.provider = Provider()
    # Keep the existing transport/archive path; only select its explicit test return.
    from football_agent.decision_risk_experiment import ConstructedTools
    tools = ConstructedTools(tmp_path, {'verify_k': answer}, clock=lambda: clock[0])
    if answer == 'mixed':
        class MixedSession:
            def get(self, url, **kwargs):
                tools.calls.append(url)
                return SimpleNamespace(content=('<main>' + ' '.join(TEXT_K.values()) + '</main>').encode(),
                    status_code=200, headers={'content-type': 'text/html'}, close=lambda: None)
            def close(self):
                pass
        tools.session_factory = MixedSession
    bridge.workflow.tools = tools
    result = bridge.run(result=parent, context_available_at=HELPER['AVAILABLE'], freeze_manifest=freeze)
    reference = {'root': str(tmp_path), 'run_id': result['run_id'], 'head_hash': result['run_head_hash'],
        'cluster_id': 'constructed-parent', 'origin': 'constructed', 'cluster_basis': 'one constructed parent'}
    args = {'handoff_root': bridge.root, 'handoff_id': result['handoff_id'], 'parent': parent,
            'reference': reference, 'as_of': clock[0].isoformat(), 'freeze_manifest': freeze}
    return args, result, tools, calls


def first_state(output):
    return next(s for s in output['evidence_states'] if s['hypothesis']['hypothesis_id'] == 'h1')


@pytest.mark.parametrize('answer,expected', [('yes', 'source_support'), ('no', 'source_challenge')])
def test_source_direction_reaches_planning_without_promoting_truth(tmp_path, answer, expected):
    args, result, tools, calls = scenario(tmp_path, answer=answer)
    before = (len(tools.calls), len(calls))
    output = project_handoff_context(**args)
    state = first_state(output)
    assert state['requirements']['source_answer_status'] == expected
    assert output['next_context'][state['context_key_prefix'] + 'source_answer'] == expected
    assert output['next_context']['qualification'] is None
    assert not output['domain_truth_updated'] and not output['prediction_updated']
    assert output['semantic_verification'] == 'unverified'
    assert output['expected_information_value'] is output['realized_decision_utility'] is None
    assert (len(tools.calls), len(calls)) == before and output['new_network_requests'] == 0
    assert all(key.startswith('inquiry.') for key in output['changed_keys'])
    assert output == project_handoff_context(**args)


def test_other_match_claim_does_not_fill_this_match_question(tmp_path):
    args, *_ = scenario(tmp_path, scope_key='another-fixture')
    output = project_handoff_context(**args)
    state = first_state(output)
    assert state['requirements']['source_answer_status'] == 'inputs_incomplete'
    assert state['requirements']['missing_types'] == ['official_lineup']


def test_context_claim_fills_input_without_deciding_proposition(tmp_path):
    args, *_ = scenario(tmp_path, relation='context')
    state = first_state(project_handoff_context(**args))
    assert state['requirements']['missing_types'] == []
    assert state['requirements']['source_answer_status'] == 'inference_required'


def test_opposed_claims_remain_present_and_request_review(tmp_path):
    args, *_ = scenario(tmp_path, answer='mixed')
    output = project_handoff_context(**args)
    state = first_state(output)
    assert state['requirements']['missing_types'] == []
    assert state['requirements']['source_answer_status'] == 'opposed_source_claims'
    prefix = state['context_key_prefix']
    assert output['next_context'][prefix + 'query_focus'] == 'review_opposed_sources'
    assert output['next_context'][prefix + 'source_direction'] == 'mixed'
    assert state['source_truth_label'] is None


def test_failed_source_is_missing_not_negative_or_zero_value(tmp_path):
    args, *_ = scenario(tmp_path, answer='failure')
    output = project_handoff_context(**args)
    state = first_state(output)
    assert state['requirements']['source_answer_status'] == 'inputs_incomplete'
    assert state['requirements']['proposition_direction'] == 'undetermined'
    assert output['query_rows'][0]['observed_return']['realized_decision_utility'] is None


def test_provider_failure_returns_failure_state_without_fabricated_acquisition(tmp_path):
    args, _, tools, _ = scenario(tmp_path, provider_failure=True)
    output = project_handoff_context(**args)
    assert output['next_context']['inquiry.state'] == 'DEFERRED'
    assert output['next_context']['inquiry.http_attempts'] == 0
    assert output['evidence_states'] == output['query_rows'] == output['documents'] == []
    assert not tools.calls and output['expected_information_value'] is None


def test_projection_before_review_does_not_leak_later_source_direction(tmp_path):
    args, result, *_ = scenario(tmp_path, answer='no')
    acquired = next(e for e in result['run']['events'] if e['type'] == 'inquiry_acquisition_returned')
    output = project_handoff_context(**{**args, 'as_of': acquired['at']})
    assert output['next_context']['inquiry.claim_count'] == 0
    assert first_state(output)['requirements']['source_answer_status'] == 'inputs_incomplete'
    assert output['documents'] and not first_state(output)['claims']
    final = project_handoff_context(**args)
    assert first_state(final)['requirements']['source_answer_status'] == 'source_challenge'
    assert output['prefix_event_sequence'] < final['prefix_event_sequence']


@pytest.mark.parametrize('as_of', ['2026-09-29T06:00:00Z', '2026-09-29T13:00:00Z'])
def test_projection_respects_original_context_time_and_cutoff(tmp_path, as_of):
    args, *_ = scenario(tmp_path)
    with pytest.raises(ValueError, match='Projection time'):
        project_handoff_context(**{**args, 'as_of': as_of})


def test_different_parent_is_rejected(tmp_path):
    args, *_ = scenario(tmp_path)
    with pytest.raises(ValueError, match='supplied parent'):
        project_handoff_context(**{**args, 'parent': replace(args['parent'], target_key='other')})


@pytest.mark.parametrize('field,value', [('run_id', 'different-child'), ('root', '/different-store'),
                                        ('head_hash', '0' * 64), ('origin', 'operational_network')])
def test_inquiry_reference_is_bound_and_constructed_origin_preserved(tmp_path, field, value):
    args, *_ = scenario(tmp_path)
    with pytest.raises(ValueError):
        project_handoff_context(**{**args, 'reference': {**args['reference'], field: value}})


def test_receipt_mutation_is_detected(tmp_path):
    args, *_ = scenario(tmp_path)
    path = args['handoff_root'] / (args['handoff_id'] + '.json')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    receipt['binding']['parent_episode_id'] = 'changed'
    atomic_write(path, receipt)
    with pytest.raises(ValueError, match='hash mismatch'):
        project_handoff_context(**args)


def test_original_frozen_configuration_is_required(tmp_path):
    args, *_ = scenario(tmp_path)
    freeze = deepcopy(args['freeze_manifest'])
    freeze['parameters']['context_inquiry_handoff']['max_requests'] += 1
    with pytest.raises(ValueError, match='frozen handoff'):
        project_handoff_context(**{**args, 'freeze_manifest': freeze})


def test_observation_can_enable_followup_planning_without_resolving_qualification(tmp_path):
    args, *_ = scenario(tmp_path, answer='no')
    output = project_handoff_context(**args)
    key = first_state(output)['context_key_prefix'] + 'source_answer'
    cutoff = CUTOFF
    board = StateQueryBoard(target_key='match-one', prediction_cutoff=cutoff,
        hypotheses=[StateHypothesis('h-followup', 'match-one', 'confirmation', ('known', 'unknown'))],
        candidates=[StateQueryCandidate(x, 'match-one', cutoff, 'confirmation', 'public', 'read', .01)
                    for x in ('review-source', 'infer-qualified')])
    routes = [{'candidate_id': 'review-source', 'route_kind': 'domain', 'required_context_keys': [key]},
              {'candidate_id': 'infer-qualified', 'route_kind': 'domain', 'required_context_keys': ['qualification']}]
    before = filter_context_routes(board, ContextState(output['previous_context']), routes)
    after = filter_context_routes(board, ContextState(output['next_context']), routes)
    assert not any(r['eligible'] for r in before)
    assert [r['candidate_id'] for r in after if r['eligible']] == ['review-source']
    assert all(r['expected_value'] is None for r in after)
