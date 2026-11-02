'''Constructed mechanism checks; none of these are real football outcome labels.'''
from copy import deepcopy
from datetime import timedelta

import pytest

from football_agent.audit import digest
from football_agent.evidence_windows import EvidenceWindows
from football_agent.fixture_bound_review import task_contract
from football_agent.inquiry_tools import OpenWebTools
from football_agent.target_navigation import TargetNavigationWorkflow
from test_information_navigation import DOC, NOW, choice as base_choice


def choice(public, action='defer', **kwargs):
    result = base_choice(action, **kwargs)
    result.update(task_contract_digest=public['task_contract_digest'],
        target_state={'status': 'unresolved', 'candidate_label': None, 'support_claim_indices': [],
                      'open_requirements': ['fixture_identity', 'all_fixture_order']},
        information_test=None, return_assessment=None)
    if action in {'read_window', 'query_source'}:
        result['information_test'] = {'requirement': 'all_fixture_order',
            'useful_if': 'An applicable complete schedule identifies the next overall fixture.',
            'irrelevant_if': 'A next-home banner alone does not establish overall ordering.'}
    prior = public['last_action_result']
    if prior:
        result['return_assessment'] = {'document_id': prior['document_id'],
            'requirement': prior['information_test']['requirement'], 'outcome': 'unread',
            'reason': 'Metadata alone does not answer the question.', 'evidence_claim_indices': []}
    return result


def query(public):
    return choice(public, 'query_source', query={'tool': 'read_url', 'url': 'https://example.com/schedule', 'query': None})


def read_new(public):
    doc = public['documents'][-1]
    return choice(public, 'read_window', document_id=doc['document_id'], window_id=doc['catalog'][0]['window_id'])


def finish(public):
    prior = public['last_action_result']
    doc = next(d for d in public['documents'] if d['document_id'] == prior['document_id'])
    window = next(w for w in doc['visible_windows'] if w['window_id'] == prior['window_id'])
    result = choice(public, claims=[{'document_id': doc['document_id'], 'window_id': window['window_id'],
        'quote': window['text'][:35], 'interpretation': 'A constructed statement, with ordering still uncertain.'}])
    result['return_assessment'].update(outcome='partial', evidence_claim_indices=[0])
    return result


def make(tmp_path, replies, *, source_error=None, **limits):
    seen, requested = [], []
    class Provider:
        def configuration(self): return {'provider': 'offline-constructed'}
        def decide(self, public):
            seen.append(deepcopy(public))
            response = replies[len(seen) - 1]
            if isinstance(response, Exception): raise response
            return {'response': response(public), 'receipt': {'real_provider_call': False}}
    class Tools(OpenWebTools):
        def execute(self, prepared, *, run_id, request_id):
            requested.append(prepared)
            if source_error: raise source_error
            return {'request_id': request_id, 'http_attempts': 1, 'document': {**deepcopy(DOC),
                'document_id': 'new-doc', 'url': 'https://example.com/schedule', 'links': [],
                'text': 'Constructed next home schedule, not overall fixture ordering. ' * 30}}
    flow = TargetNavigationWorkflow(tmp_path, Provider(), Tools(None, ['example.com', 'www.bing.com']), clock=lambda: NOW)
    second = {**deepcopy(DOC), 'document_id': 'second', 'url': 'https://example.com/other',
              'text': 'Constructed second inherited document.'}
    docs = [deepcopy(DOC), second]
    handoff = {'version': 'target-navigation-handoff-v1', 'documents': docs,
        'document_digests': {d['document_id']: digest(d) for d in docs},
        'opened': {d['document_id']: [next(iter(EvidenceWindows(d).windows))] for d in docs},
        'claims': [], 'last_choice': base_choice(), 'parent_budget': {},
        'parent_root': str(tmp_path / 'constructed-parent'), 'parent_run_id': 'synthetic',
        'parent_head_hash': 'constructed', 'parent_completed_at': NOW.isoformat(),
        'scope': 'constructed actor history, no reviewer labels'}
    task = {'task_id': 'constructed-target', 'objective': 'Find current or next overall fixture.',
        'deadline': (NOW + timedelta(minutes=10)).isoformat(), 'seed_urls': [DOC['url']]}
    contract = task_contract(task_id=task['task_id'], question=task['objective'], subject_team='Team A',
        team_aliases=[], reference_time=NOW)
    run = flow.start(task, handoff, contract, **limits)
    return flow, run['run_id'], seen, requested, handoff, contract


def test_query_body_reconsider_keeps_task_and_tests_same_question(tmp_path):
    flow, rid, seen, requested, handoff, contract = make(tmp_path, [query, read_new, finish])
    for _ in range(3): run = flow.step(rid)
    assert run['reason'] == 'AGENT_DEFER' and run['pending'] is None
    assert len(requested) == 1 and run['budget']['window_reads'] == 1
    assert all(p['task_contract'] == contract for p in seen)
    assert [d['document_id'] for d in seen[0]['documents']] == [d['document_id'] for d in handoff['documents']]
    assert seen[1]['documents'][-1]['visible_windows'] == []
    assert seen[1]['last_action_result']['information_test'] == run['decisions'][0]['choice']['information_test']
    assert run['decisions'][-1]['choice']['return_assessment']['outcome'] == 'partial'
    assert flow.step(rid) == run and len(seen) == 3
    assert flow.store.verify(rid)['status'] == 'OK'


def test_can_read_archived_body_without_network(tmp_path):
    def read_old(public):
        doc = public['documents'][0]
        return choice(public, 'read_window', document_id=doc['document_id'], window_id=doc['catalog'][1]['window_id'])
    flow, rid, _, requested, _, _ = make(tmp_path, [read_old, finish])
    flow.step(rid)
    run = flow.step(rid)
    assert run['reason'] == 'AGENT_DEFER' and not requested
    assert run['budget']['window_reads'] == 1


@pytest.mark.parametrize('mutation', ['contract', 'no_probe', 'wrong_requirement', 'stop_unresolved', 'no_unknowns', 'fake_candidate'])
def test_invalid_prospective_choices_do_not_dispatch(tmp_path, mutation):
    def invalid(public):
        response = query(public)
        if mutation == 'contract': response['task_contract_digest'] = 'other-task'
        elif mutation == 'no_probe': response['information_test'] = None
        elif mutation == 'wrong_requirement': response['information_test']['requirement'] = 'match_status'
        elif mutation == 'stop_unresolved': response = choice(public, 'stop')
        elif mutation == 'no_unknowns': response['target_state']['open_requirements'] = []
        else: response['target_state'].update(status='candidate_only', candidate_label='Made-up fixture')
        return response
    flow, rid, _, requested, _, _ = make(tmp_path, [invalid])
    run = flow.step(rid)
    assert run['reason'] == 'INVALID_NAVIGATION_RESPONSE' and not requested


@pytest.mark.parametrize('mutation', ['missing', 'different_requirement', 'different_document', 'preview_resolved'])
def test_metadata_assessment_binds_prior_question_without_claiming_body(tmp_path, mutation):
    def invalid(public):
        result = read_new(public)
        if mutation == 'missing': result['return_assessment'] = None
        elif mutation == 'different_requirement': result['return_assessment']['requirement'] = 'fixture_identity'
        elif mutation == 'different_document': result['return_assessment']['document_id'] = DOC['document_id']
        else: result['return_assessment']['outcome'] = 'resolved'
        return result
    flow, rid, _, requested, _, _ = make(tmp_path, [query, invalid])
    flow.step(rid)
    run = flow.step(rid)
    assert run['reason'] == 'INVALID_NAVIGATION_RESPONSE' and len(requested) == 1
    assert run['budget']['window_reads'] == 0


@pytest.mark.parametrize('mutation', ['no_support', 'other_document', 'still_open'])
def test_body_assessment_needs_returned_evidence_and_consistent_gap(tmp_path, mutation):
    def invalid(public):
        result = finish(public)
        if mutation == 'no_support': result['return_assessment']['evidence_claim_indices'] = []
        elif mutation == 'still_open': result['return_assessment']['outcome'] = 'resolved'
        else:
            doc = public['documents'][0]
            win = doc['visible_windows'][0]
            result['claims'] = [{'document_id': doc['document_id'], 'window_id': win['window_id'],
                                 'quote': win['text'][:30], 'interpretation': 'Different source.'}]
        return result
    flow, rid, _, _, _, _ = make(tmp_path, [query, read_new, invalid])
    for _ in range(3): run = flow.step(rid)
    assert run['reason'] == 'INVALID_NAVIGATION_RESPONSE'


@pytest.mark.parametrize('kind', ['model', 'source'])
def test_unknown_attempt_is_charged_and_not_repeated(tmp_path, kind):
    flow, rid, seen, requested, _, _ = make(tmp_path, [TimeoutError()] if kind == 'model' else [query], source_error=TimeoutError())
    run = flow.step(rid)
    assert run['pending']['kind'] == ('llm' if kind == 'model' else 'http')
    assert run['budget']['llm_calls'] == 1
    assert run['budget']['http_attempts'] == (kind == 'source')
    assert flow.step(rid) == run and len(seen) == 1


@pytest.mark.parametrize('mutation', ['reference', 'objective', 'future_document', 'digest', 'coverage'])
def test_start_rejects_task_or_evidence_time_drift(tmp_path, mutation):
    flow, rid, _, _, handoff, contract = make(tmp_path, [])
    task = flow.get(rid)['task']
    if mutation == 'reference': contract['reference_time'] = (NOW + timedelta(days=1)).isoformat()
    elif mutation == 'objective': task['objective'] = 'Silently substituted task'
    elif mutation == 'future_document':
        handoff['documents'][0]['available_at'] = (NOW + timedelta(days=1)).isoformat()
        handoff['document_digests'][DOC['document_id']] = digest(handoff['documents'][0])
    elif mutation == 'digest': handoff['documents'][0]['text'] += ' changed'
    else: handoff['opened'][DOC['document_id']] = ['unknown-window']
    with pytest.raises(ValueError): flow.start(task, handoff, contract)


def test_stop_is_provisional_not_truth_or_numeric_value(tmp_path):
    def stop(public):
        doc = public['documents'][0]
        win = doc['visible_windows'][0]
        result = choice(public, 'stop', claims=[{'document_id': doc['document_id'], 'window_id': win['window_id'],
            'quote': win['text'][:30], 'interpretation': 'Constructed provisional target, semantic truth untested.'}])
        result['target_state'].update(status='proposed_target', candidate_label='Constructed candidate',
                                      support_claim_indices=[0], open_requirements=[])
        return result
    flow, rid, _, requested, _, _ = make(tmp_path, [stop])
    run = flow.step(rid)
    assert run['reason'] == 'AGENT_STOP' and not requested
    assert run['decisions'][0]['expected_gain'] is None
    assert run['events'][-1]['data']['value_stop_proven'] is False


def replay_boundary(monkeypatch, handoff):
    import football_agent.target_navigation_verify as verifier
    monkeypatch.setattr(verifier, 'navigation_handoff', lambda *args: deepcopy(handoff))
    monkeypatch.setattr(verifier, 'verify_acquisition', lambda root,rid,config,result,*args: deepcopy(result['document']))
    return verifier


def test_replay_reconstructs_contract_and_conditional_assessments(tmp_path, monkeypatch):
    flow, rid, _, _, handoff, _ = make(tmp_path, [query, read_new, finish])
    for _ in range(3): flow.step(rid)
    proof = replay_boundary(monkeypatch, handoff).verify_target_navigation(tmp_path, rid)
    assert proof['decisions_verified'] == 3 and proof['raw_source_returns_verified'] == 1
    assert proof['decisions_after_new_body_read'] == 1 and proof['numerical_VoI_verified'] is False


@pytest.mark.parametrize('kind', ['model', 'source', 'defer'])
def test_replay_preserves_failed_attempts_and_voluntary_deferral(tmp_path, monkeypatch, kind):
    flow, rid, _, _, handoff, _ = make(tmp_path, [TimeoutError()] if kind == 'model' else [query if kind == 'source' else choice], source_error=TimeoutError())
    flow.step(rid)
    proof = replay_boundary(monkeypatch, handoff).verify_target_navigation(tmp_path, rid)
    assert proof['model_attempts'] == 1
    assert (proof['pending'] is not None) == (kind != 'defer')


def test_replay_rejects_hash_consistent_reference_substitution(tmp_path, monkeypatch):
    from test_information_navigation import rechain_constructed
    flow, rid, _, _, handoff, _ = make(tmp_path, [choice])
    run = flow.step(rid)
    event = next(e for e in run['events'] if e['type'] == 'navigation_reasoning_reserved')
    old = event['data']['input_ref']
    public = flow.store.get_blob(rid, old)
    public['task_contract']['reference_time'] = (NOW + timedelta(days=1)).isoformat()
    replacement = flow.store.put_blob(rid, public)
    def replace(value):
        if isinstance(value,dict): return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list): return [replace(v) for v in value]
        return replacement if value == old else value
    rechain_constructed(flow, replace(run))
    with pytest.raises(ValueError, match='previously visible evidence'):
        replay_boundary(monkeypatch, handoff).verify_target_navigation(tmp_path, rid)
