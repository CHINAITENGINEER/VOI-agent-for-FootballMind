'''Offline behavioral checks; constructed sources never count as live evidence.'''
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_agent.audit import digest
from football_agent.evidence_windows import EvidenceWindows
from football_agent.information_navigation import InformationNavigationWorkflow
from football_agent.inquiry_tools import OpenWebTools

NOW = datetime(2026, 9, 29, 4, 0, tzinfo=timezone.utc)
DOC = {'document_id': 'constructed-old', 'status': 'OK', 'kind': 'page',
    'url': 'https://example.com/old', 'raw_sha256': '0' * 64,
    'observed_at': NOW.isoformat(), 'available_at': NOW.isoformat(),
    'source_published_at': None, 'publication_status': 'unknown', 'text_truncated': False,
    'text': 'Old constructed source with unresolved fixture identity. ' * 60,
    'links': [{'url': 'https://example.com/schedule', 'title': 'Schedule lead, not verified'}]}


def choice(action='defer', *, document_id=None, window_id=None, query=None, claims=None):
    return {'question': 'Which evidence identifies the fixture?', 'action': action,
        'document_id': document_id, 'window_id': window_id, 'query': query,
        'reason': 'Compare remaining routes.', 'expected_decision_change': 'May change which evidence to read.',
        'unread_limits': 'Other bodies and the real fixture remain unverified.',
        'next_information_need': 'Confirm fixture identity.',
        'path_comparison': {'read_archived': 'Existing text may be sufficient.',
            'query_source': 'New source might clarify the date at request cost.',
            'defer': 'No verified identity yet.'}, 'claims': claims or []}


def make(tmp_path, replies, *, result=None, tools_error=None, **limits):
    clock, seen, requests = [NOW], [], []
    class Provider:
        def configuration(self): return {'provider': 'offline-constructed'}
        def decide(self, public):
            seen.append(deepcopy(public))
            reply = replies[len(seen) - 1]
            if isinstance(reply, Exception): raise reply
            return {'response': reply(public) if callable(reply) else deepcopy(reply),
                'receipt': {'real_provider_call': False}}
    class Tools(OpenWebTools):
        def execute(self, prepared, *, run_id, request_id):
            requests.append(prepared)
            if tools_error: raise tools_error
            returned = deepcopy(result) if result is not None else {**deepcopy(DOC),
                'document_id': 'constructed-new', 'url': 'https://example.com/schedule',
                'text': 'New constructed schedule passage with conditional details. ' * 30, 'links': []}
            return {'document': returned, 'http_attempts': 1, 'request_id': request_id,
                'latency_seconds': 0, 'money': None, 'money_status': 'constructed-offline'}
    workflow = InformationNavigationWorkflow(tmp_path, Provider(), Tools(None, ['example.com', 'www.bing.com']), clock=lambda: clock[0])
    first = next(iter(EvidenceWindows(DOC).windows))
    handoff = {'source': {'document': deepcopy(DOC), 'provenance': {'document_digest': digest(DOC)}},
        'parent_root': str(tmp_path / 'constructed-parent'), 'parent_run_id': 'constructed-parent',
        'opened': [first], 'claims': [], 'last_choice': choice(),
        'parent_completed_at': NOW.isoformat(), 'parent_head_hash': 'constructed-head',
        'scope': 'constructed prior model request, not verified fact'}
    task = {'task_id': 'constructed-navigation', 'objective': 'Constructed information routing',
        'deadline': (NOW + timedelta(minutes=10)).isoformat(), 'seed_urls': [DOC['url']]}
    run = workflow.start(task, handoff, **limits)
    return workflow, run['run_id'], seen, requests, clock


def query_schedule(public):
    return choice('query_source', query={'tool': 'read_url', 'url': 'https://example.com/schedule', 'query': None})


def read_new(public):
    doc = public['documents'][-1]
    return choice('read_window', document_id=doc['document_id'], window_id=doc['catalog'][0]['window_id'])


def test_agent_queries_then_selects_body_and_reconsiders(tmp_path):
    def finish(public):
        doc = public['documents'][-1]
        window = doc['visible_windows'][0]
        return choice('defer', claims=[{'document_id': doc['document_id'], 'window_id': window['window_id'],
            'quote': window['text'][:30], 'interpretation': 'Only a constructed source statement.'}])
    workflow, identity, seen, requests, _ = make(tmp_path, [query_schedule, read_new, finish])
    workflow.step(identity)
    workflow.step(identity)
    run = workflow.step(identity)
    assert run['reason'] == 'AGENT_DEFER' and len(requests) == 1
    assert seen[1]['documents'][-1]['visible_windows'] == []
    assert len(seen[2]['documents'][-1]['visible_windows']) == 1
    assert run['budget']['llm_calls'] == 3 and run['budget']['http_attempts'] == 1
    assert run['budget']['window_reads'] == 1 and run['budget']['delivered_body_chars'] == 1200
    assert len(run['claims']) == 1 and run['pending'] is None
    assert workflow.step(identity) == run and len(seen) == 3
    assert workflow.store.verify(identity)['status'] == 'OK'


def test_agent_can_continue_archive_instead_of_following_prior_query_need(tmp_path):
    def read_old(public):
        doc = public['documents'][0]
        return choice('read_window', document_id=doc['document_id'], window_id=doc['catalog'][2]['window_id'])
    workflow, identity, seen, requests, _ = make(tmp_path, [read_old, choice('stop')])
    workflow.step(identity)
    run = workflow.step(identity)
    assert run['reason'] == 'AGENT_STOP' and requests == []
    assert len(run['opened'][DOC['document_id']]) == 2
    assert run['budget']['window_reads'] == 1
    assert seen[0]['inherited_information_need']['next_information_need'] == 'Confirm fixture identity.'


def test_agent_can_defer_without_any_action(tmp_path):
    workflow, identity, seen, requests, _ = make(tmp_path, [choice()])
    run = workflow.step(identity)
    assert run['reason'] == 'AGENT_DEFER' and requests == []
    assert run['budget']['window_reads'] == run['budget']['http_attempts'] == 0


@pytest.mark.parametrize('route', ['model', 'source'])
def test_unknown_outcomes_stay_charged_and_are_not_repeated(tmp_path, route):
    replies = [TimeoutError()] if route == 'model' else [query_schedule]
    workflow, identity, seen, requests, _ = make(tmp_path, replies, tools_error=TimeoutError())
    run = workflow.step(identity)
    assert run['pending']['kind'] == ('llm' if route == 'model' else 'http')
    assert run['budget']['llm_calls'] == 1
    assert run['budget']['http_attempts'] == (1 if route == 'source' else 0)
    assert workflow.step(identity) == run and len(seen) == 1
    assert len(requests) == (1 if route == 'source' else 0)


@pytest.mark.parametrize('kind', ['invented_url', 'repeat_old', 'unread_quote', 'terminal_query'])
def test_invalid_actions_are_rejected_before_new_http(tmp_path, kind):
    def invalid(public):
        result = query_schedule(public)
        if kind == 'invented_url': result['query']['url'] = 'https://example.com/not-seen'
        elif kind == 'repeat_old': result['query']['url'] = DOC['url']
        elif kind == 'terminal_query': result['action'] = 'defer'
        else:
            doc = public['documents'][0]
            window = doc['catalog'][1]
            result['claims'] = [{'document_id': doc['document_id'], 'window_id': window['window_id'],
                'quote': window['preview'][:25], 'interpretation': 'Unopened statement.'}]
        return result
    workflow, identity, _, requests, _ = make(tmp_path, [invalid])
    run = workflow.step(identity)
    assert run['reason'] == 'INVALID_NAVIGATION_RESPONSE' and requests == []
    assert run['claims'] == []


def test_failed_source_is_returned_to_agent_without_becoming_counterevidence(tmp_path):
    failed = {**DOC, 'document_id': 'constructed-failed', 'url': 'https://example.com/schedule',
        'status': 'FAILED', 'text': '', 'links': [], 'http_status': 503}
    workflow, identity, seen, requests, _ = make(tmp_path, [query_schedule, choice()], result=failed)
    workflow.step(identity)
    run = workflow.step(identity)
    assert seen[1]['documents'][-1]['status'] == 'FAILED'
    assert 'catalog' not in seen[1]['documents'][-1]
    assert run['claims'] == [] and len(requests) == 1


def test_http_zero_budget_and_deadline_do_not_dispatch(tmp_path):
    workflow, identity, _, requests, _ = make(tmp_path, [query_schedule], max_http_attempts=0)
    assert workflow.step(identity)['reason'] == 'HTTP_BUDGET_EXHAUSTED'
    assert requests == []
    other, identity, seen, requests, clock = make(tmp_path / 'late', [])
    clock[0] += timedelta(minutes=11)
    assert other.step(identity)['reason'] == 'DEADLINE_REACHED' and seen == requests == []


def test_last_model_call_cannot_dispatch_without_reconsideration(tmp_path):
    def read_old(public):
        doc = public['documents'][0]
        return choice('read_window', document_id=doc['document_id'], window_id=doc['catalog'][1]['window_id'])
    workflow, identity, _, requests, _ = make(tmp_path, [read_old, query_schedule], max_llm_calls=2, max_window_reads=1)
    workflow.step(identity)
    run = workflow.step(identity)
    assert run['reason'] == 'NO_RECONSIDERATION_CAPACITY' and requests == []


def stub_parent_and_raw_boundary(monkeypatch, workflow, identity):
    # These tests validate the navigation reconstruction. Real raw archive and
    # parent provenance are independently exercised by the live offline receipt.
    import football_agent.information_navigation_verify as verifier
    run = workflow.get(identity)
    handoff = workflow.store.get_blob(identity, run['handoff_ref'])
    monkeypatch.setattr(verifier, 'reading_handoff', lambda *args: deepcopy(handoff))
    monkeypatch.setattr(verifier, 'verify_acquisition', lambda root, run_id, config, result, *args: deepcopy(result['document']))
    return verifier


def rechain_constructed(workflow, run):
    previous = None
    for event in run['events']:
        event['previous_hash'] = previous
        event['event_hash'] = digest({k:v for k,v in event.items() if k != 'event_hash'})
        previous = event['event_hash']
    run.update(deepcopy(run['events'][-1]['snapshot']))
    workflow.store.save(run)
    assert workflow.store.verify(run['run_id'])['status'] == 'OK'


def test_offline_replay_reconstructs_query_read_reconsideration(tmp_path, monkeypatch):
    workflow, identity, _, _, _ = make(tmp_path, [query_schedule, read_new, choice()])
    verifier = stub_parent_and_raw_boundary(monkeypatch, workflow, identity)
    for _ in range(3): workflow.step(identity)
    proof = verifier.verify_navigation(tmp_path, identity)
    assert proof['model_attempts'] == proof['completed_model_returns'] == 3
    assert proof['raw_source_returns_verified'] == proof['new_window_reads'] == 1
    assert proof['decisions_after_new_body_read'] == 1
    assert proof['numerical_VoI_verified'] is False and proof['pending'] is None


@pytest.mark.parametrize('case', ['model_timeout', 'source_timeout', 'defer', 'invalid', 'deadline'])
def test_replay_retains_failures_and_terminal_causes(tmp_path, monkeypatch, case):
    replies = {'model_timeout':[TimeoutError()], 'source_timeout':[query_schedule],
        'defer':[choice()], 'invalid':[choice('read_window',document_id='invented',window_id='x')],
        'deadline':[]}[case]
    workflow, identity, _, _, clock = make(tmp_path, replies, tools_error=TimeoutError())
    verifier = stub_parent_and_raw_boundary(monkeypatch, workflow, identity)
    if case == 'deadline': clock[0] += timedelta(minutes=11)
    workflow.step(identity)
    proof = verifier.verify_navigation(tmp_path, identity)
    assert proof['model_attempts'] == (0 if case == 'deadline' else 1)
    assert proof['source_attempt_reservations'] == (1 if case == 'source_timeout' else 0)
    assert (proof['pending'] is not None) == (case in {'model_timeout','source_timeout'})


def test_replay_rejects_hash_consistent_model_input_leak(tmp_path, monkeypatch):
    workflow, identity, _, _, _ = make(tmp_path, [choice()])
    verifier = stub_parent_and_raw_boundary(monkeypatch, workflow, identity)
    run = workflow.step(identity)
    event = next(e for e in run['events'] if e['type'] == 'navigation_reasoning_reserved')
    original = event['data']['input_ref']
    public = workflow.store.get_blob(identity, original)
    book = EvidenceWindows(DOC)
    public['documents'][0]['visible_windows'].append(book.read(list(book.windows)[1]))
    replacement = workflow.store.put_blob(identity, public)
    def replace(value):
        if isinstance(value,dict): return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list): return [replace(v) for v in value]
        return replacement if value == original else value
    run = replace(run)
    rechain_constructed(workflow,run)
    with pytest.raises(ValueError, match='previously visible evidence'):
        verifier.verify_navigation(tmp_path,identity)


def test_replay_rejects_hash_consistent_executed_url_swap(tmp_path, monkeypatch):
    workflow, identity, _, _, _ = make(tmp_path, [query_schedule, choice()])
    verifier = stub_parent_and_raw_boundary(monkeypatch, workflow, identity)
    run = workflow.step(identity)
    event = next(e for e in run['events'] if e['type'] == 'navigation_acquisition_reserved')
    event['data']['action']['endpoint'] = 'https://example.com/substituted'
    rechain_constructed(workflow,run)
    with pytest.raises(ValueError, match='chosen action'):
        verifier.verify_navigation(tmp_path,identity)
