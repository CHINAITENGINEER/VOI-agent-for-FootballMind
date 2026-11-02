'''Constructed offline cases; model and network services are never invoked.'''
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_agent.audit import digest
from football_agent.evidence_windows import EvidenceWindows
from football_agent.evidence_reading import EvidenceReadingWorkflow, validate_choice

NOW = datetime(2026, 9, 29, 3, 0, tzinfo=timezone.utc)
DOC = {'document_id': 'constructed-doc', 'status': 'OK', 'kind': 'page',
    'url': 'https://example.com/constructed', 'raw_sha256': '0' * 64,
    'observed_at': NOW.isoformat(), 'available_at': NOW.isoformat(),
    'source_published_at': None, 'publication_status': 'unknown',
    'text_truncated': False, 'text': ('球员名字甲😀 uses a precise source phrase. ' * 35)}


def response(action='read_window', window_id=None, claims=None):
    return {'question': 'Which source passage identifies the fixture?', 'action': action,
        'window_id': window_id, 'reason': 'Inspect the selected passage before inferring applicability.',
        'expected_decision_change': 'May clarify the fixture or require another source.',
        'unread_limits': 'Unread bodies and source freshness remain unknown.',
        'next_information_need': None, 'alternatives': [], 'claims': claims or []}


def setup(tmp_path, responses, *, max_calls=3, max_reads=2, source_provenance=None):
    seen, clock = [], [NOW]
    class Provider:
        def configuration(self): return {'provider': 'constructed-offline-reader'}
        def decide(self, public):
            seen.append(deepcopy(public))
            value = responses[len(seen) - 1]
            if isinstance(value, Exception):
                raise value
            return {'response': value(public) if callable(value) else value,
                    'receipt': {'real_provider_call': False}}
    workflow = EvidenceReadingWorkflow(tmp_path, Provider(), clock=lambda: clock[0],
        window_chars=200, preview_chars=30)
    source = {'document': deepcopy(DOC), 'provenance': {'document_digest': digest(DOC),
        'fixture_only': True, 'new_http_attempts': 0, **(source_provenance or {})}}
    task = {'task_id': 'constructed', 'objective': 'Constructed passage selection',
            'deadline': (NOW + timedelta(minutes=10)).isoformat()}
    run = workflow.start(task, source, max_llm_calls=max_calls, max_window_reads=max_reads)
    return workflow, run['run_id'], seen, clock


def test_windows_reconstruct_unicode_text_with_exact_unbroken_coverage():
    book = EvidenceWindows(DOC, window_chars=200, preview_chars=30)
    windows = list(book.windows.values())
    assert ''.join(w['text'] for w in windows) == DOC['text']
    assert all(w['text'] == DOC['text'][w['start']:w['end']] for w in windows)
    assert windows[0]['start'] == 0 and windows[-1]['end'] == len(DOC['text'])
    assert all(a['end'] == b['start'] for a, b in zip(windows, windows[1:]))
    assert all('text' not in item for item in book.catalog([]))
    assert book.public([])['visible_windows'] == []


def test_anchor_rejects_unopened_body_even_when_quotation_is_in_preview():
    book = EvidenceWindows(DOC, window_chars=200)
    first = next(iter(book.windows))
    with pytest.raises(ValueError, match='previously opened'):
        book.anchor(first, DOC['text'][:20], [])
    anchor = book.anchor(first, DOC['text'][:20], [first])
    assert DOC['text'][anchor['start']:anchor['end']] == anchor['quote']
    assert anchor['text_sha256'] == book.text_sha256


def test_extraction_and_unopened_coverage_stay_visible():
    doc = {**DOC, 'text_truncated': True}
    book = EvidenceWindows(doc, window_chars=200)
    first = next(iter(book.windows))
    public = book.public([first])
    assert public['manifest']['extraction_truncated'] is True
    assert public['manifest']['source_published_at'] is None
    assert public['manifest']['observed_at'] == DOC['observed_at']
    assert public['opened_body_chars'] == 200
    assert public['not_opened_body_ranges'][0][0] == 200


def test_agent_can_choose_nonfirst_window_then_reconsider_and_stop(tmp_path):
    def choose(public):
        return response(window_id=public['catalog'][2]['window_id'])
    def finish(public):
        window = public['visible_windows'][0]
        return response('stop', claims=[{'window_id': window['window_id'],
            'quote': window['text'][10:40], 'interpretation': 'This source contains the quoted phrase.'}])
    workflow, run_id, seen, _ = setup(tmp_path, [choose, finish])
    first = workflow.step(run_id)
    assert first['opened'] == [seen[0]['catalog'][2]['window_id']]
    assert seen[0]['visible_windows'] == []
    final = workflow.step(run_id)
    assert final['state'] == 'STOPPED' and final['reason'] == 'AGENT_STOP'
    assert len(seen[1]['visible_windows']) == 1 and len(final['claims']) == 1
    assert final['budget']['new_http_attempts'] == 0
    assert final['budget']['llm_calls'] == 2 and final['budget']['window_reads'] == 1
    assert workflow.store.verify(run_id)['status'] == 'OK'
    assert workflow.step(run_id) == final and len(seen) == 2


def test_agent_can_defer_without_forced_reading(tmp_path):
    workflow, run_id, seen, _ = setup(tmp_path, [response('defer')])
    run = workflow.step(run_id)
    assert run['reason'] == 'AGENT_DEFER' and run['opened'] == []
    assert len(seen) == 1 and run['budget']['window_reads'] == 0


def test_model_timeout_is_charged_and_restarting_does_not_repeat(tmp_path):
    workflow, run_id, seen, _ = setup(tmp_path, [TimeoutError()])
    failed = workflow.step(run_id)
    assert failed['pending']['kind'] == 'llm' and failed['budget']['llm_calls'] == 1
    assert workflow.step(run_id) == failed and len(seen) == 1
    assert failed['budget']['window_reads'] == 0


def test_final_model_call_cannot_open_body_without_interpretation_capacity(tmp_path):
    choose = lambda public: response(window_id=next(w['window_id'] for w in public['catalog'] if not w['opened']))
    workflow, run_id, _, _ = setup(tmp_path, [choose, choose], max_calls=2, max_reads=1)
    workflow.step(run_id)
    final = workflow.step(run_id)
    assert final['reason'] == 'READ_CONTINUATION_BUDGET_INSUFFICIENT'
    assert final['budget']['window_reads'] == 1
    assert final['budget']['llm_calls'] == 2


def test_future_or_mutated_source_is_rejected_before_provider_call(tmp_path):
    workflow, _, seen, _ = setup(tmp_path, [])
    source = {'document': {**DOC, 'available_at': (NOW + timedelta(seconds=1)).isoformat()}}
    source['provenance'] = {'document_digest': digest(source['document'])}
    task = {'task_id': 'future', 'objective': 'Fixture', 'deadline': (NOW + timedelta(hours=1)).isoformat()}
    with pytest.raises(ValueError, match='not available'):
        workflow.start(task, source)
    source['document']['available_at'] = DOC['available_at']
    with pytest.raises(ValueError, match='does not bind'):
        workflow.start(task, source)
    assert seen == []


def test_claim_from_newly_requested_window_is_rejected(tmp_path):
    def unseen(public):
        first = public['catalog'][0]
        return response(window_id=first['window_id'], claims=[{'window_id': first['window_id'],
            'quote': first['preview'][:15], 'interpretation': 'Premature source claim'}])
    workflow, run_id, _, _ = setup(tmp_path, [unseen])
    run = workflow.step(run_id)
    assert run['reason'] == 'INVALID_READING_RESPONSE' and run['claims'] == []
    assert run['budget']['window_reads'] == 0


@pytest.mark.parametrize('mutation', ['duplicate_read', 'unknown_window', 'selected_alternative', 'terminal_with_window'])
def test_invalid_window_choices_are_rejected(mutation):
    book = EvidenceWindows(DOC, window_chars=200)
    first = next(iter(book.windows))
    value, opened = response(window_id=first), []
    if mutation == 'duplicate_read': opened = [first]
    elif mutation == 'unknown_window': value['window_id'] = 'invented'
    elif mutation == 'selected_alternative': value['alternatives'] = [{'window_id': first, 'reason_not_selected': 'Contradiction'}]
    else: value['action'] = 'stop'
    with pytest.raises(ValueError): validate_choice(value, book, opened)


def test_deadline_and_configuration_changes_stop_before_new_model_call(tmp_path):
    workflow, run_id, seen, clock = setup(tmp_path, [])
    clock[0] += timedelta(minutes=11)
    assert workflow.step(run_id)['reason'] == 'DEADLINE_REACHED'
    assert seen == []
    other, identity, seen, _ = setup(tmp_path / 'other', [])
    other.preview_chars = 20
    assert other.step(identity)['reason'] == 'CONFIGURATION_CHANGED'
    assert seen == []


def replay_fixture(tmp_path, monkeypatch, responses, **kwargs):
    # Only this offline test substitutes prefix provenance. Production verification
    # independently replays the actual source journal and raw acquisition archive.
    import football_agent.evidence_reading_verify as verifier
    workflow, run_id, seen, clock = setup(tmp_path, responses, source_provenance={
        'parent_root': str(tmp_path / 'constructed-parent'), 'parent_run_id': 'fixture',
        'parent_head_hash': 'fixture-head', 'document_id': DOC['document_id']}, **kwargs)
    source = workflow.store.get_blob(run_id, workflow.get(run_id)['source_ref'])
    monkeypatch.setattr(verifier, 'source_prefix', lambda *args, **kwargs: deepcopy(source))
    return workflow, run_id, seen, clock


def rechain_fixture(workflow, run):
    # Recompute the entire chain to exercise semantic verification, not checksum
    # detection. This helper is used exclusively on constructed temporary runs.
    previous = None
    for event in run['events']:
        event['previous_hash'] = previous
        event['event_hash'] = digest({k: v for k, v in event.items() if k != 'event_hash'})
        previous = event['event_hash']
    run.update(deepcopy(run['events'][-1]['snapshot']))
    workflow.store.save(run)
    assert workflow.store.verify(run['run_id'])['status'] == 'OK'


def test_replay_proves_nonfirst_selection_and_prior_visibility(tmp_path, monkeypatch):
    from football_agent.evidence_reading_verify import verify_reading
    choose = lambda p: response(window_id=p['catalog'][2]['window_id'])
    def finish(p):
        w = p['visible_windows'][0]
        return response('stop', claims=[{'window_id': w['window_id'],
            'quote': w['text'][10:40], 'interpretation': 'Constructed quote only.'}])
    workflow, identity, _, _ = replay_fixture(tmp_path, monkeypatch, [choose, finish])
    workflow.step(identity)
    workflow.step(identity)
    proof = verify_reading(tmp_path, identity)
    assert proof['body_reads'] == 1 and proof['model_attempts'] == 2
    assert proof['claims_quote_checked'] == 1 and proof['agent_terminal_after_reading']
    assert proof['new_http_attempts'] == 0 and proof['numerical_value_verified'] is False


@pytest.mark.parametrize('case', ['defer', 'timeout', 'invalid', 'deadline', 'read_budget'])
def test_replay_preserves_terminal_causes_and_charged_attempts(tmp_path, monkeypatch, case):
    from football_agent.evidence_reading_verify import verify_reading
    choose = lambda p: response(window_id=next(w['window_id'] for w in p['catalog'] if not w['opened']))
    replies = {'defer': [response('defer')], 'timeout': [TimeoutError()],
        'invalid': [response(window_id='invented')], 'deadline': [], 'read_budget': [choose, choose]}[case]
    workflow, identity, _, clock = replay_fixture(tmp_path, monkeypatch, replies, max_calls=2, max_reads=1)
    if case == 'deadline':
        clock[0] += timedelta(minutes=11)
    workflow.step(identity)
    if case == 'read_budget':
        workflow.step(identity)
    proof = verify_reading(tmp_path, identity)
    assert proof['model_attempts'] == (0 if case == 'deadline' else 2 if case == 'read_budget' else 1)
    assert (proof['pending'] is not None) == (case == 'timeout')
    assert proof['completed_returns'] == (0 if case in {'timeout', 'deadline'} else proof['model_attempts'])
    assert proof['body_reads'] == (1 if case == 'read_budget' else 0)


def test_replay_rejects_hash_consistent_unread_body_leak(tmp_path, monkeypatch):
    from football_agent.evidence_reading_verify import verify_reading
    workflow, identity, _, _ = replay_fixture(tmp_path, monkeypatch, [response('defer')])
    run = workflow.step(identity)
    reservation = next(e for e in run['events'] if e['type'] == 'reading_reasoning_reserved')
    old_ref = reservation['data']['input_ref']
    leaked = workflow.store.get_blob(identity, old_ref)
    book = workflow.book(run)
    leaked['visible_windows'] = [book.read(next(iter(book.windows)))]
    new_ref = workflow.store.put_blob(identity, leaked)
    def replace(value):
        if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list): return [replace(v) for v in value]
        return new_ref if value == old_ref else value
    run = replace(run)
    rechain_fixture(workflow, run)
    with pytest.raises(ValueError, match='visible evidence'):
        verify_reading(tmp_path, identity)


def test_replay_rejects_hash_consistent_body_cost_inflation(tmp_path, monkeypatch):
    from football_agent.evidence_reading_verify import verify_reading
    choose = lambda p: response(window_id=p['catalog'][0]['window_id'])
    workflow, identity, _, _ = replay_fixture(tmp_path, monkeypatch, [choose, response('defer')])
    workflow.step(identity)
    run = workflow.step(identity)
    for event in run['events']:
        if event['snapshot']['budget']['window_reads']:
            event['snapshot']['budget']['delivered_body_chars'] += 200
    rechain_fixture(workflow, run)
    with pytest.raises(ValueError, match='reconstructed reading prefix'):
        verify_reading(tmp_path, identity)
