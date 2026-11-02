'''Constructed reviewer responses verify mechanics, not football correctness.'''
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_agent.navigation_task_review import (CRITERIA, ReviewSession,
    build_review_packet, validate_review)

NOW = datetime(2026, 9, 29, 4, 0, tzinfo=timezone.utc)
TEXT = 'NEXT HOME MATCH Team A vs Team B, October 9, 16:30. The year is unspecified.'


def packet():
    source = {'document_id': 'doc', 'url': 'https://example.com/', 'status': 'OK',
        'kind': 'page', 'observed_at': NOW.isoformat(), 'available_at': NOW.isoformat(),
        'visible_windows': [{'window_id': 'body', 'start': 5, 'end': 5 + len(TEXT), 'text': TEXT}],
        'catalog': [{'window_id': 'unopened', 'opened': False, 'preview': 'Unopened preview only.',
                     'preview_start': 100, 'preview_end': 122}],
        'not_opened_body_ranges': [[100, 500]], 'manifest': {}}
    projection = {'available_at': NOW.isoformat(), 'visible_documents': [source],
        'claims': [{'interpretation': 'POLICY_LEAK_MARKER'}],
        'information_need': {'reason': 'POLICY_LEAK_MARKER'},
        'costs': {'llm_calls': 9}, 'next_context': {'navigation.last_action': 'defer'}}
    return build_review_packet(projection, case_id='opaque1', task_question='Identify next fixture.',
                               task_reference_time=NOW)


def response():
    return {'case_id': 'opaque1', 'criteria': [{'criterion_id': key, 'status': 'partial',
        'reason': 'A home fixture clue is not a fully established next match.',
        'citations': [{'document_id': 'doc', 'window_id': 'body', 'quote': 'NEXT HOME MATCH'}]}
        for key in CRITERIA], 'proposed_answer': None,
        'missing_information': ['Year and complete schedule.'], 'summary': 'Insufficient packet.'}


def test_input_hides_policy_trace_and_keeps_opened_and_preview_distinct():
    public = packet()
    assert 'POLICY_LEAK_MARKER' not in str(public)
    assert not {'claims', 'information_need', 'costs', 'next_context'} & public.keys()
    assert public['documents'][0]['body_windows'][0]['text'] == TEXT
    assert public['documents'][0]['navigation_previews'][0]['window_id'] == 'unopened'


def test_review_preserves_partial_evidence_and_unknown_truth():
    reviewed = validate_review(packet(), response())
    assert reviewed['evidence_sufficiency'] == 'not_established'
    assert reviewed['criteria'][0]['citations'][0]['start'] == 5
    assert reviewed['criteria'][0]['citations'][0]['end'] == 20
    assert reviewed['task_outcome_label'] is None
    assert reviewed['eligible_for_value_calibration'] is reviewed['world_truth_verified'] is False


@pytest.mark.parametrize('case', ['wrong_case', 'preview', 'invented_quote', 'missing_quote', 'duplicate_criterion', 'unsupported_answer', 'one_sided_conflict'])
def test_invalid_support_is_rejected(case):
    result = response()
    first = result['criteria'][0]
    if case == 'wrong_case': result['case_id'] = 'other'
    elif case == 'preview': first['citations'] = [{'document_id': 'doc', 'window_id': 'unopened', 'quote': 'Unopened preview only.'}]
    elif case == 'invented_quote': first['citations'][0]['quote'] = 'A verified full schedule.'
    elif case == 'missing_quote': first['citations'] = []
    elif case == 'duplicate_criterion': result['criteria'][1]['criterion_id'] = first['criterion_id']
    elif case == 'unsupported_answer': result['proposed_answer'] = 'Definitely the next fixture.'
    elif case == 'one_sided_conflict': first['status'] = 'conflicting'
    with pytest.raises(ValueError): validate_review(packet(), result)


def test_full_packet_support_is_still_provisional_not_truth_or_loss():
    result = response()
    for criterion in result['criteria']: criterion['status'] = 'supported'
    result['proposed_answer'] = 'Constructed reviewer answer.'
    reviewed = validate_review(packet(), result)
    assert reviewed['evidence_sufficiency'] == 'supported_in_packet'
    assert reviewed['review_status'] == 'PROVISIONAL_REQUIRES_ADJUDICATION'
    assert reviewed['realized_decision_utility'] is reviewed['task_outcome_label'] is None


class Provider:
    def __init__(self, reply=None, error=None, clock=None):
        self.reply, self.error, self.calls, self.clock = reply, error, 0, clock
    def configuration(self): return {'provider': 'constructed-offline'}
    def decide(self, public):
        self.calls += 1
        if self.clock: self.clock[0] += timedelta(minutes=2)
        if self.error: raise self.error
        return {'response': self.reply if self.reply is not None else response(),
                'receipt': {'real_provider_call': False}}


@pytest.mark.parametrize('failure', [False, True])
def test_repeat_execution_never_spends_another_attempt(tmp_path, failure):
    provider = Provider(error=TimeoutError() if failure else None)
    session = ReviewSession(tmp_path, provider, clock=lambda: NOW)
    first = session.execute(packet(), deadline=NOW + timedelta(minutes=1))
    second = session.execute(packet(), deadline=NOW + timedelta(minutes=1))
    assert first == second and provider.calls == first['attempts'] == 1
    assert first['status'] == ('RETURN_UNKNOWN' if failure else 'REVIEWED_PROVISIONAL')
    assert (first['pending'] is not None) == failure


def test_invalid_and_late_responses_are_retained_without_labels(tmp_path):
    invalid = ReviewSession(tmp_path / 'invalid', Provider(reply={}), clock=lambda: NOW)
    result = invalid.execute(packet(), deadline=NOW + timedelta(minutes=1))
    assert result['status'] == 'INVALID_REVIEW' and result['review'] is None
    clock = [NOW]
    late = ReviewSession(tmp_path / 'late', Provider(clock=clock), clock=lambda: clock[0])
    result = late.execute(packet(), deadline=NOW + timedelta(minutes=1))
    assert result['status'] == 'LATE_RETURN_EXCLUDED' and result['review'] is None


def test_expired_task_is_not_called_and_changed_packet_is_not_retried(tmp_path):
    provider = Provider()
    session = ReviewSession(tmp_path, provider, clock=lambda: NOW)
    result = session.execute(packet(), deadline=NOW)
    assert result['attempts'] == provider.calls == 0
    assert result['status'] == 'CUTOFF_EXCLUDED'
    changed = deepcopy(packet()); changed['task']['question'] = 'A changed task.'
    with pytest.raises(ValueError, match='Frozen'):
        session.execute(changed, deadline=NOW)
