'''Constructed cases test review mechanics, not football truth or model quality.'''
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest
from jsonschema.exceptions import ValidationError

from football_agent.audit import digest
from football_agent.fixture_bound_review import task_contract
from football_agent.question_return_outcomes import VERSION, blinded_review_packet
from football_agent.question_return_review import (
    review_input, validate_review, QuestionReviewSession, evaluate_question_reviews,
)

NOW = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)
TEXT = 'Schedule filters: Select a team and date. Calendar view is available.'


@pytest.fixture
def record():
    contract = task_contract(task_id='constructed', question='Find the next fixture.', subject_team='Team A',
        team_aliases=[], reference_time=NOW - timedelta(minutes=10))
    test = {'requirement': 'all_fixture_order', 'useful_if': 'Lists dated fixtures and status.',
        'irrelevant_if': 'Only filter controls are shown.'}
    return {'schema_version': VERSION, 'record_id': 'constructedreturn',
        'task_contract': contract, 'task_contract_digest': digest(contract),
        'selected_at': (NOW - timedelta(minutes=2)).isoformat(), 'assessed_at': NOW.isoformat(),
        'prospective_question': 'Does the visible schedule establish the next fixture?',
        'information_test': test, 'information_test_digest': digest(test),
        'document': {'document_id': 'doc', 'status': 'OK', 'available_at': (NOW - timedelta(minutes=3)).isoformat()},
        'exposure': 'body', 'visible_body_window': {'window_id': 'body', 'start': 10, 'end': 10 + len(TEXT), 'text': TEXT},
        'actor_outcome': 'irrelevant', 'actor_reason': 'ACTOR_LEAK_MARKER'}


def response(label='irrelevant'):
    return {'case_id': 'opaque1', 'label': label, 'reason': 'Only filter controls are visible.',
        'citations': [{'document_id': 'doc', 'window_id': 'body', 'quote': 'Select a team and date'}],
        'remaining_unknowns': ['Actual fixture order and status.']}


class Provider:
    def __init__(self, clock, *, reply=None, error=None, delay=1):
        self.clock, self.reply, self.error, self.delay = clock, reply, error, delay
        self.calls = 0

    def configuration(self):
        return {'provider': 'constructed_offline'}

    def decide(self, public):
        self.calls += 1
        self.clock[0] += timedelta(seconds=self.delay)
        if self.error:
            raise self.error
        return {'response': deepcopy(self.reply if self.reply is not None else response()),
            'receipt': {'real_provider_call': False}}


def execute(tmp_path, record, **kwargs):
    clock = [NOW + timedelta(seconds=10)]
    provider = Provider(clock, **kwargs)
    worker = QuestionReviewSession(tmp_path, provider, clock=lambda: clock[0])
    packet = blinded_review_packet(record)
    deadline = NOW + timedelta(minutes=1)
    result = worker.execute(packet, case_id='opaque1', deadline=deadline)
    return result, worker, provider, packet, deadline


def test_input_hides_actor_and_preserves_whole_visible_window(record):
    packet = blinded_review_packet(record)
    public = review_input(packet, case_id='opaque1')
    assert 'ACTOR_LEAK_MARKER' not in str(public)
    assert public['payload']['visible_body_window']['text'] == TEXT
    packet['payload']['actor_reason'] = 'leak'
    packet['payload_digest'] = digest(packet['payload'])
    with pytest.raises(ValueError, match='extra fields'):
        review_input(packet, case_id='opaque1')


@pytest.mark.parametrize('kind', ['digest', 'task', 'status', 'offset', 'prior_verdict', 'flag'])
def test_invalid_packet_is_rejected(record, kind):
    p = blinded_review_packet(record)
    if kind == 'digest': p['payload_digest'] = 'wrong'
    if kind == 'task': p['payload']['task_contract']['question'] = 'Another task'
    if kind == 'status': p['payload']['document']['status'] = 'FAILED'
    if kind == 'offset': p['payload']['visible_body_window']['end'] += 1
    if kind == 'prior_verdict': p['verdict'] = 'irrelevant'
    if kind == 'flag': p['actor_verdict_exposed'] = True
    if kind != 'digest': p['payload_digest'] = digest(p['payload'])
    with pytest.raises(ValueError): review_input(p, case_id='opaque1')


@pytest.mark.parametrize('kind', ['case', 'window', 'quote', 'duplicate', 'conflict', 'resolved', 'extra_reward'])
def test_invalid_review_is_rejected(record, kind):
    p = review_input(blinded_review_packet(record), case_id='opaque1')
    r = response()
    if kind == 'case': r['case_id'] = 'other'
    if kind == 'window': r['citations'][0]['window_id'] = 'unopened'
    if kind == 'quote': r['citations'][0]['quote'] = 'Invented factual statement.'
    if kind == 'duplicate': r['citations'] *= 2
    if kind == 'conflict': r['label'] = 'conflicting'
    if kind == 'resolved': r['label'] = 'resolved'
    if kind == 'extra_reward': r['gain'] = 0.5
    with pytest.raises((ValueError, ValidationError)): validate_review(p, r)


def test_valid_quote_is_location_proof_not_semantic_ground_truth(record):
    public = review_input(blinded_review_packet(record), case_id='opaque1')
    r = response('resolved')
    r['remaining_unknowns'] = []
    result = validate_review(public, r)
    assert result['world_truth_verified'] is False
    assert result['independent_semantic_label'] is None
    assert result['numeric_voi_training_eligible'] is False
    assert result['citations'][0]['start'] == 10 + TEXT.index('Select a team and date')


def test_unknown_context_may_abstain_without_fabricated_quote(record):
    r = response('insufficient_context')
    r['citations'] = []
    result = validate_review(review_input(blinded_review_packet(record), case_id='opaque1'), r)
    assert result['label'] == 'insufficient_context'


def test_one_charged_attempt_and_idempotent_readback(tmp_path, record):
    s, worker, provider, packet, deadline = execute(tmp_path, record)
    assert s['status'] == 'REVIEWED_PROVISIONAL'
    assert s['available_at'] == s['completed_at']
    assert worker.execute(packet, case_id='opaque1', deadline=deadline) == s
    assert provider.calls == s['attempts'] == 1
    with pytest.raises(ValueError, match='changed'):
        worker.execute(packet, case_id='opaque1', deadline=deadline + timedelta(seconds=1))


@pytest.mark.parametrize('mode,expected', [('error', 'RETURN_UNKNOWN'), ('late', 'LATE_RETURN_EXCLUDED'), ('invalid', 'INVALID_REVIEW')])
def test_failed_or_late_review_has_no_publishable_label(tmp_path, record, mode, expected):
    kwargs = {'error': TimeoutError()} if mode == 'error' else {'delay': 70} if mode == 'late' else {'reply': response('resolved')}
    s, worker, provider, packet, deadline = execute(tmp_path, record, **kwargs)
    assert s['status'] == expected and s['available_at'] is None and s['review'] is None
    assert worker.execute(packet, case_id='opaque1', deadline=deadline) == s
    assert provider.calls == 1
    report = evaluate_question_reviews([record], [s], as_of=NOW + timedelta(hours=1))
    assert report['published_provisional_reviews'] == 0


def test_interrupted_reservation_is_not_silently_retried(tmp_path, record):
    clock = [NOW + timedelta(seconds=10)]
    provider = Provider(clock, error=KeyboardInterrupt())
    worker = QuestionReviewSession(tmp_path, provider, clock=lambda: clock[0])
    packet, deadline = blinded_review_packet(record), NOW + timedelta(minutes=1)
    with pytest.raises(KeyboardInterrupt): worker.execute(packet, case_id='opaque1', deadline=deadline)
    s = worker.execute(packet, case_id='opaque1', deadline=deadline)
    assert s['status'] == 'RESERVED' and s['pending'] == 'model_return_unknown' and provider.calls == 1


def test_future_review_is_not_in_actor_time_snapshot(tmp_path, record):
    s, *_ = execute(tmp_path, record)
    before = evaluate_question_reviews([record], [s], as_of=NOW)
    assert before['published_provisional_reviews'] == 0
    bad_future = deepcopy(s)
    bad_future['review']['label'] = 'resolved'
    assert evaluate_question_reviews([record], [bad_future], as_of=NOW) == before
    after = evaluate_question_reviews([record], [s, s], as_of=s['available_at'])
    assert after['published_provisional_reviews'] == 1
    assert after['rows'][0]['agreement_count'] == 1
    assert after['independent_semantic_labels'] == 0 and after['policy_effect'] is None
    assert after['runtime_safe'] is False and 'reviews' not in record


@pytest.mark.parametrize('field', ['review', 'raw_response', 'available_at', 'review_input'])
def test_tampered_release_or_response_rejected(tmp_path, record, field):
    s, *_ = execute(tmp_path, record)
    if field == 'review': s['review']['world_truth_verified'] = True
    if field == 'raw_response': s['raw_response']['reason'] = 'changed'
    if field == 'available_at': s['available_at'] = NOW.isoformat()
    if field == 'review_input': s['review_input']['payload']['prospective_question'] = 'Different question'
    with pytest.raises(ValueError):
        evaluate_question_reviews([record], [s], as_of=NOW + timedelta(minutes=1))


def test_insufficient_context_is_not_a_disagreement_label(tmp_path, record):
    r = response('insufficient_context')
    r['citations'] = []
    s, *_ = execute(tmp_path, record, reply=r)
    result = evaluate_question_reviews([record], [s], as_of=s['available_at'])
    assert result['rows'][0]['comparable_reviews'] == 0
    assert result['rows'][0]['agreement_count'] == 0
