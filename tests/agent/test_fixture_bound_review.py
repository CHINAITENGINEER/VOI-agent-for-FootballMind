from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_agent.fixture_bound_review import (FIELDS, CRITERIA, BoundReviewSession,
    build_bound_packet, task_contract, validate_bound_review, compare_bound_reviews)
from football_agent.fixture_review_export import attach_fixture_reviews

NOW = datetime(2026, 9, 29, 10, tzinfo=timezone.utc)
TEXT = ('Official schedule as of 2026-09-29: all competitions, home and away. '
        'Alpha FC vs Beta FC, League A, 2026-10-02 19:00 UTC+09:00, scheduled. '
        'This is Alpha FC next match across all competitions. '
        'Earlier: Alpha FC vs Gamma FC on 2026-09-27. '
        'NEXT HOME MATCH Alpha FC vs Delta FC October 9 16:30.')


def packet():
    contract = task_contract(task_id='originaltask', question='Identify current or next Alpha fixture',
        subject_team='Alpha FC', team_aliases=['Alpha'], reference_time=NOW)
    base = {'task': {'question': contract['question'], 'reference_time': contract['reference_time']},
        'evidence_cutoff': NOW.isoformat(), 'documents': [{'document_id': 'doc1',
            'available_at': NOW.isoformat(), 'body_windows': [{'window_id': 'w1', 'start': 10,
                'end': 10 + len(TEXT), 'text': TEXT}],
            'navigation_previews': [{'window_id': 'w2', 'preview': 'Unopened preview candidate only.'}]}]}
    return build_bound_packet(base, contract, case_id='case1')


def cite(quote='Alpha FC vs Beta FC, League A, 2026-10-02 19:00 UTC+09:00, scheduled.'):
    return {'document_id': 'doc1', 'window_id': 'w1', 'quote': quote}


def response(*, resolved=False):
    fields = {'home_team': 'Alpha FC', 'away_team': 'Beta FC', 'competition': 'League A',
        'scheduled_date': '2026-10-02', 'local_time': '19:00', 'utc_offset': '+09:00',
        'schedule_scope': 'all_fixtures', 'match_status': 'scheduled'}
    body = {'case_id': 'case1', 'task_contract_digest': packet()['task_contract_digest'],
        'candidates': [{'candidate_id': 'c1', 'description': 'Synthetic scheduled fixture', 'fields': fields,
            'field_evidence': [{'fields': list(FIELDS), 'citation': cite(TEXT[:240])}]}],
        'target_binding': {'candidate_id': 'c1' if resolved else None,
            'basis': 'next_overall' if resolved else 'unresolved', 'citations': [cite(TEXT[:240])],
            'reason': 'Synthetic applicability claim; not independently verified.'},
        'criteria': [{'criterion_id': key, 'status': 'supported' if resolved else 'partial',
            'subject_candidate_ids': ['c1'], 'citations': [cite()], 'reason': 'Synthetic evidence claim.'}
            for key in CRITERIA], 'missing_information': [] if resolved else ['Target still unresolved'],
        'summary': 'Mechanism fixture, not a real experiment outcome.'}
    return body


def test_candidate_facts_do_not_become_task_support():
    result = validate_bound_review(packet(), response())
    assert result['candidates'][0]['fields']['away_team'] == 'Beta FC'
    assert result['proposed_target_key'] is None
    assert result['evidence_sufficiency'] == 'not_established'
    assert result['task_outcome_label'] is None and not result['eligible_for_value_calibration']
    anchor = result['candidates'][0]['field_evidence'][0]['citation']
    assert anchor['start'] == 10 and anchor['end'] == 250


@pytest.mark.parametrize('case', ['unbound_support', 'wrong_case', 'wrong_contract', 'unknown_referent',
    'duplicate_candidate', 'missing_field_quote', 'invented_quote', 'preview_quote', 'partial_date',
    'invalid_date', 'invalid_time', 'invalid_offset', 'blank_field', 'home_is_next', 'past_is_next',
    'wrong_team', 'unknown_timezone', 'finished_is_next', 'duplicate_criterion', 'conflict_one_quote'])
def test_invalid_binding_or_support_is_rejected(case):
    p, r = packet(), response(resolved=True)
    fields = r['candidates'][0]['fields']
    if case == 'unbound_support': r['target_binding'].update(candidate_id=None, basis='unresolved')
    elif case == 'wrong_case': r['case_id'] = 'another'
    elif case == 'wrong_contract': r['task_contract_digest'] = 'another'
    elif case == 'unknown_referent': r['criteria'][0]['subject_candidate_ids'] = ['c2']
    elif case == 'duplicate_candidate': r['candidates'].append(deepcopy(r['candidates'][0]))
    elif case == 'missing_field_quote': r['candidates'][0]['field_evidence'][0]['fields'].remove('competition')
    elif case == 'invented_quote': r['candidates'][0]['field_evidence'][0]['citation']['quote'] = 'Invented quotation.'
    elif case == 'preview_quote': r['candidates'][0]['field_evidence'][0]['citation'].update(window_id='w2', quote='Unopened preview candidate only.')
    elif case == 'partial_date': fields['scheduled_date'] = '10-02'
    elif case == 'invalid_date': fields['scheduled_date'] = '2026-02-30'
    elif case == 'invalid_time': fields['local_time'] = '25:00'
    elif case == 'invalid_offset': fields['utc_offset'] = '+14:01'
    elif case == 'blank_field': fields['competition'] = ' '
    elif case == 'home_is_next': fields['schedule_scope'] = 'next_home'
    elif case == 'past_is_next': fields['scheduled_date'] = '2026-09-27'
    elif case == 'wrong_team': fields['home_team'] = 'Other FC'
    elif case == 'unknown_timezone':
        fields['utc_offset'] = None
        r['candidates'][0]['field_evidence'][0]['fields'].remove('utc_offset')
    elif case == 'finished_is_next': fields['match_status'] = 'finished'
    elif case == 'duplicate_criterion': r['criteria'][1] = deepcopy(r['criteria'][0])
    elif case == 'conflict_one_quote': r['criteria'][0]['status'] = 'conflicting'
    with pytest.raises(ValueError): validate_bound_review(p, r)


def test_ambiguous_repeat_quote_is_rejected():
    p, r = packet(), response()
    p['documents'][0]['body_windows'][0]['text'] += TEXT
    with pytest.raises(ValueError, match='Ambiguous repeated'):
        validate_bound_review(p, r)


def test_same_proposed_fixture_is_comparable_but_not_truth_or_reward():
    p, r = packet(), response(resolved=True)
    before = validate_bound_review(p, r)
    r['criteria'][-1]['status'] = 'partial'
    after = validate_bound_review(p, r)
    compared = compare_bound_reviews(before, after)
    assert compared['alignment'] == 'same_proposed_fixture'
    assert compared['criterion_statuses_changed'] == ['next_fixture_scope']
    assert compared['loss_delta'] is None and not compared['eligible_for_value_calibration']


@pytest.mark.parametrize('case', ['unresolved', 'other_fixture', 'different_kickoff', 'different_reference', 'different_scope'])
def test_noncomparable_labels_never_produce_status_delta(case):
    p, r = packet(), response(resolved=True)
    before = validate_bound_review(p, r)
    if case == 'unresolved': r = response()
    elif case == 'other_fixture': r['candidates'][0]['fields']['away_team'] = 'Other FC'
    elif case == 'different_kickoff': r['candidates'][0]['fields']['local_time'] = '20:00'
    elif case == 'different_reference':
        p['task_contract']['reference_time'] = (NOW - timedelta(days=1)).isoformat()
        from football_agent.audit import digest
        p['task_contract_digest'] = r['task_contract_digest'] = digest(p['task_contract'])
    elif case == 'different_scope':
        p['task_contract']['fixture_scope'] = 'home_only'
        from football_agent.audit import digest
        p['task_contract_digest'] = r['task_contract_digest'] = digest(p['task_contract'])
    after = validate_bound_review(p, r)
    compared = compare_bound_reviews(before, after)
    assert not compared['structurally_comparable']
    assert compared['criterion_statuses_changed'] is compared['loss_delta'] is None


def test_subject_alias_is_explicit_and_other_names_are_not_fuzzy_matched():
    p, r = packet(), response(resolved=True)
    before = validate_bound_review(p, r)
    r['candidates'][0]['fields']['home_team'] = 'Alpha'
    assert compare_bound_reviews(before, validate_bound_review(p, r))['structurally_comparable']
    r['candidates'][0]['fields']['away_team'] = 'Beta'
    assert not compare_bound_reviews(before, validate_bound_review(p, r))['structurally_comparable']


def test_bound_export_preserves_actor_and_censors_later_reviews():
    p, r = packet(), response()
    review = validate_bound_review(p, r)
    row = {'before_event_hash': 'before', 'after_event_hash': 'after', 'risk_after': None,
        'task_outcome_label': None, 'selection_probability': None, 'eligible_for_value_calibration': False}
    proposals = [{'case_id': 'case1', 'prefix_event_hash': event, 'evidence_cutoff': NOW.isoformat(),
        'available_at': (NOW + timedelta(minutes=1)).isoformat(), 'review': review}
        for event in ('before', 'after')]
    early = attach_fixture_reviews([row], proposals, as_of=NOW)[0]
    assert early['evaluator_review']['before'] is early['evaluator_review']['target_comparison'] is None
    late = attach_fixture_reviews([row], proposals, as_of=NOW + timedelta(minutes=2))[0]
    assert late['evaluator_review']['target_comparison']['alignment'] == 'target_unresolved'
    assert late['evaluator_review']['criterion_statuses_changed'] is None
    assert {k:v for k,v in late.items() if k != 'evaluator_review'} == row


class Provider:
    def __init__(self, reply=None, fail=False): self.reply, self.fail, self.calls = reply, fail, 0
    def configuration(self): return {'provider': 'synthetic-test'}
    def decide(self, packet):
        self.calls += 1
        if self.fail: raise TimeoutError('synthetic unknown return')
        return {'response': deepcopy(self.reply), 'receipt': {'real_provider_call': False}}


@pytest.mark.parametrize('fail', [False, True])
def test_attempt_reserved_and_not_repeated(tmp_path, fail):
    provider = Provider(response(), fail)
    session = BoundReviewSession(tmp_path, provider, clock=lambda: NOW)
    first = session.execute(packet(), deadline=NOW + timedelta(minutes=3))
    second = session.execute(packet(), deadline=NOW + timedelta(minutes=3))
    assert first == second and provider.calls == 1 and first['attempts'] == 1
    assert first['status'] == ('RETURN_UNKNOWN' if fail else 'REVIEWED_PROVISIONAL')
    assert (first['pending'] is not None) == fail
    changed = packet(); changed['documents'] = []
    with pytest.raises(ValueError): session.execute(changed, deadline=NOW + timedelta(minutes=3))


def test_deadline_and_invalid_return_do_not_create_review(tmp_path):
    provider = Provider(response())
    assert BoundReviewSession(tmp_path/'expired', provider, clock=lambda: NOW).execute(packet(), deadline=NOW)['attempts'] == 0
    assert provider.calls == 0
    times = iter([NOW, NOW + timedelta(minutes=4)])
    late = BoundReviewSession(tmp_path/'late', provider, clock=lambda: next(times)).execute(packet(), deadline=NOW + timedelta(minutes=3))
    assert late['status'] == 'LATE_RETURN_EXCLUDED' and late['review'] is None
    invalid = response(); invalid['criteria'][0]['status'] = 'supported'
    bad = BoundReviewSession(tmp_path/'invalid', Provider(invalid), clock=lambda: NOW).execute(packet(), deadline=NOW + timedelta(minutes=3))
    assert bad['status'] == 'INVALID_REVIEW' and bad['review'] is None


def test_modified_base_task_or_future_document_is_rejected():
    p = packet()
    base = {'task': {'question': p['task_contract']['question'], 'reference_time': p['task_contract']['reference_time']},
        'evidence_cutoff': p['evidence_cutoff'], 'documents': p['documents']}
    base['task']['question'] = 'Different objective'
    with pytest.raises(ValueError): build_bound_packet(base, p['task_contract'], case_id='other')
    base['task']['question'] = p['task_contract']['question']
    base['documents'][0]['available_at'] = (NOW + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError): build_bound_packet(base, p['task_contract'], case_id='other')
