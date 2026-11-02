from copy import deepcopy
from datetime import timedelta

import pytest

from football_agent.question_return_archive import project_bound_run
from football_agent.question_return_outcomes import (
    project_return, visible_outcomes, summarize_outcomes, blinded_review_packet,
)
from test_execution_bound_navigation import make_bound, bound_reply, raw_choice, read_new
from test_target_navigation import query, finish


def make_episode(tmp_path, monkeypatch, steps=3):
    replies = [lambda p: bound_reply(p, raw_choice(query, p)),
               lambda p: bound_reply(p, raw_choice(read_new, p)),
               lambda p: bound_reply(p, raw_choice(finish, p))]
    flow, rid, _ = make_bound(tmp_path, monkeypatch, replies)
    initial_time = flow.clock()
    for index in range(steps):
        flow.clock = lambda index=index: initial_time + timedelta(seconds=index + 1)
        flow.step(rid)
    run = flow.get(rid)
    return flow, rid, run


@pytest.fixture
def episode(tmp_path, monkeypatch):
    return make_episode(tmp_path, monkeypatch)


def archive(episode):
    flow, rid, run = episode
    return project_bound_run(flow.store.root.parent, rid, expected_head=run['events'][-1]['event_hash'])


def arguments(episode, index=1):
    flow, rid, run = episode
    decisions = [event for event in run['events'] if event['type'] == 'navigation_decision']
    before, after = decisions[index:index + 2]
    public = flow.store.get_blob(rid, after['data']['input_ref'])
    document = flow.store.get_blob(rid, run['documents'][public['last_action_result']['document_id']])
    return dict(run_id=rid, acquisition_event_hash=before['event_hash'],
        acquisition_choice=deepcopy(before['data']['decision']['choice']), acquisition_at=before['at'],
        assessment_event_hash=after['event_hash'], response=deepcopy(after['data']['decision']['choice']),
        public=deepcopy(public), assessed_at=after['at'], document=deepcopy(document),
        opened_before=deepcopy(after['data']['decision']['opened_before']),
        window_chars=run['configuration']['window_chars'], preview_chars=run['configuration']['preview_chars'])


def test_bound_episode_separates_fetch_from_body_and_never_invents_gain(episode):
    result = archive(episode)
    assert result['verified_model_returns'] == 3
    assert result['unassessed_actions'] == []
    assert [row['actor_outcome'] for row in result['records']] == ['unread', 'partial']
    summary = summarize_outcomes(result['records'], as_of=result['records'][-1]['assessed_at'])
    assert summary['assessed_acquisition_actions'] == 2
    assert summary['assessed_source_requests'] == summary['source_requests_with_ok_return'] == 1
    assert summary['assessed_body_reads'] == 1
    assert summary['independent_semantic_label_count'] == 0
    assert summary['conditional_semantic_accuracy'] is None
    assert summary['numeric_voi'] is None
    assert all(row['execution_binding_verified'] and row['realized_decision_gain'] is None for row in result['records'])


def test_successful_transport_does_not_make_unread_metadata_an_answer(episode):
    args = arguments(episode, 0)
    args['response']['return_assessment']['outcome'] = 'resolved'
    with pytest.raises(ValueError, match='unread metadata'):
        project_return(**args)


def test_failed_request_is_missing_information_not_semantic_negative(episode):
    args = arguments(episode, 0)
    args['document']['status'] = 'FAILED'
    args['public']['last_action_result']['status'] = 'FAILED'
    args['response']['return_assessment'].update(outcome='failed', evidence_claim_indices=[])
    row = project_return(**args)
    assert row['actor_outcome'] == 'failed' and row['visible_body_window'] is None
    assert row['independent_semantic_label'] is None
    assert blinded_review_packet(row) is None
    args['response']['return_assessment']['outcome'] = 'irrelevant'
    with pytest.raises(ValueError, match='failed request'):
        project_return(**args)


def test_same_requirement_with_rewritten_useful_condition_is_not_same_test(episode):
    args = arguments(episode)
    args['public']['last_action_result']['information_test']['useful_if'] = 'A different question after seeing the outcome.'
    with pytest.raises(ValueError, match='exact prospective'):
        project_return(**args)


@pytest.mark.parametrize('field', ['requirement', 'document_id'])
def test_assessment_stays_bound_to_its_original_question_and_document(episode, field):
    args = arguments(episode)
    args['response']['return_assessment'][field] = 'match_status' if field == 'requirement' else 'other-document'
    with pytest.raises(ValueError):
        project_return(**args)


def test_claim_from_other_window_does_not_support_current_return(episode):
    args = arguments(episode)
    idx = args['response']['return_assessment']['evidence_claim_indices'][0]
    args['response']['claims'][idx]['window_id'] = 'other-window'
    with pytest.raises(ValueError, match='another returned window'):
        project_return(**args)


def test_claim_must_match_exact_exposed_text(episode):
    args = arguments(episode)
    idx = args['response']['return_assessment']['evidence_claim_indices'][0]
    args['response']['claims'][idx]['quote'] = 'Invented body statement not present in the retrieved source.'
    with pytest.raises(ValueError, match='exact'):
        project_return(**args)


def test_opened_summary_does_not_replace_captured_model_exposure(episode):
    args = arguments(episode)
    for doc in args['public']['documents']:
        if doc['document_id'] == args['document']['document_id']:
            doc['visible_windows'] = []
    with pytest.raises(ValueError, match='captured actor input'):
        project_return(**args)


def test_unopened_body_is_not_accepted(episode):
    args = arguments(episode)
    args['opened_before'][args['document']['document_id']] = []
    with pytest.raises(ValueError, match='not exposed'):
        project_return(**args)


def test_body_judgment_needs_evidence_and_resolution_must_close_its_requirement(episode):
    args = arguments(episode)
    args['response']['return_assessment']['evidence_claim_indices'] = []
    with pytest.raises(ValueError, match='cited opened evidence'):
        project_return(**args)
    args = arguments(episode)
    args['response']['return_assessment']['outcome'] = 'resolved'
    with pytest.raises(ValueError, match='resolved flag'):
        project_return(**args)


def test_actor_resolved_flag_is_not_independent_semantic_truth(episode):
    args = arguments(episode)
    args['response']['return_assessment']['outcome'] = 'resolved'
    requirement = args['response']['return_assessment']['requirement']
    args['response']['target_state']['open_requirements'].remove(requirement)
    row = project_return(**args)
    assert row['actor_outcome'] == 'resolved'
    assert row['semantic_entailment_verified'] is False
    assert row['world_truth_verified'] is False
    assert row['numeric_voi_training_eligible'] is False


def test_blind_review_preserves_full_visible_window_but_hides_actor_verdict(episode):
    row = archive(episode)['records'][-1]
    packet = blinded_review_packet(row)
    assert packet['payload']['visible_body_window'] == row['visible_body_window']
    assert packet['payload']['information_test'] == row['information_test']
    assert packet['verdict'] is None and packet['review_status'] == 'PENDING_INDEPENDENT_REVIEW'
    assert packet['actor_verdict_exposed'] is False
    assert not {'actor_outcome', 'actor_reason', 'evidence_anchors', 'assessment_request_id'} & packet['payload'].keys()
    assert packet['full_source_coverage_established'] is False


def test_metadata_and_failed_requests_do_not_enter_blind_semantic_review(episode):
    row = archive(episode)['records'][0]
    assert row['document']['status'] == 'OK'
    assert blinded_review_packet(row) is None


def test_point_in_time_filter_excludes_future_actor_judgments(episode):
    rows = archive(episode)['records']
    cutoff = rows[0]['assessed_at']
    assert visible_outcomes(rows, as_of=cutoff) == [rows[0]]
    future_conflict = deepcopy(rows[1])
    future_conflict['actor_reason'] = 'Future replacement must stay invisible.'
    assert visible_outcomes(rows + [future_conflict], as_of=cutoff) == [rows[0]]
    summary = summarize_outcomes(rows, as_of=cutoff)
    assert summary['assessed_body_reads'] == 0


def test_duplicate_projection_does_not_multiply_actions_or_samples(episode):
    rows = archive(episode)['records']
    cutoff = rows[-1]['assessed_at']
    assert summarize_outcomes(rows + rows, as_of=cutoff) == summarize_outcomes(rows, as_of=cutoff)
    corrupted = deepcopy(rows[-1])
    corrupted['actor_reason'] = 'Changed record under same identity'
    with pytest.raises(ValueError, match='conflicting records'):
        visible_outcomes(rows + [corrupted], as_of=cutoff)


def test_same_timestamp_includes_all_observations_at_that_cutoff(episode):
    rows = archive(episode)['records']
    rows[1]['assessed_at'] = rows[0]['assessed_at']
    visible = visible_outcomes(rows, as_of=rows[0]['assessed_at'])
    assert {row['record_id'] for row in visible} == {row['record_id'] for row in rows}


def test_projection_returns_detached_data(episode):
    rows = archive(episode)['records']
    copied = visible_outcomes(rows, as_of=rows[-1]['assessed_at'])
    copied[-1]['information_test']['useful_if'] = 'mutated'
    assert rows[-1]['information_test']['useful_if'] != 'mutated'


def test_future_document_is_not_used_in_earlier_assessment(episode):
    args = arguments(episode)
    args['document']['available_at'] = '2099-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='predates'):
        project_return(**args)


def test_older_document_remains_distinct_from_current_request_time(episode):
    args = arguments(episode, 0)
    args['document']['available_at'] = '2020-01-01T00:00:00Z'
    row = project_return(**args)
    assert row['document']['available_at'] == '2020-01-01T00:00:00Z'
    assert row['selected_at'] != row['document']['available_at']
    assert row['actor_outcome'] == 'unread'


def test_archive_head_must_match_pinned_input(episode):
    flow, rid, _ = episode
    with pytest.raises(ValueError, match='pinned episode'):
        project_bound_run(flow.store.root.parent, rid, expected_head='wrong-head')


def test_returned_but_not_reconsidered_action_stays_unassessed(tmp_path, monkeypatch):
    episode = make_episode(tmp_path, monkeypatch, steps=1)
    result = archive(episode)
    assert result['records'] == []
    assert result['source_attempts_recorded_in_episode'] == 1
    assert result['unassessed_actions'][0]['status'] == 'RETURNED_NOT_ASSESSED'


def test_run_identity_is_not_inferred_from_quote_similarity(episode):
    args = arguments(episode)
    args['run_id'] = 'another-run'
    with pytest.raises(ValueError, match='another run'):
        project_return(**args)
