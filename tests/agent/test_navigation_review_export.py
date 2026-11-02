from copy import deepcopy
from datetime import timedelta

import pytest

from football_agent.navigation_review_export import attach_review_proposals
from football_agent.navigation_task_review import validate_review
from test_navigation_task_review import NOW, packet, response


def inputs():
    row = {'row_id': 'selected-only', 'before_event_hash': 'before', 'after_event_hash': 'after',
        'pre_action_context': {'known_at_selection': 'old evidence'},
        'risk_before': None, 'risk_after': None, 'task_outcome_label': None,
        'selection_probability': None, 'eligible_for_value_calibration': False}
    review = validate_review(packet(), response())
    proposal = {'case_id': 'opaque1', 'prefix_event_hash': 'after',
        'evidence_cutoff': NOW.isoformat(), 'available_at': (NOW + timedelta(minutes=1)).isoformat(),
        'review': review}
    return row, proposal


def test_later_review_never_backfills_earlier_actor_input():
    row, proposal = inputs()
    original = deepcopy(row)
    early = attach_review_proposals([row], [proposal], as_of=NOW)[0]
    late = attach_review_proposals([row], [proposal], as_of=NOW + timedelta(minutes=2))[0]
    assert early['evaluator_review']['after'] is None
    assert late['evaluator_review']['after']['review']['evidence_sufficiency'] == 'not_established'
    assert late['evaluator_review']['before'] is None
    assert late['evaluator_review']['criterion_statuses_changed'] is None
    assert {k: v for k, v in late.items() if k != 'evaluator_review'} == original == row


def test_matching_both_prefixes_does_not_create_reward_or_calibration_label():
    row, proposal = inputs()
    earlier = {**deepcopy(proposal), 'prefix_event_hash': 'before'}
    result = attach_review_proposals([row], [earlier, proposal], as_of=NOW + timedelta(minutes=2))[0]
    assert result['evaluator_review']['criterion_statuses_changed'] == []
    assert result['evaluator_review']['loss_delta'] is None
    assert result['evaluator_review']['criterion_change_is_utility_change'] is False
    assert result['evaluator_review']['cross_packet_fixture_alignment'] == 'not_independently_adjudicated'
    assert result['task_outcome_label'] is result['risk_after'] is None


@pytest.mark.parametrize('case', ['duplicate', 'identity', 'truth', 'future', 'overwrite'])
def test_invalid_annotation_join_is_rejected(case):
    row, proposal = inputs()
    proposals = [proposal]
    if case == 'duplicate': proposals.append(deepcopy(proposal))
    elif case == 'identity': proposal['case_id'] = 'other'
    elif case == 'truth': proposal['review']['world_truth_verified'] = True
    elif case == 'future': proposal['available_at'] = (NOW - timedelta(seconds=1)).isoformat()
    elif case == 'overwrite': row['evaluator_review'] = {}
    with pytest.raises(ValueError): attach_review_proposals([row], proposals, as_of=NOW + timedelta(minutes=2))
