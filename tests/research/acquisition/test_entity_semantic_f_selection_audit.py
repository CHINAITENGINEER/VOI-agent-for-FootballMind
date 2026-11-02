"""Tests for the non-promoting f capability-constrained selection audit."""

import pytest

from football_research.acquisition.entity_semantic_f_selection_audit import (
    capability_constrained_selection,
    summarize_response_rows,
)


def _row(mid, changed, amount):
    return {'physical_match_id': mid, 'audit': {
        'feature_vector_changed': changed,
        'max_probability_change': amount,
    }}


def test_response_summary_reports_transition_and_match_denominators():
    report = summarize_response_rows([
        _row('a', True, .02), _row('a', False, 0.), _row('b', True, .01),
    ])
    assert report['transition_rows'] == 3
    assert report['physical_matches'] == 2
    assert report['feature_vector_changed_rows'] == 2
    assert report['prediction_changed_rows'] == 2
    assert report['responsive_physical_matches'] == 2
    assert report['prediction_change_rate'] == pytest.approx(2 / 3)


def test_capability_constraint_excludes_prior_only_but_keeps_supported_candidate():
    result = capability_constrained_selection([
        {'regularization': None, 'mean_match_log_loss': .95,
         'within_one_paired_se_of_best': True,
         'response_scope': 'A_purged_forward_validation',
         'forward_response': {'prediction_changed_rows': 0, 'responsive_physical_matches': 0,
                              'transition_rows': 20, 'physical_matches': 3}},
        {'regularization': 10., 'mean_match_log_loss': .951,
         'within_one_paired_se_of_best': True,
         'response_scope': 'A_purged_forward_validation',
         'forward_response': {'prediction_changed_rows': 12, 'responsive_physical_matches': 2,
                              'transition_rows': 20, 'physical_matches': 3}},
        {'regularization': 1., 'mean_match_log_loss': .949,
         'within_one_paired_se_of_best': False,
         'response_scope': 'A_purged_forward_validation',
         'forward_response': {'prediction_changed_rows': 20, 'responsive_physical_matches': 3,
                              'transition_rows': 20, 'physical_matches': 3}},
    ])
    assert result['rule'].startswith('within_one_paired_se')
    assert result['proposed_candidate_regularization'] == 10.
    assert result['capability_feasible_count'] == 1
    assert result['promotion_allowed'] is False


def test_capability_constraint_reports_no_candidate_when_response_is_zero():
    result = capability_constrained_selection([
        {'regularization': None, 'mean_match_log_loss': .95,
         'within_one_paired_se_of_best': True,
         'response_scope': 'A_purged_forward_validation',
         'forward_response': {'prediction_changed_rows': 0, 'responsive_physical_matches': 0,
                              'transition_rows': 10, 'physical_matches': 3}},
    ])
    assert result['proposed_candidate_regularization'] is None
    assert result['no_feasible_reason']
