"""Tests for the non-promoting entity f capability audit."""

import json

from football_research.acquisition.entity_semantic_f_capability import audit_f_capability, compare_f_capabilities


def _model(coefficients, *, role=None):
    return {
        'model_version': 'f:test',
        'representation_sha256': 'r' * 64,
        'coefficients': coefficients,
        'metadata': {
            'selection': {'selected_regularization': None},
            'candidate_role': role,
            'production_ready': False,
            'gain_model_enabled': False,
        },
    }


def _audit(changed, amount, interpretation='predictor_insensitive_not_information_irrelevant'):
    return {'sample_id': 's', 'audit': {
        'feature_vector_changed': changed,
        'max_probability_change': amount,
        'interpretation': interpretation,
    }}


def _target(gain):
    return {'sample_id': 's', 'target': {
        'realized_gain_nats': gain,
        'prospective_training_eligible': False,
        'representation_interpretation': 'predictor_insensitive_not_information_irrelevant',
    }}


def test_prior_only_model_is_identified_as_degenerate():
    report = audit_f_capability(_model([[0.0, 0.0, 0.0]]), post_read_audits=[_audit(True, 0.0)],
                                target_rows=[_target(0.0)], role='formal')
    assert report['verdict'] == 'degenerate_prior_only'
    assert report['coefficient_support']['zero_coefficient'] is True
    assert report['response_support']['prediction_changed_rows'] == 0
    assert report['promotion_allowed'] is False


def test_responsive_candidate_with_variable_targets_is_reported_without_promotion():
    report = audit_f_capability(
        _model([[.2, -.1, .0]], role='diagnostic_only'),
        post_read_audits=[_audit(True, .02), _audit(True, .01)],
        target_rows=[_target(.1), _target(-.02)], role='diagnostic',
    )
    assert report['verdict'] == 'responsive_research_candidate'
    assert report['response_support']['prediction_changed_rows'] == 2
    assert report['target_signal']['nonzero_rows'] == 2
    assert report['target_signal']['negative_rows'] == 1
    assert report['promotion_allowed'] is False


def test_comparison_preserves_formal_diagnostic_boundary():
    selected = audit_f_capability(_model([[0.0, 0.0, 0.0]]), post_read_audits=[_audit(True, 0.0)],
                                  target_rows=[_target(0.0)], role='formal')
    diagnostic = audit_f_capability(_model([[.2, -.1, 0.0]], role='diagnostic_only'),
                                    post_read_audits=[_audit(True, .02)], target_rows=[_target(.1)],
                                    role='diagnostic')
    comparison = compare_f_capabilities(selected, diagnostic)
    assert comparison['interpretation']['selected_is_formal'] is True
    assert comparison['interpretation']['diagnostic_is_research_only'] is True
    assert comparison['interpretation']['diagnostic_can_not_replace_selected_without_new_selection_protocol'] is True
    assert comparison['interpretation']['selected_prediction_change_gap'] == 1
