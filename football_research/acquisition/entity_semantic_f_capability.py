"""Offline capability audit for the entity-semantic f.

This module does not select, retrain, or promote a predictor.  It measures
whether a frozen f can transmit newly revealed semantic features into a
probability change and whether the associated hindsight targets contain any
variation for a later g fit.  The selected f and a diagnostic candidate can
therefore be compared without silently changing the model-selection protocol.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


VERSION = 'entity-semantic-f-capability-audit-v1'
PREDICTION_CHANGE_THRESHOLD = 1e-12


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field} must be a finite number')
    return float(value)


def _flatten_numbers(value: Any, field: str, output: list[float]) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_numbers(item, f'{field}[{index}]', output)
        return
    output.append(_finite(value, field))


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{field} must be a JSON object')
    return value


def _summary(values: list[float]) -> dict[str, Any]:
    unique = sorted(set(values))
    return {
        'rows': len(values),
        'nonzero_rows': sum(value != 0.0 for value in values),
        'positive_rows': sum(value > 0.0 for value in values),
        'negative_rows': sum(value < 0.0 for value in values),
        'unique_count': len(unique),
        'unique_values': unique,
        'min': min(values) if values else None,
        'max': max(values) if values else None,
        'mean': sum(values) / len(values) if values else None,
    }


def audit_f_capability(
    model_artifact: Mapping[str, Any],
    *,
    post_read_audits: list[Mapping[str, Any]],
    target_rows: list[Mapping[str, Any]] | None = None,
    role: str = 'unspecified',
) -> dict[str, Any]:
    """Return a descriptive, non-promoting capability summary."""
    model = _mapping(model_artifact, 'model_artifact')
    metadata = _mapping(model.get('metadata'), 'model_artifact.metadata')
    coefficients: list[float] = []
    _flatten_numbers(model.get('coefficients'), 'model_artifact.coefficients', coefficients)
    if not coefficients:
        raise ValueError('model_artifact.coefficients cannot be empty')

    probability_changes: list[float] = []
    feature_changed = 0
    interpretations: set[str] = set()
    for index, wrapper in enumerate(post_read_audits):
        row = _mapping(wrapper, f'post_read_audits[{index}]')
        audit = _mapping(row.get('audit'), f'post_read_audits[{index}].audit')
        changed = audit.get('feature_vector_changed')
        if not isinstance(changed, bool):
            raise ValueError(f'post_read_audits[{index}].audit.feature_vector_changed must be Boolean')
        amount = _finite(audit.get('max_probability_change'),
                         f'post_read_audits[{index}].audit.max_probability_change')
        if amount < 0.0:
            raise ValueError(f'post_read_audits[{index}].audit.max_probability_change must be nonnegative')
        feature_changed += int(changed)
        probability_changes.append(amount)
        interpretation = audit.get('interpretation')
        if isinstance(interpretation, str) and interpretation:
            interpretations.add(interpretation)

    targets: list[float] = []
    target_eligible = {'true': 0, 'false': 0, 'unknown': 0}
    target_interpretations: set[str] = set()
    if target_rows is not None:
        for index, wrapper in enumerate(target_rows):
            row = _mapping(wrapper, f'target_rows[{index}]')
            target = _mapping(row.get('target'), f'target_rows[{index}].target')
            targets.append(_finite(target.get('realized_gain_nats'),
                                   f'target_rows[{index}].target.realized_gain_nats'))
            eligibility = target.get('prospective_training_eligible')
            if eligibility is True:
                target_eligible['true'] += 1
            elif eligibility is False:
                target_eligible['false'] += 1
            else:
                target_eligible['unknown'] += 1
            interpretation = target.get('representation_interpretation')
            if isinstance(interpretation, str) and interpretation:
                target_interpretations.add(interpretation)

    max_change = max(probability_changes) if probability_changes else 0.0
    prediction_changed = sum(value > PREDICTION_CHANGE_THRESHOLD for value in probability_changes)
    coefficient_max = max(abs(value) for value in coefficients)
    coefficient_nonzero = sum(abs(value) > PREDICTION_CHANGE_THRESHOLD for value in coefficients)
    target_summary = _summary(targets)

    if coefficient_nonzero == 0 and prediction_changed == 0:
        verdict = 'degenerate_prior_only'
        verdict_detail = 'f has no fitted residual support and cannot transmit query information to g'
    elif prediction_changed == 0:
        verdict = 'representation_not_transmitted'
        verdict_detail = 'coefficients exist but no audited query changed the probability output'
    elif not targets:
        verdict = 'responsive_without_realized_targets'
        verdict_detail = 'f responds in the audit, but no hindsight target file was supplied'
    elif target_summary['nonzero_rows'] == 0:
        verdict = 'responsive_but_zero_realized_gain'
        verdict_detail = 'f changes predictions, but the realized fixed-f target remains constant at zero'
    else:
        verdict = 'responsive_research_candidate'
        verdict_detail = 'f transmits some revealed information and has variable research targets'

    return {
        'schema': VERSION,
        'role': role,
        'model_version': model.get('model_version'),
        'representation_sha256': model.get('representation_sha256'),
        'coefficient_support': {
            'total_coefficients': len(coefficients),
            'nonzero_coefficients': coefficient_nonzero,
            'max_abs_coefficient': coefficient_max,
            'zero_coefficient': coefficient_nonzero == 0,
        },
        'response_support': {
            'audits': len(probability_changes),
            'feature_vector_changed_rows': feature_changed,
            'prediction_changed_rows': prediction_changed,
            'max_probability_change': max_change,
            'mean_probability_change': sum(probability_changes) / len(probability_changes)
            if probability_changes else None,
            'prediction_change_threshold': PREDICTION_CHANGE_THRESHOLD,
            'interpretations': sorted(interpretations),
        },
        'target_signal': {
            **target_summary,
            'prospective_training_eligible': target_eligible,
            'interpretations': sorted(target_interpretations),
        },
        'selection_metadata': {
            'selected_regularization': metadata.get('selection', {}).get('selected_regularization')
            if isinstance(metadata.get('selection'), Mapping) else None,
            'candidate_role': metadata.get('candidate_role'),
            'production_ready': metadata.get('production_ready'),
            'gain_model_enabled': metadata.get('gain_model_enabled'),
        },
        'verdict': verdict,
        'verdict_detail': verdict_detail,
        'promotion_allowed': False,
    }


def compare_f_capabilities(selected: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    """Make the selected/diagnostic distinction explicit for review."""
    selected = _mapping(selected, 'selected')
    diagnostic = _mapping(diagnostic, 'diagnostic')
    return {
        'schema': VERSION,
        'selected': dict(selected),
        'diagnostic': dict(diagnostic),
        'interpretation': {
            'selected_is_formal': True,
            'diagnostic_is_research_only': True,
            'diagnostic_can_not_replace_selected_without_new_selection_protocol': True,
            'selected_prediction_change_gap': diagnostic.get('response_support', {}).get('prediction_changed_rows', 0)
            - selected.get('response_support', {}).get('prediction_changed_rows', 0),
            'selected_target_variation_gap': diagnostic.get('target_signal', {}).get('nonzero_rows', 0)
            - selected.get('target_signal', {}).get('nonzero_rows', 0),
        },
    }


__all__ = ['VERSION', 'PREDICTION_CHANGE_THRESHOLD', 'audit_f_capability',
           'compare_f_capabilities']
