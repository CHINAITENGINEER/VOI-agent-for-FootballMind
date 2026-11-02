"""Audit a capability-constrained selection rule for the entity-semantic f.

The formal f in ``f-entity-semantic-20261011-v2`` was selected by a
purged-forward one-standard-error rule.  That rule is statistically simple,
but it can select a prior-only model which cannot transmit a newly revealed
document into a probability update.  This module measures that failure mode
on the A training period only and reports a *research proposal* for a
capability constraint.  It never reads B/C outcomes, fits g, changes the
formal model, or promotes a candidate.

The proposed constraint is deliberately explicit:

``within_one_paired_se_of_best AND has_nonzero_A_forward_response``

Among candidates that satisfy it, the lowest A-only validation loss is shown
as a proposed candidate.  The rule is labelled retrospective until it is
registered before a future experiment. A outcomes are used for fitting and
purged-forward loss evaluation. Each fold's response probe excludes its
validation matches from fitting; it does not use their outcome labels. This
is numerical capability, not evidence that changing a probability is useful.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .context_predictor import _digest
from .entity_semantic_f import (
    EntitySemanticF,
    PARAMETERS,
    REPRESENTATION_SHA256,
    VERSION as F_VERSION,
    _fit,
    _predict,
    vector,
)
from .historical_context_training import _discovery_panel, _state_for
from .state import parse_time
from . import entity_semantic_query as entity_query


VERSION = 'entity-semantic-f-capability-constrained-selection-audit-v2'
PREDICTION_CHANGE_THRESHOLD = 1e-12
SELECTION_RULE = 'within_one_paired_se_of_best_and_nonzero_A_forward_response'
SENSITIVITY_THRESHOLDS = (1e-12, 1e-4, 1e-3, 1e-2)


def selection_protocol(registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Serialize the fixed rule before running an audit.

    Without ``registration`` this is the historical, retrospective protocol
    used by Round125.  A future run may attach an independently created
    registration; this function only records that binding and never creates
    one implicitly.
    """
    result = {
        'schema': VERSION, 'rule': SELECTION_RULE,
        'candidate_grid': list(PARAMETERS['regularizations']),
        'response_threshold': PREDICTION_CHANGE_THRESHOLD,
        'minimum_responsive_validation_matches': 1,
        'response_scope': 'A_purged_forward_validation_matches_only',
        'tie_break': 'lowest_A_mean_log_loss_then_larger_regularization',
        'sensitivity_thresholds': list(SENSITIVITY_THRESHOLDS),
        'sensitivity_used_for_selection': False,
        'registration_status': 'retrospective_audit_proposal_not_preregistered',
        'independent_blind_test': False, 'promotion_allowed': False,
        'gain_or_harm_probability_established': False,
        'interpretation': 'numerical_nondegeneracy_not_prediction_or_information_quality',
    }
    if registration is not None:
        registration_id = registration.get('registration_id')
        registration_sha256 = _digest(registration)
        if not isinstance(registration_id, str) or not registration_id:
            raise ValueError('Registered audit requires a registration_id')
        result.update({
            'registration_status': 'pre_registered_for_future_A_batch',
            'registration_id': registration_id,
            'registration_sha256': registration_sha256,
        })
    return result


def _request(case: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Build the same declared read action used by the frozen bridge."""
    return {
        'action_id': 'article-' + str(index),
        'tool': 'read_url',
        'url': case['documents'][index]['url'],
        'cost': {'money': 0.0, 'requests': 1.0},
        'cost_status': 'simulated',
    }


def _candidate_model(template: EntitySemanticF, coefficients: Any,
                     regularization: float | None, training_rows: Sequence[Mapping[str, Any]],
                     optimizer: Mapping[str, Any], *, scope: str,
                     registration: Mapping[str, Any] | None = None) -> EntitySemanticF:
    """Create an audit-only model without changing the formal model artifact."""
    metadata = {
        'candidate_role': 'selection_capability_audit_only',
        'applied_regularization': regularization,
        'reference_formal_model_version': template.model_version,
        'selected_deployment_regularization': template.metadata.get('selection', {}).get(
            'selected_regularization'),
        'training_match_ids': sorted({r['physical_match_id'] for r in training_rows}),
        'training_rows_sha256': _digest(training_rows),
        'training_states': len(training_rows),
        'optimizer': deepcopy(optimizer),
        'representation_sha256': REPRESENTATION_SHA256,
        'evidence_origin': 'historical_reconstruction',
        'fit_execution_at': None,
        'fit_clock_note': 'actual_execution_recorded_in_audit_receipt_not_model_identity',
        'scope': scope,
        'production_ready': False,
        'gain_model_enabled': False,
        'audit_protocol': selection_protocol(registration),
    }
    return EntitySemanticF(np.asarray(coefficients).tolist(), metadata)


def build_a_only_response_rows(cases: Sequence[Mapping[str, Any]], snapshots: Mapping[str, Any],
                               bank: Any, model: EntitySemanticF,
                               a_match_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Replay every one-step reveal transition in A without using outcomes.

    ``case['outcome']`` is intentionally never read here.  It remains in the
    immutable case file because the same file is used by the historical
    training artifact, but this function is an outcome-free capability probe.
    """
    wanted = set(a_match_ids)
    observed = [case['fixture']['physical_match_id'] for case in cases]
    if len(wanted) != len(a_match_ids) or len(observed) != len(set(observed)) or set(observed) != wanted:
        raise ValueError('Response cases must contain the exact unique requested A subset')
    rows: list[dict[str, Any]] = []
    for case in cases:
        mid = case['fixture']['physical_match_id']
        for discovered_mask, acquired_mask, discovered, acquired, before in _discovery_panel(case):
            for index in discovered:
                if index in acquired:
                    continue
                sample_id = f'{mid}:{discovered_mask}:{acquired_mask}:{index}'
                query = entity_query.query_input(
                    before, _request(case, index), model=model,
                    prior_snapshot=snapshots[mid], evidence_bank=bank,
                )
                after = _state_for(
                    case, acquired_indices=acquired + [index],
                    discovered_indices=discovered,
                )
                audit = entity_query.response_audit(
                    query, after, model=model,
                    prior_snapshot=snapshots[mid], evidence_bank=bank,
                )
                rows.append({
                    'sample_id': sample_id,
                    'physical_match_id': mid,
                    'discovered_mask': discovered_mask,
                    'acquired_mask': acquired_mask,
                    'revealed_index': index,
                    'audit': audit,
                })
    return rows


def summarize_response_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize response support with both transition and match denominators."""
    changes: list[float] = []
    feature_changed = 0
    by_match: dict[str, dict[str, Any]] = {}
    seen_samples = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get('audit'), Mapping):
            raise ValueError(f'rows[{index}] must contain an audit object')
        audit = row['audit']
        value = audit.get('max_probability_change')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'rows[{index}].audit.max_probability_change must be finite in [0, 1]')
        changed = audit.get('feature_vector_changed')
        if not isinstance(changed, bool):
            raise ValueError(f'rows[{index}].audit.feature_vector_changed must be Boolean')
        mid = row.get('physical_match_id')
        if not isinstance(mid, str) or not mid:
            raise ValueError(f'rows[{index}].physical_match_id is required')
        sample_id = row.get('sample_id')
        if sample_id is not None:
            if sample_id in seen_samples:
                raise ValueError('Repeated response sample would inflate support')
            seen_samples.add(sample_id)
        if value > PREDICTION_CHANGE_THRESHOLD and not changed:
            raise ValueError('Same fixed-f feature vector cannot change probabilities')
        changes.append(float(value))
        feature_changed += int(changed)
        cell = by_match.setdefault(mid, {'transitions': 0, 'prediction_changed_rows': 0,
                                       'feature_changed_rows': 0, 'probability_changes': []})
        cell['transitions'] += 1
        cell['prediction_changed_rows'] += int(value > PREDICTION_CHANGE_THRESHOLD)
        cell['feature_changed_rows'] += int(changed)
        cell['probability_changes'].append(float(value))
    changed_rows = sum(value > PREDICTION_CHANGE_THRESHOLD for value in changes)
    responsive_matches = sum(cell['prediction_changed_rows'] > 0 for cell in by_match.values())
    return {
        'transition_rows': len(changes),
        'physical_matches': len(by_match),
        'feature_vector_changed_rows': feature_changed,
        'prediction_changed_rows': changed_rows,
        'prediction_change_rate': changed_rows / len(changes) if changes else None,
        'responsive_physical_matches': responsive_matches,
        'responsive_match_rate': responsive_matches / len(by_match) if by_match else None,
        'max_probability_change': max(changes) if changes else 0.0,
        'mean_probability_change': sum(changes) / len(changes) if changes else None,
        'equal_match_response_rate': float(np.mean([
            c['prediction_changed_rows'] / c['transitions'] for c in by_match.values()
        ])) if by_match else None,
        'equal_match_mean_probability_change': float(np.mean([
            np.mean(c['probability_changes']) for c in by_match.values()
        ])) if by_match else None,
        'threshold_sensitivity': [{
            'threshold': threshold,
            'prediction_changed_rows': sum(v > threshold for v in changes),
            'responsive_physical_matches': sum(any(v > threshold for v in c['probability_changes'])
                                              for c in by_match.values()),
        } for threshold in SENSITIVITY_THRESHOLDS],
        'prediction_change_threshold': PREDICTION_CHANGE_THRESHOLD,
        'per_match': {
            key: {k: v for k, v in by_match[key].items() if k != 'probability_changes'}
            for key in sorted(by_match)
        },
    }


def _candidate_validation_metadata(template: EntitySemanticF) -> list[dict[str, Any]]:
    selection = template.metadata.get('selection')
    if not isinstance(selection, Mapping) or not isinstance(selection.get('candidates'), list):
        raise ValueError('Formal f selection metadata is required')
    values = selection['candidates']
    expected = list(PARAMETERS['regularizations'])
    if any(not isinstance(row, Mapping) for row in values):
        raise ValueError('Formal candidate entries must be objects')
    actual = [row.get('regularization') for row in values]
    if actual != expected:
        raise ValueError('Formal candidate grid differs from the frozen entity-f protocol')
    result = []
    for index, row in enumerate(values):
        if not isinstance(row, Mapping):
            raise ValueError(f'selection.candidates[{index}] must be an object')
        mean = row.get('mean_match_log_loss')
        se = row.get('paired_standard_error_to_best')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (mean, se)) or se < 0:
            raise ValueError(f'selection.candidates[{index}] has invalid validation scores')
        result.append({'regularization': row.get('regularization'),
                       'mean_match_log_loss': float(mean),
                       'paired_standard_error_to_best': float(se)})
    best = min(row['mean_match_log_loss'] for row in result)
    for row in result:
        row['within_one_paired_se_of_best'] = bool(
            row['mean_match_log_loss'] <= best + row['paired_standard_error_to_best'] + 1e-12)
    return result


def capability_constrained_selection(candidates: Sequence[Mapping[str, Any]], *,
                                     registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply the proposed research rule to already measured candidates.

    A candidate must be within the original one-paired-SE region and have a
    nonzero A-only forward validation response. This function does not claim the rule was
    registered before the current corpus was inspected.
    """
    normalized = []
    seen_regularizations = set()
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise ValueError(f'candidates[{index}] must be an object')
        regularization = row.get('regularization')
        if regularization is not None and (isinstance(regularization, bool)
                                           or not isinstance(regularization, (int, float))
                                           or not math.isfinite(regularization) or regularization <= 0):
            raise ValueError(f'candidates[{index}].regularization is invalid')
        if regularization in seen_regularizations:
            raise ValueError('Duplicate candidate regularization')
        seen_regularizations.add(regularization)
        mean = row.get('mean_match_log_loss')
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean):
            raise ValueError(f'candidates[{index}].mean_match_log_loss is invalid')
        within = row.get('within_one_paired_se_of_best')
        if type(within) is not bool:
            raise ValueError('Validation eligibility must be Boolean')
        if row.get('response_scope') != 'A_purged_forward_validation':
            raise ValueError('Selection requires A purged-forward response, not fitted-training response')
        response = row.get('forward_response')
        if not isinstance(response, Mapping):
            raise ValueError(f'candidates[{index}].forward_response is required')
        changed = response.get('prediction_changed_rows')
        match_count = response.get('responsive_physical_matches')
        total = response.get('transition_rows')
        matches = response.get('physical_matches')
        if (any(type(v) is not int or v < 0 for v in (changed, match_count, total, matches))
                or not match_count <= changed <= total or not match_count <= matches <= total
                or bool(changed) != bool(match_count)):
            raise ValueError('Invalid forward response support counts')
        item = dict(row)
        item['nondegenerate_A_forward_response'] = bool(changed > 0 and match_count > 0)
        item['capability_feasible'] = bool(
            within and item['nondegenerate_A_forward_response'])
        normalized.append(item)
    feasible = [row for row in normalized if row['capability_feasible']]
    # Larger positive lambda is the simpler residual candidate.  None is the
    # prior-only candidate and is already excluded by nondegeneracy.
    def key(row: Mapping[str, Any]) -> tuple[float, float]:
        lam = row.get('regularization')
        simplicity = -float(lam) if lam is not None else float('inf')
        return (float(row['mean_match_log_loss']), simplicity)
    proposed = min(feasible, key=key) if feasible else None
    result = {
        'rule': SELECTION_RULE,
        'registration_status': ('pre_registered_for_future_A_batch' if registration is not None
                                else 'retrospective_audit_proposal_not_preregistered'),
        'candidate_count': len(normalized),
        'capability_feasible_count': len(feasible),
        'proposed_candidate_regularization': proposed.get('regularization') if proposed else None,
        'proposed_candidate_model_version': proposed.get('model_version') if proposed else None,
        'no_feasible_reason': None if proposed else 'no_candidate_has_both_validation_support_and_A_response',
        'promotion_allowed': False,
        'candidates': normalized,
    }
    if registration is not None:
        result.update({'registration_id': registration['registration_id'],
                       'registration_sha256': _digest(registration)})
    return result


def _validated_folds(training_rows, template, cases, snapshots, bank, a_match_ids):
    """Reject split mixing and reproduce the original forward/purge plan."""
    if template.model_version != F_VERSION + ':' + _digest(template.payload())[:24]:
        raise ValueError('Frozen formal f was modified')
    if template.metadata.get('candidate_role'):
        raise ValueError('A formal selected reference is required, not a diagnostic candidate')
    wanted = set(a_match_ids)
    case_ids = [c['fixture']['physical_match_id'] for c in cases]
    if (not wanted or len(wanted) != len(a_match_ids)
            or len(case_ids) != len(set(case_ids)) or set(case_ids) != wanted
            or not wanted <= set(snapshots)
            or set(template.metadata['training_match_ids']) != wanted
            or {r['physical_match_id'] for r in training_rows} != wanted):
        raise ValueError('Exact A-only identities required for rows, cases, priors and formal f')
    if _digest(training_rows) != template.metadata['training_rows_sha256']:
        raise ValueError('Audit requires exact original A training rows')
    if bank.sha256 != template.metadata['dataset_binding']['semantic_bank_sha256']:
        raise ValueError('Frozen semantic extraction bank differs from formal f')

    facts = {}
    for row in training_rows:
        mid = row['physical_match_id']
        fact = (row['outcome'], row['result_available_at'], row['state']['decision_at'])
        if (type(fact[0]) is not int or fact[0] not in (0, 1, 2)
                or row['state']['physical_match_id'] != mid
                or parse_time(fact[1]) <= parse_time(fact[2])
                or (mid in facts and facts[mid] != fact)):
            raise ValueError('Inconsistent A match outcome or forward clock')
        vector(row['state'])
        facts[mid] = fact
    ids = sorted(facts, key=lambda mid: (parse_time(facts[mid][2]), mid))
    expected = []
    for start in range(PARAMETERS['warmup_matches'], len(ids), PARAMETERS['validation_block_matches']):
        valid = ids[start:start + PARAMETERS['validation_block_matches']]
        boundary = min(parse_time(facts[mid][2]) for mid in valid)
        train = [mid for mid in ids[:start] if parse_time(facts[mid][1]) < boundary]
        if len(train) < PARAMETERS['minimum_fold_matches'] or len({facts[mid][0] for mid in train}) < 2:
            raise ValueError('Insufficient purged A training fold')
        expected.append((train, valid, boundary))
    folds = template.metadata['selection'].get('folds', [])
    if not expected or len(folds) != len(expected):
        raise ValueError('Formal forward fold count differs from A replay')
    for fold, (train, valid, boundary) in zip(folds, expected):
        if (fold.get('training_ids') != train or fold.get('validation_ids') != valid
                or parse_time(fold['first_validation_decision']) != boundary
                or [h['regularization'] for h in fold['hypotheses']] != PARAMETERS['regularizations']):
            raise ValueError('Formal purged-forward fold identities or candidate grid changed')
    return folds


def audit_candidate_grid(training_rows: Sequence[Mapping[str, Any]], template: EntitySemanticF,
                         cases: Sequence[Mapping[str, Any]], snapshots: Mapping[str, Any],
                         bank: Any, a_match_ids: Sequence[str], *,
                         registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Reproduce frozen losses and measure response on each held-out A fold.

    The response probe never fits on the match whose response is measured.
    Warmup matches only enter the separately labelled full-A diagnostic.
    No repeated state or reveal is counted as an independent match.
    """
    training_rows, cases = list(training_rows), list(cases)
    folds = _validated_folds(training_rows, template, cases, snapshots, bank, a_match_ids)
    validation = _candidate_validation_metadata(template)
    reports, loss_vectors = [], []
    transitions_by_candidate, forward_by_candidate, models = {}, {}, {}
    for candidate_index, row in enumerate(validation):
        lam = row['regularization']
        all_forward_rows, losses, fold_reports = [], [], []
        for fold_index, fold in enumerate(folds):
            train_ids, valid_ids = fold['training_ids'], fold['validation_ids']
            train_rows = [r for r in training_rows if r['physical_match_id'] in train_ids]
            coefficients, optimizer = _fit(train_rows, lam)
            hypothesis = fold['hypotheses'][candidate_index]
            if _digest(coefficients) != hypothesis['coefficients_sha256']:
                raise ValueError('A fold coefficient replay differs from frozen formal experiment')
            match_losses = {mid: float(np.mean([
                -np.log(_predict(coefficients, r['state'])[r['outcome']])
                for r in training_rows if r['physical_match_id'] == mid])) for mid in valid_ids}
            if (set(match_losses) != set(hypothesis['match_losses']) or any(
                    not math.isclose(loss, hypothesis['match_losses'][mid], rel_tol=0, abs_tol=1e-12)
                    for mid, loss in match_losses.items())):
                raise ValueError('A fold loss replay differs from frozen formal experiment')
            losses.extend(match_losses.values())
            model = _candidate_model(template, coefficients, lam, train_rows, optimizer,
                                     scope='A_purged_forward_validation', registration=registration)
            models[model.model_version] = {**model.payload(), 'model_version': model.model_version}
            response_rows = build_a_only_response_rows(
                [c for c in cases if c['fixture']['physical_match_id'] in valid_ids],
                snapshots, bank, model, valid_ids)
            all_forward_rows.extend([{**r, 'fold_index': fold_index} for r in response_rows])
            fold_reports.append({
                'fold_index': fold_index, 'training_ids': train_ids, 'validation_ids': valid_ids,
                'first_validation_decision': fold['first_validation_decision'],
                'model_version': model.model_version, 'coefficients_sha256': _digest(coefficients),
                'match_losses': match_losses, 'response': summarize_response_rows(response_rows),
                'frozen_coefficient_and_loss_replay_verified': True,
            })
        loss_vectors.append(losses)
        coefficients, optimizer = _fit(training_rows, lam)
        if lam == template.metadata['selection']['selected_regularization']:
            if _digest(coefficients) != _digest(template.coefficients):
                raise ValueError('Full A fit differs from frozen selected formal f')
        model = _candidate_model(template, coefficients, lam, training_rows, optimizer,
                                 scope='full_A_fit_training_response_diagnostic',
                                 registration=registration)
        models[model.model_version] = {**model.payload(), 'model_version': model.model_version}
        transitions = build_a_only_response_rows(cases, snapshots, bank, model, a_match_ids)
        transitions_by_candidate[str(lam)] = transitions
        forward_by_candidate[str(lam)] = all_forward_rows
        prior_loss = validation[0]['mean_match_log_loss']
        reports.append({**row,
                        'model_version': model.model_version,
                        'coefficients_sha256': _digest(coefficients),
                        'coefficient_max_abs': float(np.max(np.abs(coefficients))),
                        'nonzero_coefficients': int(np.sum(np.abs(coefficients) > PREDICTION_CHANGE_THRESHOLD)),
                        'optimizer': optimizer,
                        'response_scope': 'A_purged_forward_validation',
                        'forward_response': summarize_response_rows(all_forward_rows),
                        'full_A_training_response': summarize_response_rows(transitions),
                        'mean_loss_delta_vs_prior_only': row['mean_match_log_loss'] - prior_loss,
                        'forward_folds': fold_reports})
    matrix = np.asarray(loss_vectors)
    means = matrix.mean(axis=1)
    best = int(np.argmin(means))
    se = (matrix - matrix[best]).std(axis=1, ddof=1) / np.sqrt(matrix.shape[1])
    if (matrix.shape[1] != template.metadata['selection']['independent_validation_matches']
            or any(not math.isclose(float(means[i]), r['mean_match_log_loss'], rel_tol=0, abs_tol=1e-12)
                   or not math.isclose(float(se[i]), r['paired_standard_error_to_best'], rel_tol=0, abs_tol=1e-12)
                   for i, r in enumerate(reports))):
        raise ValueError('Replayed A validation aggregation differs from formal experiment')
    selection = capability_constrained_selection(reports, registration=registration)
    return {
        'schema': VERSION,
        'protocol': {**selection_protocol(registration),
            'validation_scores_source': 'exact_A_fold_coefficient_and_loss_replay',
            'validation_outcomes_used_for_loss_only': True,
            'validation_outcomes_used_for_response_or_fold_fitting': False,
            'B_C_outcomes_used_in_fit_selection_or_response': False,
            'g_fit': False, 'formal_model_changed': False,
            'standard_error_scope': 'dependent_small_development_sample_not_confidence_bound'},
        'formal_model_version': template.model_version,
        'representation_sha256': REPRESENTATION_SHA256,
        'A_match_ids': list(a_match_ids),
        'A_training_rows': len(training_rows),
        'A_forward_validation_matches': matrix.shape[1],
        'A_forward_fold_count': len(folds),
        'candidates': reports,
        'capability_constrained_selection': selection,
        'full_A_transition_rows': transitions_by_candidate,
        'forward_transition_rows': forward_by_candidate,
        'audit_models': models,
    }


__all__ = [
    'VERSION', 'PREDICTION_CHANGE_THRESHOLD', 'build_a_only_response_rows',
    'summarize_response_rows', 'capability_constrained_selection', 'audit_candidate_grid',
    'selection_protocol',
]
