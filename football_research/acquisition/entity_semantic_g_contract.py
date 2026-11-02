"""Compatibility gate for a future g trained against the entity-level f.

The entity f experiment deliberately did not fit g.  This module is the
machine-checkable hand-off between that experiment and a later g training
run.  It checks the bindings that cannot be recovered from a prediction
trace after the fact: the exact f payload, the entity representation, the
pre-query schema, the signed target and the physical-match split.

This is a research gate.  A passing artifact is still not operationally
eligible while the consumer contract says that the blind test and
prospective calibration are incomplete.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
from typing import Any, Mapping

from .context_predictor import _digest
from .entity_semantic_f import REPRESENTATION_SHA256, VERSION as F_VERSION
from .entity_semantic_query import VERSION as INPUT_VERSION, validate_query


VERSION = 'entity-semantic-g-compatibility-gate-v1'
READINESS_VERSION = 'entity-semantic-g-training-readiness-v1'
CONTRACT_SCHEMA = 'entity-semantic-f-g-consumer-v1'
TARGET = 'conditional_frozen_predictor_log_loss_change'
TARGET_ROW_SCHEMA = 'entity-semantic-fixed-f-realized-target-v1'
AUDIT_SCHEMA = 'entity-semantic-fixed-f-response-audit-v1'
ALLOWED_ORIGINS = {'historical_reconstruction', 'constructed_mechanism'}


def _issue(code: str, field: str, detail: str) -> dict[str, str]:
    return {'code': code, 'field': field, 'detail': detail}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    return value


def _ids(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f'{field} must be a list of nonempty physical-match IDs')
    if not allow_empty and not value:
        raise ValueError(f'{field} must contain at least one independent physical match')
    if len(set(value)) != len(value):
        raise ValueError(f'{field} contains duplicate physical-match IDs')
    return list(value)


def _f_payload_without_version(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact JSON payload used by ``EntitySemanticF.payload``."""
    return {key: deepcopy(value) for key, value in payload.items() if key != 'model_version'}


def _row_list(value: Any, field: str) -> list[Mapping[str, Any]]:
    """Validate a JSONL projection before the readiness gate inspects it."""
    if not isinstance(value, list):
        raise TypeError(f'{field} must be a list of JSON objects')
    if any(not isinstance(row, Mapping) for row in value):
        raise TypeError(f'{field} must contain only JSON objects')
    return list(value)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{field} must be a nonempty string')
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field} must be a finite number')
    return float(value)


def _sample_index(rows: list[Mapping[str, Any]], field: str) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    """Index sample records while retaining duplicate/malformed diagnostics."""
    indexed: dict[str, Mapping[str, Any]] = {}
    issues: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        try:
            sample_id = _nonempty_string(row.get('sample_id'), f'{field}[{index}].sample_id')
        except (TypeError, ValueError) as exc:
            issues.append(_issue('invalid_sample_id', f'{field}[{index}].sample_id', str(exc)))
            continue
        if sample_id in indexed:
            issues.append(_issue('duplicate_sample_id', f'{field}[{index}].sample_id', sample_id))
            continue
        indexed[sample_id] = row
    return indexed, issues


def check_g_training_readiness(
    contract: Mapping[str, Any],
    target_rows: list[Mapping[str, Any]],
    *,
    pre_read_rows: list[Mapping[str, Any]] | None = None,
    post_read_audits: list[Mapping[str, Any]] | None = None,
    f_artifact: Mapping[str, Any] | None = None,
    require_paired_evidence: bool = True,
) -> dict[str, Any]:
    """Decide whether fixed-f target rows can be handed to a g trainer.

    This is deliberately separate from :func:`check_g_compatibility`.  The
    latter checks a *serialized candidate g*; this gate checks the training
    material before a candidate exists.  In particular, a valid-looking
    JSONL file containing only zero gains is retained for audit but cannot be
    reported as a trained information-value model.

    ``pre_read_rows`` and ``post_read_audits`` are required by default.  The
    target file contains hashes, but the paired projections are what prove
    that each target was attached to the same frozen f query and response.
    """
    issues: list[dict[str, str]] = check_f_binding(contract, f_artifact)
    warnings: list[dict[str, str]] = []
    valid_targets: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    gains: list[float] = []
    eligible_count = 0
    ineligible_count = 0
    groups: set[str] = set()
    physical_ids: set[str] = set()
    interpretations: set[str] = set()
    audit_feature_changed = 0
    audit_prediction_changed = 0
    audit_probability_changes: list[float] = []
    audit_stats_missing = 0

    try:
        targets = _row_list(target_rows, 'target_rows')
    except (TypeError, ValueError) as exc:
        targets = []
        issues.append(_issue('invalid_target_rows', 'target_rows', str(exc)))

    target_index, index_issues = _sample_index(targets, 'target_rows')
    issues.extend(index_issues)
    expected_model = contract.get('model_version') if isinstance(contract, Mapping) else None
    expected_representation = contract.get('representation_sha256') if isinstance(contract, Mapping) else None
    expected_payload = contract.get('predictor_payload_sha256') if isinstance(contract, Mapping) else None
    f_ids = set(contract.get('f_fit_physical_match_ids', [])) if isinstance(contract, Mapping) else set()

    for index, row in enumerate(targets):
        field = f'target_rows[{index}].target'
        try:
            sample_id = _nonempty_string(row.get('sample_id'), f'target_rows[{index}].sample_id')
            target = _mapping(row.get('target'), field)
            if target.get('schema') != TARGET_ROW_SCHEMA:
                raise ValueError(f'expected {TARGET_ROW_SCHEMA}')
            if target.get('target') != TARGET:
                raise ValueError(f'expected target {TARGET}')
            if target.get('model_version') != expected_model:
                raise ValueError('target model_version differs from the frozen contract')
            if target.get('representation_sha256') != expected_representation:
                raise ValueError('target representation differs from the frozen contract')
            _nonempty_string(target.get('query_sha256'), f'{field}.query_sha256')
            _nonempty_string(target.get('audit_sha256'), f'{field}.audit_sha256')
            physical_id = _nonempty_string(target.get('physical_match_id'), f'{field}.physical_match_id')
            gain = _finite_number(target.get('realized_gain_nats'), f'{field}.realized_gain_nats')
            if target.get('prospective_training_eligible') is True:
                eligible_count += 1
            elif target.get('prospective_training_eligible') is False:
                ineligible_count += 1
            else:
                raise ValueError(f'{field}.prospective_training_eligible must be Boolean')
            if physical_id in f_ids:
                raise ValueError('target physical_match_id overlaps f training identities')
            group = row.get('group')
            if group is not None:
                _nonempty_string(group, f'target_rows[{index}].group')
                groups.add(group)
            interpretation = target.get('representation_interpretation')
            if isinstance(interpretation, str) and interpretation:
                interpretations.add(interpretation)
            gains.append(gain)
            physical_ids.add(physical_id)
            valid_targets.append((sample_id, row, target))
        except (TypeError, ValueError, KeyError) as exc:
            issues.append(_issue('invalid_target_row', field, str(exc)))

    # The target projection is not sufficient on its own: verify every hash
    # against the pre-read query and post-read audit when supplied.
    if (pre_read_rows is None) != (post_read_audits is None):
        issues.append(_issue('paired_evidence_incomplete', 'paired_evidence',
                             'pre_read_rows and post_read_audits must be supplied together'))
    elif pre_read_rows is None and require_paired_evidence:
        issues.append(_issue('paired_evidence_missing', 'paired_evidence',
                             'pre-read queries and post-read audits are required for g training readiness'))
    elif pre_read_rows is not None and post_read_audits is not None:
        try:
            pre_rows = _row_list(pre_read_rows, 'pre_read_rows')
            audit_rows = _row_list(post_read_audits, 'post_read_audits')
        except (TypeError, ValueError) as exc:
            pre_rows, audit_rows = [], []
            issues.append(_issue('invalid_paired_evidence', 'paired_evidence', str(exc)))
        pre_index, pre_issues = _sample_index(pre_rows, 'pre_read_rows')
        audit_index, audit_issues = _sample_index(audit_rows, 'post_read_audits')
        issues.extend(pre_issues + audit_issues)
        target_ids = set(target_index)
        if set(pre_index) != target_ids:
            issues.append(_issue('pre_read_target_set_mismatch', 'pre_read_rows',
                                 'sample IDs must exactly match target_rows'))
        if set(audit_index) != target_ids:
            issues.append(_issue('audit_target_set_mismatch', 'post_read_audits',
                                 'sample IDs must exactly match target_rows'))
        for sample_id, _row, target in valid_targets:
            pre = pre_index.get(sample_id)
            audit = audit_index.get(sample_id)
            if pre is None or audit is None:
                continue
            try:
                query = _mapping(pre.get('query'), f'pre_read_rows[{sample_id}].query')
                audit_obj = _mapping(audit.get('audit'), f'post_read_audits[{sample_id}].audit')
                if query.get('schema') != INPUT_VERSION:
                    raise ValueError(f'pre-read query schema must be {INPUT_VERSION}')
                if audit_obj.get('schema') != AUDIT_SCHEMA:
                    raise ValueError(f'post-read audit schema must be {AUDIT_SCHEMA}')
                if query.get('query_sha256') != target.get('query_sha256'):
                    raise ValueError('pre-read query_sha256 does not match target')
                if audit_obj.get('query_sha256') != target.get('query_sha256'):
                    raise ValueError('post-read audit query_sha256 does not match target')
                if audit_obj.get('audit_sha256') != target.get('audit_sha256'):
                    raise ValueError('post-read audit_sha256 does not match target')
                if query.get('physical_match_id') != target.get('physical_match_id'):
                    raise ValueError('pre-read physical_match_id does not match target')
                if audit_obj.get('physical_match_id') != target.get('physical_match_id'):
                    raise ValueError('post-read physical_match_id does not match target')
                if query.get('model_version') != expected_model or audit_obj.get('model_version') != expected_model:
                    raise ValueError('paired evidence model_version differs from the frozen contract')
                if query.get('representation_sha256') != expected_representation:
                    raise ValueError('pre-read representation differs from the frozen contract')
                if audit_obj.get('representation_sha256') != expected_representation:
                    raise ValueError('post-read representation differs from the frozen contract')
                if query.get('predictor_payload_sha256') != expected_payload:
                    raise ValueError('pre-read predictor payload differs from the frozen contract')
                feature_changed = audit_obj.get('feature_vector_changed')
                max_probability_change = audit_obj.get('max_probability_change')
                if not isinstance(feature_changed, bool):
                    audit_stats_missing += 1
                    raise ValueError('post-read audit feature_vector_changed must be Boolean')
                max_probability_change = _finite_number(
                    max_probability_change,
                    f'post_read_audits[{sample_id}].audit.max_probability_change',
                )
                if max_probability_change < 0.0:
                    raise ValueError('post-read max_probability_change must be nonnegative')
                if feature_changed:
                    audit_feature_changed += 1
                if max_probability_change > 1e-12:
                    audit_prediction_changed += 1
                audit_probability_changes.append(max_probability_change)
            except (TypeError, ValueError, KeyError) as exc:
                issues.append(_issue('paired_evidence_mismatch', f'paired_evidence[{sample_id}]', str(exc)))

    if audit_stats_missing:
        warnings.append(_issue('audit_response_stats_missing', 'post_read_audits',
                               f'{audit_stats_missing} paired audits did not expose validated response statistics'))

    unique_gains = sorted(set(gains))
    nonzero = [gain for gain in gains if gain != 0.0]
    positive = [gain for gain in gains if gain > 0.0]
    negative = [gain for gain in gains if gain < 0.0]
    if not gains:
        signal_status = 'no_valid_target_signal'
    elif len(unique_gains) == 1:
        signal_status = 'zero_target_signal' if unique_gains[0] == 0.0 else 'constant_target_signal'
    else:
        signal_status = 'variable_target_signal'

    if signal_status == 'zero_target_signal':
        warnings.append(_issue('zero_target_signal', 'target.realized_gain_nats',
                               'all valid fixed-f gains are zero; retain rows for audit but do not fit formal g'))
    elif signal_status == 'constant_target_signal':
        warnings.append(_issue('constant_target_signal', 'target.realized_gain_nats',
                               'all valid fixed-f gains have one identical value; no ranking signal is available'))
    if ineligible_count:
        warnings.append(_issue('prospective_rows_ineligible', 'target.prospective_training_eligible',
                               f'{ineligible_count} valid rows are historical/reconstruction-only'))
    if positive == [] or negative == []:
        warnings.append(_issue('one_sided_target_signal', 'target.realized_gain_nats',
                               'valid gains do not contain both improvement and harm; risk calibration remains limited'))
    if 'predictor_insensitive_not_information_irrelevant' in interpretations:
        warnings.append(_issue('predictor_insensitivity_interpretation', 'target.representation_interpretation',
                               'zero gain is attributed to frozen-f insensitivity, not to absence of information'))

    structural_ok = not issues
    variable = signal_status == 'variable_target_signal'
    ready_for_fit = structural_ok and variable and bool(valid_targets)
    prospective_ready = ready_for_fit and ineligible_count == 0
    if not structural_ok:
        status = 'rejected'
    elif not valid_targets:
        status = 'not_ready_no_valid_target_rows'
    elif signal_status == 'zero_target_signal':
        status = 'not_ready_zero_target_signal'
    elif signal_status == 'constant_target_signal':
        status = 'not_ready_constant_target_signal'
    elif prospective_ready:
        status = 'ready_for_fit'
    else:
        status = 'ready_for_research_fit'
    return {
        'schema': READINESS_VERSION,
        'status': status,
        'ready_for_fit': ready_for_fit,
        'research_fit_eligible': ready_for_fit,
        'prospective_fit_eligible': prospective_ready,
        'operational_eligible': False,
        'issues': issues,
        'warnings': warnings,
        'contract_model_version': expected_model,
        'target_schema': TARGET_ROW_SCHEMA,
        'target': TARGET,
        'counts': {
            'target_rows': len(targets),
            'valid_target_rows': len(valid_targets),
            'invalid_target_rows': len(targets) - len(valid_targets),
            'distinct_physical_match_ids': len(physical_ids),
            'eligible_rows': eligible_count,
            'ineligible_rows': ineligible_count,
            'nonzero_gain_rows': len(nonzero),
            'positive_gain_rows': len(positive),
            'negative_gain_rows': len(negative),
            'unique_target_count': len(unique_gains),
            'groups': sorted(groups),
        },
        'target_summary': {
            'min': min(gains) if gains else None,
            'max': max(gains) if gains else None,
            'mean': sum(gains) / len(gains) if gains else None,
            'unique_values': unique_gains,
            'signal_status': signal_status,
        },
        'required_contract_groups': {
            'gain_fit_recommended_groups': deepcopy(contract.get('gain_fit_recommended_groups'))
            if isinstance(contract, Mapping) else None,
            'development_evaluation_groups': deepcopy(contract.get('development_evaluation_groups'))
            if isinstance(contract, Mapping) else None,
        },
        'paired_evidence_required': require_paired_evidence,
        'paired_evidence_supplied': pre_read_rows is not None and post_read_audits is not None,
        'paired_response_summary': {
            'audits_checked': len(audit_probability_changes),
            'feature_vector_changed_rows': audit_feature_changed,
            'prediction_changed_rows': audit_prediction_changed,
            'max_probability_change': max(audit_probability_changes)
            if audit_probability_changes else None,
            'mean_probability_change': sum(audit_probability_changes) / len(audit_probability_changes)
            if audit_probability_changes else None,
            'prediction_change_threshold': 1e-12,
        },
        'training_physical_match_ids': sorted(physical_ids),
    }


def check_f_binding(contract: Mapping[str, Any], f_artifact: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """Check the consumer contract and, when supplied, the serialized f."""
    issues: list[dict[str, str]] = []
    required = {
        'schema', 'input_schema', 'model_version', 'predictor_payload_sha256',
        'representation_sha256', 'f_fit_physical_match_ids',
        'gain_fit_recommended_groups', 'development_evaluation_groups',
        'independent_blind_test', 'target', 'numeric_g_fit', 'prospective_eligible',
        'diagnostic_candidate_excluded',
    }
    if not isinstance(contract, Mapping):
        return [_issue('contract_not_mapping', 'contract', 'consumer contract must be a JSON object')]
    missing = sorted(required - set(contract))
    if missing:
        issues.append(_issue('missing_contract_fields', 'contract', ', '.join(missing)))
    if contract.get('schema') != CONTRACT_SCHEMA:
        issues.append(_issue('wrong_contract_schema', 'schema', f'expected {CONTRACT_SCHEMA}'))
    if contract.get('input_schema') != INPUT_VERSION:
        issues.append(_issue('wrong_input_schema', 'input_schema', f'expected {INPUT_VERSION}'))
    if contract.get('target') != TARGET:
        issues.append(_issue('wrong_target', 'target', f'expected {TARGET}'))
    if contract.get('representation_sha256') != REPRESENTATION_SHA256:
        issues.append(_issue('wrong_representation', 'representation_sha256',
                             'contract is not bound to the frozen entity representation'))
    try:
        f_ids = _ids(contract.get('f_fit_physical_match_ids'), 'f_fit_physical_match_ids')
    except (TypeError, ValueError) as exc:
        f_ids = []
        issues.append(_issue('invalid_f_fit_ids', 'f_fit_physical_match_ids', str(exc)))
    if contract.get('numeric_g_fit') is not False:
        issues.append(_issue('numeric_g_already_claimed', 'numeric_g_fit',
                             'the current contract must state that g has not yet been fitted'))
    if contract.get('prospective_eligible') is not False:
        issues.append(_issue('prospective_flag_not_frozen', 'prospective_eligible',
                             'the current research contract cannot grant prospective eligibility'))
    if contract.get('diagnostic_candidate_excluded') is not True:
        issues.append(_issue('diagnostic_f_not_excluded', 'diagnostic_candidate_excluded',
                             'the non-selected diagnostic f must remain outside the g consumer'))
    if not isinstance(contract.get('independent_blind_test'), bool):
        issues.append(_issue('invalid_blind_test_flag', 'independent_blind_test', 'must be Boolean'))

    if f_artifact is not None:
        try:
            f = _mapping(f_artifact, 'f_artifact')
            if f.get('schema') != F_VERSION:
                issues.append(_issue('wrong_f_schema', 'f_artifact.schema', f'expected {F_VERSION}'))
            if f.get('model_version') != contract.get('model_version'):
                issues.append(_issue('wrong_f_model_version', 'f_artifact.model_version',
                                     'serialized f does not match the consumer contract'))
            if f.get('representation_sha256') != contract.get('representation_sha256'):
                issues.append(_issue('wrong_f_representation', 'f_artifact.representation_sha256',
                                     'serialized f representation differs from the contract'))
            if _digest(_f_payload_without_version(f)) != contract.get('predictor_payload_sha256'):
                issues.append(_issue('wrong_f_payload_digest', 'predictor_payload_sha256',
                                     'serialized f payload digest differs from the contract'))
            metadata = _mapping(f.get('metadata'), 'f_artifact.metadata')
            if metadata.get('representation_sha256') != contract.get('representation_sha256'):
                issues.append(_issue('wrong_f_metadata_representation', 'f_artifact.metadata.representation_sha256',
                                     'f metadata and serialized f disagree'))
            if metadata.get('training_match_ids') != f_ids:
                issues.append(_issue('wrong_f_training_ids', 'f_artifact.metadata.training_match_ids',
                                     'f training identities differ from the contract'))
        except (TypeError, ValueError, KeyError) as exc:
            issues.append(_issue('invalid_f_artifact', 'f_artifact', str(exc)))
    return issues


def check_contract_readiness(
    contract: Mapping[str, Any],
    *,
    f_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the hand-off before a g file exists.

    ``waiting_for_g`` is an intentional state, not a failed model check.  It
    means the frozen f/contract pair is coherent and the next required input
    is a separately trained g artifact.
    """
    issues = check_f_binding(contract, f_artifact)
    waiting = not issues and contract.get('numeric_g_fit') is False
    return {
        'schema': VERSION,
        'status': 'waiting_for_g' if waiting else ('rejected' if issues else 'g_metadata_required'),
        'compatible': False,
        'operational_eligible': False,
        'issues': issues,
        'warnings': ([
            _issue('g_artifact_missing', 'g_artifact',
                   'no numeric g artifact is supplied; fit and validate it against this exact f contract'),
            _issue('blind_test_incomplete', 'contract.independent_blind_test',
                   'a passing g remains research-only until an independent blind test is sealed'),
        ] if waiting else []),
        'required_g_metadata': [
            'input_schema', 'predictor_payload_sha256', 'representation_sha256',
            'target', 'training_physical_match_ids',
            'predictor_training_physical_match_ids', 'evidence_origin',
        ],
        'contract_model_version': contract.get('model_version') if isinstance(contract, Mapping) else None,
    }


def check_g_compatibility(
    contract: Mapping[str, Any],
    g_artifact: Mapping[str, Any],
    *,
    f_artifact: Mapping[str, Any] | None = None,
    decision_physical_match_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic compatibility report for one candidate g.

    ``g_artifact`` may be any serialized model format as long as it exposes a
    ``metadata`` object.  The gate intentionally does not load or execute the
    estimator.  It validates the binding metadata first, so a malformed or
    accidentally old model is rejected before it can influence query choice.
    """
    issues = check_f_binding(contract, f_artifact)
    warnings: list[dict[str, str]] = []
    try:
        c = _mapping(contract, 'contract')
        g = _mapping(g_artifact, 'g_artifact')
        meta = _mapping(g.get('metadata'), 'g_artifact.metadata')
        required = {
            'input_schema', 'predictor_payload_sha256', 'representation_sha256',
            'target', 'training_physical_match_ids',
            'predictor_training_physical_match_ids', 'evidence_origin',
        }
        missing = sorted(required - set(meta))
        if missing:
            issues.append(_issue('missing_g_metadata', 'g_artifact.metadata', ', '.join(missing)))
        if meta.get('input_schema') != c.get('input_schema'):
            issues.append(_issue('g_input_schema_mismatch', 'metadata.input_schema',
                                 'g was not trained on the entity pre-query schema'))
        if meta.get('predictor_payload_sha256') != c.get('predictor_payload_sha256'):
            issues.append(_issue('g_f_payload_mismatch', 'metadata.predictor_payload_sha256',
                                 'g is bound to a different f payload'))
        if meta.get('representation_sha256') != c.get('representation_sha256'):
            issues.append(_issue('g_representation_mismatch', 'metadata.representation_sha256',
                                 'g is bound to a different entity representation'))
        if meta.get('target') != c.get('target'):
            issues.append(_issue('g_target_mismatch', 'metadata.target',
                                 'g target is not the signed fixed-f log-loss change'))
        if meta.get('evidence_origin') not in ALLOWED_ORIGINS:
            issues.append(_issue('g_evidence_origin_invalid', 'metadata.evidence_origin',
                                 'only declared historical reconstruction or constructed mechanism is allowed'))
        g_ids = _ids(meta.get('training_physical_match_ids'), 'g training_physical_match_ids')
        predictor_ids = _ids(meta.get('predictor_training_physical_match_ids'),
                             'g predictor_training_physical_match_ids')
        f_ids = _ids(c.get('f_fit_physical_match_ids'), 'f_fit_physical_match_ids')
        overlap = sorted(set(g_ids) & set(predictor_ids))
        if overlap:
            issues.append(_issue('f_g_training_overlap', 'metadata.training_physical_match_ids',
                                 'g and predictor training groups overlap: '+', '.join(overlap)))
        if predictor_ids != f_ids:
            issues.append(_issue('predictor_group_mismatch', 'metadata.predictor_training_physical_match_ids',
                                 'g must name the exact f training group, in the same identity set'))
        if g.get('model_version') is not None and (not isinstance(g.get('model_version'), str)
                                                   or not g.get('model_version')):
            issues.append(_issue('invalid_g_model_version', 'g_artifact.model_version',
                                 'when present, model_version must be a nonempty string'))
        if meta.get('model_version') is not None and meta.get('model_version') != g.get('model_version'):
            issues.append(_issue('g_model_version_binding_mismatch', 'metadata.model_version',
                                 'g metadata model_version differs from the serialized artifact'))
        if meta.get('production_ready') is True or meta.get('operational_eligible') is True:
            issues.append(_issue('premature_operational_flag', 'metadata',
                                 'current consumer contract forbids operational promotion'))
        if c.get('independent_blind_test') is not True:
            warnings.append(_issue('blind_test_incomplete', 'contract.independent_blind_test',
                                   'compatible result remains research-only until a sealed blind test is completed'))
        if c.get('numeric_g_fit') is not True:
            warnings.append(_issue('contract_not_numeric_g_ready', 'contract.numeric_g_fit',
                                   'current frozen contract records that numeric g was not yet fitted'))
        if decision_physical_match_ids is not None:
            decision_ids = _ids(decision_physical_match_ids, 'decision_physical_match_ids', allow_empty=True)
            forbidden = sorted(set(decision_ids) & (set(f_ids) | set(g_ids)))
            if forbidden:
                issues.append(_issue('decision_training_overlap', 'decision_physical_match_ids',
                                     'decision fixtures overlap f/g training identities: '+', '.join(forbidden)))
    except (TypeError, ValueError, KeyError) as exc:
        issues.append(_issue('invalid_g_artifact', 'g_artifact', str(exc)))

    compatible = not issues
    status = 'compatible_research_only' if compatible else 'rejected'
    return {
        'schema': VERSION,
        'status': status,
        'compatible': compatible,
        'operational_eligible': False,
        'issues': issues,
        'warnings': warnings,
        'contract_model_version': contract.get('model_version') if isinstance(contract, Mapping) else None,
        'g_model_version': g_artifact.get('model_version') if isinstance(g_artifact, Mapping) else None,
    }


class EntitySemanticGBinding:
    """Immutable runtime wrapper for a g that passed the entity gate.

    The wrapper deliberately accepts an already-loaded estimator object.  It
    does not guess a serialized estimator format or train anything; the
    producing thread remains responsible for loading its own g implementation
    before handing it over here.
    """

    def __init__(self, model: Any, *, contract: Mapping[str, Any], f_artifact: Mapping[str, Any] | None = None):
        metadata = getattr(model, 'metadata', None)
        model_version = getattr(model, 'model_version', None)
        artifact = {'model_version': model_version, 'metadata': deepcopy(metadata)}
        report = check_g_compatibility(contract, artifact, f_artifact=f_artifact)
        if not report['compatible']:
            codes = ', '.join(row['code'] for row in report['issues']) or 'unknown_gate_failure'
            raise ValueError('Entity semantic g compatibility gate rejected: '+codes)
        if not isinstance(model_version, str) or not model_version:
            raise ValueError('A bound entity g needs a nonempty model_version')
        if not callable(getattr(model, 'predict', None)):
            raise TypeError('A bound entity g needs a predict(query) method')
        self._model = model
        self._contract = deepcopy(dict(contract))
        self.metadata = deepcopy(dict(metadata))
        self.model_version = model_version
        self.compatibility_report = deepcopy(report)
        self._binding = _digest({'version': model_version, 'metadata': self.metadata})

    def _check_unchanged(self) -> None:
        current = {'version': getattr(self._model, 'model_version', None),
                   'metadata': getattr(self._model, 'metadata', None)}
        if _digest(current) != self._binding:
            raise ValueError('Bound entity g was modified')

    def predict(self, query: Mapping[str, Any]) -> dict[str, Any]:
        self._check_unchanged()
        validate_query(query)
        if query.get('predictor_payload_sha256') != self._contract['predictor_payload_sha256']:
            raise ValueError('Entity g query uses a different frozen f payload')
        if query.get('representation_sha256') != self._contract['representation_sha256']:
            raise ValueError('Entity g query uses a different representation')
        forbidden = set(self.metadata['training_physical_match_ids']) | set(
            self.metadata['predictor_training_physical_match_ids'])
        if query.get('physical_match_id') in forbidden:
            raise ValueError('Entity g query overlaps its training identities')
        result = self._model.predict(deepcopy(dict(query)))
        if not isinstance(result, Mapping):
            raise TypeError('Entity g predict() must return a mapping')
        return deepcopy(dict(result))


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return value


__all__ = [
    'VERSION', 'READINESS_VERSION', 'TARGET', 'TARGET_ROW_SCHEMA', 'AUDIT_SCHEMA',
    'check_contract_readiness', 'check_g_training_readiness', 'check_f_binding',
    'check_g_compatibility', 'EntitySemanticGBinding', 'load_json',
]
