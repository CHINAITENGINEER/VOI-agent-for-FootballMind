"""Pre-register the next capability-constrained f selection experiment.

The Round125 audit found that the formally selected prior-only f cannot
transmit a newly revealed semantic document into its probabilities.  That
finding was retrospective.  This module creates a *future* registration so
the same observation cannot be used to silently promote a candidate after
looking at its response.

The registration is deliberately a design contract, not a model artifact:
it binds the candidate grid, response rule, formal reference f and the
already-seen A identities.  A later audit must use a disjoint A batch and
carry this registration ID.  Promotion remains a separate, explicit step.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from .context_predictor import _digest
from .entity_semantic_f import EntitySemanticF, PARAMETERS, VERSION as F_VERSION
from .entity_semantic_f_selection_audit import (
    PREDICTION_CHANGE_THRESHOLD,
    SELECTION_RULE,
    SENSITIVITY_THRESHOLDS,
    VERSION as AUDIT_VERSION,
)
from .state import parse_time


VERSION = 'entity-semantic-f-selection-registration-v1'
PURPOSE = 'future_f_capability_constrained_selection'
REGISTRATION_STATUS = 'pre_registered_for_future_A_batch'
MINIMUM_NEW_MATCHES = 12
MINIMUM_VALIDATION_MATCHES = 6
MINIMUM_RESPONSIVE_VALIDATION_MATCHES = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value.strip()


def _match_ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} must be a nonempty list')
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f'{name} must contain nonempty strings')
    if len(set(value)) != len(value):
        raise ValueError(f'{name} must contain unique identities')
    return list(value)


def _canonical_without_id(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in plan.items() if key != 'registration_id'}


def _check_finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return float(value)


def validate_registration(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a saved registration and return a defensive copy.

    This function intentionally rejects registrations which already bind an
    audit report or a candidate artifact.  Those belong to the post-run
    receipt and cannot be used to backdate the design.
    """
    if not isinstance(plan, Mapping) or plan.get('schema') != VERSION:
        raise ValueError('Unknown f selection registration schema')
    if plan.get('purpose') != PURPOSE or plan.get('registration_status') != REGISTRATION_STATUS:
        raise ValueError('Registration purpose or status is invalid')
    created = parse_time(_string(plan.get('created_at'), 'created_at'))
    if created > datetime.now(timezone.utc):
        raise ValueError('Registration cannot be dated in the future')
    if plan.get('audit_version') != AUDIT_VERSION:
        raise ValueError('Registration audit version differs')
    if plan.get('f_version') != F_VERSION:
        raise ValueError('Registration f version differs')
    if plan.get('selection_rule') != SELECTION_RULE:
        raise ValueError('Registration selection rule differs')
    grid = plan.get('candidate_grid')
    if grid != list(PARAMETERS['regularizations']):
        raise ValueError('Registration candidate grid differs from frozen f protocol')
    if plan.get('response_scope') != 'A_purged_forward_validation_matches_only':
        raise ValueError('Registration response scope must be held-out A validation')
    if _check_finite(plan.get('response_threshold'), 'response_threshold', minimum=0) != PREDICTION_CHANGE_THRESHOLD:
        raise ValueError('Registration response threshold differs')
    if plan.get('minimum_responsive_validation_matches') != MINIMUM_RESPONSIVE_VALIDATION_MATCHES:
        raise ValueError('Registration responsive-match minimum differs')
    if plan.get('sensitivity_thresholds') != list(SENSITIVITY_THRESHOLDS):
        raise ValueError('Registration sensitivity thresholds differ')
    _string(plan.get('future_batch_label'), 'future_batch_label')
    _string(plan.get('future_data_scope'), 'future_data_scope')
    _match_ids(plan.get('known_A_match_ids'), 'known_A_match_ids')
    if (type(plan.get('minimum_new_A_matches')) is not int
            or plan['minimum_new_A_matches'] < MINIMUM_NEW_MATCHES):
        raise ValueError('minimum_new_A_matches is too small')
    if (type(plan.get('minimum_validation_matches')) is not int
            or plan['minimum_validation_matches'] < MINIMUM_VALIDATION_MATCHES):
        raise ValueError('minimum_validation_matches is too small')
    reference = plan.get('reference_formal_model')
    if not isinstance(reference, Mapping):
        raise ValueError('reference_formal_model is required')
    for key in ('path', 'sha256', 'model_version'):
        _string(reference.get(key), f'reference_formal_model.{key}')
    if len(reference['sha256']) != 64:
        raise ValueError('reference formal model digest must be SHA-256')
    if plan.get('promotion_allowed') is not False or plan.get('g_fit') is not False:
        raise ValueError('Registration cannot authorize promotion or g fitting')
    forbidden = {'audit_report', 'candidate_model', 'selection_result'}
    if forbidden & set(plan):
        raise ValueError('Registration must not contain post-run audit artifacts')
    registration_id = plan.get('registration_id')
    expected = VERSION + ':' + _digest(_canonical_without_id(plan))[:24]
    if registration_id != expected:
        raise ValueError('Registration digest mismatch')
    return deepcopy(dict(plan))


def create_registration(*, formal_model_path: str | Path, known_A_match_ids: list[str],
                        future_batch_label: str, future_data_scope: str,
                        minimum_new_A_matches: int = MINIMUM_NEW_MATCHES,
                        minimum_validation_matches: int = MINIMUM_VALIDATION_MATCHES,
                        created_at: str | datetime | None = None) -> dict[str, Any]:
    """Create a future registration without reading any future outcomes."""
    path = Path(formal_model_path).resolve()
    if not path.is_file():
        raise ValueError(f'formal model does not exist: {path}')
    model = EntitySemanticF.load(path)
    known = _match_ids(known_A_match_ids, 'known_A_match_ids')
    if type(minimum_new_A_matches) is not int or minimum_new_A_matches < MINIMUM_NEW_MATCHES:
        raise ValueError('minimum_new_A_matches is too small')
    if type(minimum_validation_matches) is not int or minimum_validation_matches < MINIMUM_VALIDATION_MATCHES:
        raise ValueError('minimum_validation_matches is too small')
    at = parse_time(created_at or datetime.now(timezone.utc)).isoformat()
    plan: dict[str, Any] = {
        'schema': VERSION,
        'purpose': PURPOSE,
        'registration_status': REGISTRATION_STATUS,
        'created_at': at,
        'audit_version': AUDIT_VERSION,
        'f_version': F_VERSION,
        'selection_rule': SELECTION_RULE,
        'candidate_grid': list(PARAMETERS['regularizations']),
        'response_scope': 'A_purged_forward_validation_matches_only',
        'response_threshold': PREDICTION_CHANGE_THRESHOLD,
        'minimum_responsive_validation_matches': MINIMUM_RESPONSIVE_VALIDATION_MATCHES,
        'sensitivity_thresholds': list(SENSITIVITY_THRESHOLDS),
        'future_batch_label': _string(future_batch_label, 'future_batch_label'),
        'future_data_scope': _string(future_data_scope, 'future_data_scope'),
        'known_A_match_ids': known,
        'minimum_new_A_matches': minimum_new_A_matches,
        'minimum_validation_matches': minimum_validation_matches,
        'reference_formal_model': {
            'path': str(path), 'sha256': _sha256(path), 'model_version': model.model_version,
        },
        'outcome_use': {
            'fit_and_log_loss': 'A outcomes only within purged forward folds',
            'response_probe': 'outcomes forbidden; held-out validation states only',
            'B_C_outcomes': 'forbidden for f selection',
        },
        'promotion_allowed': False,
        'g_fit': False,
        'independent_blind_test': False,
        'scope': 'pre_registered_design_only; future_A_batch_required; no_current_audit_backfill',
    }
    plan['registration_id'] = VERSION + ':' + _digest(plan)[:24]
    return validate_registration(plan)


def validate_registered_audit(plan: Mapping[str, Any], report: Mapping[str, Any], *,
                              audit_started_at: str | datetime | None = None) -> dict[str, Any]:
    """Check that a future audit actually obeyed the registered design.

    The current Round125 retrospective report must fail this check because it
    has no registration ID and its A identities are already known.  That
    failure is intentional and is tested as a regression boundary.
    """
    registered = validate_registration(plan)
    if not isinstance(report, Mapping):
        raise ValueError('Audit report is required')
    started = parse_time(audit_started_at or datetime.now(timezone.utc))
    if started < parse_time(registered['created_at']):
        raise ValueError('Audit started before registration')
    protocol = report.get('protocol')
    if not isinstance(protocol, Mapping):
        raise ValueError('Audit protocol is missing')
    if (protocol.get('registration_status') != REGISTRATION_STATUS
            or protocol.get('registration_id') != registered['registration_id']
            or protocol.get('registration_sha256') != _digest(registered)):
        raise ValueError('Audit does not carry this pre-registration')
    for key in ('rule', 'response_scope', 'response_threshold',
                'minimum_responsive_validation_matches', 'candidate_grid'):
        expected = {
            'rule': registered['selection_rule'],
            'response_scope': registered['response_scope'],
            'response_threshold': registered['response_threshold'],
            'minimum_responsive_validation_matches': registered['minimum_responsive_validation_matches'],
            'candidate_grid': registered['candidate_grid'],
        }[key]
        if protocol.get(key) != expected:
            raise ValueError(f'Audit protocol field differs: {key}')
    ids = _match_ids(report.get('A_match_ids'), 'audit A_match_ids')
    known = set(registered['known_A_match_ids'])
    if known.intersection(ids):
        raise ValueError('Future audit A identities overlap registered known A identities')
    if len(ids) < registered['minimum_new_A_matches']:
        raise ValueError('Future audit has too few new A matches')
    if report.get('A_forward_validation_matches', 0) < registered['minimum_validation_matches']:
        raise ValueError('Future audit has too few held-out validation matches')
    if report.get('formal_model_version') != registered['reference_formal_model']['model_version']:
        raise ValueError('Future audit formal f differs from registered reference')
    source_artifact = report.get('source_artifact')
    if (not isinstance(source_artifact, Mapping)
            or source_artifact.get('formal_model_sha256') != registered['reference_formal_model']['sha256']):
        raise ValueError('Future audit formal f digest differs from registered reference')
    if report.get('capability_constrained_selection', {}).get('promotion_allowed') is not False:
        raise ValueError('Audit cannot authorize promotion')
    return {
        'status': 'PASS',
        'registration_id': registered['registration_id'],
        'new_A_match_count': len(ids),
        'validation_match_count': report['A_forward_validation_matches'],
        'independent_from_known_A': True,
        'promotion_allowed': False,
    }


__all__ = [
    'VERSION', 'PURPOSE', 'REGISTRATION_STATUS', 'MINIMUM_NEW_MATCHES',
    'MINIMUM_VALIDATION_MATCHES', 'create_registration', 'validate_registration',
    'validate_registered_audit',
]
