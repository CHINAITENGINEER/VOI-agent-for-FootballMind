"""Regression tests for the future f-selection pre-registration gate."""

from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json

import pytest

from football_research.acquisition.entity_semantic_f_selection_registration import (
    REGISTRATION_STATUS,
    VERSION,
    create_registration,
    validate_registered_audit,
    validate_registration,
)


def _formal_model(tmp_path):
    source = tmp_path / 'model.json'
    source.write_text(json.dumps({
        'schema': 'quoted-entity-count-semantic-match-residual-f-v1',
        'parameters': {}, 'representation_sha256': 'x', 'coefficients': [],
        'metadata': {}, 'model_version': 'f:test',
    }), encoding='utf-8')
    return source


@pytest.fixture
def registration(tmp_path, monkeypatch):
    # Avoid constructing a real model for the schema-focused tests while still
    # exercising the public factory's artifact binding.
    from football_research.acquisition import entity_semantic_f_selection_registration as module
    monkeypatch.setattr(module.EntitySemanticF, 'load', lambda path: type('M', (), {
        'model_version': 'quoted-entity-count-semantic-match-residual-f-v1:test',
    })())
    path = _formal_model(tmp_path)
    return create_registration(
        formal_model_path=path, known_A_match_ids=['a1', 'a2'],
        future_batch_label='future-a-v1', future_data_scope='new independent A batch',
        created_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )


def test_registration_is_digest_bound_and_non_promoting(registration):
    assert registration['registration_status'] == REGISTRATION_STATUS
    assert registration['promotion_allowed'] is False
    assert validate_registration(registration)['registration_id'] == registration['registration_id']


def test_mutated_registration_is_rejected(registration):
    mutated = deepcopy(registration)
    mutated['candidate_grid'] = [10.0]
    with pytest.raises(ValueError, match='digest mismatch|candidate grid'):
        validate_registration(mutated)


def test_current_retrospective_report_cannot_be_backfilled(registration):
    report = {
        'protocol': {'registration_status': 'retrospective_audit_proposal_not_preregistered'},
        'A_match_ids': ['new-1'] * 0,
    }
    with pytest.raises(ValueError, match='does not carry this pre-registration'):
        validate_registered_audit(registration, report)


def test_registered_audit_requires_disjoint_new_batch(registration):
    protocol = {
        'registration_status': REGISTRATION_STATUS,
        'registration_id': registration['registration_id'],
        'registration_sha256': __import__(
            'football_research.acquisition.context_predictor', fromlist=['_digest'])._digest(registration),
        'rule': registration['selection_rule'],
        'response_scope': registration['response_scope'],
        'response_threshold': registration['response_threshold'],
        'minimum_responsive_validation_matches': registration['minimum_responsive_validation_matches'],
        'candidate_grid': registration['candidate_grid'],
    }
    report = {
        'protocol': protocol,
        'A_match_ids': ['a2'] + [f'new-{i}' for i in range(12)],
        'A_forward_validation_matches': 6,
        'formal_model_version': registration['reference_formal_model']['model_version'],
        'source_artifact': {'formal_model_sha256': registration['reference_formal_model']['sha256']},
        'capability_constrained_selection': {'promotion_allowed': False},
    }
    with pytest.raises(ValueError, match='overlap'):
        validate_registered_audit(registration, report)
