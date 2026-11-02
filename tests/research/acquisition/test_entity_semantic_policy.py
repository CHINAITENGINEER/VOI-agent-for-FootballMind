"""Entity f/g boundary tests for the shared query policy."""

import pytest

from football_agent.semantic_query_policy import SemanticQueryPolicy
from football_research.acquisition import entity_semantic_query
from tests.research.acquisition.test_entity_semantic_query import _fixture_setup


def _query():
    state, prior, _a, b, bank, model = _fixture_setup()
    action = {
        'action_id': 'read-b',
        'tool': 'read_url',
        'url': b['url'],
        'cost': {'money': 0.0, 'requests': 1},
        'cost_status': 'simulated',
    }
    return entity_semantic_query.query_input(
        state, action, model=model, prior_snapshot=prior, evidence_bank=bank,
    )


def test_shared_policy_accepts_entity_protocol_and_defers_unknown_value():
    query = _query()
    policy = SemanticQueryPolicy(
        predictor_payload_sha256=query['predictor_payload_sha256'],
        input_schema=entity_semantic_query.VERSION,
    )
    plan = policy.plan(
        [query],
        {'limits': {'money': 0.0, 'requests': 1.0}, 'spent': {}},
        candidate_generation_complete=True,
    )
    assert plan['selection']['decision'] == 'defer'
    assert plan['selection']['reason'] == 'context_gain_support_unknown'
    assert plan['candidates'][0]['value']['status'] == 'INELIGIBLE'
    assert plan['candidates'][0]['value']['expected_signed_gain'] is None
    assert plan['candidates'][0]['reason'] == 'no_g_bound_to_this_semantic_f'


def test_old_policy_cannot_consume_entity_query_bytes():
    query = _query()
    policy = SemanticQueryPolicy(
        predictor_payload_sha256=query['predictor_payload_sha256'],
    )
    with pytest.raises(ValueError, match='modified'):
        policy.plan(
            [query],
            {'limits': {'money': 0.0, 'requests': 1.0}, 'spent': {}},
        )


def test_entity_policy_rejects_g_bound_to_old_binary_representation():
    query = _query()

    class WrongRepresentationG:
        model_version = 'wrong-representation-g'
        metadata = {
            'input_schema': entity_semantic_query.VERSION,
            'predictor_payload_sha256': query['predictor_payload_sha256'],
            'representation_sha256': '0' * 64,
            'target': 'conditional_frozen_predictor_log_loss_change',
            'training_physical_match_ids': ['g:1'],
            'predictor_training_physical_match_ids': ['f:1'],
            'evidence_origin': 'constructed_mechanism',
        }

    import pytest
    with pytest.raises(ValueError, match='entity representation'):
        SemanticQueryPolicy(
            predictor_payload_sha256=query['predictor_payload_sha256'],
            input_schema=entity_semantic_query.VERSION,
            value_model=WrongRepresentationG(),
        )
