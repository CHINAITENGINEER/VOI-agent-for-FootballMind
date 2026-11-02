"""Tests for the entity f/g hand-off gate."""

from copy import deepcopy
import json

from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.entity_semantic_g_contract import (
    EntitySemanticGBinding, check_contract_readiness, check_f_binding, check_g_compatibility,
    check_g_training_readiness,
)
from football_agent.semantic_query_policy import SemanticQueryPolicy
from football_research.acquisition import entity_semantic_query
from tests.research.acquisition.test_entity_semantic_query import _fixture_setup


ROOT = 'var/f-entity-semantic-20261011-v2'


def _artifacts():
    contract = json.loads(open(f'{ROOT}/g-consumer-contract.json', encoding='utf-8').read())
    f_model = json.loads(open(f'{ROOT}/model.json', encoding='utf-8').read())
    return contract, f_model


def _target_material():
    def read_jsonl(name):
        return [json.loads(line) for line in open(f'{ROOT}/{name}', encoding='utf-8') if line.strip()]
    return (read_jsonl('selected-f-hindsight-targets.jsonl'),
            read_jsonl('selected-f-pre-read.jsonl'),
            read_jsonl('selected-f-post-read-audits.jsonl'))


def _g(contract, f_model):
    f_ids = contract['f_fit_physical_match_ids']
    return {
        'schema': 'future-entity-g-test-v1',
        'model_version': 'future-entity-g-test-v1:fixture',
        'metadata': {
            'input_schema': contract['input_schema'],
            'predictor_payload_sha256': contract['predictor_payload_sha256'],
            'representation_sha256': contract['representation_sha256'],
            'target': contract['target'],
            'training_physical_match_ids': ['understat:29001', 'understat:29002'],
            'predictor_training_physical_match_ids': list(f_ids),
            'evidence_origin': 'historical_reconstruction',
            'production_ready': False,
        },
    }


def test_current_f_artifact_matches_consumer_contract():
    contract, f_model = _artifacts()
    assert check_f_binding(contract, f_model) == []


def test_readiness_reports_waiting_for_g_without_inventing_a_model():
    contract, f_model = _artifacts()
    report = check_contract_readiness(contract, f_artifact=f_model)
    assert report['status'] == 'waiting_for_g'
    assert report['compatible'] is False
    assert any(row['code'] == 'g_artifact_missing' for row in report['warnings'])


def test_training_readiness_rejects_current_all_zero_targets_without_losing_audit_signal():
    contract, f_model = _artifacts()
    targets, pre, audits = _target_material()
    report = check_g_training_readiness(
        contract, targets, pre_read_rows=pre, post_read_audits=audits, f_artifact=f_model,
    )
    assert report['status'] == 'not_ready_zero_target_signal'
    assert report['ready_for_fit'] is False
    assert report['research_fit_eligible'] is False
    assert report['counts']['target_rows'] == 166
    assert report['counts']['valid_target_rows'] == 166
    assert report['counts']['nonzero_gain_rows'] == 0
    assert report['target_summary']['signal_status'] == 'zero_target_signal'
    assert report['paired_response_summary']['audits_checked'] == 166
    assert report['paired_response_summary']['feature_vector_changed_rows'] == 101
    assert report['paired_response_summary']['prediction_changed_rows'] == 0
    assert report['paired_response_summary']['max_probability_change'] == 0.0
    codes = {row['code'] for row in report['warnings']}
    assert {'zero_target_signal', 'predictor_insensitivity_interpretation'} <= codes


def test_training_readiness_requires_the_paired_query_and_audit_projections():
    contract, f_model = _artifacts()
    targets, _pre, _audits = _target_material()
    report = check_g_training_readiness(contract, targets, f_artifact=f_model)
    assert report['status'] == 'rejected'
    assert any(row['code'] == 'paired_evidence_missing' for row in report['issues'])


def test_training_readiness_allows_a_variable_research_target_but_reports_scope():
    contract, f_model = _artifacts()
    targets, pre, audits = _target_material()
    # One B and one C row provide a minimal synthetic variation test without
    # changing the frozen artifact on disk.  Both rows remain historical
    # reconstruction, so the result is research-only rather than prospective.
    selected = [deepcopy(targets[0]), deepcopy(targets[46])]
    selected_ids = {row['sample_id'] for row in selected}
    for index, row in enumerate(selected):
        row['target']['realized_gain_nats'] = .1 if index == 0 else -.05
    pre_selected = [deepcopy(row) for row in pre if row['sample_id'] in selected_ids]
    audit_selected = [deepcopy(row) for row in audits if row['sample_id'] in selected_ids]
    report = check_g_training_readiness(
        contract, selected, pre_read_rows=pre_selected, post_read_audits=audit_selected,
        f_artifact=f_model,
    )
    assert report['status'] == 'ready_for_research_fit'
    assert report['ready_for_fit'] is True
    assert report['prospective_fit_eligible'] is False
    assert report['counts']['positive_gain_rows'] == 1
    assert report['counts']['negative_gain_rows'] == 1
    assert any(row['code'] == 'prospective_rows_ineligible' for row in report['warnings'])


def test_training_readiness_rejects_a_hash_mismatch_in_the_pair():
    contract, f_model = _artifacts()
    targets, pre, audits = _target_material()
    selected = [deepcopy(targets[0])]
    selected_id = selected[0]['sample_id']
    pre_selected = [deepcopy(row) for row in pre if row['sample_id'] == selected_id]
    audit_selected = [deepcopy(row) for row in audits if row['sample_id'] == selected_id]
    pre_selected[0]['query']['query_sha256'] = 'tampered'
    report = check_g_training_readiness(
        contract, selected, pre_read_rows=pre_selected, post_read_audits=audit_selected,
        f_artifact=f_model,
    )
    assert report['status'] == 'rejected'
    assert any(row['code'] == 'paired_evidence_mismatch' for row in report['issues'])


def test_compatible_g_is_research_only_until_blind_test():
    contract, f_model = _artifacts()
    report = check_g_compatibility(contract, _g(contract, f_model), f_artifact=f_model,
                                   decision_physical_match_ids=['understat:29999'])
    assert report['status'] == 'compatible_research_only'
    assert report['compatible'] is True
    assert report['operational_eligible'] is False
    assert any(row['code'] == 'blind_test_incomplete' for row in report['warnings'])


def test_wrong_entity_representation_is_rejected():
    contract, f_model = _artifacts()
    candidate = _g(contract, f_model)
    candidate['metadata']['representation_sha256'] = '0' * 64
    report = check_g_compatibility(contract, candidate, f_artifact=f_model)
    assert report['status'] == 'rejected'
    assert any(row['code'] == 'g_representation_mismatch' for row in report['issues'])


def test_training_identity_overlap_is_rejected():
    contract, f_model = _artifacts()
    candidate = _g(contract, f_model)
    candidate['metadata']['training_physical_match_ids'] = [contract['f_fit_physical_match_ids'][0]]
    report = check_g_compatibility(contract, candidate, f_artifact=f_model)
    assert report['status'] == 'rejected'
    assert any(row['code'] == 'f_g_training_overlap' for row in report['issues'])


def test_decision_fixture_cannot_be_from_training_groups():
    contract, f_model = _artifacts()
    candidate = _g(contract, f_model)
    report = check_g_compatibility(contract, candidate, f_artifact=f_model,
                                   decision_physical_match_ids=['understat:29001'])
    assert any(row['code'] == 'decision_training_overlap' for row in report['issues'])


def test_old_g_schema_and_old_f_digest_are_rejected():
    contract, f_model = _artifacts()
    candidate = _g(contract, f_model)
    candidate['metadata'].update(input_schema='semantic-fixed-f-query-input-v1',
                                 predictor_payload_sha256='1' * 64)
    report = check_g_compatibility(contract, candidate, f_artifact=f_model)
    assert report['status'] == 'rejected'
    codes = {row['code'] for row in report['issues']}
    assert {'g_input_schema_mismatch', 'g_f_payload_mismatch'} <= codes


def test_g_model_version_metadata_must_bind_to_serialized_artifact():
    contract, f_model = _artifacts()
    candidate = _g(contract, f_model)
    candidate['metadata']['model_version'] = 'g:metadata-version'
    report = check_g_compatibility(contract, candidate, f_artifact=f_model)
    assert any(row['code'] == 'g_model_version_binding_mismatch' for row in report['issues'])


def test_runtime_binding_allows_only_the_checked_entity_query_protocol():
    state, prior, _a, b, bank, f_model = _fixture_setup()
    action = {
        'action_id': 'read-b', 'tool': 'read_url', 'url': b['url'],
        'cost': {'money': 0.0, 'requests': 1}, 'cost_status': 'simulated',
    }
    query = entity_semantic_query.query_input(
        state, action, model=f_model, prior_snapshot=prior, evidence_bank=bank,
    )
    contract = {
        'schema': 'entity-semantic-f-g-consumer-v1',
        'input_schema': entity_semantic_query.VERSION,
        'model_version': 'constructed-f',
        'predictor_payload_sha256': query['predictor_payload_sha256'],
        'representation_sha256': query['representation_sha256'],
        'f_fit_physical_match_ids': ['f:1'],
        'gain_fit_recommended_groups': ['B'], 'development_evaluation_groups': ['C'],
        'independent_blind_test': False, 'diagnostic_candidate_excluded': True,
        'target': 'conditional_frozen_predictor_log_loss_change',
        'numeric_g_fit': False, 'prospective_eligible': False,
    }

    class ConstructedG:
        model_version = 'constructed-g'
        metadata = {
            'input_schema': entity_semantic_query.VERSION,
            'predictor_payload_sha256': query['predictor_payload_sha256'],
            'representation_sha256': query['representation_sha256'],
            'target': 'conditional_frozen_predictor_log_loss_change',
            'training_physical_match_ids': ['g:1', 'g:2'],
            'predictor_training_physical_match_ids': ['f:1'],
            'evidence_origin': 'constructed_mechanism',
        }

        def __init__(self):
            self.calls = []

        def predict(self, value):
            self.calls.append(value)
            return {'support_status': 'supported', 'expected_signed_gain': .1,
                    'reason': 'constructed_runtime_binding_test'}

    raw = ConstructedG()
    bound = EntitySemanticGBinding(raw, contract=contract)
    policy = SemanticQueryPolicy(
        predictor_payload_sha256=query['predictor_payload_sha256'],
        input_schema=entity_semantic_query.VERSION, value_model=bound,
    )
    plan = policy.plan([query], {'limits': {'money': 0.0, 'requests': 1}, 'spent': {}})
    assert plan['selection']['decision'] == 'acquire'
    assert len(raw.calls) == 1

    raw.metadata['representation_sha256'] = '0' * 64
    try:
        bound.predict(query)
    except ValueError as exc:
        assert 'modified' in str(exc)
    else:
        raise AssertionError('A modified bound g must be rejected')
