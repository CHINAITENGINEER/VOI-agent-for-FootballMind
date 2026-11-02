from copy import deepcopy
import runpy
from pathlib import Path

from football_agent.acquisition_boundary import PolicyChoice, policy_identity
from football_agent.acquisition_replay import verify_run
from football_agent.audit import stored_digest
from football_agent.experience import ExperienceService
from football_agent.query_value_contract import QueryValueAssessment
from football_agent.value_selection_policy import (LEGACY_VERSION, VERSION,
                                                    ValueAwareSelectionPolicy,
                                                    migrate_saved_identity)


helpers = runpy.run_path(str(Path(__file__).with_name('test_match_experience.py')))
match_decision = helpers['match_decision']


def modeled(_, candidate):
    return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=0.4, cost=0.05)


def legacy_identity(*, model_descriptor=None):
    configuration = {
        'method': 'LOCAL_VALUE_AWARE_SELECTION',
        'numeric_voi_calibrated': False,
        'unmeasured_policy': 'DEFER_OPEN_WORLD_VALUE_UNKNOWN',
    }
    if model_descriptor is not None:
        configuration['model_descriptor'] = deepcopy(model_descriptor)
    identity = {'version': LEGACY_VERSION, 'configuration': configuration}
    identity['fingerprint'] = stored_digest(identity)
    return identity


def test_supported_v1_identity_migrates_only_to_unbounded_v2():
    current = policy_identity(ValueAwareSelectionPolicy(modeled))
    saved = legacy_identity()
    assert migrate_saved_identity(saved, current) == current


def test_migration_rejects_tampering_model_change_and_cost_cap():
    current = policy_identity(ValueAwareSelectionPolicy(modeled))

    tampered = deepcopy(legacy_identity())
    tampered['configuration']['unmeasured_policy'] = 'STOP_UNKNOWN'
    assert migrate_saved_identity(tampered, current) is None

    broken_fingerprint = legacy_identity()
    broken_fingerprint['fingerprint'] = '0' * 64
    assert migrate_saved_identity(broken_fingerprint, current) is None

    extra_field = legacy_identity()
    extra_field['configuration']['value_budget'] = 0.1
    extra_field['fingerprint'] = stored_digest({k: extra_field[k] for k in ('version', 'configuration')})
    assert migrate_saved_identity(extra_field, current) is None

    capped = policy_identity(ValueAwareSelectionPolicy(modeled, value_budget=0.1))
    assert migrate_saved_identity(legacy_identity(), capped) is None


def test_resume_migrates_before_proposer_call_and_records_event(tmp_path):
    observed = []
    service = None
    run_id = None

    class Proposer:
        def propose(self, public):
            observed.append(service.get(run_id))
            return {'candidates': [match_decision(public)], 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(tmp_path, proposer_factory=lambda *_: Proposer(),
                                selection_policy=ValueAwareSelectionPolicy(modeled), mode='test_fixture')
    try:
        run_id = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        service.mutate(run_id, lambda state: state['config'].update(selection_policy=legacy_identity()))

        service.step(run_id)
        state = service.get(run_id)
        assert len(observed) == 1
        assert observed[0]['config']['selection_policy']['version'] == VERSION
        assert any(event['type'] == 'selection_policy_migrated' for event in observed[0]['events'])
        assert state['config']['selection_policy'] == policy_identity(service.selection_policy)
        assert state['spent']['llm_calls'] == 1
        assert state['spent']['http_calls'] == 0
        assert state['status'] == 'ready'
    finally:
        service.close()


def test_migration_mismatch_still_pauses_without_proposer_call(tmp_path):
    calls = []

    class Proposer:
        def propose(self, public):
            calls.append(public)
            return {'candidates': [match_decision(public)], 'recommended_index': 0, 'test_double': True}

    service = ExperienceService(tmp_path, proposer_factory=lambda *_: Proposer(),
                                selection_policy=ValueAwareSelectionPolicy(modeled, value_budget=0.1), mode='test_fixture')
    try:
        run_id = service.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        service.mutate(run_id, lambda state: state['config'].update(selection_policy=legacy_identity()))
        service.step(run_id)
        state = service.get(run_id)
        assert state['status'] == 'paused'
        assert 'POLICY_OR_PROPOSER_CHANGED' in state['last_error']
        assert calls == []
        assert state['spent'] == {'llm_calls': 0, 'http_calls': 0}
        assert not any(event['type'] == 'selection_policy_migrated' for event in state['events'])
    finally:
        service.close()


def test_migrated_run_replays_historical_v1_selection_and_current_v2_selection(tmp_path):
    tools_factory = lambda folder: helpers['MatchTools']()

    class LegacyPolicy:
        version = LEGACY_VERSION
        configuration = {
            'method': 'LOCAL_VALUE_AWARE_SELECTION',
            'numeric_voi_calibrated': False,
            'unmeasured_policy': 'DEFER_OPEN_WORLD_VALUE_UNKNOWN',
        }

        def select(self, context, batch):
            return PolicyChoice(context.task_id, context.snapshot_digest,
                                candidate_id=batch.candidates[0].candidate_id,
                                reason='构造 v1 历史选择。')

    class Proposer:
        def propose(self, public):
            response = match_decision(public)
            response['query'] = '迁移回放查询-' + str(len(public['documents']))
            return {'candidates': [response], 'recommended_index': 0, 'test_double': True}

    old = ExperienceService(tmp_path, proposer_factory=lambda *_: Proposer(),
                             selection_policy=LegacyPolicy(), tools_factory=tools_factory,
                             mode='test_fixture')
    try:
        run_id = old.create({'match': '构造甲队 vs 构造乙队'})['run_id']
        old.step(run_id)
        old.step(run_id)
    finally:
        old.close()

    current = ExperienceService(tmp_path, proposer_factory=lambda *_: Proposer(),
                                selection_policy=ValueAwareSelectionPolicy(modeled),
                                tools_factory=tools_factory, mode='test_fixture')
    try:
        current.step(run_id)
        state = current.get(run_id)
        assert state['config']['selection_policy'] == policy_identity(current.selection_policy)
        assert any(event['type'] == 'selection_policy_migrated' for event in state['events'])
        report = verify_run(current.folder(run_id))
        assert report['status'] == 'ready'
        assert report['accepted_decisions_verified'] == 2
        assert report['linked_http_receipts'] == 1
    finally:
        current.close()
