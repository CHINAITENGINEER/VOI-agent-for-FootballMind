"""Independent R196 execution audit; every source/receipt is constructed."""
from copy import deepcopy
from datetime import timedelta
import json
import socket

import pytest

from football_agent.context_voi_policy import ContextVOIPolicy
from football_agent.learned_voi_runtime import LearnedVOIRuntime
from football_agent.learned_voi_replay import replay_saved_run
from football_research.acquisition.joint_planner import UNITS
from football_research.acquisition.state import parse_time
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from test_context_runtime import collected, frozen, read


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('AUDIT_NETWORK_DISABLED')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)


@pytest.fixture
def source(frozen, tmp_path, monkeypatch):
    folder, *rest = collected(frozen, tmp_path, monkeypatch)
    return folder, read(folder/'configuration.json'), read(folder/'run.json'), frozen


def rerun(source, tmp_path, *, status='FAILED', times=None, exploration_count=2,
          receipts=None, resource_requests=1, execution_error=False,
          exploration_limits=None, actual_overrides=None):
    folder, config, original, frozen = source
    base = MarketPriorPolicy.from_manifest(config['policy_manifest']['base_policy'])
    policy = ContextVOIPolicy(base, frozen[0],
        physical_match_id=config['policy_manifest']['physical_match_id'],
        predictor_artifact=(folder/'context-predictor.json').read_bytes())
    actions = [c['selected_action'] for c in original['cycles'] if c['selected_action']]
    stamp = parse_time(config['created_at'])
    clock_values = iter(times or [stamp+timedelta(seconds=i) for i in range(12)])
    executed = []

    def propose(public):
        index = len(public['attempts'])
        return [deepcopy(actions[index])] if index < len(actions) else []

    def execute(action, public):
        executed.append(deepcopy(public))
        if execution_error:
            raise OSError('constructed unreconciled transport failure')
        if receipts is not None:
            return deepcopy(receipts[len(executed)-1])
        return {'status': status, 'actual_cost': {**{u: resource_requests if u == 'requests' else 0 for u in UNITS},
                **(actual_overrides or {})},
                'receipt': {'scope': 'constructed_audit', 'hidden_outcome': 'SUPERVISOR_ONLY_SENTINEL'}}

    runtime = LearnedVOIRuntime(policy, seed=197, exploration_limits=exploration_limits or config['exploration_limits'],
        max_exploration_actions=exploration_count, max_steps=4)
    result = runtime.run(config['initial_state'], run_id='constructed-execution-audit',
        limits=config['limits'], propose=propose, execute=execute, output=tmp_path/'audit-run',
        provider_version='constructed-audit', generator_version='constructed-audit',
        clock=lambda: next(clock_values), evidence_match=config['evidence_projection']['match'],
        evidence_version=config['evidence_projection']['version'])
    return result, executed


@pytest.mark.parametrize('status', ['FAILED', 'NO_DATA'])
def test_context_failed_returns_spend_budget_and_update_immediately(source, tmp_path, status):
    result, executed = rerun(source, tmp_path, status=status)
    assert len(executed) == result['exploration_actions'] == 2
    assert result['reserved_cost']['requests'] == 2
    assert result['exploration_reserved_cost']['requests'] == 2
    assert result['measured_resources']['requests'] == 2
    assert result['documents'] == []
    assert all(a['status'] == status for a in result['attempts'])
    for cycle in result['cycles'][:2]:
        assert cycle['execution_status'] == 'returned'
        assert cycle['public_after']['decision_at'] == cycle['returned_at']
        assert cycle['prediction_after'] == cycle['public_after']['prediction']['probabilities']
    assert executed[1]['attempts'] == result['attempts'][:1]
    assert 'SUPERVISOR_ONLY_SENTINEL' not in json.dumps(executed)
    assert all(c['terminal']['stop_certificate'] is None for c in result['cycles'] if 'terminal' in c)


def test_context_zero_exploration_never_queries_unknown(source, tmp_path):
    result, executed = rerun(source, tmp_path, exploration_count=0)
    assert executed == []
    assert result['exploration_actions'] == result['reserved_cost']['requests'] == 0
    assert result['terminal_selection']['decision'] == 'defer'
    assert result['terminal_selection']['stop_certificate'] is None


def test_context_actual_request_overrun_halts_before_next_query(source, tmp_path):
    result, executed = rerun(source, tmp_path, resource_requests=3)
    assert len(executed) == 1
    assert result['reserved_cost']['requests'] == result['exploration_reserved_cost']['requests'] == 3
    assert result['terminal_selection']['reason'] == 'actual_resource_budget_overrun'
    assert result['cycles'][0]['execution_status'] == 'returned'
    assert result['cycles'][0]['public_after'] is None
    assert result['cycles'][0]['update']['reason'] == 'actual_resource_budget_overrun'
    assert result['final_probabilities'] == result['cycles'][0]['prediction_before']


def test_context_unreconciled_failure_keeps_reservation_and_never_retries(source, tmp_path):
    result, executed = rerun(source, tmp_path, execution_error=True)
    assert len(executed) == 1
    assert result['reserved_cost']['requests'] == result['exploration_reserved_cost']['requests'] == 1
    assert result['attempts'] == []
    assert result['cycles'][0]['execution_status'] == 'execution_outcome_unknown'
    assert result['terminal_selection']['reason'] == 'execution_reconciliation_required'


def test_context_exploration_only_resource_cap_honors_actual_spend(source, tmp_path):
    _, config, _, _ = source
    result, executed = rerun(source, tmp_path,
        exploration_limits={**config['exploration_limits'], 'latency_ms': 1},
        actual_overrides={'latency_ms': 3})
    assert len(executed) == 1, 'Exploration-only resource cap must halt before another actual request'
    assert result['exploration_reserved_cost']['latency_ms'] == 3
    assert result['terminal_selection']['reason'] == 'actual_resource_budget_overrun'


def test_context_late_return_preserves_last_prematch_forecast(source, tmp_path):
    folder, config, original, _ = source
    cutoff = parse_time(config['initial_state']['prediction_cutoff'])
    receipt = next(read(folder/f"receipt-{c['step']:03}.json") for c in original['cycles']
        if c['selected_action'] and read(folder/f"receipt-{c['step']:03}.json")['status'] == 'OK')
    receipt['document']['available_at'] = (cutoff+timedelta(seconds=1)).isoformat()
    receipt['document']['observed_at'] = cutoff.isoformat()
    result, executed = rerun(source, tmp_path, receipts=[receipt], times=[
        cutoff-timedelta(seconds=2), cutoff-timedelta(seconds=1),
        cutoff+timedelta(seconds=2), cutoff+timedelta(seconds=3)])
    cycle = result['cycles'][0]
    assert len(executed) == 1
    assert result['documents'] == []
    assert cycle['public_after'] is None
    assert cycle['admission']['reason'] == 'after_information_cutoff'
    assert cycle['update']['reason'] == 'after_information_cutoff'
    assert result['final_probabilities'] == cycle['prediction_before']
    assert result['terminal_selection']['reason'] == 'forecast_cutoff_elapsed'


def test_context_future_document_is_not_visible_at_actual_return(source, tmp_path):
    folder, config, original, _ = source
    stamp = parse_time(config['created_at'])
    receipt = next(read(folder/f"receipt-{c['step']:03}.json") for c in original['cycles']
        if c['selected_action'] and read(folder/f"receipt-{c['step']:03}.json")['status'] == 'OK')
    receipt['document']['observed_at'] = (stamp+timedelta(seconds=1)).isoformat()
    receipt['document']['available_at'] = (stamp+timedelta(seconds=3)).isoformat()
    result, executed = rerun(source, tmp_path, receipts=[receipt], exploration_count=1)
    assert len(executed) == 1
    assert result['documents'] == []
    assert result['cycles'][0]['admission']['reason'] == 'after_actual_return_clock'
    assert result['cycles'][0]['public_after']['documents'] == []


def test_context_clock_cannot_move_back_before_previous_return(source, tmp_path):
    _, config, _, _ = source
    stamp = parse_time(config['created_at'])
    with pytest.raises(ValueError, match='Decision clock cannot precede'):
        rerun(source, tmp_path, times=[stamp+timedelta(seconds=n)
            for n in [0, 1, 3, 2, 4, 5, 6]])


def test_context_finished_clock_cannot_precede_the_final_return(source, tmp_path):
    _, config, _, _ = source
    stamp = parse_time(config['created_at'])
    with pytest.raises(ValueError, match='Finished clock cannot precede'):
        rerun(source, tmp_path, times=[stamp+timedelta(seconds=n)
            for n in [0, 1, 2, 3, 4, 5, 3]])


@pytest.mark.parametrize('artifact', ['context-predictor.json', 'context-calibration.json'])
def test_context_snapshot_tamper_rejected_before_replay(source, tmp_path, artifact):
    folder, *_ = source
    path = folder/artifact
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='SHA|snapshot|binding'):
        replay_saved_run(folder, output=tmp_path/'tamper-replay', project_root=tmp_path)


def test_context_exact_replay_checks_immediate_post_snapshot(source, tmp_path):
    folder, config, original, _ = source
    cycle = original['cycles'][0]
    cycle['public_after']['prediction']['probabilities'][0] += .01
    (folder/'run.json').write_text(json.dumps(original), encoding='utf-8')
    (folder/'cycle-000.json').write_text(json.dumps(cycle), encoding='utf-8')
    with pytest.raises(ValueError, match='Full replay differs'):
        replay_saved_run(folder, output=tmp_path/'tamper-replay', project_root=tmp_path)

