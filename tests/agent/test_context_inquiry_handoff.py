'''Local constructed providers exercise the real coordinator and audit chain.'''
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import json

import pytest

from football_agent.autonomous_context_loop import run_autonomous_context_loop
from football_agent.context_inquiry_handoff import ContextInquiryHandoff
from football_agent.decision_risk_experiment import ConstructedTools, proposal
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_evidence_state import EvidenceStateProtocol
from football_agent.inquiry_exploration import UniformInquiryExploration
from football_agent.inquiry_selection import LearningExplorationSelector
from football_agent.inquiry_verify import verify
from football_agent.query_value_contract import QueryValueAssessment, QueryValueGate
from football_agent.state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate
from football_agent.return_dataset import export_run
from football_research.evaluation.policy_experiment_protocol import manifest_hash

NOW = datetime(2026, 9, 29, 8, tzinfo=timezone.utc)
CUTOFF = '2026-09-29T12:00:00Z'
AVAILABLE = '2026-09-29T07:00:00Z'


def parent(*, mode='unknown', initial=None, episode='unknown-parent'):
    board = StateQueryBoard(target_key='match-one', prediction_cutoff=CUTOFF,
        hypotheses=[StateHypothesis('h-state', 'match-one', 'qualification', ('known', 'unknown'))],
        candidates=[StateQueryCandidate('qualification', 'match-one', CUTOFF, 'qualification', 'public', 'read', .01)])
    def policy(board, context, budget):
        value = (QueryValueAssessment.modeled('qualification', expected_value=0, cost=.01) if mode == 'modeled'
                 else QueryValueAssessment.unmeasured('qualification', cost=.01))
        return QueryValueGate.decide(board, [value], remaining_budget=budget, view='agent')
    def unexpected(plan, context):
        raise AssertionError('Parent test must not acquire information')
    return run_autonomous_context_loop(board=board, initial_context=initial or {'competition': 'fixture'},
        routes=[{'candidate_id': 'qualification', 'route_kind': 'context', 'provides_context_keys': ['qualification']}],
        episode_id=episode, decision_provider=policy, execute=unexpected, budget=0 if mode == 'budget' else .10)


def setup(tmp_path, *, cap=2, requests=2, keep_querying=False):
    seen = []
    clock = [NOW]
    class Provider:
        def configuration(self):
            return {'provider': 'constructed_handoff_provider', 'configured': True}
        def decide(self, public):
            seen.append(deepcopy(public))
            if not public['action_history']:
                answer = proposal(['fill_j', 'verify_k'])
            elif keep_querying:
                previous = public['action_history'][-1]['parameters']['url'].rsplit('/', 1)[-1]
                answer = proposal(['verify_k' if previous == 'fill_j' else 'fill_j'])
            else:
                answer = proposal([], kind='stop')
            return {'response': answer, 'receipt': {'real_provider_call': False}}
    tools = ConstructedTools(tmp_path, {'fill_j': 'yes', 'verify_k': 'no'}, clock=lambda: clock[0])
    workflow = InquiryWorkflow(tmp_path/'runs', Provider(), tools, clock=lambda: clock[0],
        protocol=EvidenceStateProtocol(), selector=LearningExplorationSelector(
            UniformInquiryExploration(7766554433), max_exploration_requests=cap))
    bridge = ContextInquiryHandoff(tmp_path/'handoffs', workflow, context_keys=['competition', 'qualification'],
        seed_urls=['https://example.com/fill_j', 'https://example.com/verify_k'],
        max_requests=requests, max_llm_calls=4)
    payload = {'parameters': {'context_inquiry_handoff': bridge.configuration()},
               'thresholds': {key: bridge.configuration()[key] for key in
                              ('max_requests', 'max_llm_calls', 'max_exploration_requests')}}
    freeze = {**payload, 'parameter_hash': manifest_hash(payload), 'frozen': True,
              'frozen_at': '2026-09-29T06:00:00Z', 'fit_partition': 'train',
              'selection_partition': 'validation', 'evaluation_partition': 'test'}
    return bridge, tools, seen, clock, freeze


def execute(bridge, freeze, result=None):
    return bridge.run(result=result or parent(), context_available_at=AVAILABLE, freeze_manifest=freeze)


def test_unknown_parent_executes_only_selected_generated_query_and_exports_real_trace(tmp_path):
    bridge, tools, seen, clock, freeze = setup(tmp_path)
    output = execute(bridge, freeze)
    run = output['run']
    selected = run['decisions'][0]['exploration_record']
    assert tools.calls == ['https://example.com/' + selected['chosen_id']]
    assert selected['probabilities'] == {'fill_j': .5, 'verify_k': .5}
    assert selected['expected_information_value'] is None
    assert run['state'] == 'STOPPED'
    assert run['budget']['http_attempts'] == 1
    assert 'qualification' in str(seen[0]['task']['initial_facts'])
    assert '7766554433' not in str(seen)
    assert not output['parent_context_updated'] and not output['prediction_updated']
    assert not output['independent_confirmation']
    proof = verify(tmp_path, run['run_id'])
    assert proof['new_network_requests'] == 0
    rows, source = export_run({'root': str(tmp_path), 'run_id': run['run_id'],
        'head_hash': output['run_head_hash'], 'cluster_id': 'constructed-match-one',
        'origin': 'constructed', 'cluster_basis': 'one constructed parent scenario'}, as_of=NOW.isoformat())
    assert len(rows) == 1 and rows[0]['pre_query']['selection_probability'] == .5
    assert rows[0]['observed_return']['realized_decision_utility'] is None
    assert rows[0]['pre_query']['documents'] == []


def test_start_persists_binding_without_calling_provider_or_tool(tmp_path):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    started = bridge.start(result=parent(), context_available_at=AVAILABLE, freeze_manifest=freeze)
    assert started['state'] == 'BOUND' and started['run']['state'] == 'READY'
    assert tools.calls == seen == []
    assert (bridge.root / (started['handoff_id'] + '.json')).is_file()


def test_repeat_run_reuses_finished_inquiry_without_refilling_budget(tmp_path):
    bridge, tools, seen, clock, freeze = setup(tmp_path)
    original = execute(bridge, freeze)
    calls = (list(tools.calls), len(seen))
    clock[0] += timedelta(days=1)
    restored = execute(bridge, freeze)
    assert restored['reused'] and restored['run_id'] == original['run_id']
    assert restored['run_head_hash'] == original['run_head_hash']
    assert (tools.calls, len(seen)) == calls


def test_exploration_cap_defers_without_inventing_numeric_value(tmp_path):
    bridge, tools, seen, _, freeze = setup(tmp_path, cap=1, requests=3, keep_querying=True)
    output = execute(bridge, freeze)
    assert len(tools.calls) == 1
    assert output['run_state'] == 'DEFERRED'
    decision = output['run']['decisions'][-1]
    assert decision['expected_gain'] is None
    assert decision['risk_assessment']['stop_certificate'] is None


@pytest.mark.parametrize('mode', ['modeled', 'budget'])
def test_non_unknown_value_defer_does_not_create_extra_budget(tmp_path, mode):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    with pytest.raises(ValueError):
        execute(bridge, freeze, parent(mode=mode))
    assert not tools.calls and not seen and not list((tmp_path/'runs').glob('*/run.json'))


@pytest.mark.parametrize('field,value', [('selected_candidates', ('prior',)), ('execution_entries', 2), ('declared_cost_spent', .01)])
def test_replay_returns_are_not_promoted_to_operational_context(tmp_path, field, value):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    with pytest.raises(ValueError, match='without prior replay queries'):
        execute(bridge, freeze, replace(parent(), **{field: value}))
    assert tools.calls == seen == []


@pytest.mark.parametrize('stamp', ['2026-09-29T08:01:00Z', '2026-09-30T00:00:00Z'])
def test_future_context_availability_stops_before_inquiry_creation(tmp_path, stamp):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    with pytest.raises(ValueError, match='availability'):
        bridge.run(result=parent(), context_available_at=stamp, freeze_manifest=freeze)
    assert tools.calls == seen == []


def test_expired_parent_cutoff_never_starts_a_new_inquiry(tmp_path):
    bridge, tools, seen, clock, freeze = setup(tmp_path)
    clock[0] = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='cutoff passed'):
        execute(bridge, freeze)
    assert tools.calls == seen == []


@pytest.mark.parametrize('change', ['hash', 'budget', 'threshold', 'future_freeze'])
def test_freeze_mismatch_stops_before_any_provider_or_tool_call(tmp_path, change):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    if change == 'hash': freeze['parameter_hash'] = '0' * 64
    elif change == 'budget': freeze['parameters']['context_inquiry_handoff']['max_requests'] += 1
    elif change == 'threshold': freeze['thresholds']['max_requests'] += 1
    else: freeze['frozen_at'] = '2026-09-29T09:00:00Z'
    if change in {'budget', 'threshold'}:
        freeze['parameter_hash'] = manifest_hash({key: freeze[key] for key in ('parameters', 'thresholds')})
    with pytest.raises(ValueError): execute(bridge, freeze)
    assert tools.calls == seen == []


def test_changed_parent_cannot_reset_an_existing_inquiry(tmp_path):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    original = execute(bridge, freeze)
    calls = (list(tools.calls), len(seen))
    with pytest.raises(ValueError, match='immutable inputs'):
        execute(bridge, freeze, parent(initial={'competition': 'changed'}))
    assert (tools.calls, len(seen)) == calls


def test_receipt_tampering_is_detected_before_resuming(tmp_path):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    original = execute(bridge, freeze)
    path = bridge.root / (original['handoff_id'] + '.json')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    receipt['run_id'] = 'wrong'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    calls = (list(tools.calls), len(seen))
    with pytest.raises(ValueError, match='receipt hash mismatch'):
        execute(bridge, freeze)
    assert (tools.calls, len(seen)) == calls


def test_interrupted_binding_recovers_existing_child_without_new_budget(tmp_path, monkeypatch):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    original = bridge._write
    def interrupt(path, body):
        if body['state'] == 'BOUND': raise KeyboardInterrupt('constructed interruption')
        original(path, body)
    monkeypatch.setattr(bridge, '_write', interrupt)
    with pytest.raises(KeyboardInterrupt):
        bridge.start(result=parent(), context_available_at=AVAILABLE, freeze_manifest=freeze)
    assert len(list((tmp_path/'runs').glob('*/run.json'))) == 1
    assert tools.calls == seen == []
    monkeypatch.setattr(bridge, '_write', original)
    output = execute(bridge, freeze)
    assert output['reused'] and len(tools.calls) == 1
    assert len(list((tmp_path/'runs').glob('*/run.json'))) == 1


def test_changed_actual_selector_configuration_is_rejected(tmp_path):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    bridge.workflow.selector = LearningExplorationSelector(UniformInquiryExploration(55), max_exploration_requests=2)
    with pytest.raises(ValueError, match='configuration changed'):
        execute(bridge, freeze)
    assert tools.calls == seen == []


def test_only_frozen_context_projection_reaches_the_query_generator(tmp_path):
    bridge, tools, seen, _, freeze = setup(tmp_path)
    source = parent(initial={'competition': 'fixture', 'unrequested_note': 'DO_NOT_SEND_THIS_NOTE'})
    output = execute(bridge, json.loads(json.dumps(freeze)), source)
    assert output['run_state'] == 'STOPPED'
    assert 'DO_NOT_SEND_THIS_NOTE' not in str(seen)
    assert 'qualification' in str(seen[0]['task']['initial_facts'])


def test_operational_clock_advancing_during_proposal_prevents_late_source_request(tmp_path, monkeypatch):
    bridge, tools, seen, clock, freeze = setup(tmp_path)
    original = bridge.workflow.provider.decide
    def slow(public):
        output = original(public)
        clock[0] = datetime(2026, 9, 29, 12, 1, tzinfo=timezone.utc)
        return output
    monkeypatch.setattr(bridge.workflow.provider, 'decide', slow)
    output = execute(bridge, freeze)
    assert len(seen) == 1 and tools.calls == []
    assert output['run_state'] in {'STOPPED', 'DEFERRED'}
    assert output['http_attempts'] == 0
