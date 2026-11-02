'''Behavioral tests using constructed navigation, never live outcome labels.'''
from copy import deepcopy
from datetime import timedelta

import pytest

from football_agent.context_transition import ContextTransition, ContextTransitionLedger
from football_agent.navigation_feedback import project_navigation_feedback
from test_information_navigation import (NOW, DOC, make, choice, query_schedule,
    read_new, stub_parent_and_raw_boundary)


def reference(flow, identity):
    run = flow.get(identity)
    return {'root': str(flow.store.root.parent), 'run_id': identity,
            'head_hash': run['events'][-1]['event_hash']}


def completed(tmp_path, monkeypatch, *, result=None):
    final = choice('defer')
    final['next_information_need'] = 'Verify year and all-competition coverage.'
    flow, identity, seen, requests, clock = make(tmp_path, [query_schedule, read_new, final], result=result)
    stub_parent_and_raw_boundary(monkeypatch, flow, identity)
    for _ in range(3):
        clock[0] += timedelta(seconds=1)
        flow.step(identity)
    return flow, identity, seen, requests, clock


def project(flow, identity, **kwargs):
    return project_navigation_feedback(reference(flow, identity),
        as_of=kwargs.pop('as_of', flow.get(identity)['events'][-1]['at']), **kwargs)


def test_real_interfaces_receive_changed_need_without_fake_value(tmp_path, monkeypatch):
    flow, identity, seen, requests, _ = completed(tmp_path, monkeypatch)
    report = project(flow, identity)
    rows = report['action_rows']
    assert [r['selected_action']['action'] for r in rows] == ['query_source', 'read_window', 'defer']
    assert rows[1]['reconsideration']['need_text_changed'] is True
    assert rows[1]['reconsideration']['information_need'] == 'Verify year and all-competition coverage.'
    assert rows[0]['reconsideration']['need_text_changed'] is False
    assert all(r['selection_probability'] is None and r['task_outcome_label'] is None
               and r['realized_decision_utility'] is None and r['expected_information_value'] is None
               and r['eligible_for_value_calibration'] is False for r in rows)
    assert all(key.startswith('navigation.') for key in report['next_context'])
    assert report['new_model_calls'] == report['new_network_requests'] == 0
    assert len(seen) == 3 and len(requests) == 1


def test_count_units_are_separate_and_never_recharge_parent(tmp_path, monkeypatch):
    flow, identity, *_ = completed(tmp_path, monkeypatch)
    report = project(flow, identity)
    costs = report['costs']['spent']
    assert costs == {'llm_calls': 3, 'http_attempts': 1, 'window_reads': 1, 'delivered_body_chars': 1200}
    assert all(sum(r['observed_cost'][k] for r in report['action_rows']) == value for k, value in costs.items())
    assert not any(report['costs_without_accepted_choice'].values())
    assert report['costs']['actual_money'] is report['costs']['loss_unit_conversion'] is None
    before = report['action_rows'][0]['pre_action_value_gate']['groups']
    after = report['action_rows'][1]['pre_action_value_gate']['groups']
    assert before[0]['value_gate']['selection_basis'] == 'withhold_open_world_value_unknown'
    assert after[0]['value_gate']['selection_basis'] == 'budget_gate'
    assert after[1]['value_gate']['selection_basis'] == 'withhold_open_world_value_unknown'
    for group in before + after:
        assert group['value_gate']['action'] == 'DEFER'
        assert all(a['expected_value'] is None and a['net_value'] is None
                   for a in group['value_gate']['assessments'])


def test_causal_prefix_separates_source_arrival_body_access_and_later_need(tmp_path, monkeypatch):
    flow, identity, *_ = completed(tmp_path, monkeypatch)
    events = flow.get(identity)['events']
    initial = project(flow, identity, through_sequence=1, as_of=events[0]['at'])
    assert len(initial['visible_documents']) == 1 and initial['action_rows'] == []
    assert initial['costs']['spent']['llm_calls'] == 0
    returned = next(e for e in events if e['type'] == 'navigation_acquisition_returned')
    arrival = project(flow, identity, through_sequence=returned['seq'], as_of=returned['at'])
    assert arrival['visible_documents'][-1]['visible_windows'] == []
    assert arrival['action_rows'][0]['reconsideration'] is None
    assert arrival['next_context']['navigation.information_need'] == 'Confirm fixture identity.'
    opened = next(e for e in events if e['type'] == 'navigation_window_opened')
    reading = project(flow, identity, through_sequence=opened['seq'], as_of=opened['at'])
    assert len(reading['visible_documents'][-1]['visible_windows']) == 1
    assert reading['action_rows'][1]['reconsideration'] is None
    assert reading['next_context']['navigation.information_need'] == 'Confirm fixture identity.'
    assert 'Verify year' not in str(initial) + str(arrival) + str(reading)


@pytest.mark.parametrize('route', ['model', 'source'])
def test_unknown_failures_retain_costs_and_pending_without_negative_labels(tmp_path, monkeypatch, route):
    replies = [TimeoutError()] if route == 'model' else [query_schedule]
    flow, identity, _, _, _ = make(tmp_path, replies, tools_error=TimeoutError())
    stub_parent_and_raw_boundary(monkeypatch, flow, identity)
    flow.step(identity)
    report = project(flow, identity)
    assert report['costs']['spent']['llm_calls'] == 1
    assert report['costs']['spent']['http_attempts'] == (route == 'source')
    assert report['pending']['kind'] == ('llm' if route == 'model' else 'http')
    assert 'existing_call_requires_reconciliation' in report['value_reassessment']['operational_guards']
    if route == 'model':
        assert report['action_rows'] == []
        assert report['costs_without_accepted_choice']['llm_calls'] == 1
    else:
        assert report['action_rows'][0]['source_returns'] == []
        assert report['action_rows'][0]['realized_decision_utility'] is None


def test_failed_http_return_is_an_observation_not_a_resolved_fact(tmp_path, monkeypatch):
    failed = {**deepcopy(DOC), 'document_id': 'constructed-failed',
        'url': 'https://example.com/schedule', 'status': 'FAILED', 'text': '', 'links': [], 'http_status': 503}
    flow, identity, _, _, _ = make(tmp_path, [query_schedule, choice()], result=failed)
    stub_parent_and_raw_boundary(monkeypatch, flow, identity)
    flow.step(identity); flow.step(identity)
    report = project(flow, identity)
    row = report['action_rows'][0]
    assert row['source_returns'][0]['status'] == 'FAILED'
    assert row['task_outcome_label'] is row['risk_after'] is None
    assert report['domain_truth_updated'] is False


def test_pre_action_gate_sees_only_exposed_routes_and_unopened_windows(tmp_path, monkeypatch):
    flow, identity, *_ = completed(tmp_path, monkeypatch)
    initial = project(flow, identity, through_sequence=1)
    options = [o for g in initial['value_reassessment']['groups'] for o in g['options']]
    assert not any(o.get('url') == DOC['url'] for o in options)
    assert any(o.get('url') == 'https://example.com/schedule' for o in options)
    assert all(o.get('document_id') != 'constructed-new' for o in options)
    assert not any(o.get('window_id') == initial['visible_documents'][0]['catalog'][0]['window_id'] for o in options)


def test_projection_uses_pinned_head_and_explicit_time(tmp_path, monkeypatch):
    flow, identity, *_ = completed(tmp_path, monkeypatch)
    ref = reference(flow, identity)
    with pytest.raises(ValueError, match='head changed'):
        project_navigation_feedback({**ref, 'head_hash': '0' * 64}, as_of=NOW + timedelta(seconds=9))
    with pytest.raises(ValueError, match='available'):
        project(flow, identity, as_of=NOW - timedelta(seconds=1))
    with pytest.raises(ValueError, match='available'):
        project(flow, identity, through_sequence=14, as_of=NOW)
    with pytest.raises(ValueError, match='existing event'):
        project(flow, identity, through_sequence=True)


def test_existing_context_ledger_replays_projected_transitions(tmp_path, monkeypatch):
    flow, identity, *_ = completed(tmp_path, monkeypatch)
    report = project(flow, identity)
    records = [ContextTransition(**{k: v for k, v in r.items()
        if k not in {'schema_version', 'hidden_result_included'}}) for r in report['transitions']]
    ledger = ContextTransitionLedger(tmp_path / 'context')
    ledger.append(records)
    replay = ledger.replay_episode(identity)
    assert replay['final_context'] == report['next_context']
    assert len(ledger.entries()) == len(report['transitions'])
    assert 'existing_run_terminal' in report['value_reassessment']['operational_guards']
    assert report['value_reassessment']['optimal_stop_proven'] is False
