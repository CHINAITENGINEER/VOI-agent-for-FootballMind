from copy import deepcopy
import inspect
import importlib.util
import json
import math
from pathlib import Path

import pytest

from football_agent.budgeted_acquisition_comparison import (
    NO_RETURN, POLICIES, choose_action, compare_case, enumerate_tapes,
    execute_tape, experiment_cases, experiment_manifest, residual_target_risk,
    run_experiment, with_dropout,
)
from football_agent.decision_value import FiniteDecisionModel, ObservationTool


@pytest.fixture(scope='module')
def cases():
    return experiment_cases()


@pytest.fixture(scope='module')
def report():
    return run_experiment(experiment_manifest())


def rows(report, case, policy):
    return [r for r in report['cases'][case]['trajectories'] if r['policy'] == policy]


def test_policy_api_does_not_accept_realized_tapes_or_truth():
    assert set(inspect.signature(choose_action).parameters) == {
        'model', 'state', 'policy', 'remaining_budget', 'queries_left', 'order'}


def test_hidden_return_changes_cannot_change_pre_exposure_choice(cases):
    model = cases['qualified_rotation_unknown']['model']
    tapes = list(enumerate_tapes(model))
    runs = [execute_tape(model, t, policy='adaptive_voi_2', budget=.08,
                         max_queries=2, order=tuple(model.tools)) for t in tapes]
    assert all(r['trace'][0]['decision'] == runs[0]['trace'][0]['decision'] for r in runs)
    # Two private reserve qualities, identical exposed starters: identical full trace.
    starters = [r for t, r in zip(tapes, runs) if t['returns']['lineup'] == 'starters']
    assert len(starters) == 2 and starters[0]['trace'] == starters[1]['trace']
    encoded = json.dumps(runs[0]['trace'])
    for forbidden in ('theta_index', 'conditional_loss', 'tape_index', 'realized_loss'):
        assert forbidden not in encoded


def test_returns_change_second_action_and_stopping(report):
    seen = {}
    for r in rows(report, 'qualified_rotation_unknown', 'adaptive_voi_2'):
        first = r['trace'][0]
        assert first['decision']['action_id'] == 'lineup'
        seen.setdefault(first['returned'], set()).add(r['trace'][1]['decision']['action_id'])
    assert seen == {'starters': {'STOP'}, 'reserves': {'reserve_quality'}}


def test_replanning_ablation_keeps_initial_values_and_spends_more(report):
    aggregate = report['cases']['qualified_rotation_unknown']['aggregate']
    frozen, adaptive = aggregate['frozen_voi_order'], aggregate['adaptive_voi_2']
    assert frozen['expected_acquisition_cost'] == pytest.approx(.035)
    assert adaptive['expected_acquisition_cost'] == pytest.approx(.025)
    assert adaptive['expected_query_count'] == pytest.approx(1.5)
    assert adaptive['expected_target_loss'] == pytest.approx(frozen['expected_target_loss'])


def test_same_caps_common_tapes_and_full_probability_mass(report):
    assert report['all_budget_violations'] == 0
    for case in report['cases'].values():
        assert case['private_tape_mass'] == pytest.approx(1)
        assert case['all_trajectories_stopped']
        for policy in POLICIES:
            group = [r for r in case['trajectories'] if r['policy'] == policy]
            assert set(r['tape_index'] for r in group) == set(range(case['private_tape_count']))
            assert math.fsum(r['weight'] for r in group) == pytest.approx(1)
            for run in group:
                assert run['budget'] == case['budget'] and run['max_queries'] == case['max_queries']
                assert run['spent'] <= run['budget'] + 1e-12
                assert run['spent'] == pytest.approx(sum(e.get('cost', 0) for e in run['trace']))
                assert run['query_count'] <= run['max_queries']


def test_random_strategy_integrates_all_orders_without_cherry_picked_seed(report):
    group = rows(report, 'rotation_tight_budget', 'random_order')
    for index in range(4):
        same = [r for r in group if r['tape_index'] == index]
        assert {tuple(r['order']) for r in same} == {('lineup', 'reserve_quality'), ('reserve_quality', 'lineup')}
        assert [r['weight'] for r in same] == pytest.approx([.125, .125])
    aggregate = report['cases']['rotation_tight_budget']['aggregate']['random_order']
    assert aggregate['expected_acquisition_cost'] == pytest.approx((.015 + .02) / 2)
    assert aggregate['model_target_resolution_rate'] == pytest.approx(.25)


def test_one_step_can_lose_to_fixed_and_two_step_recovers_complementarity(report):
    aggregate = report['cases']['complementary_xor_reference']['aggregate']
    assert aggregate['adaptive_voi_1']['expected_query_count'] == 0
    assert aggregate['adaptive_voi_2']['expected_query_count'] == 2
    assert aggregate['adaptive_voi_1']['expected_total_risk'] > aggregate['fixed_order']['expected_total_risk']
    assert aggregate['adaptive_voi_2']['expected_total_risk'] == pytest.approx(aggregate['fixed_order']['expected_total_risk'])
    # Cost with zero immediate value is NOT automatically wasted: it unlocks context_b.
    assert aggregate['adaptive_voi_2']['expected_zero_one_step_value_cost'] == pytest.approx(.02)


def test_finite_two_step_optimum_respects_query_cap(cases):
    model = cases['complementary_xor_reference']['model']
    one = compare_case(model, budget=.04, max_queries=1)
    assert one['aggregate']['adaptive_voi_2']['expected_query_count'] == 0
    assert one['budget_violations'] == 0


def test_dropout_is_costly_but_does_not_shift_belief(cases, report):
    model = cases['lineup_source_degraded']['model']
    state = model.initial_state()
    after = model.observe(state, 'lineup', NO_RETURN)
    assert after.belief == pytest.approx(state.belief)
    for r in rows(report, 'lineup_source_degraded', 'fixed_order'):
        if r['trace'][0]['returned'] == NO_RETURN:
            assert r['trace'][0]['cost'] == .015
            assert r['trace'][0]['prediction_after'] == pytest.approx(r['trace'][0]['decision']['prediction_before'])
    agg = report['cases']['lineup_source_degraded']['aggregate']
    assert agg['fixed_order']['expected_failed_request_cost'] == pytest.approx(.9 * .015)
    assert {r['selected_actions'][0] for r in rows(report, 'lineup_source_degraded', 'adaptive_voi_2')} == {'reserve_quality'}


def test_known_total_unavailability_stops_without_falsely_resolving(report):
    adaptive = rows(report, 'all_returns_missing', 'adaptive_voi_2')
    assert all(r['spent'] == 0 and r['terminal_status'] == 'UNRESOLVED_AT_STOP' for r in adaptive)
    fixed = report['cases']['all_returns_missing']['aggregate']['fixed_order']
    assert fixed['expected_failed_request_cost'] == pytest.approx(.035)


def test_reasonable_stop_is_not_zero_uncertainty_or_factual_correctness(report):
    group = rows(report, 'qualified_reserves_known_strong', 'adaptive_voi_2')
    assert all(r['query_count'] == 0 and r['remaining_reducible_risk'] > 0 for r in group)
    assert all(r['terminal_status'] == 'UNRESOLVED_AT_STOP' for r in group)
    assert report['real_question_resolution_rate'] is None


def test_target_irrelevant_news_can_resolve_latent_but_no_target_question(cases, report):
    model = cases['irrelevant_high_entropy_news']['model']
    assert residual_target_risk(model, model.initial_state()) == pytest.approx(0)
    group = report['cases']['irrelevant_high_entropy_news']['aggregate']
    assert group['fixed_order']['expected_zero_one_step_value_cost'] == pytest.approx(.001)
    assert group['adaptive_voi_2']['expected_query_count'] == 0


def test_duplicate_reports_do_not_double_count_target_information(report):
    group = report['cases']['duplicate_shared_report']['aggregate']
    assert group['frozen_voi_order']['expected_query_count'] == 2
    assert group['adaptive_voi_2']['expected_query_count'] == 1
    assert group['frozen_voi_order']['expected_target_loss'] == pytest.approx(group['adaptive_voi_2']['expected_target_loss'])


def test_noisy_same_evidence_alias_has_shared_tape_and_cannot_be_read_twice():
    tool = ObservationTool(('yes', 'no'), {'a': (.8, .2), 'b': (.2, .8)}, .01, evidence_id='shared')
    model = FiniteDecisionModel({'a': (.8, .1, .1), 'b': (.1, .1, .8)}, {'a': .5, 'b': .5}, {'x': tool, 'y': tool})
    tapes = list(enumerate_tapes(model))
    assert len(tapes) == 4
    assert all(t['returns']['x'] == t['returns']['y'] for t in tapes)
    assert math.fsum(t['weight'] for t in tapes) == pytest.approx(1)
    report = compare_case(model, budget=.04)
    assert all(r['query_count'] <= 1 for r in report['trajectories'])
    broken = deepcopy(tapes[0])
    broken['returns']['y'] = 'no' if broken['returns']['x'] == 'yes' else 'yes'
    with pytest.raises(ValueError, match='same evidence ID'):
        execute_tape(model, broken, policy='fixed_order', budget=.04, max_queries=2, order=('x', 'y'))


def test_brier_and_log_loss_are_not_mixed_and_updater_is_unchanged(cases):
    original = cases['qualified_rotation_unknown']['model'].manifest()
    altered = deepcopy(original)
    altered.update(loss='brier', cost_unit='expected_multiclass_brier')
    model = FiniteDecisionModel.from_manifest(altered)
    before = deepcopy(model.manifest())
    report = compare_case(model, budget=.04)
    assert report['cost_unit'] == 'expected_multiclass_brier'
    assert model.manifest() == before
    for row in report['aggregate'].values():
        assert math.isfinite(row['expected_total_risk'])


def test_zero_cap_zero_budget_and_empty_tool_set():
    model = FiniteDecisionModel({'a': (.8, .1, .1)}, {'a': 1}, {})
    report = compare_case(model, budget=0, max_queries=0)
    assert len(report['trajectories']) == len(POLICIES)
    assert all(r['spent'] == 0 and r['query_count'] == 0 for r in report['trajectories'])


def test_zero_budget_excludes_positive_cost_tools(cases):
    report = compare_case(cases['qualified_rotation_unknown']['model'], budget=0)
    assert all(r['spent'] == 0 and r['query_count'] == 0 for r in report['trajectories'])


@pytest.mark.parametrize('budget,cap', [(-1, 2), (True, 2), (float('nan'), 2), (float('inf'), 2), (.04, -1), (.04, 3), (.04, True)])
def test_invalid_budget_or_horizon_is_rejected(cases, budget, cap):
    with pytest.raises(ValueError):
        compare_case(cases['qualified_rotation_unknown']['model'], budget=budget, max_queries=cap)


def test_unknown_policy_order_and_tape_are_rejected(cases):
    model = cases['qualified_rotation_unknown']['model']
    with pytest.raises(ValueError, match='every action'):
        compare_case(model, budget=.04, fixed_order=('lineup', 'lineup'))
    with pytest.raises(ValueError, match='unknown acquisition'):
        choose_action(model, model.initial_state(), policy='oracle', remaining_budget=.04, queries_left=2, order=tuple(model.tools))
    with pytest.raises(ValueError, match='cover'):
        execute_tape(model, {'theta_index': 0, 'returns': {}}, policy='fixed_order', budget=.04, max_queries=2, order=tuple(model.tools))
    tape = next(enumerate_tapes(model))
    tape['returns']['lineup'] = 'invented'
    with pytest.raises(ValueError, match='support'):
        execute_tape(model, tape, policy='fixed_order', budget=.04, max_queries=2, order=tuple(model.tools))


def test_dropout_rate_validation_and_no_implicit_health_estimation(cases):
    model = cases['qualified_rotation_unknown']['model']
    for rates in ({}, {'lineup': -1, 'reserve_quality': 1}, {'lineup': True, 'reserve_quality': 1}):
        with pytest.raises(ValueError):
            with_dropout(model, rates)
    with pytest.raises(ValueError, match='already contains'):
        with_dropout(cases['all_returns_missing']['model'], {'lineup': 1, 'reserve_quality': 1})


def test_manifest_experiment_is_reproducible_and_honest_about_evidence(report):
    manifest = experiment_manifest()
    assert run_experiment(manifest) == report
    assert report['case_count'] == 9
    assert report['independent_real_tasks'] == report['new_source_requests'] == report['model_calls'] == 0
    assert report['empirical_policy_benefit'] is None
    assert report['production_policy_changed'] is False and report['runtime_safe'] is False
    assert 'not independent preregistration' in manifest['freeze_scope']
    assert 'aggregate' not in manifest and 'trajectories' not in manifest
    with pytest.raises(ValueError, match='manifest'):
        run_experiment({**manifest, 'policy_order': ['oracle']})


def test_saved_manifest_roundtrip_preserves_every_trajectory_and_metric(report):
    original = experiment_manifest()
    persisted = json.loads(json.dumps(original, sort_keys=True))
    assert run_experiment(persisted) == report
    # Reverse keys at every object level, while preserving explicit array order.
    def reorder(value):
        if isinstance(value, dict):
            return {k: reorder(value[k]) for k in reversed(value)}
        if isinstance(value, list):
            return [reorder(v) for v in value]
        return value
    assert run_experiment(reorder(original)) == report


def test_readable_report_roundtrip_uses_explicit_display_order(report):
    path = Path(__file__).resolve().parents[2] / 'var/parallel/agent/round111/run_comparison.py'
    spec = importlib.util.spec_from_file_location('round111_display_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = module.render_report(report)
    after = module.render_report(json.loads(json.dumps(report, sort_keys=True)))
    assert before == after
    for name in report['cases']:
        positions = [before.index('| ' + name + ' | ' + policy + ' |') for policy in POLICIES]
        assert positions == sorted(positions)
