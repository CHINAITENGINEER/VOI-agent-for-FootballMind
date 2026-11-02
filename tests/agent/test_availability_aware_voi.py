from dataclasses import replace
import json

import pytest

from football_agent.availability_aware_voi import (
    AvailabilityAwareVoIPolicy, CalibrationSnapshot, SourceBinding,
)
from football_agent.decision_value import FiniteDecisionModel, ObservationTool, synthetic_scenarios
from football_agent.evidence_ledger import EvidenceLedgerEntry
from football_agent.source_health_calibration import SourceHealthCalibration, calibrate_source_health


TRAIN_END = '2026-09-27T23:00:00Z'
EVAL_START = '2026-09-28T00:00:00Z'
EVAL_END = '2026-09-28T12:00:00Z'
RELEASE = '2026-09-28T13:00:00Z'
AS_OF = '2026-09-28T14:00:00Z'
ASSUMPTION = 'state_independent_return_availability'


def report(source='lineup-source', *, train=10, evaluation=10, n=10, version='v1'):
    def counts(successes):
        return {'available': successes, 'late': 0, 'failed': n - successes, 'duplicate': 0}
    return SourceHealthCalibration(
        schema_version='source-health-calibration-v1', source_id=source, source_version=version,
        train_end=TRAIN_END, eval_start=EVAL_START, eval_end=EVAL_END,
        train_attempts=n, train_independent_dependency_count=n, train_duplicate_attempts=0,
        train_status_counts=counts(train), train_attempt_status_counts=counts(train),
        train_available_rate=train / n if n else None, train_available_lower_bound=None,
        eval_attempts=n, eval_independent_dependency_count=n, eval_duplicate_attempts=0,
        eval_status_counts=counts(evaluation), eval_attempt_status_counts=counts(evaluation),
        eval_available_rate=evaluation / n if n else None,
        availability_rate_delta=(evaluation - train) / n if n else None,
        drift_status='STABLE' if train == evaluation else 'DEGRADED',
        operational_support_status='DECLARED_SYNTHETIC_COUNTS',
        semantic_accuracy_status='UNMEASURED', numeric_voi_status='UNMEASURED',
        limitations=('Manually specified mechanism fixture, not observed provider reliability.',),
    )


def snapshot(row=None, *, release=RELEASE, kind='SYNTHETIC_ASSUMPTIONS'):
    return CalibrationSnapshot.from_calibration(
        row or report(), available_at=release, evidence_kind=kind,
    )


def policy(model=None, snapshots=(), **kwargs):
    model = model or synthetic_scenarios()['qualified_rotation_unknown']['model']
    bindings = {name: SourceBinding(name + '-source', 'v1', 'Same endpoint and acquisition contract assumed.')
                for name in model.tools}
    return AvailabilityAwareVoIPolicy(model, bindings, snapshots, dropout_assumption=ASSUMPTION, **kwargs)


def evaluate(p, *, as_of=AS_OF, budget=.08, state=None):
    return p.evaluate(state or p.model.initial_state(), as_of=as_of, remaining_budget=budget)


def candidate(result, action='lineup'):
    return next(row for row in result['candidates'] if row['action_id'] == action)


def test_good_return_scenarios_reproduce_semantic_policy_without_changing_beliefs():
    p = policy(snapshots=[snapshot(report(name + '-source')) for name in ('lineup', 'reserve_quality')])
    result = evaluate(p)
    base = p.model.evaluate(p.model.initial_state(), remaining_budget=.08)
    assert result['action_id'] == base['action_id'] == 'lineup'
    assert result['prediction'] == base['prediction']
    assert result['current_risk'] == base['current_risk']
    assert candidate(result)['lower_net_value'] == pytest.approx(candidate(base)['immediate_net_value'])


def test_degraded_lineup_switches_to_reserve_information_at_the_same_belief():
    p = policy(snapshots=[snapshot(report(evaluation=1)), snapshot(report('reserve_quality-source'))])
    result = evaluate(p)
    assert result['action_id'] == 'reserve_quality'
    assert candidate(result)['lower_net_value'] < 0 < candidate(result)['upper_net_value']
    assert candidate(result, 'reserve_quality')['lower_net_value'] > 0
    assert result['prediction'] == list(p.model.prediction(p.model.initial_state()))


def test_no_availability_evidence_defers_instead_of_claiming_zero_value():
    result = evaluate(policy())
    row = candidate(result)
    assert result['action_id'] == 'DEFER'
    assert row['lower_net_value'] == -row['cost']
    assert row['upper_net_value'] > 0
    assert row['support']['status'] == 'NO_VISIBLE_CALIBRATION'


def test_irrelevant_information_stops_even_with_unknown_return_rate():
    model = synthetic_scenarios()['irrelevant_high_entropy_news']['model']
    result = evaluate(policy(model))
    assert result['action_id'] == 'STOP'
    assert candidate(result, 'news')['conditional_evsi_on_return'] == pytest.approx(0)


def test_recompute_after_reserve_evidence_stops_without_querying_lineup():
    p = policy(snapshots=[snapshot(report(evaluation=1)), snapshot(report('reserve_quality-source'))])
    assert evaluate(p)['action_id'] == 'reserve_quality'
    state = p.model.observe(p.model.initial_state(), 'reserve_quality', 'strong')
    result = evaluate(p, state=state, budget=.06)
    assert result['action_id'] == 'STOP'
    assert candidate(result, 'reserve_quality')['reason'] == 'already_observed'
    assert candidate(result)['upper_net_value'] < 0
    assert result['observations'] == [['reserve_quality', 'strong']]


def test_future_calibration_and_its_identity_are_invisible_to_actor():
    future = snapshot(release='2026-09-28T15:00:00Z')
    conflict = snapshot(report(evaluation=1), release='2026-09-28T15:00:00Z')
    assert evaluate(policy(snapshots=[future, conflict])) == evaluate(policy())
    assert future.digest not in json.dumps(evaluate(policy(snapshots=[future])))


def test_release_boundary_is_inclusive_and_eval_completion_is_not_release():
    s = snapshot()
    before = evaluate(policy(snapshots=[s]), as_of='2026-09-28T12:59:59Z')
    at = evaluate(policy(snapshots=[s]), as_of=RELEASE)
    assert candidate(before)['support']['status'] == 'NO_VISIBLE_CALIBRATION'
    assert candidate(at)['support']['status'] == 'OBSERVED_FREQUENCY_SCENARIOS'


@pytest.mark.parametrize('release', ['2026-09-27T00:00:00Z', '2026-09-28T11:59:59Z'])
def test_rejects_backdated_release_before_evaluation_end(release):
    with pytest.raises(ValueError, match='evaluation completion'):
        snapshot(release=release)


def test_expiry_uses_observation_window_end_even_if_report_is_republished():
    s = snapshot(release='2026-09-30T12:00:00Z')
    result = evaluate(policy(snapshots=[s]), as_of='2026-09-30T13:00:00Z')
    assert candidate(result)['support']['status'] == 'EXPIRED_DIAGNOSTIC'
    assert result['action_id'] == 'DEFER'


def test_wrong_source_version_does_not_borrow_older_support():
    result = evaluate(policy(snapshots=[snapshot(report(version='v2'))]))
    assert candidate(result)['support']['status'] == 'NO_VISIBLE_CALIBRATION'


def test_unversioned_channel_is_unknown_even_with_many_successes():
    p = policy()
    bindings = {name: replace(binding, source_version='UNVERSIONED') for name, binding in p.bindings.items()}
    p = AvailabilityAwareVoIPolicy(p.model, bindings, [snapshot(report(version='UNVERSIONED'))],
                                  dropout_assumption=ASSUMPTION)
    assert candidate(evaluate(p))['support']['status'] == 'UNKNOWN_SOURCE_VERSION'


@pytest.mark.parametrize('n', [0, 1, 4])
def test_small_dependency_group_support_remains_unknown(n):
    result = evaluate(policy(snapshots=[snapshot(report(train=n, evaluation=n, n=n))]))
    assert candidate(result)['support']['status'] == 'INSUFFICIENT_DEPENDENCY_GROUPS'
    assert result['action_id'] == 'DEFER'


def test_many_retries_do_not_create_support():
    row = report(train=1, evaluation=1, n=1)
    row = replace(row, train_attempts=100, train_duplicate_attempts=99,
                  train_attempt_status_counts={'available': 100, 'late': 0, 'failed': 0, 'duplicate': 0},
                  eval_attempts=100, eval_duplicate_attempts=99,
                  eval_attempt_status_counts={'available': 100, 'late': 0, 'failed': 0, 'duplicate': 0})
    assert candidate(evaluate(policy(snapshots=[snapshot(row)])))['support']['status'] == 'INSUFFICIENT_DEPENDENCY_GROUPS'


def test_archived_fixture_is_not_a_population_even_if_perfect_and_large():
    result = evaluate(policy(snapshots=[snapshot(kind='ARCHIVED_FIXTURE')]))
    assert candidate(result)['support']['status'] == 'FIXTURE_IS_NOT_A_CALIBRATION_POPULATION'
    assert result['action_id'] == 'DEFER'


def test_legacy_concentration_bound_does_not_enter_acquisition_value():
    first = evaluate(policy(snapshots=[snapshot(replace(report(), train_available_lower_bound=0))]))
    second = evaluate(policy(snapshots=[snapshot(replace(report(), train_available_lower_bound=1))]))
    for key in ('action_id', 'prediction', 'current_risk', 'certificate'):
        assert first[key] == second[key]
    assert candidate(first)['scenario_net_values'] == candidate(second)['scenario_net_values']
    support = candidate(first)['support']
    assert support['statistical_coverage_claim'] is False
    assert support['dependency_groups_are_statistically_independent'] == 'NOT_ESTABLISHED'


def test_snapshot_detaches_mutable_upstream_counts_and_returned_dictionary():
    row = report()
    s = snapshot(row)
    digest = s.digest
    row.train_status_counts['available'] = 0
    returned = s.report
    returned['eval_status_counts']['available'] = 0
    assert s.digest == digest
    assert s.report['train_status_counts']['available'] == 10
    assert s.report['eval_status_counts']['available'] == 10


def test_inconsistent_rates_rejected_on_snapshot_import():
    with pytest.raises(ValueError, match='rate disagrees'):
        snapshot(replace(report(), train_available_rate=.1))


def test_conflicting_visible_snapshot_releases_rejected_instead_of_order_picking():
    with pytest.raises(ValueError, match='conflicting calibration'):
        evaluate(policy(snapshots=[snapshot(), snapshot(report(evaluation=1))]))
    assert evaluate(policy(snapshots=[snapshot(), snapshot()]))['action_id'] == 'lineup'


def test_latest_release_with_insufficient_support_does_not_fallback_to_better_old_rate():
    newer = snapshot(report(train=1, evaluation=1, n=1), release=AS_OF)
    result = evaluate(policy(snapshots=[snapshot(), newer]))
    assert candidate(result)['support']['snapshot_sha256'] == newer.digest
    assert candidate(result)['support']['status'] == 'INSUFFICIENT_DEPENDENCY_GROUPS'


def test_budget_and_shared_evidence_identity_are_preserved():
    shared = ObservationTool(('a', 'b'), {'a': (1, 0), 'b': (0, 1)}, .01, evidence_id='one-report')
    model = FiniteDecisionModel({'a': (.8, .1, .1), 'b': (.1, .1, .8)}, {'a': .5, 'b': .5},
                                {'report_a': shared, 'report_b': shared})
    p = policy(model)
    result = evaluate(p, budget=.005)
    assert result['action_id'] == 'STOP'
    assert all(row['reason'] == 'budget_exhausted' for row in result['candidates'])
    state = model.observe(model.initial_state(), 'report_a', 'a')
    result = evaluate(p, state=state)
    assert candidate(result, 'report_b')['reason'] == 'evidence_already_observed'
    assert result['action_id'] == 'STOP'


@pytest.mark.parametrize('horizon', [0, 2, True, 1.0])
def test_no_claim_of_common_robust_multistep_policy(horizon):
    p = policy()
    with pytest.raises(ValueError, match='horizon=1'):
        p.evaluate(p.model.initial_state(), as_of=AS_OF, remaining_budget=.08, horizon=horizon)


@pytest.mark.parametrize('loss', ['log_loss', 'brier'])
@pytest.mark.parametrize('q', [0, .3, 1])
def test_availability_formula_matches_explicit_missing_observation_model(loss, q):
    hypotheses = {'strong': (.8, .15, .05), 'weak': (.2, .3, .5)}
    prior = {'strong': .5, 'weak': .5}
    conditional = ObservationTool(('positive', 'negative'), {'strong': (.9, .1), 'weak': (.2, .8)}, .01)
    model = FiniteDecisionModel(hypotheses, prior, {'lineup': conditional}, loss=loss)
    missing = ObservationTool(('positive', 'negative', 'missing'),
        {name: tuple(q * v for v in values) + (1 - q,) for name, values in conditional.likelihoods.items()}, .01)
    full = FiniteDecisionModel(hypotheses, prior, {'lineup': missing}, loss=loss)
    p = policy(model, snapshots=[snapshot(report(train=int(q * 10), evaluation=int(q * 10)))])
    row = candidate(evaluate(p))
    expected = candidate(full.evaluate(full.initial_state(), remaining_budget=.08))
    assert row['lower_net_value'] == pytest.approx(expected['immediate_net_value'])
    assert row['upper_net_value'] == row['lower_net_value']
    if q == 0:
        assert row['lower_net_value'] == -.01
    if q < 1:
        assert full.observe(full.initial_state(), 'lineup', 'missing').belief == full.prior


def test_policy_assumptions_and_synthetic_limits_remain_explicit():
    result = evaluate(policy(snapshots=[snapshot()]))
    certificate = result['certificate']
    assert certificate['dropout_assumption'] == ASSUMPTION
    assert certificate['numeric_voi_empirically_calibrated'] is False
    assert certificate['empirical_policy_benefit'] == 'UNMEASURED'
    assert certificate['global_claim'] is False
    assert certificate['stop_is_global_optimality_certificate'] is False
    assert candidate(result)['support']['evidence_kind'] == 'SYNTHETIC_ASSUMPTIONS'


def test_pipeline_consumes_actual_ledger_aggregation_and_ignores_repeated_attempts():
    entries = []
    for split, captured in [('train', '2026-09-27T12:00:00Z'), ('eval', '2026-09-28T10:00:00Z')]:
        for i in range(6):
            available = split == 'train' or i == 0
            row = {'candidate_id': f'{split}-{i}', 'source_id': 'lineup-source',
                   'dependency_key': f'{split}-{i}', 'quality': 'VALIDATED' if available else 'PROVIDER_ERROR',
                   'availability': 'available' if available else 'failed',
                   'observed_at': captured if available else None,
                   'operational_available_at': captured if available else None,
                   'evidence_refs': [f'raw:{split}-{i}'], 'content_hash': f'hash:{split}-{i}'}
            for attempt in range(2):
                entries.append(EvidenceLedgerEntry.from_row(row, episode_id='synthetic-episode',
                    target_key='synthetic-match', action_id='lineup', request_id=f'{split}-{i}-{attempt}',
                    sequence=len(entries), prediction_cutoff=AS_OF, captured_at=captured,
                    requested_at=captured, source_version='v1'))
    calibrated = calibrate_source_health(entries, train_end=TRAIN_END, eval_start=EVAL_START, eval_end=EVAL_END)[0]
    result = evaluate(policy(snapshots=[snapshot(calibrated), snapshot(report('reserve_quality-source'))]))
    assert calibrated.train_attempts == 12
    assert calibrated.train_independent_dependency_count == 6
    assert candidate(result)['support']['q_scenarios'] == [1 / 6, 1.0]
    assert result['action_id'] == 'reserve_quality'


def test_bindings_detach_from_caller_and_iterable_snapshots_are_retained():
    model = synthetic_scenarios()['qualified_rotation_unknown']['model']
    bindings = {name: SourceBinding(name + '-source', 'v1', 'Declared matching endpoint.') for name in model.tools}
    p = AvailabilityAwareVoIPolicy(model, bindings, (s for s in [snapshot()]), dropout_assumption=ASSUMPTION)
    bindings.clear()
    assert evaluate(p)['action_id'] == 'lineup'
    with pytest.raises(TypeError):
        p.bindings['lineup'] = SourceBinding('different', 'v1', 'different')


@pytest.mark.parametrize('kwargs', [
    {'min_dependency_groups': 0}, {'min_dependency_groups': True},
    {'max_age_seconds': float('inf')}, {'max_age_seconds': 0},
])
def test_policy_rejects_invalid_support_configuration(kwargs):
    with pytest.raises(ValueError):
        policy(**kwargs)


def test_state_dependent_dropout_requires_a_different_joint_model():
    p = policy()
    with pytest.raises(ValueError, match='state-independent'):
        AvailabilityAwareVoIPolicy(p.model, p.bindings, (), dropout_assumption='depends_on_match_state')
