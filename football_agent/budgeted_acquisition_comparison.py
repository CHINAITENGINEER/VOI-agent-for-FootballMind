'''Paired, exact comparisons of information acquisition under declared models.

Every arm uses the same updater, costs, resource caps and evaluator-only return
tapes. This is a synthetic mechanism experiment, not an online LLM evaluation.
Private tapes enter the executor only after the policy has chosen an action.
'''
from __future__ import annotations

from itertools import permutations, product
import json
import math

from .decision_value import FiniteDecisionModel, ObservationTool, VALUE_TOLERANCE, synthetic_scenarios

VERSION = 'budgeted-acquisition-comparison-v1'
POLICIES = ('fixed_order', 'random_order', 'frozen_voi_order', 'adaptive_voi_1', 'adaptive_voi_2')
NO_RETURN = '__NO_RETURN__'
RESOLUTION_TOLERANCE = 1e-10


def _limits(budget, max_queries):
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
        raise ValueError('budget must be finite and nonnegative')
    if type(max_queries) is not int or not 0 <= max_queries <= 2:
        raise ValueError('max_queries must be an integer in 0..2')


def _order(model, order):
    order = tuple(order)
    if len(order) != len(model.tools) or set(order) != set(model.tools):
        raise ValueError('order must contain every action exactly once')
    return order


def choose_action(model, state, *, policy, remaining_budget, queries_left, order):
    '''Actor API: there is no tape, latent state, result or evaluator metric here.'''
    _limits(remaining_budget, queries_left)
    if policy not in POLICIES:
        raise ValueError('unknown acquisition policy')
    order = _order(model, order)
    horizon = min(2 if policy == 'adaptive_voi_2' else 1, queries_left)
    evaluated = model.evaluate(state, remaining_budget=remaining_budget, horizon=horizon)
    compact = [{key: row[key] for key in ('action_id', 'feasible', 'reason', 'cost',
                'immediate_evsi', 'immediate_net_value', 'planning_net_value') if key in row}
               for row in evaluated['candidates']]
    if queries_left == 0:
        action, reason = 'STOP', 'query_cap_reached'
    elif policy.startswith('adaptive_voi_'):
        action, reason = evaluated['action_id'], evaluated['reason']
    else:
        feasible = {row['action_id'] for row in evaluated['candidates'] if row['feasible']}
        eligible = order
        if policy == 'frozen_voi_order':
            # Initial values are deliberately frozen for the no-replanning ablation.
            initial = model.evaluate(model.initial_state(), remaining_budget=remaining_budget, horizon=1)
            scores = {row['action_id']: row['immediate_net_value'] for row in initial['candidates'] if row['feasible']}
            eligible = tuple(sorted((a for a in order if scores.get(a, 0) > VALUE_TOLERANCE),
                                    key=lambda a: (-scores[a], a)))
        action = next((a for a in eligible if a in feasible), 'STOP')
        reason = 'predeclared_order' if action != 'STOP' else 'no_eligible_affordable_action'
    return {'action_id': action, 'reason': reason, 'policy': policy, 'horizon': horizon,
            'remaining_budget': remaining_budget, 'queries_left': queries_left,
            'observed_before': [list(pair) for pair in state.observations],
            'prediction_before': list(model.prediction(state)), 'candidates': compact}


def _loss(model, prediction, theta_index):
    truth_distribution = model.outcome_probabilities[theta_index]
    if model.loss == 'log_loss':
        return -math.fsum(p * math.log(q) for p, q in zip(truth_distribution, prediction) if p > 0)
    return 1 + math.fsum(q * q for q in prediction) - 2 * math.fsum(p * q for p, q in zip(truth_distribution, prediction))


def _intrinsic_risk(model, theta_index):
    return _loss(model, model.outcome_probabilities[theta_index], theta_index)


def residual_target_risk(model, state):
    '''Remaining reducible predictive risk; not a factual question gold label.'''
    residual = model.bayes_risk(state) - math.fsum(
        p * _intrinsic_risk(model, i) for i, p in enumerate(state.belief))
    if residual < -RESOLUTION_TOLERANCE:
        raise ArithmeticError('negative reducible risk')
    return max(0.0, residual)


def enumerate_tapes(model):
    '''Exact evaluator support under the model factorization, shared by all arms.

    One evidence ID is sampled once even when multiple tools expose it. Other
    residual correlations must be encoded in the full latent hypothesis.
    '''
    groups = {}
    for action, tool in model.tools.items():
        key = ('evidence', tool.evidence_id) if tool.evidence_id is not None else ('action', action)
        groups.setdefault(key, []).append(action)
    for theta_index, theta in enumerate(model.hypothesis_names):
        if model.prior[theta_index] == 0:
            continue
        axes = []
        for actions in groups.values():
            tool = model.tools[actions[0]]
            axes.append([(outcome, p) for outcome, p in zip(tool.outcomes, tool.likelihoods[theta]) if p > 0])
        for combination in product(*axes):
            weight = model.prior[theta_index] * math.prod(p for _, p in combination)
            tape = {action: outcome for actions, (outcome, _) in zip(groups.values(), combination) for action in actions}
            yield {'theta_index': theta_index, 'weight': weight, 'returns': tape}


def execute_tape(model, tape, *, policy, budget, max_queries, order):
    '''Execute chosen actions only. Evaluator truth is scored after STOP.'''
    _limits(budget, max_queries)
    order = _order(model, order)
    if set(tape['returns']) != set(model.tools):
        raise ValueError('return tape must cover the declared action set')
    theta_index = tape['theta_index']
    if type(theta_index) is not int or not 0 <= theta_index < len(model.hypothesis_names):
        raise ValueError('invalid evaluator latent index')
    theta = model.hypothesis_names[theta_index]
    if model.prior[theta_index] == 0:
        raise ValueError('return tape uses a zero-prior latent state')
    shared = {}
    for action, value in tape['returns'].items():
        tool = model.tools[action]
        if value not in tool.outcomes or tool.likelihoods[theta][tool.outcomes.index(value)] <= 0:
            raise ValueError('return tape is outside declared support')
        if tool.evidence_id is not None:
            if tool.evidence_id in shared and shared[tool.evidence_id] != value:
                raise ValueError('same evidence ID has inconsistent private returns')
            shared[tool.evidence_id] = value
    state = model.initial_state()
    remaining, spent = float(budget), 0.0
    trace = []
    failed_cost, zero_target_value_cost, no_change_cost = 0.0, 0.0, 0.0
    for step in range(max_queries + 1):
        decision = choose_action(model, state, policy=policy, remaining_budget=remaining,
                                 queries_left=max_queries - step, order=order)
        event = {'step': step, 'decision': decision}
        action = decision['action_id']
        if action == 'STOP':
            trace.append(event)
            break
        candidate = next(row for row in decision['candidates'] if row['action_id'] == action)
        tool = model.tools[action]
        # First exposure of this action's private return happens AFTER selection.
        returned = tape['returns'][action]
        before = model.prediction(state)
        state = model.observe(state, action, returned)
        after = model.prediction(state)
        spent = math.fsum((spent, tool.cost))
        remaining = max(0.0, budget - spent)
        failed_cost += tool.cost if returned == NO_RETURN else 0.0
        zero_target_value_cost += tool.cost if candidate['immediate_evsi'] <= VALUE_TOLERANCE else 0.0
        no_change_cost += tool.cost if max(abs(a - b) for a, b in zip(before, after)) <= RESOLUTION_TOLERANCE else 0.0
        event.update({'returned': returned, 'cost': tool.cost, 'remaining_budget_after': remaining,
                      'prediction_after': list(after)})
        trace.append(event)
    residual = residual_target_risk(model, state)
    prediction = model.prediction(state)
    return {'policy': policy, 'budget': budget, 'max_queries': max_queries, 'order': list(order),
            'trace': trace, 'spent': spent, 'query_count': len(state.observations),
            'selected_actions': [a for a, _ in state.observations], 'stop_reason': trace[-1]['decision']['reason'],
            'final_prediction': list(prediction), 'posterior_target_risk': model.bayes_risk(state),
            'remaining_reducible_risk': residual,
            'target_resolved_under_model': residual <= RESOLUTION_TOLERANCE,
            'terminal_status': 'RESOLVED_UNDER_MODEL' if residual <= RESOLUTION_TOLERANCE else 'UNRESOLVED_AT_STOP',
            'failed_request_cost': failed_cost, 'zero_one_step_value_cost': zero_target_value_cost,
            'unchanged_prediction_cost': no_change_cost,
            'evaluator_only': {'conditional_loss': _loss(model, prediction, theta_index),
                               'theta_index': theta_index, 'runtime_safe': False}}


def compare_case(model, *, budget, max_queries=2, fixed_order=None):
    '''Integrate all latent/return tapes and all uniform random permutations.'''
    _limits(budget, max_queries)
    fixed_order = _order(model, tuple(model.tools) if fixed_order is None else fixed_order)
    if len(model.tools) > 6:
        raise ValueError('exact comparator supports at most six tools')
    tapes = list(enumerate_tapes(model))
    total_weight = math.fsum(t['weight'] for t in tapes)
    if abs(total_weight - 1) > 1e-10:
        raise ArithmeticError('private tape probability mass must equal one')
    rows, aggregates = [], {}
    initial_risk = model.bayes_risk(model.initial_state())
    for policy in POLICIES:
        orders = list(permutations(fixed_order)) if policy == 'random_order' else [fixed_order]
        weighted = []
        for index, tape in enumerate(tapes):
            for order in orders:
                run = execute_tape(model, tape, policy=policy, budget=budget,
                                   max_queries=max_queries, order=order)
                weight = tape['weight'] / len(orders)
                weighted.append((weight, run))
                rows.append({'tape_index': index, 'weight': weight, **run})
        def mean(key):
            return math.fsum(weight * run[key] for weight, run in weighted)
        risk = math.fsum(weight * run['evaluator_only']['conditional_loss'] for weight, run in weighted)
        aggregate = {'expected_target_loss': risk, 'initial_target_loss': initial_risk,
                     'expected_acquisition_cost': mean('spent'), 'expected_query_count': mean('query_count'),
                     'expected_remaining_reducible_risk': mean('remaining_reducible_risk'),
                     'model_target_resolution_rate': mean('target_resolved_under_model'),
                     'expected_failed_request_cost': mean('failed_request_cost'),
                     'expected_zero_one_step_value_cost': mean('zero_one_step_value_cost'),
                     'expected_unchanged_prediction_cost': mean('unchanged_prediction_cost'),
                     'early_stop_probability': math.fsum(w for w, r in weighted if r['query_count'] < max_queries),
                     'support_weight': math.fsum(w for w, _ in weighted)}
        aggregate['expected_total_risk'] = risk + aggregate['expected_acquisition_cost']
        aggregate['expected_net_gain'] = initial_risk - aggregate['expected_total_risk']
        aggregates[policy] = aggregate
        if abs(risk - mean('posterior_target_risk')) > 1e-9:
            raise ArithmeticError('evaluator and posterior risk integration disagree')
    comparisons = []
    for adaptive in ('adaptive_voi_1', 'adaptive_voi_2'):
        for baseline in ('fixed_order', 'random_order', 'frozen_voi_order'):
            a, b = aggregates[adaptive], aggregates[baseline]
            comparisons.append({'policy': adaptive, 'baseline': baseline,
                                'total_risk_reduction': b['expected_total_risk'] - a['expected_total_risk'],
                                'cost_reduction': b['expected_acquisition_cost'] - a['expected_acquisition_cost'],
                                'resolution_rate_change': a['model_target_resolution_rate'] - b['model_target_resolution_rate']})
    return {'budget': budget, 'max_queries': max_queries, 'cost_unit': model.cost_unit,
            'fixed_order': list(fixed_order), 'private_tape_count': len(tapes), 'private_tape_mass': total_weight,
            'aggregate': aggregates, 'paired_comparisons': comparisons, 'trajectories': rows,
            'budget_violations': sum(r['spent'] > budget + VALUE_TOLERANCE or r['query_count'] > max_queries for r in rows),
            'all_trajectories_stopped': all(r['trace'][-1]['decision']['action_id'] == 'STOP' for r in rows),
            'runtime_safe': False}


def with_dropout(model, rates):
    '''Declared state-independent missing return. Failure is an observed outcome.

    Rate inputs are assumptions, not estimated from the one archived trajectory.
    Costs are unchanged and the updater learns nothing from state-independent
    dropout. Same-evidence aliases must have identical rates.
    '''
    if set(rates) != set(model.tools):
        raise ValueError('rates must exactly cover all tools')
    tools = {}
    for action, tool in model.tools.items():
        q = rates[action]
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('availability must be in [0,1]')
        if NO_RETURN in tool.outcomes:
            raise ValueError('model already contains the reserved missing observation')
        tools[action] = ObservationTool((*tool.outcomes, NO_RETURN),
            {theta: tuple(q * p for p in law) + (1 - q,) for theta, law in tool.likelihoods.items()},
            tool.cost, tool.description, tool.evidence_id)
    return FiniteDecisionModel(dict(zip(model.hypothesis_names, model.outcome_probabilities)),
        dict(zip(model.hypothesis_names, model.prior)), tools, outcome_labels=model.outcome_labels,
        loss=model.loss, cost_unit=model.cost_unit,
        assumptions=(*model.assumptions, 'State-independent return availability is manually declared; all attempted requests pay full cost.'))


def experiment_cases():
    cases = synthetic_scenarios()
    base = cases['qualified_rotation_unknown']
    for name, rates in [('lineup_source_degraded', {'lineup': .1, 'reserve_quality': 1.0}),
                        ('all_returns_missing', {'lineup': 0.0, 'reserve_quality': 0.0})]:
        cases[name] = {**base, 'model': with_dropout(base['model'], rates),
                       'description': 'Constructed availability stress: ' + name}
    cases['rotation_tight_budget'] = {**base, 'budget': .02,
                                     'description': 'Same rotation model with a loss-unit budget that permits at most one query.'}
    return cases


def experiment_manifest():
    return {'schema_version': VERSION, 'evidence_class': 'SYNTHETIC_PAIRED_ACQUISITION_EXPERIMENT',
            'policy_order': list(POLICIES), 'randomization': 'exact uniform integration over every tool-order permutation, independent of private tape',
            'same_budget_means': 'equal ceilings, not forced equal spending',
            'resolution_definition': 'zero remaining reducible predictive risk under the declared model; not independently verified factual correctness',
            'freeze_scope': 'configuration frozen before recorded execution; preflight and tests may inspect fixture outcomes; not independent preregistration or a held-out benchmark',
            'cases': {name: {'model': case['model'].manifest(), 'description': case['description'],
                             'context': case['context'], 'budget': case['budget'], 'max_queries': 2,
                             'fixed_order': list(case['model'].tools)} for name, case in experiment_cases().items()}}


def run_experiment(manifest):
    if manifest.get('schema_version') != VERSION or manifest.get('policy_order') != list(POLICIES):
        raise ValueError('unexpected comparison manifest')
    # Object key insertion order is not part of JSON's semantic contract. Fix it
    # before building latent indexes, tapes and trajectories on every execution.
    # Explicit arrays (outcomes, fixed action order) retain their declared order.
    manifest = json.loads(json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False))
    cases = {}
    for name, spec in manifest['cases'].items():
        model = FiniteDecisionModel.from_manifest(spec['model'])
        cases[name] = compare_case(model, budget=spec['budget'], max_queries=spec['max_queries'], fixed_order=spec['fixed_order'])
    return {'schema_version': VERSION, 'evidence_class': manifest['evidence_class'], 'cases': cases,
            'case_count': len(cases), 'independent_real_tasks': 0, 'new_source_requests': 0, 'model_calls': 0,
            'production_policy_changed': False, 'runtime_safe': False,
            'all_budget_violations': sum(case['budget_violations'] for case in cases.values()),
            'empirical_policy_benefit': None, 'real_question_resolution_rate': None,
            'limitations': [
                'Finite tools and exact semantic likelihoods are supplied assumptions; open-web discovery and learned VoI are untested.',
                'Exact support integration removes Monte Carlo noise, not model misspecification or development-set bias.',
                'No per-case tuning, selected favorable cases, pooled league table or independent-match sample count is implied.',
                'An unchanged prediction may follow a worthwhile failed query; zero one-step value can hide complementary information.',
                'All arms share posterior updates and information budgets; LLM reasoning latency and token cost are not measured.',
                'STOP may leave target uncertainty unresolved; two-step optimality applies only to this two-query declared-model experiment.',
                'The single archived online trajectory provides no counterfactual returns for unselected actions and is excluded from efficacy estimation.']}
