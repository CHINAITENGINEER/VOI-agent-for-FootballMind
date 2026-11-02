"""Exact small-graph budget comparisons, not prospective policy evaluation.

Scorers see only action IDs and pre-query features. Outcome-dependent fields
are accessed after a decision solely to score the reconstructed trajectory.
Uniform-random paths are enumerated, not labelled as logged propensities.
"""
from copy import deepcopy
import math


def rollout(graph, start, *, policy, budget, cost, scorer=None, menu_selector=None):
    if type(budget) is not int or budget < 0 or not math.isfinite(cost) or cost < 0:
        raise ValueError('Nonnegative budget and cost required')
    if policy not in ('stop', 'fixed', 'random', 'g', 'g_forced', 'menu'):
        raise ValueError('Unknown comparison policy')
    if policy in ('g', 'g_forced') and scorer is None:
        raise ValueError('Pre-query scorer required')
    if policy == 'menu' and menu_selector is None:
        raise ValueError('Pre-query menu selector required')
    paths = []

    def visit(state, steps, probability, visited):
        if state in visited:
            raise ValueError('Acquisition graph contains a cycle')
        menu = sorted(graph.get(state, []), key=lambda t: t['action_id'])
        if len({t['action_id'] for t in menu}) != len(menu):
            raise ValueError('Duplicate action in menu')
        reason = ('stop_baseline' if policy == 'stop' else 'budget_exhausted' if len(steps) >= budget
                  else 'no_feasible_candidate' if not menu else None)
        scores = None; menu_decision = None
        if reason is None and policy == 'menu':
            public_menu = deepcopy([{'action_id': t['action_id'], 'query': t['query']} for t in menu])
            menu_decision = menu_selector(public_menu, budget-len(steps), cost)
            selected_id = menu_decision['action_id']
            if selected_id is None:
                reason = menu_decision['reason']
                choices = []
            else:
                ids = [t['action_id'] for t in menu]
                if selected_id not in ids:
                    raise ValueError('Menu selector chose an unavailable action')
                choices = [(ids.index(selected_id), 1.)]
        elif reason is None and policy in ('g', 'g_forced'):
            scores = [float(scorer(deepcopy({'action_id': t['action_id'], 'query': t['query']}))) for t in menu]
            if not all(math.isfinite(v) for v in scores):
                raise ValueError('Finite pre-query scores required')
            best = max(range(len(menu)), key=lambda i: scores[i])
            if policy == 'g' and scores[best] <= cost:
                reason = 'estimated_net_gain_nonpositive'
            choices = [(best, 1.)]
        else:
            choices = [(i, 1/len(menu)) for i in range(len(menu))] if policy == 'random' and menu else [(0, 1.)]
        if reason:
            gain = sum(step['signed_gain'] for step in steps)
            paths.append({'simulation_path_probability': probability, 'steps': steps,
                          'reads': len(steps), 'gain': gain, 'net': gain-cost*len(steps),
                          'end_state_id': state, 'stop_reason': reason})
            if menu_decision is not None:
                paths[-1]['stop_menu_decision'] = menu_decision
            return
        for i, chance in choices:
            edge = menu[i]
            step = {'action_id': edge['action_id'], 'state_id': state, 'next_state_id': edge['next_state_id'],
                    'signed_gain': edge['signed_gain'], 'estimated_gain': scores[i] if scores else None}
            if menu_decision is not None:
                step['menu_decision'] = menu_decision
            visit(edge['next_state_id'], steps+[step], probability*chance, visited | {state})

    visit(start, [], 1., set())
    if not math.isclose(sum(p['simulation_path_probability'] for p in paths), 1., abs_tol=1e-12):
        raise ValueError('Path probabilities must sum to one')
    result = {'policy': policy, 'budget': budget, 'cost_loss_units': cost, 'paths': paths,
              'logged_propensity': None, 'scope': 'enumerated_historical_graph_research'}
    for key in ('reads', 'gain', 'net'):
        result['expected_'+key] = sum(p['simulation_path_probability']*p[key] for p in paths)
    return result
