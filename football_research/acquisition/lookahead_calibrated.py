"""Two-step mask approximation permitting a feasible negative-value first step.

Not an observation-integrated EVSI planner: continuation values condition on a
projected mask and the CURRENT prediction. No imputed hypothetical observation
is passed through the predictor, and no hidden response or label is consulted.
"""
from __future__ import annotations

from .lookahead import mask_transition_state
from .policy import select_or_stop


def select_two_step_v2(state, candidates, *, predictor, estimator, budget, lambda_cost,
                       risk_level="mean", min_support=12):
    prediction = predictor(state)
    first = estimator.estimate_value(state, candidates, prediction=prediction, risk_level=risk_level)
    scored = []
    for estimate in first:
        action = estimate['action']
        decision = select_or_stop(state, [estimate], budget, lambda_cost,
                                 conservative=risk_level == 'conservative', min_support=min_support)
        # Eligibility (budget/support/availability) differs from one-step value.
        # A negative immediate net value may unlock a positive continuation.
        eligible = decision.get('candidates', [])
        if not eligible:
            scored.append({'action_id':action['action_id'], 'eligible':False,
                           'reason':decision['reason'], 'net_gain':None})
            continue
        first_net = eligible[0]['net_gain']
        projected = mask_transition_state(state, action)
        remaining = [a for a in candidates if a['action_id'] != action['action_id']]
        remaining_budget = {unit: max(0., value-action['cost'].get('latency_ms' if unit=='deadline_ms_remaining' else unit, 0.))
                            for unit,value in budget.items()}
        continuations = estimator.estimate_value(projected, remaining, prediction=prediction, risk_level=risk_level)
        following = select_or_stop(projected, continuations, remaining_budget, lambda_cost,
                                   conservative=risk_level=='conservative', min_support=min_support)
        future = following['net_gain'] if following['decision']=='action' else 0.
        scored.append({'action_id':action['action_id'], 'action':action, 'eligible':True,
            'first_net_gain':first_net,'continuation_net_gain':future,'net_gain':first_net+future,
            'continuation':following})
    usable = sorted((r for r in scored if r['eligible']), key=lambda r:(-r['net_gain'],r['action_id']))
    best = usable[0] if usable else None
    acquire = best is not None and best['net_gain'] > 0
    return {'decision':'action' if acquire else 'stop','action_id':best['action_id'] if acquire else None,
        'action':best['action'] if acquire else None,'net_gain':best['net_gain'] if best else None,
        'reason':'positive_two_step_net_gain' if acquire else 'nonpositive_two_step_net_gain',
        'two_step':scored,'policy_version':'two-step-feasible-mask-v2',
        'limitation':'fixed-current-prediction mask approximation; no observation distribution integration'}
