"""Bounded acquisition while empirical information value is unmeasured.

The allowance is opt-in, finite and reserved on accepted decisions. Its cost
ledger uses frozen declared loss costs; actual HTTP/LLM resources remain in the
executor's separate ledger. Cheapest-first exploration is not a VoI estimate.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
import math

from .acquisition_boundary import packed
from .audit import digest
from .query_value_contract import VALUE_UNMEASURED
from .state_bound_context_value import (CONTEXT_UNKNOWN, NO_MATCH, INSUFFICIENT_SUPPORT,
                                        SUPPORTED_INTERVAL)

VERSION = 'runtime-bounded-exploration-v1'
POLICY_VERSION = 'state-bound-runtime-policy-v4'
STATE_FIELD = 'bounded_exploration_state'
RESERVATION = 'exploration_reservation'
BASIS = 'bounded_unmeasured_exploration'


def configuration(options, unit):
    if not isinstance(options, dict) or set(options) != {'max_actions', 'max_declared_cost'}:
        raise ValueError('EXPLORATION_EXACT_LIMITS_REQUIRED')
    count, cost = options['max_actions'], options['max_declared_cost']
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError('EXPLORATION_ACTION_LIMIT_1_TO_16')
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0:
        raise ValueError('EXPLORATION_POSITIVE_FINITE_COST_BUDGET_REQUIRED')
    return {'version': VERSION, 'max_actions': count, 'max_declared_cost': float(cost),
        'cost_unit': unit, 'method': 'lowest_declared_cost_then_route_id',
        'scope': 'generated_exact_registered_unmeasured_routes',
        'reservation_scope': 'accepted_decisions_including_failed_or_unexecuted_acquisitions',
        'deduplication': 'all_accepted_source_version_action_groups',
        'numeric_voi_calibrated': False, 'selection_probability': None}


def descriptor(identity):
    config = (identity or {}).get('configuration', {})
    saved = config.get('bounded_exploration')
    if saved is None:
        return None
    expected = configuration({k: saved.get(k) for k in ('max_actions', 'max_declared_cost')},
                             config['model_descriptor']['loss_unit'])
    from .runtime_discovered_routes import descriptor as discovery_descriptor
    dynamic = discovery_descriptor(identity)
    if dynamic is not None:
        expected['scope'] = 'generated_registered_or_capability_bound_unmeasured_routes'
    from .runtime_source_support import policy_contract
    strict = policy_contract(identity)
    if saved != expected or (identity.get('version') != POLICY_VERSION and dynamic is None and strict is None):
        raise ValueError('EXPLORATION_CONFIGURATION_MISMATCH')
    return saved


def group(route):
    return digest({k: route[k] for k in ('source_id', 'source_version', 'action_id')})


def _matching(routes, response, match_input):
    from .state_bound_runtime_policy import _matches
    return [route for route in routes if _matches(route, response, match_input)]


def project_state(identity, operations, responses, match_input):
    """Project the complete accepted history; model output cannot reset it."""
    config = descriptor(identity)
    if config is None:
        return None
    reservations = []
    for op in operations:
        record = op.get(RESERVATION)
        if record is None:
            continue
        if (op.get('kind') != 'model' or not op.get('decision_digest')
                or record.get('operation_id') != op.get('operation_id')
                or record.get('decision_digest') != op['decision_digest']):
            raise ValueError('EXPLORATION_UNACCEPTED_RESERVATION')
        reservations.append(deepcopy(record))
    costs = [Decimal(str(r['declared_cost'])) for r in reservations]
    total = sum(costs, Decimal(0))
    if len(reservations) > config['max_actions'] or total > Decimal(str(config['max_declared_cost'])):
        raise ValueError('EXPLORATION_RESERVATION_LIMIT_EXCEEDED')
    routes = identity['configuration']['model_descriptor']['routes']
    used = {group(route) for response in responses for route in _matching(routes, response, match_input)}
    result = {'version': VERSION, 'reserved_actions': len(reservations),
        'reserved_declared_cost': float(total), 'cost_unit': config['cost_unit'],
        'remaining_actions': config['max_actions'] - len(reservations),
        'remaining_declared_cost': float(Decimal(str(config['max_declared_cost'])) - total),
        'used_route_groups': sorted(used), 'reservations': reservations,
        'realized_information_gain': None, 'numeric_voi_training_eligible': False}
    from .runtime_discovered_routes import descriptor as discovery_descriptor, accepted_keys
    dynamic = discovery_descriptor(identity)
    if dynamic is not None:
        from .runtime_discovered_routes import LOCAL_VERSION
        result['used_discovered_action_keys'] = accepted_keys(responses, include_local=dynamic['version'] == LOCAL_VERSION)
    return result


def _eligible(public, base, voi, config, routes):
    state = public.get(STATE_FIELD)
    if not isinstance(state, dict) or state.get('version') != VERSION:
        raise ValueError('EXPLORATION_STATE_REQUIRED')
    eligible, exclusions = [], {}
    if (base.get('selection_basis') != 'open_world_value_withheld'
            or voi.get('blocked_reason') is not None):
        return [], {}, 'base_policy_takes_precedence'
    if state['remaining_actions'] <= 0:
        return [], {}, 'exploration_action_budget_exhausted'
    if (public.get('final_decision_only') or public.get('seconds_to_deadline', 0) <= 0
            or public.get('remaining', {}).get('llm_calls', 0) <= 1):
        return [], {}, 'operational_budget_or_deadline'
    by_id = {route['route_id']: route for route in routes}
    dynamic = voi.get('discovered_routes', {})
    by_id.update({route['route_id']:route for route in dynamic.values()})
    bindings = voi['route_bindings']
    # An alias with unknown value never reopens a declared information group
    # which has supported nonpositive value in the same candidate menu.
    nonpositive = set()
    for row in base.get('candidates', []):
        value = row.get('value') or {}
        binding = bindings.get(row['candidate_id'], {})
        route = by_id.get(binding.get('route_id'))
        if route and value.get('net_value') is not None and value['net_value'] <= 0:
            nonpositive.add(group(route))
    for row in base.get('candidates', []):
        cid, value = row['candidate_id'], row.get('value') or {}
        binding = bindings.get(cid, {})
        route = by_id.get(binding.get('route_id'))
        reasons = []
        local_action = row.get('action') == 'query_acquired' and cid in dynamic
        if not row.get('feasible') or (row.get('action') not in {'read_url', 'search_web'} and not local_action):
            reasons.append('not_feasible_source_action')
        if value.get('value_status') != VALUE_UNMEASURED:
            reasons.append('not_unmeasured')
        is_dynamic = binding.get('status')=='capability_bound_unmeasured' and cid in dynamic
        if (binding.get('status') != 'exact_route' and not is_dynamic) or route is None:
            reasons.append('exact_registered_route_required')
        interval = voi['intervals'].get(cid, {})
        status = interval.get('support_status')
        incomplete = status in {CONTEXT_UNKNOWN, NO_MATCH, INSUFFICIENT_SUPPORT}
        crossing = (status == SUPPORTED_INTERVAL and route is not None
                    and interval.get('lower_bound') is not None and interval.get('upper_bound') is not None
                    and interval['lower_bound'] <= route['cost'] < interval['upper_bound'])
        if not incomplete and not crossing and not is_dynamic:
            reasons.append('interval_not_eligible_for_exploration')
        if route:
            if 'used_discovered_action_keys' in state:
                from .runtime_discovered_routes import action_key, registered_request_payload
                key = (route['action_key'] if is_dynamic else
                       action_key(registered_request_payload(route, public['match_input'])))
                if key in state['used_discovered_action_keys']:
                    reasons.append('normalized_action_already_accepted')
                if any(guard['action_key']==key for guard in voi.get('discovery_action_guards', [])):
                    reasons.append('registered_nonpositive_action_guard')
            if group(route) in state['used_route_groups']:
                reasons.append('information_group_already_accepted')
            if group(route) in nonpositive:
                reasons.append('information_group_supported_nonpositive')
            if Decimal(str(route['cost'])) > Decimal(str(state['remaining_declared_cost'])):
                reasons.append('declared_cost_allowance_exceeded')
            resource = 'local_reads' if local_action else 'http_calls'
            if public.get('remaining', {}).get(resource, 0) <= 0:
                reasons.append(resource + '_exhausted')
        if reasons:
            exclusions[cid] = reasons
        else:
            eligible.append({'candidate_id': cid, 'route_id': route['route_id'],
                'route_group': group(route), 'declared_cost': route['cost'], 'cost_unit': route['unit']})
    eligible.sort(key=lambda r: (r['declared_cost'], r['route_id'], r['candidate_id']))
    return eligible, exclusions, 'eligible' if eligible else 'no_eligible_unmeasured_route'


def select(public, choice, voi, config, routes):
    base = json.loads(choice.diagnostics_json)
    eligible, exclusions, status = _eligible(public, base, voi, config, routes)
    diagnostic = {'version': VERSION, 'status': status, 'eligible': eligible,
        'exclusions': exclusions, 'state_before': deepcopy(public[STATE_FIELD]),
        'cost_semantics': 'frozen_declared_loss_cost_not_realized_resource_cost',
        'selection_probability': None, 'candidate_generation_complete': False,
        'numeric_voi_calibrated': False, 'global_optimum_claim': False}
    if not eligible:
        base['bounded_exploration'] = diagnostic
        return replace(choice, diagnostics_json=packed(base))
    selected = eligible[0]
    value = next(row['value'] for row in base['candidates'] if row['candidate_id'] == selected['candidate_id'])
    diagnostic['status'] = 'reserved_on_acceptance'
    return replace(choice, candidate_id=selected['candidate_id'], terminal_action=None,
        reason='当前候选价值尚未测量；在独立有限探索额度内选择声明成本最低且尚未获取的信息动作。',
        diagnostics_json=packed({'selection_basis': BASIS, 'selected_value': value,
            'candidates': base['candidates'],
            'evaluated_candidate_ids': base['evaluated_candidate_ids'],
            'unmeasured_candidate_ids': base['unmeasured_candidate_ids'],
            'selection_scope': config['scope'], 'bounded_exploration': diagnostic,
            'value_selection_before_exploration': base, 'query_value_decision': None,
            'query_value_decision_omission_reason': 'unmeasured_exploration_not_positive_voi',
            'candidate_generation_complete': False, 'global_optimum_claim': False}))


def reservation(public, response, choice, operation_id, *, batch=None):
    """Recheck the allowance from frozen routes before atomic acceptance/replay.

    Interval diagnostics retain their existing policy-reported evidence status;
    this validates reservation accounting, not empirical/causal utility.
    """
    identity = public.get('selection_policy')
    config = descriptor(identity)
    diag = choice['diagnostics']
    if diag.get('selection_basis') != BASIS:
        return None
    if config is None or choice.get('terminal_action') is not None:
        raise ValueError('EXPLORATION_UNEXPECTED_SELECTION')
    routes = identity['configuration']['model_descriptor']['routes']
    from . import runtime_discovered_routes as discovery
    dynamic_config = discovery.descriptor(identity)
    if dynamic_config is not None:
        if batch is None:
            raise ValueError('DISCOVERY_REPLAY_CANDIDATE_BATCH_REQUIRED')
        bound, excluded = discovery.bindings(public,batch,dynamic_config,routes)
        voi = diag['state_bound_voi']
        guards = discovery.registered_guards(routes,voi['registered_route_intervals'],public['match_input'])
        if (voi['discovered_routes']!=bound or voi['discovery_exclusions']!=excluded
                or voi['discovery_action_guards']!=guards):
            raise ValueError('DISCOVERY_RESERVATION_BINDING_MISMATCH')
    eligible, _, _ = _eligible(public, diag['value_selection_before_exploration'],
                               diag['state_bound_voi'], config, routes)
    matches = _matching(routes, response, public['match_input'])
    selected_dynamic = diag['state_bound_voi'].get('discovered_routes',{}).get(choice['candidate_id'])
    if selected_dynamic is not None:
        matches = [discovery.bind(public,response,dynamic_config)]
    if (not eligible or choice['candidate_id'] != eligible[0]['candidate_id']
            or len(matches) != 1 or matches[0]['route_id'] != eligible[0]['route_id']
            or diag['selected_value'].get('expected_value') is not None):
        raise ValueError('EXPLORATION_RESERVATION_SELECTION_MISMATCH')
    record = {'version': VERSION, 'operation_id': operation_id,
        'decision_digest': digest(response), 'snapshot_digest': digest(public),
        **eligible[0], 'value_status': VALUE_UNMEASURED, 'realized_information_gain': None,
        'numeric_voi_training_eligible': False}
    if selected_dynamic is not None:
        record['discovered_route'] = matches[0]
    return record
