"""Public protocol projection; never reads archives, hidden snapshots or labels.

The action catalog is capability metadata. Costs below describe an explicitly
simulated reveal experiment over an already cached provider response, not HTTP
savings or repeated supplier charges for the same underlying response.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Mapping

from football_platform.information import ActionSpec, CostVector, ExperimentConfig, simulated_cost
from football_platform.information.contracts import PROTOCOL_VERSION
from .state import GROUP_FIELDS, parse_time, sanitize_state

VERSION = 'research-public-information-bridge-v1'
GROUP_MAP = {'strength': 'elo', 'recent_form': 'recent_form', 'rolling_xg': 'xg'}
DIMENSIONS = ('requests', 'money', 'latency_ms', 'tokens')
MASK_MAP = {'UNACQUIRED': 'not_acquired', 'OBSERVED': 'observed',
            'SOURCE_MISSING': 'source_missing', 'STALE': 'stale', 'FAILED': 'source_missing'}


def _dump(value):
    return value.model_dump(mode='json') if hasattr(value, 'model_dump') else dict(value)


def _inert(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}', value)) and not re.match(r'^[A-Za-z]:', value)


def _cost(value):
    # A real unknown component remains null. The public validator enforces the
    # amount/basis correspondence; reject nonfinite quantities explicitly too.
    cost = CostVector.model_validate(_dump(value)).model_dump(mode='json')
    if any(part['amount'] is not None and not math.isfinite(part['amount']) for part in cost.values()):
        raise ValueError('Cost amounts must be finite or explicitly unknown')
    bases = {part['basis'] for part in cost.values()}
    status = 'unknown' if 'unknown' in bases else 'simulated' if 'simulated' in bases else 'estimated' if 'estimated' in bases else 'known'
    return {name: cost[name]['amount'] for name in DIMENSIONS}, status, cost


def to_research_actions(public_actions):
    """Project ActionSpec objects/dicts; unsupported groups remain non-capable.

    Actions with no model-supported group are omitted from numerical candidates.
    Their public catalog entry remains in experiment manifests. A partially
    unsupported action is retained but disabled, never silently split/underpriced.
    """
    actions = []
    for value in public_actions:
        source = _dump(value)
        # Select domain fields before validation so arbitrary returned values,
        # source payloads or unknown target availability never reach the policy.
        domain = {k: source[k] for k in ActionSpec.model_fields if k in source}
        action = ActionSpec.model_validate(domain).model_dump(mode='json')
        groups = [GROUP_MAP[group] for group in action['group_ids'] if group in GROUP_MAP]
        if not groups:
            continue
        costs, status, detail = _cost(action['expected_cost'])
        all_supported = len(groups) == len(action['group_ids'])
        actions.append({'action_id': action['action_id'], 'group_ids': groups,
            'available': action['capability_status'] == 'AVAILABLE' and all_supported,
            'capability_status': action['capability_status'] if all_supported else 'UNSUPPORTED_GROUPS',
            'cost': costs, 'cost_status': status, 'cost_components': detail,
            'cost_scope': action['cost_scope'], 'source': action['source'], 'tool': action['tool'],
            'required_inputs': action['required_inputs'], 'prerequisites': action['prerequisites'],
            'billing_key': action['billing_key'], 'dependency_key': action['dependency_key'],
            'cache_policy': action['cache_policy'], 'idempotency': action['idempotency'],
            'retry_allowed': action['refresh_supported'], 'refresh_allowed': action['refresh_supported'],
            'max_attempts': 2 if action['refresh_supported'] else 1,
            'bridge_version': VERSION})
    return actions


def experiment_actions(seed=7):
    """Return two public-config-bound simulated reveal actions and manifest.

    Money uses simulation_credit, requests uses projection_call. lambda.money
    therefore has units log-loss nats per simulation_credit. No actual supplier
    fee, HTTP request, online latency or token reduction is asserted.
    """
    config = ExperimentConfig(mode='simulated_reveal',
        bindings={'results_view': ['strength', 'recent_form'], 'xg_view': ['rolling_xg']},
        costs={'results_view': simulated_cost(money=.02, requests=1, latency_ms=0, tokens=0),
               'xg_view': simulated_cost(money=.03, requests=1, latency_ms=0, tokens=0)},
        budget={'money': .05, 'requests': 2}, seed=seed, experiment_role='exploratory')
    catalog = [ActionSpec(action_id=aid, source='understat', tool='frozen_evidence_projection',
        group_ids=groups, expected_cost=config.cost(aid), cost_scope='simulated_reveal',
        billing_key=f'simulated-reveal:{aid}', dependency_key='understat_league_response',
        cache_policy='shared_response') for aid, groups in config.bindings.items()]
    serialized = [item.model_dump(mode='json') for item in catalog]
    digest = hashlib.sha256(json.dumps(serialized, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    manifest = {'bridge_version': VERSION, 'protocol_version': PROTOCOL_VERSION,
        'config_type': 'football_platform.information.ExperimentConfig', 'config': config.model_dump(mode='json'),
        'config_version': config.version, 'catalog_type': 'football_platform.information.ActionSpec',
        'catalog': serialized, 'catalog_version': f'research-bound-public-catalog-v1:{digest}',
        'group_map': GROUP_MAP, 'cost_interpretation': 'simulated_reveal_of_cached_views_not_HTTP_savings',
        'dependency_semantics': 'Underlying Understat response contains all three groups. results_view jointly reveals strength and recent_form at one simulated charge. These are controlled reveal costs, not repeated real supplier fees.',
        'source_bundle_reference': 'A physical source request must bind all three groups once; public ExperimentConfig source_bundle enforces this.',
        'cache_views_reference': 'Public cache_views mode records zero incremental external query fees for existing offline cache.',
        'lambda_money_unit': 'log_loss_nats_per_simulation_credit'}
    return to_research_actions(catalog), manifest


def to_research_state(public_state):
    """Project public VisibleState or dict to model/policy-visible input only.

    Mask is authoritative. Unknown/non-observed group payloads are never parsed,
    hashed, copied or exposed. Opaque evidence identifiers are inert; paths and
    arbitrary manifests are dropped. Action receipts preserve bounded retry
    history; FAILED becomes source_missing with its public failure reason.
    """
    source = _dump(public_state)
    fixture = _dump(source['fixture'])
    if source['match_id'] != fixture['match_id']:
        raise ValueError('Visible fixture and state match IDs differ')
    masks = source.get('mask', {})
    public_groups = source.get('groups', {})
    history = []
    for value in source.get('history', []):
        row = _dump(value)
        aid, status = row.get('action_id'), row.get('status')
        if not _inert(aid) or not _inert(status):
            raise ValueError('Action history requires inert identifiers and status')
        entry = {'action_id': aid, 'status': status,
                 'group_ids': [GROUP_MAP[g] for g in row.get('group_ids', []) if g in GROUP_MAP]}
        for key in ('request_id', 'reason'):
            if _inert(row.get(key)):
                entry[key] = row[key]
        if type(row.get('sequence')) is int:
            entry['sequence'] = row['sequence']
        if row.get('charged_cost') is not None:
            costs, basis, _ = _cost(row['charged_cost'])
            entry.update(cost=costs, cost_status=basis)
        history.append(entry)
    groups, reasons = {}, {}
    for public_id, research_id in GROUP_MAP.items():
        mask = masks.get(public_id, 'UNACQUIRED')
        if mask not in MASK_MAP:
            raise ValueError('Unknown public evidence mask')
        status = MASK_MAP[mask]
        group = {'status': status, 'values': {}}
        if status == 'observed':
            supplied = _dump(public_groups[public_id])
            if supplied.get('group_id') != public_id:
                raise ValueError('Visible group identity conflicts with key')
            quality = supplied.get('quality')
            if quality in {'INVALID', 'SOURCE_MISSING'}:
                group['status'] = 'source_missing'
                reasons[research_id] = quality
            elif quality == 'STALE':
                group['status'] = 'stale'
                reasons[research_id] = quality
            else:
                times = _dump(supplied['times'])
                if source['information_mode'] == 'operational' and any(
                    parse_time(times[key]) > parse_time(source['prediction_cutoff']) for key in
                    ('observed_at', 'ingested_at', 'operational_available_at', 'source_published_at') if times.get(key)):
                    raise ValueError('Operational evidence is unavailable at prediction cutoff')
                group.update(values={field: supplied.get('values', {}).get(field) for field in GROUP_FIELDS[research_id]},
                    available_at=times['feature_available_at'],
                    version=supplied.get('version') if _inert(supplied.get('version')) else 'unspecified',
                    evidence_refs=[ref for ref in supplied.get('evidence_refs', []) if _inert(ref)])
        if mask == 'FAILED':
            reasons[research_id] = next((r.get('reason', r['status']) for r in reversed(history)
                                        if research_id in r['group_ids']), 'FAILED')
        groups[research_id] = group
    clean = sanitize_state({'match_id': source['match_id'], 'prediction_cutoff': source['prediction_cutoff'],
        'information_mode': source['information_mode'],
        'base': {k: fixture[k] for k in ('competition_id', 'home_team_id', 'away_team_id', 'kickoff')},
        'groups': groups})
    for group, reason in reasons.items():
        clean['groups'][group]['reason'] = reason
    clean['action_history'] = history
    public_budget = _dump(source.get('budget', {}))
    budget = {}
    for part in ('limits', 'spent'):
        values = public_budget.get(part, {})
        budget[part] = {}
        for key in DIMENSIONS:
            if key not in values:
                continue
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Visible budget quantities must be finite and nonnegative')
            budget[part][key] = float(value)
    budget['unknown_components'] = [name for name in public_budget.get('unknown_components', []) if name in DIMENSIONS]
    clean['budget'] = budget
    clean['bridge_version'] = VERSION
    return clean
