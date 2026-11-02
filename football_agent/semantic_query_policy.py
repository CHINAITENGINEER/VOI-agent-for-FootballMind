"""Research-only semantic f/g boundary using the existing menu selector.

No transport, extraction, model training or outcome lookup lives here. A g
trained against another f or a different input schema cannot be substituted.
"""
from copy import deepcopy
import math

from football_research.acquisition.context_predictor import _digest
from football_research.acquisition.joint_planner import UNITS, _budget, _number
from football_research.acquisition.semantic_query import VERSION as INPUT_VERSION, validate_query
from .query_value_selection import predictor_value_blocker, select_query_candidates

VERSION = 'semantic-query-research-policy-v1'
TARGET = 'conditional_frozen_predictor_log_loss_change'


class SemanticQueryPolicy:
    def __init__(self, *, predictor_payload_sha256, lambda_cost=None, value_model=None,
                 input_schema=INPUT_VERSION):
        # Explicit opt-in: old archived policy manifests and query bytes stay
        # unchanged. New f inputs cannot accidentally bind the old g contract.
        if input_schema == INPUT_VERSION:
            self.validate_query = validate_query
        else:
            from football_research.acquisition.entity_semantic_query import (
                VERSION as ENTITY_INPUT_VERSION, validate_query as validate_entity_query)
            if input_schema != ENTITY_INPUT_VERSION:
                raise ValueError('Unknown semantic query input schema')
            self.validate_query = validate_entity_query
        weights = lambda_cost or {}
        if set(weights) - set(UNITS):
            raise ValueError('Unknown loss-conversion unit')
        self.lambda_cost = {u: _number(weights.get(u, 0.), 'lambda '+u) for u in UNITS}
        self.predictor_payload_sha256 = predictor_payload_sha256
        self.value_model = value_model
        self.value_binding = None
        if value_model is not None:
            meta = value_model.metadata
            entity_representation = None
            if input_schema != INPUT_VERSION:
                from football_research.acquisition.entity_semantic_f import (
                    REPRESENTATION_SHA256,
                )
                entity_representation = REPRESENTATION_SHA256
            if (meta.get('input_schema') != input_schema
                    or meta.get('predictor_payload_sha256') != predictor_payload_sha256
                    or meta.get('target') != TARGET
                    or (entity_representation is not None
                        and meta.get('representation_sha256') != entity_representation)
                    or not isinstance(meta.get('training_physical_match_ids'), list)
                    or not isinstance(meta.get('predictor_training_physical_match_ids'), list)
                    or set(meta['training_physical_match_ids']) & set(meta['predictor_training_physical_match_ids'])
                    or meta.get('evidence_origin') not in {'historical_reconstruction', 'constructed_mechanism'}):
                raise ValueError('g requires this exact f, entity representation, pre-query schema and independent training groups')
            self.value_binding = _digest({'version': value_model.model_version, 'metadata': meta})
        self.manifest = {'schema': VERSION, 'input_schema': input_schema,
            'predictor_payload_sha256': predictor_payload_sha256,
            'value_model_version': value_model.model_version if value_model is not None else None,
            'lambda_cost': self.lambda_cost, 'operational_eligible': False,
            'unknown_policy': 'existing_separately_bounded_exploration',
            'target': TARGET, 'production_ready': False}
        self.version = VERSION + ':' + _digest(self.manifest)[:24]

    def plan(self, queries, budget, *, candidate_generation_complete=False):
        if self.value_model is not None and self.value_binding != _digest({
                'version': self.value_model.model_version, 'metadata': self.value_model.metadata}):
            raise ValueError('Frozen g binding was modified')
        limits, spent = _budget(budget)
        candidates, ids, common = [], set(), None
        for query in queries:
            self.validate_query(query)
            if query['predictor_payload_sha256'] != self.predictor_payload_sha256:
                raise ValueError('Query and policy use different frozen predictors')
            if query['information_mode'] != 'historical_reconstruction':
                raise ValueError('This bounded semantic adapter has not been promoted to live execution')
            fingerprint = _digest({k: v for k, v in query.items() if k not in {'candidate', 'query_sha256'}})
            if common is not None and fingerprint != common:
                raise ValueError('All candidates must share the same actual before state')
            common = fingerprint
            a = query['candidate']
            if a['action_id'] in ids:
                raise ValueError('Duplicate candidate action identity')
            ids.add(a['action_id'])
            why = ('action_unavailable' if not a['available'] else 'already_attempted' if a['already_attempted'] else
                   'physical_budget_exhausted' if any(spent[u]+a['cost'][u] > cap+1e-12
                                                     for u, cap in limits.items()) else None)
            value = None
            if why is None:
                blocker = predictor_value_blocker({'model_version': query['model_version'],
                                                   'predictor_support': query['current']['predictor_support']})
                estimate = {'support_status': 'UNKNOWN', 'expected_signed_gain': None,
                            'reason': blocker or 'no_g_bound_to_this_semantic_f'}
                if not blocker and self.value_model is not None:
                    meta = self.value_model.metadata
                    if query['physical_match_id'] in (meta['training_physical_match_ids']
                                                      + meta['predictor_training_physical_match_ids']):
                        raise ValueError('Decision fixture overlaps f/g training groups')
                    estimate = self.value_model.predict(deepcopy(query))
                supported = estimate.get('support_status') == 'supported'
                gain = estimate.get('expected_signed_gain') if supported else None
                if supported and (type(gain) not in (int, float) or not math.isfinite(gain)):
                    raise ValueError('Supported fixed-f gain must be a finite signed number')
                charge = math.fsum(self.lambda_cost[u]*a['cost'][u] for u in UNITS)
                value = {'status': 'COMPUTED' if supported else 'INELIGIBLE',
                    'reason': estimate['reason'], 'expected_signed_gain': gain,
                    'cost': charge, 'net_voi': gain-charge if supported else None,
                    'estimate': deepcopy(estimate), 'risk_of_harm': None,
                    'value_model_version': self.manifest['value_model_version'],
                    'scope': 'fixed_f_signed_loss_reduction_not_global_information_value'}
            candidates.append({'action_id': a['action_id'], 'query_sha256': query['query_sha256'],
                'cost': deepcopy(a['cost']), 'cost_status': a['cost_status'],
                'feasible': why is None, 'reason': why or value['reason'], 'value': value})
        result = {'schema': VERSION, 'policy_version': self.version, 'candidates': candidates,
            'selection': select_query_candidates(candidates,
                candidate_generation_complete=candidate_generation_complete),
            'candidate_generation_complete': candidate_generation_complete,
            'physical_budget': {'limits': limits, 'spent': spent}, 'operational_eligible': False}
        return {**result, 'decision_digest': _digest(result)}
