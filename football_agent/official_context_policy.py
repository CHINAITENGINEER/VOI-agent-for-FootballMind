"""Optional-prior prediction and signed query gain for verified official fixtures.

The initial source prior remains absent. A fitted predictor may supply its
training-derived base, disclosed separately from the original source state.
Unknown values use only the existing runner's bounded exploration allowance.
"""
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math

from football_research.acquisition.optional_prior_predictor import OptionalPriorContextPredictor
from football_research.acquisition.official_context_models import is_role_predictor
from football_research.acquisition.joint_planner import UNITS, _budget, _cost, _number
from football_research.acquisition.state import parse_time
from .context_voi_policy import ContextVOIPolicy, predictor_bytes, _digest
from .evidence_only_policy import EvidenceOnlyPolicy


VERSION = 'official-optional-context-query-policy-v1'
ROLE_VERSION = 'official-role-context-query-policy-v1'


class OfficialContextVOIPolicy(ContextVOIPolicy):
    """Research policy with a frozen optional-prior f and an optional independent g."""

    def __init__(self, identity, predictor, *, value_model=None, predictor_artifact=None,
                 value_artifact=None, lambda_cost=None, role_identity=None):
        self.base_policy = EvidenceOnlyPolicy(identity)
        self.role_context_mode = is_role_predictor(predictor)
        if not isinstance(predictor, OptionalPriorContextPredictor) and not self.role_context_mode:
            raise ValueError('Official context requires the optional-prior predictor')
        if self.role_context_mode:
            from football_research.data_adapters.fixture_role_identity import verify_fixture_role_identity
            if role_identity is None or role_identity.get('binding') != identity:
                raise ValueError('Role context requires the same raw-verified official fixture identity')
            verify_fixture_role_identity(role_identity, as_of=identity['verified_at'])
            self.role_identity = deepcopy(role_identity)
        elif role_identity is not None:
            raise ValueError('The original predictor cannot silently acquire a role representation')
        predictor._check()
        self.identity = deepcopy(self.base_policy.identity)
        self.physical_match_id = self.base_policy.physical_match_id
        self.initial_state = deepcopy(self.base_policy.initial_state)
        self.initial = deepcopy(self.initial_state)
        self.evidence_match = deepcopy(self.base_policy.evidence_match)
        self.model, self.predictor, self.value_model = self.base_policy.model, predictor, value_model
        self.predictor_bytes = predictor_bytes(predictor) if predictor_artifact is None else predictor_artifact
        if json.loads(self.predictor_bytes) != {**predictor._payload(), 'model_version': predictor.model_version}:
            raise ValueError('Predictor artifact bytes do not represent the frozen model')
        self.predictor_sha256 = sha256(self.predictor_bytes).hexdigest()
        if self.physical_match_id in predictor.metadata['training_physical_match_ids']:
            raise ValueError('The decision fixture was used to train the outcome predictor')
        weights = {} if lambda_cost is None else lambda_cost
        if not isinstance(weights, Mapping) or set(weights) - set(UNITS):
            raise ValueError('Unknown loss-conversion unit')
        self.lambda_cost = {unit: _number(weights.get(unit, 0), 'Lambda '+unit) for unit in UNITS}
        self.value_bytes = None
        if value_model is not None:
            from football_research.acquisition.optional_context_value import OptionalContextValueModel
            if not isinstance(value_model, OptionalContextValueModel):
                raise ValueError('Official context requires the optional-context gain model')
            from football_research.acquisition.role_context_value import RoleContextValueModel
            if isinstance(value_model, RoleContextValueModel) != self.role_context_mode:
                raise ValueError('The outcome and query-value models must share their role representation')
            value_model._check()
            meta = value_model.metadata
            predictor_ids = set(predictor.metadata['training_physical_match_ids'])
            value_ids = set(meta['training_physical_match_ids'])
            if (meta['predictor_sha256'] != self.predictor_sha256
                    or meta['predictor_model_version'] != predictor.model_version
                    or meta['evidence_origin'] != predictor.metadata['evidence_origin']
                    or set(meta['predictor_training_physical_match_ids']) != predictor_ids
                    or predictor_ids & value_ids
                    or self.physical_match_id in value_ids):
                raise ValueError('The gain model must bind this f and physically independent A, B and decision fixtures')
            self.value_bytes = predictor_bytes(value_model) if value_artifact is None else value_artifact
            if json.loads(self.value_bytes) != {**value_model._payload(), 'model_version': value_model.model_version}:
                raise ValueError('Value artifact bytes do not represent the frozen model')
        elif value_artifact is not None:
            raise ValueError('A value artifact requires its fitted value model')
        self.manifest = {'schema': VERSION, 'base_policy': deepcopy(self.base_policy.manifest),
            'physical_match_id': self.physical_match_id, 'predictor_version': predictor.model_version,
            'predictor_sha256': self.predictor_sha256,
            'value_model_version': value_model.model_version if value_model else None,
            'value_model_sha256': sha256(self.value_bytes).hexdigest() if self.value_bytes is not None else None,
            'lambda_cost': self.lambda_cost, 'evidence_origin': predictor.metadata['evidence_origin'],
            'prior_status': 'prior_unavailable', 'prior_probabilities': None,
            'base_scope': 'training_derived_base_disclosed_separately_from_source_prior',
            'value_scope': 'expected_logged_fixed_predictor_signed_log_loss_reduction',
            'support_scope': 'empirical_context_support_not_bayes_evsi_or_coverage_guarantee',
            'unknown_policy': 'separately_bounded_exploration', 'production_ready': False}
        if self.role_context_mode:
            self.manifest['schema'] = ROLE_VERSION
            self.manifest['context_representation'] = 'verified_fixture_role_text_blocks'
        self.version = self.manifest['schema']+':'+_digest(self.manifest)[:24]

    def _visible(self, public, actions=()):
        from football_research.acquisition.visible_state import encode_visible_state
        visible = encode_visible_state(public, actions)
        if self.role_context_mode:
            from football_research.acquisition.fixture_roles import bind_fixture_roles
            visible = bind_fixture_roles(visible, self.role_identity)
        return visible

    def _state(self, state):
        return self.base_policy._state(state)

    def public_state(self, state):
        return self.base_policy.public_state(state)

    def check_start(self, at):
        self.base_policy.check_start(at)
        at = parse_time(at)
        if at > parse_time(self.initial_state['prediction_cutoff']):
            raise ValueError('An official context run must start before its forecast cutoff')
        for model in (self.predictor, self.value_model):
            if model is not None and not (parse_time(model.metadata['training_as_of'])
                    <= parse_time(model.metadata['fit_execution_at']) < at):
                raise ValueError('Actual model fitting must precede the first query')
        if self.value_model is not None and not (
                parse_time(self.predictor.metadata['fit_execution_at'])
                < parse_time(self.value_model.metadata['training_as_of'])):
            raise ValueError('Value training must follow the actual frozen outcome predictor fit')

    def predict_public(self, public):
        self._state(public['state'])
        visible = self._visible(public)
        if visible['before']['prior_probabilities'] is not None:
            raise ValueError('An official optional-context run cannot introduce a source prior')
        result = self.predictor.predict(visible, physical_match_id=self.physical_match_id)
        if result['prior_probabilities'] is not None or result['prior_domain'] != 'missing_prior':
            raise ValueError('The predictor must preserve the absent original prior')
        return {**result, 'initial_model_version': None, 'initial_probabilities': None,
            'prior_status': 'prior_unavailable',
            'probability_basis': 'frozen_optional_context_with_training_derived_base'}

    def plan(self, state, actions, budget, *, candidate_generation_complete=False):
        raise ValueError('Official context planning requires the complete current public snapshot')

    def expired_plan(self, public):
        result = super().expired_plan(public)
        result['schema'] = self.manifest['schema']
        result['decision_digest'] = _digest({k: v for k, v in result.items() if k != 'decision_digest'})
        return result

    def plan_public(self, public, actions, budget, *, candidate_generation_complete=False):
        limits, spent = _budget(budget)
        if any(action.get('group_ids') or action.get('tool') not in {'read_url', 'search_web'} for action in actions):
            raise ValueError('Official context values only support public web information actions')
        visible = self._visible(public, actions)
        prediction = public['prediction']
        if prediction != self.predict_public(public):
            raise ValueError('Query values require the actual current f prediction')
        candidates = []
        for action in actions:
            cost = _cost(action)
            reason = ('action_unavailable' if not action.get('available', True) else
                'already_attempted' if visible['candidates'][action['action_id']]['already_attempted'] else
                'physical_budget_exhausted' if any(spent[u]+cost[u] > cap+1e-12 for u, cap in limits.items()) else None)
            value = None
            if reason is None:
                estimate = {'support_status': 'UNKNOWN', 'expected_signed_gain': None,
                    'reason': 'optional_context_gain_model_not_fitted'}
                if not prediction['prior_domain_context_supported']:
                    estimate['reason'] = 'missing_prior_context_domain_unsupported'
                elif self.value_model is not None:
                    if self.role_context_mode:
                        from football_research.acquisition.role_context_query import query_input
                    else:
                        from football_research.acquisition.optional_context_query import query_input
                    inputs = query_input(visible, action, prediction)
                    estimate = self.value_model.predict(inputs, physical_match_id=self.physical_match_id,
                        predictor_sha256=self.predictor_sha256)
                supported = estimate['support_status'] == 'supported'
                gross = estimate['expected_signed_gain'] if supported else None
                if supported and (type(gross) not in (int, float) or not math.isfinite(gross)):
                    raise ValueError('Supported empirical gain must be finite')
                loss_cost = math.fsum(self.lambda_cost[u]*cost[u] for u in UNITS)
                value = {'schema_version': 'official-optional-context-query-value-v1',
                    'candidate_id': action['action_id'], 'status': 'COMPUTED' if supported else 'INELIGIBLE',
                    'reason': estimate['reason'], 'expected_signed_gain': gross, 'gross_evsi': None,
                    'cost': loss_cost, 'net_voi': gross-loss_cost if supported else None,
                    'predictor_version': self.predictor.model_version,
                    'value_model_version': self.value_model.model_version if self.value_model else None,
                    'estimate': estimate, 'calibration_status': 'not_established'}
            candidates.append({**deepcopy(action), 'feasible': reason is None, 'cost': cost,
                'reason': reason or value['reason'], 'value': value})
        feasible = [c for c in candidates if c['feasible']]
        positive = [c for c in feasible if c['value']['status'] == 'COMPUTED' and c['value']['net_voi'] > 1e-12]
        selection = {'candidate_id': None, 'stop_certificate': None}
        if positive:
            selected = min(positive, key=lambda c: (-c['value']['net_voi'], c['action_id']))
            selection.update(decision='acquire', reason='positive_empirical_context_gain', candidate_id=selected['action_id'])
        elif not feasible and any(c['reason'] == 'physical_budget_exhausted' for c in candidates):
            selection.update(decision='halt', reason='physical_budget_exhausted')
        elif any(c['value']['status'] != 'COMPUTED' for c in feasible):
            selection.update(decision='defer', reason='context_gain_support_unknown')
        elif not candidate_generation_complete:
            selection.update(decision='defer', reason='candidate_generation_incomplete')
        else:
            selection.update(decision='stop', reason='declared_menu_nonpositive_empirical_mean')
        result = {'schema': self.manifest['schema'], 'policy_version': self.version, 'model_version': self.predictor.model_version,
            'match_id': public['state']['match_id'], 'state_digest': _digest(public), 'prediction': prediction,
            'candidates': candidates, 'selection': selection, 'physical_budget': {'limits': limits, 'spent': spent},
            'candidate_generation_complete': candidate_generation_complete,
            'selection_scope': 'current_feasible_supported_one_step_menu',
            'calibration_status': 'not_established', 'numeric_voi_training_eligible': False}
        result['decision_digest'] = _digest(result)
        return result


__all__ = ['VERSION', 'OfficialContextVOIPolicy']
