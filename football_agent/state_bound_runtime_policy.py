"""Empirical interval selection inside the existing durable match runtime.

Only explicit, stable tool routes may borrow historical support. The LLM owns
candidate discovery, but its recommendation and semantic self-assessment are
not value labels or observed football context. Opt-in v2 model packages may use
subject-bound source assertions as provisional context through a separate
validated projection. Opt-in v3 packages use fixture-scoped structured reports
under their own frozen generation contract, without actor feature extraction.
No executor is started here.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import re
from urllib.parse import urlsplit

from .acquisition_boundary import packed, PolicyChoice
from .query_value_contract import QueryValueAssessment
from .runtime_voi_model import (RuntimeVoIModel, SOURCE_CLAIM_VERSION, SOURCE_SUPPORT_VERSION,
                               SOURCE_CLAIM_CONTEXT_KEYS)
from .state_bound_context_value import SUPPORTED_INTERVAL
from .value_selection_policy import ValueAwareSelectionPolicy
from .experience_sources import VERSION as TRANSPORT_VERSION
from .structured_sources import PROJECTION_VERSION as SPFTLL_VERSION, registered_request
from .runtime_fixture_identity import VERSION as IDENTITY_VERSION, resolve_match_identity
from .runtime_fixture_bridge import VERSION as BRIDGE_VERSION, validate_fixture_identity
from .runtime_identity_discovery import (VERSION as DISCOVERY_VERSION, MAX_SOURCE_ACTIONS,
                                         discovery_candidate)
from .runtime_identity_pages import VERSION as LOCAL_IDENTITY_VERSION, local_identity_candidate

VERSION = 'state-bound-runtime-policy-v2'
PROJECTION_VERSION = 'runtime-observed-context-v1'
SOURCE_CLAIM_POLICY_VERSION = 'state-bound-runtime-policy-v3'
SOURCE_CLAIM_PROJECTION_VERSION = 'runtime-observed-context-v2'
SOURCE_SUPPORT_POLICY_VERSION = 'state-bound-runtime-policy-v6'
SOURCE_SUPPORT_PROJECTION_VERSION = 'runtime-observed-context-v3'
TARGET = re.compile(r'spftll:[1-9][0-9]*\Z')


def _time(value):
    if not isinstance(value, str):
        raise ValueError('RUNTIME_VOI_TIMESTAMP_REQUIRED')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('RUNTIME_VOI_TIMESTAMP_INVALID') from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('RUNTIME_VOI_TIMESTAMP_TIMEZONE')
    return result.astimezone(timezone.utc)


def observed_runtime_context(public):
    """Technical observations only; football hypotheses remain unknown.

    These feature names describe an acquisition state, not verified world
    facts. Source body text, scores, plans and provisional semantic reviews
    never enter the empirical estimator.
    """
    cutoff = _time(public.get('snapshot_as_of'))
    documents = public.get('documents', [])
    if not isinstance(documents, list):
        raise ValueError('RUNTIME_VOI_DOCUMENTS_REQUIRED')
    observed, excluded = [], []
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError('RUNTIME_VOI_DOCUMENT_OBJECT')
        try:
            timestamp = _time(document.get('observed_at'))
        except ValueError:
            excluded.append({'document_index': index, 'reason': 'missing_or_invalid_observation_time'})
            continue
        if timestamp > cutoff:
            excluded.append({'document_index': index, 'reason': 'after_snapshot'})
            continue
        observed.append(document)
    last = observed[-1] if observed else None
    surface = (last.get('return_observation') or {}).get('evidence_surface') if last else 'none'
    # Arbitrary content cannot introduce a new feature value via this field.
    allowed = {'body_text', 'structured_records', 'local_structured_records',
               'discovery_metadata', 'none'}
    surface = surface if surface in allowed else 'unknown'
    values = {'operation_document_count': len(observed),
              'operation_readable_document_count': sum(d.get('status') == 'readable' for d in observed),
              'operation_last_return_surface': surface,
              'qualification_state': 'unknown', 'tactical_layout': 'unknown'}
    return {'version': PROJECTION_VERSION, 'snapshot_as_of': public['snapshot_as_of'],
            'context': values, 'excluded_documents': excluded,
            'semantics': 'technical_observations_not_football_facts',
            'actor_semantic_assessments_used': False}


def _matches(route, response, match_input):
    pattern = route['match']
    if response.get('action') != pattern['action']:
        return False
    if pattern['action'] == 'read_url':
        return response.get('url') == pattern['url']
    if pattern['action'] == 'search_web':
        return (response.get('engine') == pattern['engine']
                and response.get('query') == pattern['query_template'].replace('{match_input}', match_input))
    return False


def _source_route(response):
    if response['action'] == 'search_web':
        return 'search:' + response['engine'], TRANSPORT_VERSION
    if registered_request(response['url']) is not None:
        return 'spftll', SPFTLL_VERSION
    return 'web:' + urlsplit(response['url']).netloc, TRANSPORT_VERSION


class StateBoundVoISelectionPolicy:
    version = VERSION
    # ExperienceService fixes this clock before model invocation and archives it.
    requires_snapshot_as_of = True
    requires_fixture_identity = True

    def __init__(self, model: RuntimeVoIModel, *, allow_synthetic=False, exploration=None, discovered_routes=None):
        if not isinstance(model, RuntimeVoIModel):
            raise ValueError('RUNTIME_VOI_MODEL_REQUIRED')
        if type(allow_synthetic) is not bool:
            raise ValueError('RUNTIME_VOI_SYNTHETIC_FLAG')
        self.model, self.allow_synthetic = model, allow_synthetic
        if self.requires_source_claims:
            self.version = SOURCE_CLAIM_POLICY_VERSION
        self._exploration = None
        if exploration is not None:
            from .runtime_exploration import configuration, POLICY_VERSION
            if not self.requires_source_claims:
                raise ValueError('EXPLORATION_REQUIRES_SOURCE_CLAIM_MODEL_V2')
            self._exploration = configuration(exploration, model.loss_unit)
            self.version = POLICY_VERSION
        self._discovered_routes = None
        if discovered_routes is not None:
            from .runtime_discovered_routes import configuration, POLICY_VERSION
            if self._exploration is None:
                raise ValueError('DISCOVERY_REQUIRES_EXPLICIT_EXPLORATION')
            self._discovered_routes = configuration(discovered_routes, model.loss_unit)
            if any(r['route_id'].startswith('discovered:') for r in model.routes):
                raise ValueError('DISCOVERY_ROUTE_NAMESPACE_RESERVED')
            self._exploration['scope'] = 'generated_registered_or_capability_bound_unmeasured_routes'
            self.version = POLICY_VERSION

        if self.source_support_contract is not None:
            self.version = SOURCE_SUPPORT_POLICY_VERSION

    @property
    def requires_source_claims(self):
        return self.model.version in (SOURCE_CLAIM_VERSION, SOURCE_SUPPORT_VERSION)

    @property
    def source_support_contract(self):
        return self.model.source_context_contract

    @property
    def configuration(self):
        configuration = {'method': 'STATE_BOUND_EMPIRICAL_INTERVAL_SELECTION',
                'numeric_voi_calibrated': False,
                'model_descriptor': self.model.configuration(),
                'context_projection': PROJECTION_VERSION,
                'transport_version': TRANSPORT_VERSION, 'spftll_projection_version': SPFTLL_VERSION,
                'allow_synthetic': self.allow_synthetic,
                'target_identity': 'canonical_input_or_archive_bound_source_fixture',
                'fixture_identity_version': IDENTITY_VERSION,
                'fixture_bridge_version': BRIDGE_VERSION,
                'identity_discovery': {'version': DISCOVERY_VERSION,
                    'max_source_actions': MAX_SOURCE_ACTIONS,
                    'value_status': 'UNMEASURED_PREREQUISITE'},
                'local_identity_discovery': {'version': LOCAL_IDENTITY_VERSION,
                    'value_status': 'UNMEASURED_PREREQUISITE'},
                'unknown_policy': 'DEFER_NO_QUALITATIVE_FALLBACK',
                'selection_scope': 'proposed_routes_with_supported_context'}
        if self.requires_source_claims:
            from .runtime_source_claims import VERSION as CLAIMS_VERSION
            configuration.update(context_projection=SOURCE_CLAIM_PROJECTION_VERSION,
                                 source_claims_version=CLAIMS_VERSION,
                                 context_semantics=self.model.context_semantics)
        if self._exploration is not None:
            configuration['bounded_exploration'] = dict(self._exploration)
            configuration['unknown_policy'] = 'EXPLICIT_BOUNDED_EXPLORATION_THEN_DEFER'
        if self._discovered_routes is not None:
            from copy import deepcopy
            configuration['discovered_routes'] = deepcopy(self._discovered_routes)
        if self.source_support_contract is not None:
            configuration.update(context_projection=SOURCE_SUPPORT_PROJECTION_VERSION,
                                 source_context_contract=self.source_support_contract)
        return configuration

    @property
    def requires_test_fixture(self):
        return self.allow_synthetic and self.model.evidence_basis == 'synthetic'

    def select(self, context, batch):
        if context != batch.context:
            raise ValueError('RUNTIME_VOI_BATCH_CONTEXT_MISMATCH')
        public = context.state
        projection = observed_runtime_context(public)
        cutoff = _time(public['snapshot_as_of'])
        created = _time(public.get('reference_utc'))
        if cutoff < created:
            raise ValueError('RUNTIME_VOI_SNAPSHOT_BEFORE_CREATION')
        identity = (validate_fixture_identity(public) if 'fixture_identity' in public
                    else resolve_match_identity(public))
        target = identity['target_key']
        source_claim_projection = None
        source_claim_capability_missing = False
        if self.requires_source_claims:
            from .runtime_source_claims import CAPABILITY_FIELD, project_context
            projection['version'] = SOURCE_CLAIM_PROJECTION_VERSION
            projection['context'].update({key: 'unknown' for key in sorted(SOURCE_CLAIM_CONTEXT_KEYS)})
            source_claim_capability_missing = CAPABILITY_FIELD not in public
            if self.source_support_contract is not None:
                from . import runtime_source_support as support
                if public.get(support.CAPABILITY_FIELD) != self.source_support_contract:
                    raise ValueError('RUNTIME_SOURCE_SUPPORT_CAPABILITY_MISMATCH')
                support.validate_support(public)
                projection['version'] = SOURCE_SUPPORT_PROJECTION_VERSION
            elif 'source_support_capability' in public:
                raise ValueError('RUNTIME_SOURCE_SUPPORT_MODEL_SEMANTICS_MISMATCH')
            if not source_claim_capability_missing:
                source_claim_projection = project_context(public, response=batch.candidates[0].response)
                projection['context'].update({key: source_claim_projection['context'][key]
                                              for key in sorted(SOURCE_CLAIM_CONTEXT_KEYS)})
            projection['semantics'] = 'technical_observations_and_actor_source_claims_not_world_facts'
            if self.source_support_contract is not None:
                projection['semantics'] = self.model.context_semantics
                projection['actor_semantic_assessments_used'] = False
            projection['source_claims_used'] = source_claim_projection is not None
        block = None
        if public.get('task_type') != 'match_analysis':
            block = 'match_task_required'
        elif source_claim_capability_missing:
            block = 'source_claim_capability_required'
        elif self.model.evidence_basis == 'synthetic' and not self.allow_synthetic:
            block = 'synthetic_history_disabled'
        elif _time(self.model.calibration_as_of) > created:
            block = 'calibration_after_run_creation'
        elif projection['excluded_documents']:
            block = 'incomplete_observation_clocks'
        elif target is None:
            block = 'canonical_target_unresolved'
        routes = self.model.routes
        estimator = self.model.estimator
        intervals, bindings = {}, {}
        discovered, discovery_excluded, registry_intervals, action_guards = {}, {}, {}, []
        if self._discovered_routes is not None and block is None:
            from . import runtime_discovered_routes as discovery
            discovered, discovery_excluded = discovery.bindings(public,batch,self._discovered_routes,routes)
            registry_intervals = {route['route_id']:estimator.estimate_candidate(
                route['route_id'],projection['context'],cutoff=public['snapshot_as_of'],
                target_key=target,cost=route['cost'],cost_unit=route['unit']).as_dict() for route in routes}
            action_guards = discovery.registered_guards(routes,registry_intervals,public.get('match_input',''))

        def assess(state, candidate):
            matches = [route for route in routes if _matches(route, candidate.response, public.get('match_input', ''))]
            reason = block or ('route_unregistered' if not matches else
                               'ambiguous_route' if len(matches) != 1 else None)
            if reason:
                if reason == 'route_unregistered' and candidate.candidate_id in discovered:
                    route = discovered[candidate.candidate_id]
                    bindings[candidate.candidate_id] = {'status':'capability_bound_unmeasured',
                        'route_id':route['route_id'],'source_id':route['source_id'],
                        'source_version':route['source_version'],'action_id':route['action_id']}
                    return QueryValueAssessment.unmeasured(candidate.candidate_id,cost=route['cost'],
                        reason='new_tool_action_without_empirical_history')
                bindings[candidate.candidate_id] = {'status': reason,
                    'matched_route_ids': [route['route_id'] for route in matches]}
                return QueryValueAssessment.unmeasured(candidate.candidate_id, reason=reason)
            route = matches[0]
            if (route['source_id'], route['source_version']) != _source_route(candidate.response):
                bindings[candidate.candidate_id] = {'status': 'runtime_source_route_mismatch',
                    'route_id': route['route_id']}
                return QueryValueAssessment.unmeasured(candidate.candidate_id,
                    reason='runtime_source_route_mismatch')
            interval = estimator.estimate_candidate(
                route['route_id'], projection['context'], cutoff=public['snapshot_as_of'],
                target_key=target, cost=route['cost'], cost_unit=route['unit'])
            intervals[candidate.candidate_id] = interval.as_dict()
            bindings[candidate.candidate_id] = {'status': 'exact_route',
                'route_id': route['route_id'], 'snapshot_digest': context.snapshot_digest,
                'source_id': route['source_id'], 'source_version': route['source_version'],
                'action_id': route['action_id']}
            supported = interval.support_status == SUPPORTED_INTERVAL
            if supported and interval.lower_bound > route['cost']:
                value, reason = interval.lower_bound, 'empirical_lower_bound_above_cost'
            elif supported and interval.upper_bound <= route['cost']:
                value, reason = max(0.0, interval.upper_bound), 'empirical_upper_bound_at_or_below_cost'
            else:
                return QueryValueAssessment.unmeasured(candidate.candidate_id, cost=route['cost'],
                    reason='interval_crosses_cost' if supported else interval.reason)
            return QueryValueAssessment.modeled(candidate.candidate_id, expected_value=value,
                                                cost=route['cost'], reason=reason)

        prerequisite = (discovery_candidate(public, batch, identity)
                        if block == 'canonical_target_unresolved' and 'fixture_identity' in public else None)
        prerequisite_kind = 'catalogue_source' if prerequisite is not None else None
        if prerequisite is None and block == 'canonical_target_unresolved' and 'fixture_identity' in public:
            prerequisite = local_identity_candidate(public, batch, identity)
            prerequisite_kind = 'local_query' if prerequisite is not None else None
        if prerequisite is not None:
            choice = PolicyChoice(context.task_id, context.snapshot_digest,
                candidate_id=prerequisite.candidate_id,
                reason=('在本地读取预算内查询已归档赛程，确认用户所指比赛；该前置动作的信息价值尚未测量。'
                        if prerequisite_kind == 'local_query' else
                        '在一次赛程来源动作限额内确认用户所指比赛；该前置动作的信息价值尚未测量。'),
                diagnostics_json=packed({
                    'selection_basis': ('identity_prerequisite_local_query' if prerequisite_kind == 'local_query'
                                        else 'identity_prerequisite_discovery'),
                    'selection_scope': ('proposed_literal_team_query_within_local_budget' if prerequisite_kind == 'local_query'
                                        else 'one_proposed_catalogue_fixture_action_per_run'),
                    'candidate_generation_complete': False,
                    'selected_value': QueryValueAssessment.unmeasured(
                        prerequisite.candidate_id, reason='identity_prerequisite_not_empirical_voi').as_dict(),
                    'query_value_decision': None,
                    'query_value_decision_omission_reason': 'unmeasured_identity_prerequisite',
                    'global_optimum_claim': False,
                }))
        else:
            choice = ValueAwareSelectionPolicy(assess).select(context, batch)
        diagnostics = json.loads(choice.diagnostics_json)
        diagnostics['state_bound_voi'] = {
            'version': self.version, 'snapshot_digest': context.snapshot_digest,
            'model_descriptor': self.model.configuration(), 'projection': projection,
            'target_key': target, 'target_identity_basis': identity['identity_basis'],
            'fixture_identity': identity,
            'identity_prerequisite_selected': prerequisite is not None,
            'identity_prerequisite_kind': prerequisite_kind,
            'blocked_reason': block, 'route_bindings': bindings, 'intervals': intervals,
            'numeric_voi_calibrated': False,
            'evidence_basis': self.model.evidence_basis,
            'estimator_scope': 'conditional_observational_history_not_causal_proof',
            'interval_coverage_scope': 'one_selection_not_anytime_valid',
            'global_optimum_claim': False,
        }
        if self.requires_source_claims:
            diagnostics['state_bound_voi']['source_claim_projection'] = source_claim_projection
        if self._discovered_routes is not None:
            diagnostics['state_bound_voi']['discovered_routes'] = discovered
            diagnostics['state_bound_voi']['discovery_exclusions'] = discovery_excluded
            diagnostics['state_bound_voi']['registered_route_intervals'] = registry_intervals
            diagnostics['state_bound_voi']['discovery_action_guards'] = action_guards
        if self._exploration is not None:
            from .runtime_exploration import select
            explored = select(public, choice, diagnostics['state_bound_voi'], self._exploration, routes)
            updated = json.loads(explored.diagnostics_json)
            updated['state_bound_voi'] = diagnostics['state_bound_voi']
            return replace(explored, diagnostics_json=packed(updated))
        return replace(choice, diagnostics_json=packed(diagnostics))


__all__ = ['VERSION', 'StateBoundVoISelectionPolicy', 'observed_runtime_context']
