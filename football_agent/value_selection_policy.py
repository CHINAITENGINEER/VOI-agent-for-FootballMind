"""Bounded local value-aware selection over Agent-proposed actions."""
from __future__ import annotations

from collections.abc import Mapping
import json
import math

from .acquisition_boundary import CandidateBatch, PolicyChoice, SelectionContext
from .audit import digest, stored_digest
from .query_value_contract import (
    BUDGET_EXHAUSTED,
    DEFER,
    MODELED_DECLARED_ASSUMPTIONS,
    MODEL_SCOPED_STOP,
    NO_FEASIBLE_REGISTERED_ACTION,
    QUERY,
    STOP,
    VERSION as QUERY_VALUE_VERSION,
    WITHHELD_OPEN_WORLD_VALUE_UNKNOWN,
    QueryValueAssessment,
    QueryValueDecision,
    StopCertificate,
)

VERSION = "value-aware-selection-v2"
LEGACY_VERSION = "value-aware-selection-v1"
ACQUISITION_ACTIONS = frozenset({'search_web', 'read_url', 'query_acquired'})
TERMINAL_ACTIONS = frozenset({'stop', 'defer'})


def migrate_saved_identity(saved, current):
    """Return *current* only for the one explicitly supported v1 migration.

    v1 did not persist a cost cap, so migration is deliberately limited to an
    unbounded v2 policy.  The saved descriptor is checked before its fingerprint
    is trusted; unknown fields or a changed model descriptor leave the task
    paused instead of silently changing the selection semantics.
    """
    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        return None
    if saved.get('version') != LEGACY_VERSION or current.get('version') != VERSION:
        return None
    if set(saved) != {'version', 'configuration', 'fingerprint'}:
        return None
    legacy = saved.get('configuration')
    target = current.get('configuration')
    if not isinstance(legacy, Mapping) or not isinstance(target, Mapping):
        return None
    if set(legacy) - {'method', 'numeric_voi_calibrated', 'unmeasured_policy', 'model_descriptor'}:
        return None
    if set(target) - {'method', 'numeric_voi_calibrated', 'unmeasured_policy',
                      'terminal_proposals', 'value_budget', 'value_budget_scope',
                      'model_descriptor'}:
        return None
    if (legacy.get('method') != 'LOCAL_VALUE_AWARE_SELECTION'
            or legacy.get('numeric_voi_calibrated') is not False
            or legacy.get('unmeasured_policy') != 'DEFER_OPEN_WORLD_VALUE_UNKNOWN'):
        return None
    if (target.get('method') != 'LOCAL_VALUE_AWARE_SELECTION'
            or target.get('numeric_voi_calibrated') is not False
            or target.get('unmeasured_policy') != 'DEFER_OPEN_WORLD_VALUE_UNKNOWN'
            or target.get('terminal_proposals') != 'EXCLUDED_FROM_INFORMATION_VALUE'
            or target.get('value_budget') is not None
            or target.get('value_budget_scope') != 'PER_SELECTION_COST_CAP'):
        return None
    if ('model_descriptor' in legacy) != ('model_descriptor' in target):
        return None
    if legacy.get('model_descriptor') != target.get('model_descriptor'):
        return None
    base = {'version': saved['version'], 'configuration': json.loads(json.dumps(legacy, ensure_ascii=False, sort_keys=True))}
    fingerprint = saved.get('fingerprint')
    if not isinstance(fingerprint, str) or fingerprint != stored_digest(base) or fingerprint != digest(base):
        return None
    current_base = {'version': current['version'],
                    'configuration': json.loads(json.dumps(target, ensure_ascii=False, sort_keys=True))}
    if current.get('fingerprint') != digest(current_base):
        return None
    return json.loads(json.dumps(current, ensure_ascii=False, sort_keys=True))


class ValueAwareSelectionPolicy:
    """Choose locally from one Agent-generated candidate batch."""

    version = VERSION

    def __init__(self, assessor, *, value_budget=None, model_descriptor=None):
        if not callable(assessor):
            raise ValueError("assessor callable required")
        if value_budget is not None and (isinstance(value_budget, bool) or not isinstance(value_budget, (int, float))
                                         or not math.isfinite(value_budget) or value_budget < 0):
            raise ValueError("value_budget must be finite and nonnegative")
        self.assessor = assessor
        if model_descriptor is not None and not isinstance(model_descriptor, Mapping):
            raise ValueError('model_descriptor must be a mapping')
        self.model_descriptor = None if model_descriptor is None else json.loads(json.dumps(model_descriptor, ensure_ascii=False, sort_keys=True))
        self._value_budget = None if value_budget is None else float(value_budget)

    @property
    def value_budget(self):
        return self._value_budget

    @property
    def configuration(self):
        descriptor = {"method": "LOCAL_VALUE_AWARE_SELECTION", "numeric_voi_calibrated": False,
            "unmeasured_policy": "DEFER_OPEN_WORLD_VALUE_UNKNOWN",
            "terminal_proposals": "EXCLUDED_FROM_INFORMATION_VALUE",
            "value_budget": self.value_budget, "value_budget_scope": "PER_SELECTION_COST_CAP"}
        if self.model_descriptor is not None:
            descriptor['model_descriptor'] = json.loads(json.dumps(self.model_descriptor, ensure_ascii=False, sort_keys=True))
        return descriptor

    def _assessment(self, state, candidate):
        assessment = self.assessor(state, candidate)
        if not isinstance(assessment, QueryValueAssessment):
            raise ValueError("assessor must return QueryValueAssessment")
        if assessment.candidate_id != candidate.candidate_id:
            raise ValueError("VALUE_ASSESSMENT_CANDIDATE_MISMATCH")
        return assessment

    @staticmethod
    def _slot(response):
        if response.get("action") in {"search_web", "read_url"}:
            return "http_calls"
        if response.get("action") == "query_acquired":
            return "local_reads"
        return None

    @staticmethod
    def _formal_decision(action, basis, assessments, *, candidate_id=None,
                         kind=None, scope=None, reason=None,
                         actionable_unknown_count=0, evaluated_candidate_ids=(),
                         remaining_budget=None, maximum_modeled_net_value=None):
        """Build the replayable query-value view without changing selection.

        The headless match runtime has named operational resources while the
        older ``QueryValueGate`` accepts one scalar budget.  ``StopCertificate``
        now preserves either representation, so this bridge can expose the
        same evidence without collapsing resource units.
        """
        certificate = None
        if action != QUERY:
            certificate = StopCertificate(
                kind, scope, False, reason, actionable_unknown_count,
                tuple(evaluated_candidate_ids), remaining_budget,
                maximum_modeled_net_value,
            )
        decision = QueryValueDecision(
            action, candidate_id, basis, tuple(assessments), certificate)
        return decision.as_dict()

    def select(self, context: SelectionContext, batch: CandidateBatch) -> PolicyChoice:
        if batch.context != context:
            raise ValueError("candidate batch belongs to another context")
        state = context.state
        remaining = state.get("remaining", {})
        if not isinstance(remaining, Mapping):
            raise ValueError("remaining budget must be a mapping")
        positive, unknown, rows, ignored, assessed = [], [], [], [], []
        evaluated_ids = []
        feasible_count = 0
        for candidate in batch.candidates:
            action = candidate.response.get('action')
            if action in TERMINAL_ACTIONS:
                # A suggested control action is not evidence that acquisition
                # has zero value. The selector derives its own stop/defer.
                ignored.append(candidate.candidate_id)
                rows.append({'candidate_id': candidate.candidate_id, 'action': action,
                    'candidate_role': 'terminal_control', 'evaluation_status': 'NOT_APPLICABLE',
                    'value': None, 'slot': None, 'feasible': False,
                    'exclusion_reasons': ['terminal_proposal_not_information']})
                continue
            if action not in ACQUISITION_ACTIONS and action != 'ask_user':
                raise ValueError('VALUE_POLICY_UNSUPPORTED_ACTION')
            assessment = self._assessment(state, candidate)
            evaluated_ids.append(candidate.candidate_id)
            assessed.append(assessment)
            slot = self._slot(candidate.response)
            exclusions = []
            if slot is not None and remaining.get(slot, 0) <= 0:
                exclusions.append(slot + '_exhausted')
            if action in ACQUISITION_ACTIONS and state.get('final_decision_only', False):
                exclusions.append('final_decision_only')
            value_ok = self.value_budget is None or assessment.cost is None or assessment.cost <= self.value_budget
            if not value_ok:
                exclusions.append('declared_cost_exceeds_selection_cap')
            feasible = not exclusions
            if feasible:
                feasible_count += 1
            rows.append({"candidate_id": candidate.candidate_id, 'action': action,
                'candidate_role': 'identity_clarification' if action == 'ask_user' else 'acquisition',
                'evaluation_status': 'ASSESSED', "value": assessment.as_dict(), "slot": slot,
                "feasible": feasible, 'exclusion_reasons': exclusions})
            if not feasible:
                continue
            if assessment.value_status == MODELED_DECLARED_ASSUMPTIONS:
                if assessment.net_value is not None and assessment.net_value > 1e-12:
                    positive.append((assessment.net_value, candidate, assessment))
            else:
                unknown.append(candidate.candidate_id)
        diagnostics = {"schema_version": VERSION, "candidates": rows, "remaining": dict(remaining),
            "value_budget": self.value_budget, 'value_budget_scope': 'PER_SELECTION_COST_CAP',
            'ignored_terminal_candidate_ids': ignored, 'evaluated_candidate_ids': evaluated_ids,
            'unmeasured_candidate_ids': unknown, 'global_optimum_claim': False,
            'candidate_generation_complete': False,
            'query_value_contract_version': QUERY_VALUE_VERSION,
            'query_value_contract_scope': 'acquisition_actions_only'}
        if positive:
            _, selected, assessment = max(positive, key=lambda item: (item[0], item[1].candidate_id))
            diagnostics.update(selection_basis="modeled_positive_net_value", selected_value=assessment.as_dict())
            diagnostics['selection_scope'] = 'modeled_feasible_candidates'
            if selected.response.get('action') in ACQUISITION_ACTIONS:
                diagnostics['query_value_decision'] = self._formal_decision(
                    QUERY, 'declared_finite_model_positive_net_value', assessed,
                    candidate_id=selected.candidate_id)
            else:
                diagnostics['query_value_decision'] = None
                diagnostics['query_value_decision_omission_reason'] = (
                    'identity_clarification_outside_acquisition_contract')
            return PolicyChoice(context.task_id, context.snapshot_digest, candidate_id=selected.candidate_id, reason="选择声明模型已覆盖且可执行的候选中净价值最高的下一动作。", diagnostics_json=json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))
        if unknown:
            diagnostics.update(selection_basis="open_world_value_withheld", stop_scope="open_world", stop_kind=WITHHELD_OPEN_WORLD_VALUE_UNKNOWN, unmeasured_candidate_ids=unknown)
            diagnostics['query_value_decision'] = self._formal_decision(
                DEFER, 'withhold_open_world_value_unknown', assessed,
                kind=WITHHELD_OPEN_WORLD_VALUE_UNKNOWN, scope='open_world',
                reason='No supported positive finite-model action is available; remaining routes have unmeasured or non-deployable value.',
                actionable_unknown_count=len(unknown), evaluated_candidate_ids=evaluated_ids,
                remaining_budget=remaining)
            return PolicyChoice(context.task_id, context.snapshot_digest, terminal_action="defer", reason="仍有候选信息价值未测量，暂缓而不把未知价值当作零。", diagnostics_json=json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))
        if not evaluated_ids:
            diagnostics.update(selection_basis='no_information_candidates', stop_scope='proposed_candidate_set',
                               stop_kind=NO_FEASIBLE_REGISTERED_ACTION)
            diagnostics['query_value_decision'] = self._formal_decision(
                DEFER, 'semantic_candidate_scope', assessed,
                kind=NO_FEASIBLE_REGISTERED_ACTION, scope='proposed_candidate_set',
                reason='No registered candidate addresses a currently visible unknown state.',
                evaluated_candidate_ids=(), remaining_budget=remaining,
                maximum_modeled_net_value=0.0)
            return PolicyChoice(context.task_id, context.snapshot_digest, terminal_action='defer',
                reason='本次只提出停止或暂缓，缺少可评估的信息动作，保留未知并暂缓。',
                diagnostics_json=json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))
        if feasible_count == 0:
            diagnostics.update(selection_basis="resource_budget_gate", stop_scope="operational_budget", stop_kind=BUDGET_EXHAUSTED)
            diagnostics['query_value_decision'] = self._formal_decision(
                DEFER, 'budget_gate', assessed,
                kind=BUDGET_EXHAUSTED, scope='operational_budget',
                reason='Every assessed candidate is excluded by the remaining operational resources or selection cap.',
                actionable_unknown_count=len(assessed), evaluated_candidate_ids=evaluated_ids,
                remaining_budget=remaining)
            return PolicyChoice(context.task_id, context.snapshot_digest, terminal_action="defer", reason="没有候选满足当前资源预算，保留未知并暂缓。", diagnostics_json=json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))
        diagnostics.update(selection_basis="modeled_nonpositive_or_unavailable", stop_scope="declared_finite_model", stop_kind=MODEL_SCOPED_STOP)
        maximum = max(row['value']['net_value'] for row in rows if row['feasible'])
        diagnostics['maximum_modeled_net_value'] = maximum
        diagnostics['query_value_decision'] = self._formal_decision(
            STOP, 'declared_finite_model_no_positive_net_value', assessed,
            kind=MODEL_SCOPED_STOP, scope='declared_finite_model',
            reason='All feasible candidates are covered by the declared finite model and have nonpositive net value.',
            actionable_unknown_count=0, evaluated_candidate_ids=evaluated_ids,
            remaining_budget=remaining, maximum_modeled_net_value=maximum)
        return PolicyChoice(context.task_id, context.snapshot_digest, terminal_action="stop", reason="声明模型内没有正净价值的可执行查询，停止仅在该模型范围内成立。", diagnostics_json=json.dumps(diagnostics, ensure_ascii=False, sort_keys=True))


__all__ = ["VERSION", "LEGACY_VERSION", "ValueAwareSelectionPolicy", "migrate_saved_identity"]
