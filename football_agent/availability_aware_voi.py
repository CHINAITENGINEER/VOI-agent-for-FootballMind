'''One-step acquisition with point-in-time operational stress scenarios.

Availability and semantic informativeness are separate quantities. Observed
frequencies define stress scenarios, not statistical confidence limits.
Different dependency IDs do not establish statistical independence.
'''
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence
from types import MappingProxyType

from .decision_value import BeliefState, FiniteDecisionModel, VALUE_TOLERANCE
from .source_health_calibration import SourceHealthCalibration, VERSION as CALIBRATION_VERSION
from .types import utc

VERSION = 'availability-aware-voi-v1'
EVIDENCE_KINDS = {'SYNTHETIC_ASSUMPTIONS', 'ARCHIVED_FIXTURE', 'OPERATIONAL_OBSERVATIONS'}


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    return value.strip()


@dataclass(frozen=True)
class SourceBinding:
    source_id: str
    source_version: str
    transfer_assumption: str

    def __post_init__(self):
        for field in ('source_id', 'source_version', 'transfer_assumption'):
            object.__setattr__(self, field, _text(getattr(self, field), field))


@dataclass(frozen=True)
class CalibrationSnapshot:
    # Immutable serialization detaches upstream mutable count dictionaries.
    report_json: str
    available_at: str
    evidence_kind: str

    def __post_init__(self):
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError('unknown calibration evidence kind')
        report = SourceHealthCalibration(**json.loads(self.report_json))
        if report.schema_version != CALIBRATION_VERSION:
            raise ValueError('unsupported calibration schema')
        if not utc(report.train_end) < utc(report.eval_start) <= utc(report.eval_end):
            raise ValueError('invalid calibration split')
        available = utc(self.available_at)
        if available < utc(report.eval_end):
            raise ValueError('calibration availability precedes evaluation completion')
        for prefix in ('train', 'eval'):
            count = getattr(report, f'{prefix}_independent_dependency_count')
            rate = getattr(report, f'{prefix}_available_rate')
            expected = getattr(report, f'{prefix}_status_counts')['available'] / count if count else None
            if (rate is None) != (expected is None) or (rate is not None and abs(rate - expected) > 1e-10):
                raise ValueError('availability rate disagrees with dependency-group counts')
        object.__setattr__(self, 'available_at', available.isoformat())
        object.__setattr__(self, 'report_json', json.dumps(report.as_dict(), sort_keys=True, separators=(',', ':')))

    @classmethod
    def from_calibration(cls, report, *, available_at, evidence_kind):
        if not isinstance(report, SourceHealthCalibration):
            raise ValueError('expected SourceHealthCalibration')
        return cls(json.dumps(report.as_dict()), available_at, evidence_kind)

    @property
    def report(self):
        return json.loads(self.report_json)

    @property
    def digest(self):
        payload = json.dumps([self.report_json, self.available_at, self.evidence_kind], separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class AvailabilityAwareVoIPolicy:
    '''Receding one-step policy over an explicitly supplied semantic model.

    Under state-independent dropout: EVSI_with_return = q * EVSI_on_return.
    Request cost is incurred even on failures. Positive minimum scenario value
    permits acquisition. Unresolved positive upper value produces DEFER.
    '''

    def __init__(self, model: FiniteDecisionModel, bindings: Mapping[str, SourceBinding],
                 snapshots: Sequence[CalibrationSnapshot], *, min_dependency_groups=5,
                 max_age_seconds=86400, dropout_assumption: str):
        if not isinstance(model, FiniteDecisionModel):
            raise ValueError('expected FiniteDecisionModel')
        if not isinstance(bindings, Mapping) or set(bindings) != set(model.tools) or any(not isinstance(b, SourceBinding) for b in bindings.values()):
            raise ValueError('bindings must exactly cover tools')
        snapshots = tuple(snapshots)
        if any(not isinstance(s, CalibrationSnapshot) for s in snapshots):
            raise ValueError('expected CalibrationSnapshot values')
        if type(min_dependency_groups) is not int or min_dependency_groups < 1:
            raise ValueError('min_dependency_groups must be a positive integer')
        if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float)) or not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError('max_age_seconds must be finite and positive')
        if dropout_assumption != 'state_independent_return_availability':
            raise ValueError('explicit state-independent return availability assumption required')
        if 'DEFER' in model.tools:
            raise ValueError('DEFER is reserved for unresolved acquisition value')
        self.model, self.bindings, self.snapshots = model, MappingProxyType(dict(bindings)), snapshots
        self.minimum, self.max_age_seconds = min_dependency_groups, float(max_age_seconds)
        self.dropout_assumption = dropout_assumption

    def _support(self, binding, as_of):
        unknown = {'status': 'NO_VISIBLE_CALIBRATION', 'q_scenarios': [0.0, 1.0],
                   'range_kind': 'UNKNOWN_RETURN_AVAILABILITY',
                   'semantic_accuracy': 'UNMEASURED', 'statistical_coverage_claim': False}
        # Hide future snapshot identity and evaluation status from the actor.
        visible = [s for s in self.snapshots if utc(s.available_at) <= as_of
                   and (s.report['source_id'], s.report['source_version']) == (binding.source_id, binding.source_version)]
        if not visible:
            return unknown
        latest_time = max(utc(s.available_at) for s in visible)
        latest = [s for s in visible if utc(s.available_at) == latest_time]
        if len({s.digest for s in latest}) != 1:
            raise ValueError('conflicting calibration snapshots share one release time')
        snapshot, data = latest[0], latest[0].report
        metadata = {'snapshot_sha256': snapshot.digest, 'available_at': snapshot.available_at,
                    'evaluation_end': data['eval_end'], 'evidence_kind': snapshot.evidence_kind}
        reason = None
        if snapshot.evidence_kind == 'ARCHIVED_FIXTURE':
            reason = 'FIXTURE_IS_NOT_A_CALIBRATION_POPULATION'
        elif binding.source_version.upper() == 'UNVERSIONED':
            reason = 'UNKNOWN_SOURCE_VERSION'
        elif (as_of - utc(data['eval_end'])).total_seconds() > self.max_age_seconds:
            reason = 'EXPIRED_DIAGNOSTIC'
        elif min(data['train_independent_dependency_count'], data['eval_independent_dependency_count']) < self.minimum:
            reason = 'INSUFFICIENT_DEPENDENCY_GROUPS'
        if reason:
            return {**unknown, **metadata, 'status': reason}
        frequencies = [data['train_available_rate'], data['eval_available_rate']]
        return {**unknown, **metadata, 'status': 'OBSERVED_FREQUENCY_SCENARIOS',
                'q_scenarios': sorted(set(frequencies)),
                'range_kind': 'DECLARED_EMPIRICAL_STRESS_SCENARIOS',
                'past_drift_status': data['drift_status'],
                'dependency_groups_are_statistically_independent': 'NOT_ESTABLISHED',
                'transfer_assumption': binding.transfer_assumption}

    def evaluate(self, state: BeliefState, *, as_of, remaining_budget: float, horizon=1):
        if type(horizon) is not int or horizon != 1:
            raise ValueError('this adapter supports horizon=1; shared robust continuation is not implemented')
        cutoff = utc(as_of)
        reference = self.model.evaluate(state, remaining_budget=remaining_budget, horizon=1)
        rows = []
        for candidate in reference['candidates']:
            if not candidate['feasible']:
                rows.append({k: candidate[k] for k in ('action_id', 'feasible', 'reason')})
                continue
            action_id = candidate['action_id']
            support = self._support(self.bindings[action_id], cutoff)
            evsi, cost = candidate['immediate_evsi'], candidate['cost']
            values = [q * evsi - cost for q in support['q_scenarios']]
            rows.append({'action_id': action_id, 'feasible': True, 'cost': cost,
                         'conditional_evsi_on_return': evsi, 'support': support,
                         'scenario_net_values': values, 'lower_net_value': min(values),
                         'upper_net_value': max(values)})
        feasible = [row for row in rows if row['feasible']]
        eligible = [row for row in feasible if row['lower_net_value'] > VALUE_TOLERANCE]
        if eligible:
            best = max(eligible, key=lambda row: (row['lower_net_value'], row['action_id']))
            action, reason = best['action_id'], 'positive_value_in_all_declared_availability_scenarios'
        elif any(row['upper_net_value'] > VALUE_TOLERANCE for row in feasible):
            action, reason = 'DEFER', 'acquisition_may_help_but_positive_lower_value_is_unestablished'
        else:
            action, reason = 'STOP', 'no_positive_one_step_value_under_declared_model_or_no_feasible_action'
        return {'schema_version': VERSION, 'action_id': action, 'reason': reason,
                'as_of': cutoff.isoformat(), 'horizon': 1, 'remaining_budget': remaining_budget,
                'prediction': reference['prediction'], 'current_risk': reference['current_risk'],
                'cost_unit': self.model.cost_unit, 'candidates': rows,
                'observations': [list(pair) for pair in state.observations],
                'certificate': {'scope': 'declared_one_step_semantic_and_availability_scenarios',
                    'global_claim': False, 'empirical_policy_benefit': 'UNMEASURED',
                    'semantic_law_status': 'EXPLICIT_MODEL_ASSUMPTION',
                    'numeric_voi_empirically_calibrated': False,
                    'formula': 'q * conditional_EVSI - request_cost',
                    'dropout_assumption': self.dropout_assumption,
                    'statistical_coverage_claim': False,
                    'source_level_transfer_is_assumed': True,
                    'stop_is_global_optimality_certificate': False}}
