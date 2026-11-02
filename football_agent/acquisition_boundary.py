'''Algorithm-neutral selection boundary; no estimator or numeric VoI is supplied.

Candidate generation may use a model. Selection is a separate, local operation.
Persist the pre-acquisition input, candidates and choice to replay selection
without access to later source returns or match outcomes.
'''
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from .audit import stored_digest as digest, redact
from .query_value_contract import VALUE_UNMEASURED

VERSION = 'acquisition-boundary-v1'
BASELINE_VERSION = 'qualitative-proposed-choice-v1'
SHARED_FIELDS = ('task_id', 'assessment', 'previous_return', 'analysis_plan', 'match_identity')
OPTIONAL_SHARED_FIELDS = ('return_assessment', 'source_claims')


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


@dataclass(frozen=True)
class SelectionContext:
    task_id: str
    snapshot_digest: str
    snapshot_json: str

    @classmethod
    def from_public(cls, public):
        if packed(public)!=packed(redact(public)):
            raise ValueError('SELECTION_CONTEXT_NOT_ARCHIVE_STABLE')
        return cls(public['task_id'], digest(public), packed(public))

    @property
    def state(self):
        # A fresh copy prevents a selector from mutating the archived input.
        return json.loads(self.snapshot_json)


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    response_json: str

    @property
    def response(self):
        return json.loads(self.response_json)

    def record(self):
        response = self.response
        record = {'candidate_id': self.candidate_id, 'response': response,
                'value_status': VALUE_UNMEASURED,
                'resource_cost': {
                    'http_request_lower_bound': int(response['action'] in {'search_web', 'read_url'}),
                    'http_request_upper_bound': None,
                    'latency_seconds_estimate': None, 'money_estimate': None,
                    'basis': 'tool contract only; redirects and failures consume actual budget',
                    'decision_loss_equivalent': None}}
        if response['action'] == 'query_acquired':
            record['resource_cost'].update(http_request_upper_bound=0,local_read_attempts=1,
                basis='local archive processing; no new HTTP; elapsed time measured, decision cost uncalibrated')
        return record


@dataclass(frozen=True)
class CandidateBatch:
    context: SelectionContext
    candidates: tuple[CandidateAction, ...]
    recommended_id: str | None
    origin: str

    @classmethod
    def build(cls, public, responses, *, recommended_index=None, origin):
        if not isinstance(responses, list) or not 1 <= len(responses) <= 32:
            raise ValueError('CANDIDATE_BATCH_SIZE')
        context = SelectionContext.from_public(public)
        shared, actions, candidates = None, set(), []
        for response in responses:
            if not isinstance(response, dict) or response.get('task_id') != context.task_id:
                raise ValueError('CANDIDATE_TASK_MISMATCH')
            if packed(response)!=packed(redact(response)):
                raise ValueError('CANDIDATE_NOT_ARCHIVE_STABLE')
            current = {k: response.get(k) for k in SHARED_FIELDS}
            current.update({k: response[k] for k in OPTIONAL_SHARED_FIELDS if k in response})
            if shared is not None and current != shared:
                raise ValueError('CANDIDATES_REQUIRE_SHARED_ANALYSIS')
            shared = current
            action_fields = {k: response.get(k) for k in ('action', 'url', 'query', 'engine', 'question_to_user')}
            if response.get('action') == 'query_acquired':
                action_fields['local_query'] = response.get('local_query')
            action_key = digest(action_fields)
            if action_key in actions:
                raise ValueError('DUPLICATE_CANDIDATE_ACTION')
            actions.add(action_key)
            candidate_id = 'candidate-' + digest({'snapshot': context.snapshot_digest, 'response': response})
            candidates.append(CandidateAction(candidate_id, packed(response)))
        if recommended_index is not None and (type(recommended_index) is not int
                                               or not 0 <= recommended_index < len(candidates)):
            raise ValueError('RECOMMENDED_CANDIDATE_INDEX')
        return cls(context, tuple(candidates),
                   candidates[recommended_index].candidate_id if recommended_index is not None else None, origin)

    def record(self):
        return {'version': VERSION, 'task_id': self.context.task_id,
                'snapshot_digest': self.context.snapshot_digest, 'origin': self.origin,
                'recommended_id': self.recommended_id,
                'candidates': [candidate.record() for candidate in self.candidates]}


@dataclass(frozen=True)
class PolicyChoice:
    task_id: str
    snapshot_digest: str
    candidate_id: str | None = None
    terminal_action: str | None = None
    reason: str = ''
    diagnostics_json: str = '{}'

    def record(self):
        diagnostics = json.loads(self.diagnostics_json)
        if not isinstance(diagnostics, dict):
            raise ValueError('POLICY_DIAGNOSTICS_OBJECT_REQUIRED')
        packed(diagnostics)
        if packed(diagnostics)!=packed(redact(diagnostics)):
            raise ValueError('POLICY_DIAGNOSTICS_NOT_ARCHIVE_STABLE')
        return {'task_id': self.task_id, 'snapshot_digest': self.snapshot_digest,
                'candidate_id': self.candidate_id, 'terminal_action': self.terminal_action,
                'reason': self.reason, 'diagnostics': diagnostics}


class SelectionPolicy(Protocol):
    version: str
    configuration: dict

    def select(self, context: SelectionContext, batch: CandidateBatch) -> PolicyChoice: ...


class ProposedChoiceBaseline:
    '''Preserve the existing LLM choice; never advertise this as numeric VoI.'''
    version = BASELINE_VERSION
    configuration = {'method': 'LLM_QUALITATIVE_BASELINE', 'numeric_voi_calibrated': False}

    def select(self, context, batch):
        if batch.recommended_id is None:
            raise ValueError('BASELINE_REQUIRES_EXPLICIT_RECOMMENDATION')
        selected = next(c for c in batch.candidates if c.candidate_id == batch.recommended_id)
        return PolicyChoice(context.task_id, context.snapshot_digest, candidate_id=selected.candidate_id,
                            reason=selected.response['reason'])


def policy_identity(policy):
    version, configuration = getattr(policy, 'version', None), getattr(policy, 'configuration', None)
    if not isinstance(version, str) or not version.strip() or not isinstance(configuration, dict):
        raise ValueError('POLICY_IDENTITY_REQUIRED')
    # JSON roundtrip makes the persisted descriptor independent of mutable policy objects.
    descriptor = {'version': version, 'configuration': json.loads(packed(configuration))}
    if packed(descriptor)!=packed(redact(descriptor)):
        raise ValueError('POLICY_CONFIGURATION_NOT_ARCHIVE_STABLE')
    return {**descriptor, 'fingerprint': digest(descriptor)}


def materialize_choice(context, batch, choice):
    if not isinstance(choice, PolicyChoice):
        raise ValueError('POLICY_CHOICE_REQUIRED')
    if (choice.task_id != context.task_id or choice.snapshot_digest != context.snapshot_digest
            or batch.context != context):
        raise ValueError('STALE_OR_CROSS_TASK_POLICY_CHOICE')
    if not isinstance(choice.reason, str) or not choice.reason.strip():
        raise ValueError('POLICY_REASON_REQUIRED')
    choice.record()
    if choice.terminal_action is None:
        candidate = next((c for c in batch.candidates if c.candidate_id == choice.candidate_id), None)
        if candidate is None:
            raise ValueError('POLICY_SELECTED_UNPROPOSED_ACTION')
        response = candidate.response
        response['reason'] = choice.reason
        return response
    if choice.candidate_id is not None or choice.terminal_action not in {'stop', 'defer'}:
        raise ValueError('POLICY_TERMINAL_ACTION_INVALID')
    # Stop/defer is always available to the selector without another model call.
    # Analysis is shared across candidates; stopping never upgrades its support.
    response = batch.candidates[0].response
    response.update(action=choice.terminal_action, url=None, query=None, engine=None,
                    question_to_user=None, reason=choice.reason, why_worth_cost=choice.reason, alternatives=[])
    if 'local_query' in response:
        response['local_query'] = None
    if 'active_question_id' in response:
        response.update(active_question_id=None, clarification_field='none')
    return response


def replay_boundary(public, proposal_record, selection_record):
    '''Verify a saved selection only; does not rerun the policy or any tools.'''
    responses = [item['response'] for item in proposal_record['candidates']]
    ids = [item['candidate_id'] for item in proposal_record['candidates']]
    recommended = proposal_record['recommended_id']
    if recommended is not None and recommended not in ids:
        raise ValueError('REPLAY_RECOMMENDATION_MISMATCH')
    batch = CandidateBatch.build(public, responses,
        recommended_index=ids.index(recommended) if recommended is not None else None,
        origin=proposal_record['origin'])
    if batch.record() != proposal_record or digest(proposal_record) != selection_record['proposal_digest']:
        raise ValueError('REPLAY_PROPOSAL_MISMATCH')
    policy = selection_record['policy']
    if digest({k: policy[k] for k in ('version', 'configuration')}) != policy['fingerprint']:
        raise ValueError('REPLAY_POLICY_IDENTITY_MISMATCH')
    c = selection_record['choice']
    choice = PolicyChoice(c['task_id'], c['snapshot_digest'], c['candidate_id'],
                          c['terminal_action'], c['reason'], packed(c['diagnostics']))
    response = materialize_choice(batch.context, batch, choice)
    if digest(response) != selection_record['decision_digest']:
        raise ValueError('REPLAY_DECISION_MISMATCH')
    return response
