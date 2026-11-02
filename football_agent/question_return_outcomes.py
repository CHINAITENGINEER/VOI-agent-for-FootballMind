'''Question-scoped observations over existing execution-bound navigation.

Actor judgments remain actor judgments. Exact quote checks establish location,
not entailment, correctness, risk reduction or source-wide reliability.
'''
from collections import Counter
from copy import deepcopy

from jsonschema import Draft202012Validator

from .audit import digest
from .evidence_windows import EvidenceWindows
from .execution_bound_navigation import SCHEMA
from .target_navigation import validate_contract
from .types import utc

VERSION = 'question-return-outcome-v1'
BODY_OUTCOMES = frozenset({'partial', 'resolved', 'irrelevant', 'conflicting'})


def project_return(*, run_id, acquisition_event_hash, acquisition_choice,
                   acquisition_at, assessment_event_hash, response, public,
                   assessed_at, document, opened_before, window_chars=1200,
                   preview_chars=80):
    '''Project one already bound response, without interpreting source prose.

The archive adapter verifies execution binding and chronological event pairing.
This function validates the prospective test, task, exposure and literal support.
'''
    if any(not isinstance(value, str) or not value.strip() for value in (
            run_id, acquisition_event_hash, assessment_event_hash)):
        raise ValueError('explicit run and event identity required')
    validate_contract(public['task_contract'], public['task'])
    if public['execution_context']['run_id'] != run_id:
        raise ValueError('assessment belongs to another run')
    Draft202012Validator(SCHEMA).validate(acquisition_choice)
    Draft202012Validator(SCHEMA).validate(response)
    prior = public['last_action_result']
    assessment = response['return_assessment']
    action = acquisition_choice['action']
    if action not in {'query_source', 'read_window'}:
        raise ValueError('return must follow an acquisition action')
    test = acquisition_choice['information_test']
    if not prior or not assessment or not test or prior['information_test'] != test:
        raise ValueError('assessment must retain the exact prospective information test')
    if test['requirement'] not in acquisition_choice['target_state']['open_requirements']:
        raise ValueError('prospective test did not address an open requirement')
    if document['document_id'] != prior['document_id'] or assessment['document_id'] != prior['document_id']:
        raise ValueError('assessment document differs from actual return')
    if assessment['requirement'] != test['requirement'] or prior['status'] != document['status']:
        raise ValueError('return status or requirement changed')
    exposure = 'body' if action == 'read_window' else 'metadata'
    if prior['exposure'] != exposure:
        raise ValueError('source metadata is not a body read')
    if not utc(acquisition_at) <= utc(assessed_at) or utc(document['available_at']) > utc(assessed_at):
        raise ValueError('assessment predates its action or returned document')
    # Document availability may predate this request (cached or previously
    # published evidence). The archive adapter separately checks the actual
    # request/return event ordering; an old document is not a future leak.
    if action == 'read_window' and (acquisition_choice['document_id'] != prior['document_id']
                                   or acquisition_choice['window_id'] != prior['window_id']):
        raise ValueError('assessed body differs from selected window')
    if exposure == 'metadata' and prior['window_id'] is not None:
        raise ValueError('metadata exposure has no body window')
    outcome, indices = assessment['outcome'], assessment['evidence_claim_indices']
    anchors, window = [], None
    if document['status'] != 'OK':
        if outcome != 'failed' or indices:
            raise ValueError('failed request remains missing information')
    elif exposure == 'metadata':
        if outcome != 'unread' or indices:
            raise ValueError('unread metadata has no semantic outcome')
    else:
        if outcome not in BODY_OUTCOMES or not indices:
            raise ValueError('body judgment needs cited opened evidence')
        windows = EvidenceWindows(document, window_chars=window_chars, preview_chars=preview_chars)
        opened = opened_before.get(document['document_id'], [])
        if prior['window_id'] not in opened:
            raise ValueError('assessment window was not exposed to actor')
        window = windows.read(prior['window_id'])
        views = [row for row in public['documents'] if row['document_id'] == document['document_id']]
        if len(views) != 1 or window not in views[0].get('visible_windows', []):
            raise ValueError('cited body is absent from captured actor input')
        if views[0].get('raw_sha256') != document['raw_sha256']:
            raise ValueError('captured source digest differs from assessed document')
        for index in indices:
            if index >= len(response['claims']):
                raise ValueError('assessment cites a missing claim')
            claim = response['claims'][index]
            if claim['document_id'] != document['document_id'] or claim['window_id'] != prior['window_id']:
                raise ValueError('claim belongs to another returned window')
            anchors.append(windows.anchor(claim['window_id'], claim['quote'], opened))
        if outcome == 'resolved' and test['requirement'] in response['target_state']['open_requirements']:
            raise ValueError('actor resolved flag conflicts with its retained requirement')
    record = {
        'schema_version': VERSION,
        'record_id': 'return-' + digest({'run_id': run_id, 'acquisition_event_hash': acquisition_event_hash}),
        'run_id': run_id, 'task_contract': deepcopy(public['task_contract']),
        'task_contract_digest': digest(public['task_contract']),
        'acquisition_event_hash': acquisition_event_hash, 'assessment_event_hash': assessment_event_hash,
        'selected_at': utc(acquisition_at).isoformat(), 'assessed_at': utc(assessed_at).isoformat(),
        'action': action, 'prospective_question': acquisition_choice['question'],
        'information_test': deepcopy(test), 'information_test_digest': digest(test),
        'request': deepcopy(acquisition_choice['query']),
        'document': {key: deepcopy(document.get(key)) for key in (
            'document_id', 'url', 'kind', 'status', 'raw_sha256', 'available_at', 'observed_at')},
        'exposure': exposure, 'actor_outcome': outcome, 'actor_reason': assessment['reason'],
        'evidence_anchors': anchors, 'visible_body_window': window,
        'source_request_count': int(action == 'query_source'),
        'local_body_read_count': int(action == 'read_window'),
        'scope': 'this_action_return_for_this_prospective_question_only',
        'independent_semantic_label': None, 'semantic_entailment_verified': False,
        'world_truth_verified': False, 'realized_decision_gain': None,
        'numeric_voi_training_eligible': False,
    }
    return record


def visible_outcomes(records, *, as_of):
    '''Return only observations recorded by the actor no later than as_of.

Later analytic projection does not change the original actor observation clock.
New independent reviewer labels would need a separate release clock and path.
'''
    cutoff = utc(as_of)
    visible = {}
    for record in records:
        if utc(record['assessed_at']) > cutoff:
            continue
        if record['schema_version'] != VERSION:
            raise ValueError('unsupported question outcome schema')
        identity = record['record_id']
        if identity in visible and visible[identity] != record:
            raise ValueError('conflicting records for one acquisition action')
        visible[identity] = deepcopy(record)
    return sorted(visible.values(), key=lambda row: (utc(row['assessed_at']), row['record_id']))


def summarize_outcomes(records, *, as_of):
    rows = visible_outcomes(records, as_of=as_of)
    counts = Counter(row['actor_outcome'] for row in rows)
    return {'schema_version': 'question-return-summary-v1', 'as_of': utc(as_of).isoformat(),
        'assessed_acquisition_actions': len(rows),
        'assessed_source_requests': sum(row['source_request_count'] for row in rows),
        'source_requests_with_ok_return': sum(row['action'] == 'query_source' and row['document']['status'] == 'OK' for row in rows),
        'source_requests_with_failed_return': sum(row['action'] == 'query_source' and row['document']['status'] != 'OK' for row in rows),
        'assessed_body_reads': sum(row['local_body_read_count'] for row in rows),
        'actor_outcome_counts': {key: counts[key] for key in ('unread', 'partial', 'resolved', 'irrelevant', 'conflicting', 'failed')},
        'independent_semantic_label_count': 0,
        'conditional_semantic_accuracy': None, 'numeric_voi': None,
        'scope': 'descriptive_counts_of_observed_actions_not_causal_policy_value_or_independent_samples'}


def blinded_review_packet(record):
    '''Prepare independent question-level review without the actor verdict.

Expose exactly the selected body window, not future/unopened page content.
Transport failures and unread metadata do not enter semantic-label denominators.
'''
    if record['exposure'] != 'body' or record['document']['status'] != 'OK':
        return None
    if record['visible_body_window'] is None:
        raise ValueError('body review needs an exposed source window')
    payload = {key: deepcopy(record[key]) for key in (
        'record_id', 'task_contract', 'task_contract_digest', 'selected_at',
        'prospective_question', 'information_test', 'information_test_digest',
        'document', 'visible_body_window')}
    return {'schema_version': 'blinded-question-review-v1', 'review_status': 'PENDING_INDEPENDENT_REVIEW',
        'payload': payload, 'payload_digest': digest(payload),
        'review_prompt': 'Assess whether this visible return answers the declared information test. Cite an exact visible span; retain insufficient_context when the window is incomplete. Do not infer world truth or source-wide reliability.',
        'allowed_labels': ['resolved', 'partial', 'irrelevant', 'conflicting', 'insufficient_context'],
        'verdict': None, 'gain_label': None, 'actor_verdict_exposed': False,
        'future_evidence_exposed': False, 'full_source_coverage_established': False}
