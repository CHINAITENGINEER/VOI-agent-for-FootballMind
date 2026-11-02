'''Read-only projection of question-level judgments from bound run archives.'''
from pathlib import Path

from .audit import RunStore, digest
from .evidence_windows import EvidenceWindows
from .execution_bound_navigation import request_record, verify_response_binding, ExecutionBoundProtocol
from .question_return_outcomes import project_return
from .target_navigation import validate_contract
from .types import utc


def project_bound_run(root, run_id, *, expected_head):
    store = RunStore(Path(root) / 'runs')
    run = store.load(run_id)
    audit = store.verify(run_id)
    if audit['head_hash'] != expected_head:
        raise ValueError('archive head differs from pinned episode')
    if run['configuration']['protocol'] != ExecutionBoundProtocol.version:
        raise ValueError('execution-bound navigation archive required')
    validate_contract(run['target_contract'], run['task'])
    if digest(run['target_contract']) != run['target_contract_digest']:
        raise ValueError('archived task contract changed')
    decision_events = [event for event in run['events'] if event['type'] == 'navigation_decision']
    if [event['data']['decision'] for event in decision_events] != run['decisions']:
        raise ValueError('archived decision summary differs from events')
    documents = {key: store.get_blob(run_id, value) for key, value in run['documents'].items()}
    rows, request_ids = [], set()
    previous = returned = reservation = reasoning = reasoning_reservation = None
    prior_at = None
    for event in run['events']:
        at = utc(event['at'])
        if prior_at is not None and at < prior_at:
            raise ValueError('archive event times run backwards')
        prior_at = at
        kind, data = event['type'], event['data']
        if kind == 'navigation_reasoning_reserved':
            reasoning_reservation = event
        elif kind == 'navigation_reasoning_returned':
            public = store.get_blob(run_id, data['input_ref'])
            reply = store.get_blob(run_id, data['output_ref'])
            identity = reply['execution_binding']['request_id']
            if identity in request_ids:
                raise ValueError('reused model request identity')
            request_ids.add(identity)
            if not reasoning_reservation or data['input_ref'] != reasoning_reservation['data']['input_ref']:
                raise ValueError('model return lacks its captured reservation')
            if public['execution_context'] != {'run_id': run_id, 'attempt': event['snapshot']['budget']['llm_calls']}:
                raise ValueError('model reply belongs to another run or attempt')
            if public['task'] != run['task'] or public['task_contract'] != run['target_contract']:
                raise ValueError('model input task changed')
            verify_response_binding(reply, public, request_record(public, identity))
            reasoning = (event, public, reply)
        elif kind == 'navigation_acquisition_reserved':
            reservation = event
        elif kind in {'navigation_acquisition_returned', 'navigation_window_opened'}:
            if previous is None or returned is not None:
                raise ValueError('unpaired or duplicate acquisition return')
            expected = ('navigation_acquisition_returned' if previous['data']['decision']['choice']['action'] == 'query_source'
                        else 'navigation_window_opened')
            if kind != expected:
                raise ValueError('returned operation differs from selected action')
            returned = event
        elif kind == 'navigation_decision':
            if reasoning is None:
                raise ValueError('decision has no bound model return')
            reasoning_event, public, reply = reasoning
            choice = data['decision']['choice']
            if any(data[key] != reasoning_event['data'][key] for key in ('input_ref', 'output_ref')) or choice != reply['response']:
                raise ValueError('decision differs from bound raw response')
            if previous is not None:
                selected = previous['data']['decision']['choice']
                if selected['action'] not in {'query_source', 'read_window'}:
                    raise ValueError('a terminal decision has a subsequent action')
                if returned is None or public['last_choice'] != selected:
                    raise ValueError('assessment lacks its actual preceding return')
                if not utc(previous['at']) <= utc(returned['at']) <= utc(public['now']) <= utc(reasoning_event['at']) <= at:
                    raise ValueError('acquisition is not visible before assessment')
                document_id = returned['data']['document_id']
                document = documents[document_id]
                result = store.get_blob(run_id, returned['data']['output_ref'])
                if selected['action'] == 'query_source':
                    if not reservation or not utc(previous['at']) <= utc(reservation['at']) <= utc(returned['at']):
                        raise ValueError('source return lacks its request reservation')
                    if result['request_id'] != reservation['data']['request_id'] or result['http_attempts'] != 1:
                        raise ValueError('source return has the wrong request or attempt count')
                    receipts = [row for row in run['receipts'] if row['result_ref'] == returned['data']['output_ref']]
                    if len(receipts) != 1 or receipts[0]['action'] != selected['query'] or receipts[0]['document_id'] != document_id:
                        raise ValueError('source receipt differs from selected request')
                    if result['document'] != document:
                        raise ValueError('source document differs from its return receipt')
                else:
                    expected_window = EvidenceWindows(document,
                        window_chars=run['configuration']['window_chars'],
                        preview_chars=run['configuration']['preview_chars']).read(selected['window_id'])
                    if result != expected_window or returned['data']['new_http_attempts'] != 0:
                        raise ValueError('body exposure differs from archived window or was counted as a fetch')
                row = project_return(run_id=run_id, acquisition_event_hash=previous['event_hash'],
                    acquisition_choice=selected, acquisition_at=previous['at'],
                    assessment_event_hash=event['event_hash'], response=choice, public=public,
                    assessed_at=event['at'], document=document, opened_before=data['decision']['opened_before'],
                    window_chars=run['configuration']['window_chars'], preview_chars=run['configuration']['preview_chars'])
                row.update(return_event_hash=returned['event_hash'], return_recorded_at=utc(returned['at']).isoformat(),
                           execution_binding_verified=True, assessment_request_id=reply['execution_binding']['request_id'])
                rows.append(row)
            elif public['last_action_result'] is not None or choice['return_assessment'] is not None:
                raise ValueError('first decision assesses an unpaired return')
            previous, returned, reservation, reasoning = event, None, None, None
    unassessed = []
    if previous and previous['data']['decision']['choice']['action'] in {'query_source', 'read_window'}:
        unassessed.append({'acquisition_event_hash': previous['event_hash'],
            'action': previous['data']['decision']['choice']['action'],
            'status': 'RETURNED_NOT_ASSESSED' if returned else 'PENDING_OR_NOT_EXECUTED',
            'reason': 'No later admitted actor judgment exists; excluded from semantic outcome counts.'})
    return {'schema_version': 'bound-question-return-projection-v1', 'run_id': run_id,
        'archive_head': audit['head_hash'], 'records': rows, 'unassessed_actions': unassessed,
        'verified_model_returns': len(request_ids),
        'source_attempts_recorded_in_episode': run['budget']['http_attempts'],
        'body_reads_recorded_in_episode': run['budget']['window_reads'],
        'new_model_requests': 0, 'new_source_requests': 0,
        'independent_semantic_verification': False, 'archive_modified': False,
        'scope': 'offline_projection_of_original_actor_judgments_not_new_labels_or_live_policy_feedback'}
