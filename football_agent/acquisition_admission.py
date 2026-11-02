"""Task-lifetime admission of acquisition returns, separate from raw retrieval."""
from __future__ import annotations

from datetime import datetime

from .audit import digest


VERSION = 'acquisition-return-admission-v1'
VERSION_FIELD = 'return_admission_version'
STATE_FIELD = 'acquisition_admissions'


def _clock(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('RETURN_ADMISSION_CLOCK_INVALID') from exc


def make_record(*, acquisition_id, selection, action, result, evaluated_at,
                task_deadline, cancel_requested, event_sequence):
    """Use the local admission clock; source publication/capture is unrelated."""
    if type(cancel_requested) is not bool:
        raise ValueError('RETURN_ADMISSION_CANCEL_INVALID')
    expired = _clock(evaluated_at) >= _clock(task_deadline)
    reason = 'cancelled' if cancel_requested else 'deadline_expired' if expired else 'within_task_lifetime'
    admitted = not cancel_requested and not expired
    document = result.get('document')
    if document is not None and (not isinstance(document, dict) or
                                 not isinstance(document.get('document_id'), str)):
        raise ValueError('RETURN_ADMISSION_DOCUMENT_INVALID')
    body = {
        'version': VERSION, 'acquisition_id': acquisition_id,
        'selection_operation_id': selection.get('operation_id') if selection else None,
        'decision_digest': selection.get('decision_digest') if selection else None,
        'action_digest': digest(action), 'result_digest': digest(result),
        'evaluated_at': evaluated_at, 'task_deadline': task_deadline,
        'cancel_requested': cancel_requested, 'status': 'admitted' if admitted else 'rejected',
        'reason': reason, 'event_sequence': event_sequence,
        'admitted_document_ids': [document['document_id']] if admitted and document else [],
        'pending_return_admitted': admitted and 'return_context' in result,
    }
    return {**body, 'admission_digest': digest(body)}


def visible(state, acquisition_id, document_id=None):
    """Keep previously admitted evidence visible after a later stop/deadline."""
    if not acquisition_id:
        return True
    records = [record for record in state.get(STATE_FIELD, [])
               if record.get('acquisition_id') == acquisition_id]
    if len(records) != 1:
        return not records and VERSION_FIELD not in state['config']
    record = records[0]
    return record.get('status') == 'admitted' and (
        document_id in record.get('admitted_document_ids', []) if document_id is not None
        else record.get('pending_return_admitted') is True)


def verify_record(record, *, state, acquisition_id, selection, action, result, operations):
    """Verify an explicit saved admission without inventing historical clocks."""
    if not isinstance(record, dict):
        raise ValueError('REPLAY_RETURN_ADMISSION_INVALID')
    try:
        expected = make_record(acquisition_id=acquisition_id, selection=selection,
            action=action, result=result, evaluated_at=record['evaluated_at'],
            task_deadline=record['task_deadline'], cancel_requested=record['cancel_requested'],
            event_sequence=record['event_sequence'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('REPLAY_RETURN_ADMISSION_INVALID') from exc
    # Raw result bytes have their own provider/local projection checks during
    # replay. Keep those diagnostics specific (REPLAY_LOCAL_* / receipt),
    # while still validating the admission clock, action binding and state.
    comparable = {key: value for key, value in record.items()
                  if key not in {'result_digest', 'admission_digest'}}
    expected_comparable = {key: value for key, value in expected.items()
                           if key not in {'result_digest', 'admission_digest'}}
    if comparable != expected_comparable:
        raise ValueError('REPLAY_RETURN_ADMISSION_MISMATCH')
    body = {key: value for key, value in record.items() if key != 'admission_digest'}
    if digest(body) != record.get('admission_digest'):
        raise ValueError('REPLAY_RETURN_ADMISSION_DIGEST_MISMATCH')
    saved = [item for item in state.get(STATE_FIELD, []) if item.get('acquisition_id') == acquisition_id]
    if saved != [record]:
        raise ValueError('REPLAY_RETURN_ADMISSION_STATE_MISMATCH')
    try:
        sequence = record['event_sequence']
        events = state.get('events', [])
        if type(sequence) is not int or not 1 <= sequence <= len(events):
            raise ValueError('REPLAY_RETURN_ADMISSION_EVENT_MISMATCH')
        event = events[sequence - 1]
        if (event.get('type') != 'return_admission' or event.get('sequence') != sequence
                or event.get('data') != {'acquisition_id': acquisition_id,
                                        'admission_digest': record['admission_digest']}):
            raise ValueError('REPLAY_RETURN_ADMISSION_EVENT_MISMATCH')
        if any(item.get('type') == 'cancel_requested' for item in events[:sequence - 1]) != record['cancel_requested']:
            raise ValueError('REPLAY_RETURN_ADMISSION_CANCEL_MISMATCH')
        checked = _clock(record['evaluated_at'])
        if checked < _clock(state['created_at']) or _clock(event['at']) < checked:
            raise ValueError('REPLAY_RETURN_ADMISSION_CLOCK_MISMATCH')
        for operation in operations:
            if (operation.get('admission_digest') != record['admission_digest']
                    or operation.get('finished_at') != record['evaluated_at']
                    or _clock(operation['started_at']) > checked):
                raise ValueError('REPLAY_RETURN_ADMISSION_OPERATION_MISMATCH')
    except ValueError:
        raise
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('REPLAY_RETURN_ADMISSION_INVALID') from exc
    return record['status'] == 'admitted'
