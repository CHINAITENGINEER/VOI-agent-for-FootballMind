'''Replay stored reading visibility, choices and cost counters; no model/network.'''
from copy import deepcopy
from pathlib import Path

from .audit import RunStore, digest
from .evidence_reading import (EvidenceReadingWorkflow, ReadingProtocol, SCHEMA, PROMPT,
                               source_prefix, validate_choice)
from .evidence_windows import EvidenceWindows
from .inquiry_contract import utc


def verify_reading(root, run_id):
    root = Path(root)
    store = RunStore(root / 'runs')
    audit, run = store.verify(run_id), store.load(run_id)
    initial = run['events'][0]['snapshot']
    config = initial['configuration']
    if config['version'] != EvidenceReadingWorkflow.version or config['protocol'] != ReadingProtocol.version:
        raise ValueError('Unknown reading configuration')
    if config['protocol_digest'] != digest({'schema': SCHEMA, 'prompt': PROMPT}):
        raise ValueError('Reading contract changed')
    source = store.get_blob(run_id, initial['source_ref'])
    p = source['provenance']
    expected_source = source_prefix(p['parent_root'], p['parent_run_id'], p['parent_head_hash'],
        p['document_id'], now=run['events'][0]['at'])
    if source != expected_source:
        raise ValueError('Source differs from the completed acquisition prefix')
    book = EvidenceWindows(source['document'], window_chars=config['window_chars'], preview_chars=config['preview_chars'])
    reader = EvidenceReadingWorkflow(root, None, window_chars=config['window_chars'], preview_chars=config['preview_chars'])
    budget = deepcopy(initial['budget'])
    if (budget['llm_calls'] or budget['window_reads'] or budget['delivered_body_chars']
            or budget['new_http_attempts'] or budget['llm_usage'] or budget['actual_money'] is not None):
        raise ValueError('Nonempty starting cost record')
    if not 2 <= budget['max_llm_calls'] <= 5 or not 1 <= budget['max_window_reads'] < budget['max_llm_calls']:
        raise ValueError('Invalid starting budgets')
    opened, claims, decisions, call_ids = [], [], [], set()
    pending, phase, reply, choice = None, 'initial', None, None
    state, reason = 'READY', 'STARTED'
    previous, reservation_ref = None, None
    returned_count, input_sizes = 0, []
    declared_only_stop_conditions = []
    for event in run['events']:
        kind, data, snapshot = event['type'], event['data'], event['snapshot']
        if any(snapshot[k] != initial[k] for k in ('run_id', 'task', 'configuration', 'source_ref')):
            raise ValueError('Frozen reading identity or configuration changed')
        if previous and utc(event['at']) < utc(previous['at']):
            raise ValueError('Reading event time went backwards')
        if phase == 'finished':
            raise ValueError('Events appeared after a terminal state')
        if kind == 'reading_started':
            if event['seq'] != 1 or data != {'source_ref': initial['source_ref'], 'provenance': p}:
                raise ValueError('Invalid reading start provenance')
            phase = 'ready'
        elif kind == 'reading_reasoning_reserved':
            if phase != 'ready' or pending or state != 'READY' or budget['llm_calls'] >= budget['max_llm_calls']:
                raise ValueError('Unexpected model reservation')
            if utc(event['at']) >= utc(initial['task']['deadline']):
                raise ValueError('Model reservation after deadline')
            public = store.get_blob(run_id, data['input_ref'])
            if public != reader.public(previous['snapshot'], now=public['now']):
                raise ValueError('Model input differs from visible evidence at selection time')
            if not utc(previous['at']) <= utc(public['now']) <= utc(event['at']):
                raise ValueError('Input timestamp outside its recorded event interval')
            pending = snapshot['pending']
            if not isinstance(pending, dict) or set(pending) != {'kind', 'call_id', 'input_ref'}:
                raise ValueError('Malformed pending call')
            if pending['kind'] != 'llm' or pending['input_ref'] != data['input_ref'] or pending['call_id'] in call_ids:
                raise ValueError('Unbound or reused reasoning reservation')
            call_ids.add(pending['call_id'])
            budget['llm_calls'] += 1
            reservation_ref = data['input_ref']
            input_sizes.append(len(__import__('json').dumps(public,ensure_ascii=False,separators=(',',':')).encode('utf-8')))
            phase = 'reserved'
        elif kind == 'reading_reasoning_returned':
            if phase != 'reserved' or not pending or data['input_ref'] != reservation_ref:
                raise ValueError('Unexpected model return')
            reply = store.get_blob(run_id, data['output_ref'])
            budget['llm_usage'].append(reply.get('receipt', {}))
            returned_count += 1
            pending, phase = None, 'returned'
        elif kind == 'reading_reasoning_failed':
            if phase != 'reserved' or data.get('outcome') != 'unknown':
                raise ValueError('Unexpected provider failure record')
            phase = 'failed'
        elif kind == 'reading_response_rejected':
            if phase != 'returned':
                raise ValueError('Unexpected response rejection')
            try:
                validate_choice(reply['response'], book, opened)
            except Exception:
                pass
            else:
                raise ValueError('A valid response was labelled invalid')
            phase = 'rejected'
        elif kind == 'reading_decision':
            if phase != 'returned' or utc(event['at']) >= utc(initial['task']['deadline']):
                raise ValueError('Decision has no timely model return')
            if data['input_ref'] != reservation_ref or store.get_blob(run_id, data['output_ref']) != reply:
                raise ValueError('Decision is not bound to its actual model call')
            choice = reply['response']
            updates = validate_choice(choice, book, opened)
            expected = {'choice': deepcopy(choice), 'opened_before': list(opened), 'claims': updates,
                        'expected_gain': None, 'selection_basis': 'qualitative_bounded_reading'}
            if data['decision'] != expected:
                raise ValueError('Decision differs from validated model response')
            for c in updates:
                if c not in claims:
                    claims.append(c)
            decisions.append(expected)
            phase = 'decided'
        elif kind == 'reading_window_opened':
            if phase != 'decided' or choice['action'] != 'read_window' or data['window_id'] != choice['window_id']:
                raise ValueError('Body read does not follow the selected window')
            if budget['window_reads'] >= budget['max_window_reads'] or budget['llm_calls'] >= budget['max_llm_calls']:
                raise ValueError('Body read exceeds continuation budget')
            body = book.read(choice['window_id'])
            if store.get_blob(run_id, data['output_ref']) != body or data['new_http_attempts'] != 0:
                raise ValueError('Window contents or acquisition cost changed')
            opened.append(body['window_id'])
            budget['window_reads'] += 1
            budget['delivered_body_chars'] += len(body['text'])
            phase = 'ready'
        elif kind == 'reading_finished':
            stop = data['reason']
            valid = False
            expected_state = 'DEFERRED'
            if stop in {'AGENT_STOP', 'AGENT_DEFER'}:
                valid = phase == 'decided' and choice['action'].upper() == stop[6:]
                expected_state = 'STOPPED' if stop == 'AGENT_STOP' else 'DEFERRED'
            elif stop == 'READ_CONTINUATION_BUDGET_INSUFFICIENT':
                valid = phase == 'decided' and choice['action'] == 'read_window' and (
                    budget['window_reads'] >= budget['max_window_reads'] or budget['llm_calls'] >= budget['max_llm_calls'])
            elif stop == 'PROVIDER_FAILED_OR_RESPONSE_UNKNOWN':
                valid = phase == 'failed' and pending is not None
            elif stop == 'INVALID_READING_RESPONSE':
                valid = phase == 'rejected'
            elif stop == 'LATE_RESPONSE':
                valid = phase == 'returned' and utc(event['at']) >= utc(initial['task']['deadline'])
                expected_state = 'STOPPED'
            elif stop == 'DEADLINE_REACHED':
                valid = phase == 'ready' and utc(event['at']) >= utc(initial['task']['deadline'])
                expected_state = 'STOPPED'
            elif stop == 'MODEL_BUDGET_EXHAUSTED':
                valid = phase == 'ready' and budget['llm_calls'] >= budget['max_llm_calls']
            elif stop == 'UNKNOWN_CALL_REQUIRES_RECONCILIATION':
                valid = phase == 'reserved' and pending is not None
            elif stop == 'CONFIGURATION_CHANGED':
                valid = phase == 'ready'
                declared_only_stop_conditions.append(stop)
            if not valid or data['state'] != expected_state or data['value_stop_proven'] is not False:
                raise ValueError('Unsupported reading terminal transition')
            state, reason, phase = data['state'], stop, 'finished'
        else:
            raise ValueError('Unknown reading event type')
        expected_fields = {'state': state, 'reason': reason, 'pending': pending,
            'opened': opened, 'claims': claims, 'decisions': decisions, 'budget': budget}
        if any(snapshot[k] != value for k,value in expected_fields.items()):
            raise ValueError('Snapshot differs from reconstructed reading prefix')
        previous = event
    return {'status': 'VERIFIED_READING_TRACE_NOT_INFORMATION_VALUE', 'audit': audit,
        'model_attempts': budget['llm_calls'], 'completed_returns': returned_count,
        'body_reads': len(opened), 'new_http_attempts': 0, 'claims_quote_checked': len(claims),
        'decisions_verified': len(decisions), 'input_json_utf8_bytes': input_sizes,
        'window_read_then_reconsider_observed': any(d['opened_before'] for d in decisions),
        'agent_terminal_after_reading': any(d['opened_before'] and d['choice']['action'] in {'stop','defer'} for d in decisions),
        'source_prefix': p, 'pending': pending,
        'declared_only_stop_conditions': declared_only_stop_conditions,
        'source_semantics_verified': False, 'numerical_value_verified': False,
        'independent_policy_benefit': None}
