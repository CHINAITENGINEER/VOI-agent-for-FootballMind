'''Project verified navigation into context and value interfaces without inventing utility.

This is an offline, time-indexed return adapter. It uses the existing query
gate only for unmeasured value and homogeneous count budgets; counts never
enter a loss-valued or money-valued finite model. A textual need change is an
observed planning response, not a causal benefit or a completed task label.
'''
from copy import deepcopy
from pathlib import Path

from .audit import RunStore, digest
from .context_transition import ContextTransition
from .contextual_adaptive_voi import context_fingerprint, normalize_context
from .information_navigation import InformationNavigationWorkflow
from .information_navigation_verify import verify_navigation
from .inquiry_contract import utc
from .inquiry_tools import public_url
from .query_value_contract import QueryValueAssessment, QueryValueGate, VALUE_UNMEASURED
from .state_query_contract import StateHypothesis, StateQueryBoard, StateQueryCandidate

VERSION = 'navigation-feedback-v1'
COUNTS = ('llm_calls', 'http_attempts', 'window_reads', 'delivered_body_chars')


class _StoredToolConfiguration:
    '''Read-only configuration facade; no executor or network method.'''
    def __init__(self, configuration):
        self._configuration = deepcopy(configuration)

    def configuration(self):
        return deepcopy(self._configuration)


def _need(public):
    choice = public['last_choice'] or public['inherited_information_need']
    return {key: choice.get(key) for key in
            ('question', 'next_information_need', 'unread_limits', 'reason', 'action')}


def _context(public, snapshot):
    need = _need(public)
    docs = public['documents']
    return normalize_context({
        'navigation.task_id': public['task']['task_id'],
        'navigation.question': need['question'],
        'navigation.information_need': need['next_information_need'],
        'navigation.need_status': ('model_reported_unresolved' if need['next_information_need']
                                   else 'not_stated_not_verified_resolved'),
        'navigation.unread_limits': need['unread_limits'],
        'navigation.reason': need['reason'],
        'navigation.last_action': need['action'],
        'navigation.state': snapshot['state'],
        'navigation.pending': (snapshot['pending'] or {}).get('kind'),
        'navigation.document_count': len(docs),
        'navigation.opened_windows': sum(len(d.get('visible_windows', [])) for d in docs),
        'navigation.claim_count': len(public['claims']),
        'navigation.claims_digest': digest(public['claims']),
        'navigation.visible_evidence_digest': digest(docs),
        'navigation.semantic_verification': 'unverified',
        'navigation.value_status': VALUE_UNMEASURED,
        **{'navigation.' + key: snapshot['budget'][key] for key in COUNTS},
    })


def _costs(snapshot):
    b = snapshot['budget']
    return {
        'spent': {key: b[key] for key in COUNTS},
        'remaining': {key: b['max_' + key] - b[key]
                      for key in ('llm_calls', 'http_attempts', 'window_reads')},
        'units': {'llm_calls': 'CLI_attempts_including_unknown_returns',
                  'http_attempts': 'source_attempt_reservations',
                  'window_reads': 'new_window_openings',
                  'delivered_body_chars': 'unicode_code_points_delivered_as_body'},
        'scope': 'current_navigation_run_only_inherited_costs_not_recharged',
        'actual_money': None, 'loss_unit_conversion': None,
    }


def _options(public):
    '''Enumerate exposed navigation affordances, not a complete semantic menu.'''
    rows = []
    urls = list(public['task']['seed_urls'])
    archived = {d['url'] for d in public['documents']}
    for doc in public['documents']:
        for window in doc.get('catalog', []):
            if not window['opened']:
                rows.append({'action': 'read_window', 'document_id': doc['document_id'],
                    'window_id': window['window_id'], 'resource': 'window_reads'})
        urls.extend(link['url'] for link in doc.get('links', []))
    seen = set()
    for url in urls:
        try:
            url = public_url(url, public['tools']['allowed_hosts'])
        except ValueError:
            continue
        if url not in seen and url not in archived:
            rows.append({'action': 'query_source', 'tool': 'read_url',
                         'url': url, 'resource': 'http_attempts'})
        seen.add(url)
    rows.append({'action': 'query_source', 'tool': 'search_web',
                 'proposal_required': True, 'resource': 'http_attempts'})
    return [{**row, 'candidate_id': 'nav-' + digest(row)} for row in rows]


def assess_navigation_options(public, snapshot, *, as_of):
    '''Re-use the gate separately for each count unit, with no modeled values.

The capability assessment is diagnostic: it does not replace or retrospectively
claim to have controlled the recorded LLM policy. A concrete search query still
has to be proposed and validated by the existing navigation executor.
'''
    options, costs = _options(public), _costs(snapshot)
    task, groups = public['task'], []
    guards = []
    if snapshot['state'] != 'READY': guards.append('existing_run_terminal')
    if snapshot['pending']: guards.append('existing_call_requires_reconciliation')
    if utc(as_of) >= utc(task['deadline']): guards.append('deadline_reached')
    if costs['remaining']['llm_calls'] < 2:
        guards.append('insufficient_calls_to_select_and_reconsider')
    for resource in ('http_attempts', 'window_reads'):
        selected = [row for row in options if row['resource'] == resource]
        variable = 'navigation.information_need'
        board = StateQueryBoard(target_key=task['task_id'], prediction_cutoff=task['deadline'],
            hypotheses=[StateHypothesis('navigation-need', task['task_id'], variable,
                ('resolved', 'unresolved'), scope='information_task')],
            candidates=[StateQueryCandidate(row['candidate_id'], task['task_id'],
                task['deadline'], variable, 'navigation_affordance', row['action'], 1,
                endpoint=row.get('url'), action_id=row['candidate_id']) for row in selected])
        assessments = [QueryValueAssessment.unmeasured(row['candidate_id'], cost=1,
            reason='no_independent_task_loss_and_conditional_return_calibration') for row in selected]
        decision = QueryValueGate.decide(board, assessments,
            remaining_budget=costs['remaining'][resource], view='agent')
        groups.append({'resource': resource, 'cost_unit': costs['units'][resource],
            'remaining_count': costs['remaining'][resource],
            'options': selected, 'value_gate': decision.as_dict(),
            'operational_guards': list(guards),
            'new_action_admitted': False,
            'scope': 'unmeasured_only_no_conversion_to_risk_or_currency'})
    return {'groups': groups, 'operational_guards': guards,
        'enumeration_scope': 'visible_windows_links_and_search_capability_not_all_possible_information',
        'value_status': VALUE_UNMEASURED, 'optimal_stop_proven': False,
        'automatic_budget_extension': False, 'recorded_policy_changed': False}


def project_navigation_feedback(reference, *, as_of, through_sequence=None):
    '''Read an explicitly pinned run and export only the requested causal prefix.'''
    root, run_id = Path(reference['root']).resolve(), reference['run_id']
    proof = verify_navigation(root, run_id)
    store = RunStore(root / 'runs')
    run = store.load(run_id)
    if proof['audit']['head_hash'] != reference['head_hash'] or run['events'][-1]['event_hash'] != reference['head_hash']:
        raise ValueError('Navigation head changed; pin the intended run')
    cutoff = utc(as_of)
    if through_sequence is not None and (type(through_sequence) is not int or
            not 1 <= through_sequence <= len(run['events'])):
        raise ValueError('through_sequence must identify an existing event')
    if through_sequence and utc(run['events'][through_sequence - 1]['at']) > cutoff:
        raise ValueError('Requested event was not available as_of')
    events = [e for e in run['events'] if utc(e['at']) <= cutoff and
              (through_sequence is None or e['seq'] <= through_sequence)]
    if not events: raise ValueError('No navigation state available as_of')
    flow = InformationNavigationWorkflow(root, None,
        _StoredToolConfiguration(run['configuration']['tools']))
    contexts, publics, transitions = {}, {}, []
    previous = {}
    for event in events:
        snap = event['snapshot']
        public = flow.public(snap, now=event['at'])
        if any(utc(d['available_at']) > utc(event['at']) for d in public['documents']):
            raise ValueError('Visible document was not yet available')
        context = _context(public, snap)
        publics[event['seq']], contexts[event['seq']] = public, context
        kind = ('INITIALIZED' if event['seq'] == 1 else
                'STOPPED' if event['type'] == 'reading_finished' else
                'REPLANNED' if event['type'] == 'navigation_decision' else 'CONTEXT_UPDATED')
        transition = ContextTransition.from_transition(episode_id=run_id,
            target_key=run['task']['task_id'], sequence=event['seq'], event_type=kind,
            previous_context=previous, next_context=context, trigger=event['type'],
            reason='observed_navigation_event_not_verified_domain_state',
            prediction_cutoff=run['task']['deadline'], captured_at=event['at'],
            evidence_refs=(event['event_hash'],))
        transitions.append(transition.as_dict())
        previous = context
    # One action row per accepted model choice. Operational counters end before
    # the next model reservation, so reconsideration cost is never counted twice.
    rows = []
    decision_events = [e for e in events if e['type'] == 'navigation_decision']
    for event in decision_events:
        reservation = next(e for e in reversed(events[:event['seq'] - 1])
                           if e['type'] == 'navigation_reasoning_reserved')
        before = events[reservation['seq'] - 2]
        next_reservation = next((e for e in events[event['seq']:]
                                if e['type'] == 'navigation_reasoning_reserved'), None)
        after = events[next_reservation['seq'] - 2] if next_reservation else events[-1]
        later = next((e for e in decision_events if e['seq'] > event['seq']), None)
        choice = event['data']['decision']['choice']
        after_need = (later['data']['decision']['choice']['next_information_need'] if later else None)
        spent = {key: after['snapshot']['budget'][key] - before['snapshot']['budget'][key]
                 for key in COUNTS}
        effect_events = events[event['seq']:after['seq']]
        returned = [e for e in effect_events if e['type'] == 'navigation_acquisition_returned']
        row = {
            'row_id': event['event_hash'], 'origin': 'verified_archived_navigation',
            'sampling_scope': 'selected_only_no_propensity_or_counterfactual_observation',
            'decision_at': event['at'], 'before_event_hash': before['event_hash'],
            'after_event_hash': after['event_hash'], 'observed_through': after['at'],
            'pre_action_context': contexts[before['seq']],
            'pre_action_context_key': context_fingerprint(contexts[before['seq']]),
            'pre_action_value_gate': assess_navigation_options(publics[before['seq']],
                before['snapshot'], as_of=before['at']),
            'selected_action': deepcopy(choice),
            'post_action_context': contexts[after['seq']],
            'observed_cost': spent, 'cost_units': _costs(after['snapshot'])['units'],
            'source_returns': [{'event_hash': e['event_hash'],
                'document_id': e['data']['document_id'],
                'status': store.get_blob(run_id, e['snapshot']['documents'][e['data']['document_id']])['status']}
                for e in returned],
            'new_body_openings': sum(e['type'] == 'navigation_window_opened' for e in effect_events),
            'pending': deepcopy(after['snapshot']['pending']),
            'reconsideration': None if later is None else {
                'event_hash': later['event_hash'], 'available_at': later['at'],
                'next_action': later['data']['decision']['choice']['action'],
                'information_need': after_need,
                'need_text_changed': after_need != choice['next_information_need'],
                'meaning': 'observed_text_change_not_causal_or_loss_reduction'},
            'selection_probability': None, 'counterfactual_return': None,
            'task_outcome_label': None, 'risk_before': None, 'risk_after': None,
            'expected_information_value': None, 'realized_decision_utility': None,
            'eligible_for_value_calibration': False,
        }
        rows.append(row)
    last, first = events[-1], events[0]
    snapshot, public = last['snapshot'], publics[last['seq']]
    attributed = {key: sum(row['observed_cost'][key] for row in rows) for key in COUNTS}
    unattributed = {key: snapshot['budget'][key] - attributed[key] for key in COUNTS}
    if any(value < 0 for value in unattributed.values()):
        raise ValueError('Action costs overlap')
    return {'schema_version': VERSION, 'reference': deepcopy(reference),
        'as_of': cutoff.isoformat(), 'through_sequence': last['seq'],
        'prefix_event_hash': last['event_hash'], 'available_at': last['at'],
        'replay_scope': 'verified_full_history_projection_of_requested_prefix',
        'task': deepcopy(run['task']), 'initial_context': contexts[first['seq']],
        'next_context': contexts[last['seq']],
        'context_key': context_fingerprint(contexts[last['seq']]),
        'information_need': _need(public), 'visible_documents': public['documents'],
        'claims': public['claims'], 'transitions': transitions, 'action_rows': rows,
        'costs': _costs(snapshot), 'costs_without_accepted_choice': unattributed,
        'value_reassessment': assess_navigation_options(public, snapshot, as_of=cutoff),
        'pending': deepcopy(snapshot['pending']),
        'structural_verification': 'existing_raw_archive_input_and_navigation_replay',
        'semantic_verification': 'unverified', 'prediction_updated': False,
        'domain_truth_updated': False, 'numerical_VoI_verified': False,
        'independent_policy_benefit': None, 'new_model_calls': 0, 'new_network_requests': 0}
