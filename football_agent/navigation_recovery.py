'''Explicit recovery admission after a failed first model call, never a refund.'''
from copy import deepcopy

from .audit import digest


def recovery_plan(prior, *, task_id, configuration):
    if prior['state'] != 'DEFERRED' or prior['reason'] != 'PROVIDER_FAILED_OR_RESPONSE_UNKNOWN':
        raise ValueError('Recovery requires a recorded unknown provider outcome')
    if not prior['pending'] or prior['pending']['kind'] != 'llm':
        raise ValueError('Require preserved unresolved model reservation')
    if prior['decisions'] or prior['receipts'] or prior['budget']['http_attempts'] or prior['budget']['window_reads']:
        raise ValueError('This recovery only admits a first-decision failure, without source effects')
    if prior['budget']['llm_calls'] < 1 or prior['budget']['llm_usage']:
        raise ValueError('Unknown attempt must remain charged with no accepted return')
    if configuration != prior['configuration']:
        raise ValueError('Recovery must not change model, protocol or execution tools')
    budget = prior['budget']
    remaining = budget['max_llm_calls'] - budget['llm_calls']
    if remaining < 2:
        raise ValueError('Insufficient original budget for action and reconsideration')
    if not isinstance(task_id,str) or not task_id.strip() or task_id == prior['task']['task_id']:
        raise ValueError('Explicit new recovery task identity required')
    contract = deepcopy(prior['target_contract'])
    contract['task_id'] = task_id
    return {'version': 'first-decision-recovery-v1',
        'prior_run_id': prior['run_id'], 'prior_head_hash': prior['events'][-1]['event_hash'],
        'prior_task_contract_digest': digest(prior['target_contract']),
        'prior_unknown_attempts_charged': budget['llm_calls'],
        'original_llm_attempt_cap': budget['max_llm_calls'],
        'task_contract': contract, 'task_contract_digest': digest(contract),
        'limits': {'max_llm_calls': remaining, 'max_http_attempts': budget['max_http_attempts'],
                   'max_window_reads': min(budget['max_window_reads'], remaining-1)},
        'scope': 'explicit new attempt; old terminal/pending state retained; no accepted result replayed',
        'reference_time_changed': False, 'old_deadline_extended': False, 'budget_refunded': False}
