'''Technical observation of a returned source, without semantic judgment.

The autonomous selector needs to know what the tool actually returned before it
decides whether to continue. HTTP success, a page title, a matching team name,
or a normalized record is not a proof that the return answers the active
question. This projection makes those boundaries explicit while keeping
semantic relevance and information gain unassigned to a heuristic.
'''
from __future__ import annotations

from copy import deepcopy


VERSION = 'return-observation-v1'
CONTENT_STATUSES = frozenset({'readable', 'empty_structured', 'empty_text'})


def _surface(document):
    kind = document.get('content_kind')
    if kind == 'platform_structured':
        return 'structured_records'
    if kind == 'platform_local_query':
        return 'local_structured_records'
    if kind == 'discovery':
        return 'discovery_metadata'
    if kind in {'html', 'json'}:
        return 'body_text'
    return 'none'


def _coverage(document, surface):
    signals = set(document.get('signals') or [])
    if surface == 'none' or document.get('status') not in CONTENT_STATUSES:
        return 'none'
    if document.get('status') in {'empty_structured', 'empty_text'}:
        return 'empty'
    if document.get('text_truncated') or signals & {
            'source_coverage_incomplete', 'bounded_source_order_projection_not_complete_result_set',
            'sparse_static_text_possible_dynamic_page'}:
        return 'partial'
    return 'unknown'


def project_return_observation(document, *, action=None, acquisition_id=None,
                               selection_operation_id=None):
    '''Project only observed technical facts; semantic relevance stays open.'''
    if not isinstance(document, dict):
        raise ValueError('RETURN_DOCUMENT_REQUIRED')
    if not document.get('document_id') or not document.get('url'):
        raise ValueError('RETURN_DOCUMENT_IDENTITY_REQUIRED')
    surface = _surface(document)
    status = document.get('status')
    signals = list(dict.fromkeys(str(value) for value in (document.get('signals') or [])))
    if status in CONTENT_STATUSES and document.get('http_status') == 200:
        transport = 'returned'
    elif status in CONTENT_STATUSES:
        transport = 'returned_with_unknown_http_status'
    elif status in {'unavailable', 'cancelled_or_budget_exhausted'} or document.get('error'):
        transport = 'failed_or_unavailable'
    else:
        transport = 'returned_without_usable_content'
    return {
        'schema_version': VERSION,
        'document_id': document['document_id'],
        'url': document['url'],
        'action': action.get('action') if isinstance(action, dict) else None,
        'acquisition_id': acquisition_id,
        'selection_operation_id': selection_operation_id,
        'transport_status': transport,
        'technical_status': status,
        'http_status': document.get('http_status'),
        'content_kind': document.get('content_kind'),
        'evidence_surface': surface,
        'coverage': _coverage(document, surface),
        'signals': deepcopy(signals),
        'semantic_relevance': 'unassessed',
        'information_gain': None,
        'decision_answered': None,
        'truth_status': 'unverified',
        'requires_agent_assessment': surface != 'none',
        'assessment_basis': 'technical_return_only; no term_match_or_http_success_inference',
    }
