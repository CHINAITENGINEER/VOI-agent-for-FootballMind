'''Explicit, bounded queries over a task's already acquired immutable evidence.'''
from copy import deepcopy
import json
import unicodedata

from jsonschema import validate

from .structured_sources import detail_links, verify_platform_document

VERSION = 'acquired-platform-query-v2'
LEGACY_VERSION = 'acquired-platform-query-v1'
ACTION = 'query_acquired'
WINDOW = 4500
QUERY_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'document_id': {'type': 'string', 'pattern': '^doc-[0-9a-f-]{36}$'},
        'record_kind': {'type': 'string', 'enum': ['fixture', 'odds', 'all']},
        'source_match_id': {'type': ['string', 'null'], 'pattern': '^[0-9]{1,20}$'},
        'team_name': {'type': ['string', 'null'], 'minLength': 1, 'maxLength': 120},
        'team_match': {'type': 'string', 'enum': ['exact', 'contains', 'normalized_exact']},
        'offset': {'type': 'integer', 'minimum': 0, 'maximum': 1000000},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 20},
    },
}
QUERY_SCHEMA['required'] = list(QUERY_SCHEMA['properties'])


def action_payload(response):
    action = {'tool': response['action'], 'url': response['url'],
              'query': response['query'], 'engine': response['engine']}
    if response['action'] == ACTION:
        action['local_query'] = deepcopy(response['local_query'])
    return action


def capability(*, legacy=False):
    return {'version': LEGACY_VERSION if legacy else VERSION, 'action': ACTION, 'max_reads': 8, 'window_chars': WINDOW,
            'scope': 'already_acquired_platform_documents_in_this_task_only',
            'matching': ('literal_case_sensitive_no_fuzzy_alias_or_relevance_ranking' if legacy else
                         'literal_or_nfkc_casefold_whitespace_exact_no_fuzzy_alias_or_ranking'),
            'ordering': 'source_order_after_explicit_filters',
            'http_requests': 0, 'latency_seconds_estimate': None,
            'decision_loss_equivalent': None, 'numeric_voi_calibrated': False}


def eligible(document):
    return (document.get('content_kind') == 'platform_structured'
            and document.get('status') in {'readable', 'empty_structured', 'unsupported_content'}
            and (document.get('platform') or {}).get('quality') in
            {'normalized', 'partial', 'no_data', 'projection_window_exceeded'})


def validate_query(query, documents):
    validate(query, QUERY_SCHEMA)
    if type(query['offset']) is not int or type(query['limit']) is not int:
        raise ValueError('LOCAL_QUERY_INTEGER_REQUIRED')
    if query['team_name'] is not None:
        if not query['team_name'].strip() or query['record_kind'] != 'fixture':
            raise ValueError('LOCAL_TEAM_FILTER_REQUIRES_FIXTURE_AND_LITERAL_NAME')
    parents = [d for d in documents if d.get('document_id') == query['document_id']]
    if len(parents) != 1 or not eligible(parents[0]):
        raise ValueError('LOCAL_SOURCE_NOT_ACQUIRED_OR_QUERYABLE')
    return parents[0]


def normalize_team_name(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def project_query(archive, documents, query, operation_id, *, legacy=False):
    '''No fetching, hidden enrichment, match ranking or probability updates.'''
    parent = validate_query(query, documents)
    if legacy and query['team_match'] == 'normalized_exact':
        raise ValueError('LOCAL_QUERY_LEGACY_MATCHING')
    records = []
    verify_platform_document(archive, parent, record_sink=records)
    filtered = []
    for row in records:
        if query['record_kind'] != 'all' and row['kind'] != query['record_kind']:
            continue
        source_id = row.get('source_id') if row['kind'] == 'fixture' else row.get('source_match_id')
        if query['source_match_id'] is not None and source_id != query['source_match_id']:
            continue
        if query['team_name'] is not None:
            names = [row['home']['name'], row['away']['name']]
            term = query['team_name']
            if query['team_match'] == 'normalized_exact':
                matches = any(normalize_team_name(term) == normalize_team_name(name) for name in names)
            else:
                matches = any(term == name if query['team_match'] == 'exact' else term in name for name in names)
            if not matches:
                continue
        filtered.append(row)
    offset = query['offset']
    lines, displayed, size = [], [], 0
    for row in filtered[offset:offset + query['limit']]:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if size + len(line) + 1 > WINDOW:
            break
        lines.append(line)
        displayed.append(row)
        size += len(line) + 1
    blocked = bool(offset < len(filtered) and not displayed)
    consumed = len(displayed) + int(blocked)
    next_offset = offset + consumed if offset + consumed < len(filtered) else None
    links = []
    for row in displayed:
        if row['kind'] == 'fixture':
            for link in detail_links(row['source_id']):
                if link not in links and len(links) < 20:
                    links.append(link)
    metadata = {'version': LEGACY_VERSION if legacy else VERSION, 'source_document_id': parent['document_id'],
                'source_observation_id': parent['platform']['observation_id'],
                'source_acquisition_id': parent.get('acquisition_id'),
                'query': deepcopy(query), 'records_in_source': len(records),
                'records_matched': len(filtered), 'records_returned': len(displayed),
                'offset': offset, 'next_offset': next_offset,
                'omitted_oversized_offsets': [offset] if blocked else [],
                'coverage': parent['platform']['coverage'],
                'availability_basis': parent['platform']['availability_basis'],
                'source_quality': parent['platform']['quality'],
                'selection': 'explicit_literal_filter_source_order_not_relevance_ranking'}
    return {'document_id': 'local-doc-' + operation_id, 'url': parent['url'],
            'title': parent['title'] + ' / 已采集记录查询',
            'status': 'unsupported_content' if blocked else 'readable' if displayed else 'empty_structured',
            'content_kind': 'platform_local_query', 'text': chr(10).join(lines),
            'text_truncated': next_offset is not None or blocked, 'links': links,
            'signals': ['local_query_no_new_observation', 'source_assertions_not_independently_verified',
                        'no_result_is_not_evidence_of_match_absence'] +
                       (['oversized_record_omitted_explicitly'] if blocked else []),
            'observed_at': parent['observed_at'],
            'source_published_at': parent['source_published_at'],
            'raw_sha256': parent['raw_sha256'], 'request_id': None,
            'semantic_relevance': 'unassessed', 'local_evidence': metadata}
