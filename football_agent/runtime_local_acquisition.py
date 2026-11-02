"""Bind proposed local reads to visible, archive-verified prematch evidence.

No IO, candidate generation, odds ranking or gain estimate. The executor and
replay supply the independently reconstructed fixture bridge. A local read
reveals more of an existing observation; it never refreshes that observation.
"""
from copy import deepcopy
from datetime import datetime, timezone

from jsonschema import validate, ValidationError

from .local_evidence import ACTION, QUERY_SCHEMA, VERSION as QUERY_VERSION, capability, validate_query
from .runtime_fixture_bridge import validate_fixture_identity

VERSION = 'runtime-local-acquisition-v1'


def enabled(public):
    from .runtime_discovered_routes import descriptor, LOCAL_VERSION
    config = descriptor(public.get('selection_policy'))
    return config is not None and config['version'] == LOCAL_VERSION and config['query_acquired'] is not None


def extend_protocol(protocol):
    return type('LocalAcquisitionProtocol', (), {
        'version': protocol.version + '-resolved-local-v1', 'schema': protocol.schema,
        'prompt': protocol.prompt + '''
OPTIONAL RESOLVED-MATCH LOCAL INFORMATION ACQUISITION:
After the fixture is source-resolved, you may propose query_acquired for unseen
odds in an already acquired, archive-verified spftll document. This is optional;
compare it with other meaningful actions. Never manufacture a need for odds.
Use the resolved source_match_id, record_kind=odds, team_name=null and
team_match=exact. Start at offset=0; continue only from that same query's
verified next_offset, never offset+limit. Use limit 1..20. A completed query
or wholly visible parent provides no new page. Local reads consume their own
resource and the frozen exploration allowance, even when HTTP cost is zero.
The source observation must precede the current prematch snapshot. Local
reading preserves the old observed_at; it is not a fresh source observation.
No read has proven positive value merely because it is available or cheap.
Odds alone do not establish qualification, rotation, lineup or tactical intent.
'''})


def payload(response):
    query = response.get('local_query')
    try:
        validate(query, QUERY_SCHEMA)
    except ValidationError as exc:
        raise ValueError('LOCAL_ACQUISITION_QUERY_SCHEMA') from exc
    if (response.get('action') != ACTION or type(query['offset']) is not int
            or type(query['limit']) is not int or query['record_kind'] != 'odds'
            or query['source_match_id'] is None or query['team_name'] is not None
            or query['team_match'] != 'exact'
            or any(response.get(key) is not None for key in ('url', 'query', 'engine'))):
        raise ValueError('LOCAL_ACQUISITION_EXACT_ODDS_QUERY_REQUIRED')
    return {'action': ACTION, 'local_query': deepcopy(query)}


def key_payload(response):
    request = payload(response)
    # Changing the page size or the question cannot reopen an accepted offset.
    request['local_query'].pop('limit')
    return request


def _time(value):
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError()
        return stamp.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('LOCAL_ACQUISITION_OBSERVATION_CLOCK') from exc


def _selection(query):
    return {k: v for k, v in query.items() if k not in {'offset', 'limit', 'team_match'}}


def bind(public, response):
    request = payload(response)
    query = request['local_query']
    if public.get('local_query_capability') != capability():
        raise ValueError('LOCAL_ACQUISITION_CAPABILITY_MISMATCH')
    identity = validate_fixture_identity(public)
    bridge = public['fixture_identity']
    if (identity['status'] != 'resolved' or identity['target_key'] != 'spftll:' + query['source_match_id']
            or bridge['errors'] or 'verified_local_sources' not in bridge):
        raise ValueError('LOCAL_ACQUISITION_RESOLVED_TARGET_REQUIRED')
    parent = validate_query(query, public['documents'])
    meta = parent.get('platform') or {}
    verified = {row['document_id'] for row in bridge['verified_sources']}
    if (parent['document_id'] not in verified or meta.get('source') != 'spftll'
            or meta.get('quality') not in {'normalized', 'projection_window_exceeded'}
            or meta.get('issues') != 0):
        raise ValueError('LOCAL_ACQUISITION_VERIFIED_PARENT_REQUIRED')
    cutoff, observed = _time(public.get('snapshot_as_of')), _time(parent.get('observed_at'))
    fixtures = [row for row in identity['candidates'] if row['target_key'] == identity['target_key']]
    if not fixtures or any(not observed <= cutoff < _time(row['kickoff']) for row in fixtures):
        raise ValueError('LOCAL_ACQUISITION_PREMATCH_OBSERVATION_REQUIRED')
    if not (parent.get('text_truncated') or parent.get('excerpt_truncated')):
        raise ValueError('LOCAL_ACQUISITION_PARENT_ALREADY_VISIBLE')
    if meta.get('quotes') == 0:
        raise ValueError('LOCAL_ACQUISITION_PARENT_HAS_NO_QUOTES')

    local_ids = {row['document_id'] for row in bridge['verified_local_sources']}
    pages = [doc for doc in public['documents'] if doc['document_id'] in local_ids
             and _selection(doc['local_evidence']['query']) == _selection(query)]
    pages.sort(key=lambda doc: (doc['local_evidence']['offset'], doc['document_id']))
    expected, previous = 0, None
    for page in pages:
        local = page['local_evidence']
        if (expected is None or local['offset'] != expected or local['version'] != QUERY_VERSION
                or page['observed_at'] != parent['observed_at']):
            raise ValueError('LOCAL_ACQUISITION_PAGE_CHAIN_INVALID')
        following = local['next_offset']
        if following is not None and (type(following) is not int or following <= expected):
            raise ValueError('LOCAL_ACQUISITION_PAGE_CHAIN_INVALID')
        expected, previous = following, page['document_id']
    if expected is None:
        raise ValueError('LOCAL_ACQUISITION_QUERY_COMPLETE')
    if query['offset'] != expected:
        raise ValueError('LOCAL_ACQUISITION_NEXT_OFFSET_REQUIRED')
    return {'kind': 'acquired_platform_page', 'document_id': parent['document_id'],
        'source_observation_id': meta['observation_id'], 'raw_sha256': parent['raw_sha256'],
        'observed_at': parent['observed_at'], 'source_published_at': parent['source_published_at'],
        'target_key': identity['target_key'], 'record_kind': 'odds', 'offset': expected,
        'previous_page_document_id': previous, 'binding_version': VERSION,
        'availability_basis': meta['availability_basis'], 'new_source_observation': False}
