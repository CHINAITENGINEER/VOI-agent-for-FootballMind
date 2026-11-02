"""Bind fixture identity to archived responses and visible local query pages.

The bridge is stricter than the runtime projection. A local ``query_acquired``
page becomes resolver evidence only after its parent observation, literal
query, page coverage and visible prefix all replay exactly.
"""
from copy import deepcopy
import json
import re
from uuid import UUID

from football_spider.models.contracts import Observation
from .audit import stored_digest as digest
from .runtime_fixture_identity import resolve_match_identity
from .structured_sources import PROJECTION_VERSION, registered_request, verify_platform_document
from .local_evidence import VERSION as LOCAL_QUERY_VERSION, LEGACY_VERSION as LOCAL_LEGACY_VERSION, project_query
from .runtime_identity_coverage import coverage_views

LEGACY_VERSION = 'runtime-fixture-bridge-v1'
VERSION = 'runtime-fixture-bridge-v2'
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_IDENTITY_FIELDS = ('document_id', 'url', 'status', 'content_kind', 'observed_at',
    'source_published_at', 'raw_sha256', 'request_id', 'platform', 'excerpt',
    'excerpt_truncated', 'text_truncated', 'structured_records')
_LEGACY_WRAPPER_FIELDS = {'version', 'input_digest', 'verified_sources', 'errors', 'resolution'}
_V2_WRAPPER_FIELDS = _LEGACY_WRAPPER_FIELDS | {'verified_local_sources', 'local_coverage'}


def _input(public):
    return {key: public.get(key) for key in ('match_input', 'snapshot_as_of', 'documents')}


def _platform(document):
    if not isinstance(document, dict) or document.get('content_kind') != 'platform_structured':
        return False
    meta = document.get('platform')
    return ((isinstance(meta, dict) and meta.get('source') == 'spftll')
            or registered_request(document.get('url')) is not None)


def _view(original):
    """Exactly the complete JSON-line prefix used by Experience.public_state."""
    view = {key: deepcopy(original.get(key)) for key in _IDENTITY_FIELDS}
    text = original['text']
    lines, length = [], 0
    for line in text.splitlines():
        if length + len(line) + 1 > 4500:
            break
        lines.append(line)
        length += len(line) + 1
    view.update(excerpt=chr(10).join(lines), structured_records=[json.loads(line) for line in lines],
        text_truncated=bool(original.get('text_truncated', False)),
        excerpt_truncated=bool(original.get('text_truncated', False)) or len(lines) < len(text.splitlines()))
    view['platform']['records_visible'] = len(lines)
    return view


def _local_view(original):
    """The bounded visible projection used for an acquired query page."""
    view = {key: deepcopy(original.get(key)) for key in
            ('document_id','url','title','status','content_kind','signals','observed_at',
             'source_published_at','raw_sha256','request_id','semantic_relevance',
             )}
    text = original.get('text') if isinstance(original.get('text'), str) else ''
    lines, length = [], 0
    for line in text.splitlines():
        if length + len(line) + 1 > 4500:
            break
        lines.append(line)
        length += len(line) + 1
    view.update(excerpt=chr(10).join(lines),
                excerpt_truncated=bool(original.get('text_truncated', False))
                    or len(lines) < len(text.splitlines()),
                text_truncated=bool(original.get('text_truncated', False)),
                links=deepcopy(original.get('links', []))[:20],
                local_evidence=deepcopy(original.get('local_evidence')),
                structured_records=[json.loads(line) for line in lines])
    return view


def _local_operation_id(document):
    did = document.get('document_id', '')
    if not isinstance(did, str) or not did.startswith('local-doc-'):
        raise ValueError('local_document_id_invalid')
    return did[len('local-doc-'):]


def _duplicate_raw_ids(archive, document):
    oid = str(UUID(document['platform']['observation_id']))
    paths = list((archive.root / 'raw' / 'spftll').glob('*/*_' + oid + '.observation.json'))
    if len(paths) != 1:
        raise ValueError('archive_document_verification_failed')
    observation = Observation.model_validate_json(paths[0].read_bytes())
    payload = json.loads(archive.read_body(observation))
    data = payload.get('data')
    if data is None:
        return False
    endpoint = observation.endpoint
    if endpoint.endswith('GetList'):
        rows = [row for group in data for row in group.get('matchList') or []]
    elif endpoint.endswith('GetSimpleMatchsAll'):
        rows = [row for group in data for row in group.get('list') or []]
    elif endpoint.endswith('GetInfo'):
        rows = [data] + [row for key in ('battleList', 'homeBattleList', 'awayBattleList')
                         for row in data.get(key) or []]
    else:
        rows = []
    seen = set()
    for row in rows:
        if isinstance(row, dict) and row.get('matchId') is not None:
            key = str(row['matchId'])
            if key in seen:
                return True
            seen.add(key)
    return False


def _resolution(public, receipts, errors, *, local_documents=(), replaced=()):
    resolver_public = deepcopy(public)
    if local_documents:
        parents = [deepcopy(d) for d in public.get('documents', []) if _platform(d)]
        for parent in parents:
            if parent['document_id'] in replaced:
                # Complete query coverage discharges the matching-row window
                # gap, not already observed evidence. Preserve every visible
                # parent row so filtering cannot erase source-ID conflicts.
                parent.update(status='readable', text_truncated=False, excerpt_truncated=False)
                parent['platform']['quality'] = 'normalized'
        resolver_public['documents'] = [*parents, *list(local_documents)]
    resolution = resolve_match_identity(resolver_public,
        verified_document_ids=[entry['document_id'] for entry in receipts]
            + [entry['document_id'] for entry in public.get('_verified_local_sources', [])])
    if errors:
        resolution.update(status='incomplete', target_key=None, identity_basis='unresolved',
                          reason='archive_identity_verification_failed', evidence=[])
    return resolution


def _index_documents(public):
    documents = public.get('documents')
    if not isinstance(documents, list):
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_COLLECTIONS')
    indexed = {}
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get('document_id'), str):
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_DOCUMENT')
        did = document['document_id']
        if not did or did in indexed:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_DUPLICATE_DOCUMENT')
        indexed[did] = document
    return indexed


def _validate_errors(errors):
    if not isinstance(errors, list):
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_COLLECTIONS')
    for error in errors:
        if (not isinstance(error, dict) or set(error) != {'document_id', 'reason'}
                or error['document_id'] is not None and not isinstance(error['document_id'], str)
                or not isinstance(error['reason'], str) or not error['reason']):
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_ERROR_SCHEMA')


def _validate_platform_receipts(indexed, receipts):
    if not isinstance(receipts, list):
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_COLLECTIONS')
    seen = set()
    for entry in receipts:
        if not isinstance(entry, dict) or set(entry) != {'document_id', 'public_document_digest', 'raw_receipt'}:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_RECEIPT_SCHEMA')
        did, raw = entry['document_id'], entry['raw_receipt']
        if not isinstance(did, str) or did in seen or did not in indexed:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_RECEIPT_ID')
        seen.add(did)
        shown = indexed[did]
        if (not _platform(shown) or entry['public_document_digest'] != digest(shown)
                or not isinstance(raw, dict) or set(raw) != {'observation_id', 'raw_sha256', 'projection_version'}):
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_RECEIPT_BINDING')
        try:
            oid = str(UUID(raw['observation_id']))
            valid_hash = isinstance(raw['raw_sha256'], str) and _HEX.fullmatch(raw['raw_sha256']) is not None
            valid = (oid == raw['observation_id'] and did == 'doc-' + oid and valid_hash
                     and raw['raw_sha256'] == shown['raw_sha256']
                     and raw['projection_version'] == PROJECTION_VERSION
                     and shown['platform']['observation_id'] == oid
                     and shown['platform']['projection_version'] == PROJECTION_VERSION)
        except (ValueError, TypeError, KeyError, AttributeError):
            valid = False
        if not valid:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_RAW_BINDING')
    return seen


def _prepare_platform(archive, public, original_documents):
    errors, receipts = [], []
    documents = public.get('documents')
    if not isinstance(documents, list) or not isinstance(original_documents, list):
        errors.append({'document_id': None, 'reason': 'document_collection_invalid'})
        documents, original_documents = [], []
    originals, visible = {}, {}
    for collection, target in ((original_documents, originals), (documents, visible)):
        for document in collection:
            if not isinstance(document, dict):
                errors.append({'document_id': None, 'reason': 'document_object_invalid'})
                continue
            did = document.get('document_id')
            if not isinstance(did, str) or not did:
                errors.append({'document_id': None, 'reason': 'document_id_invalid'})
                continue
            if did in target:
                errors.append({'document_id': did, 'reason': 'duplicate_document_id'})
            target[did] = document
    targets = {did for did, document in originals.items() if _platform(document)} | {
        did for did, document in visible.items() if _platform(document)}
    for did in sorted(targets):
        original, shown = originals.get(did), visible.get(did)
        if original is None or shown is None:
            errors.append({'document_id': did, 'reason': 'source_document_not_in_both_views'})
            continue
        try:
            raw_receipt = verify_platform_document(archive, original)
            expected = _view(original)
            if any(key not in shown or shown[key] != value for key, value in expected.items()):
                errors.append({'document_id': did, 'reason': 'public_identity_projection_mismatch'})
                continue
            if _duplicate_raw_ids(archive, original):
                errors.append({'document_id': did, 'reason': 'duplicate_raw_source_ids'})
                continue
            receipts.append({'document_id': did, 'public_document_digest': digest(shown),
                             'raw_receipt': raw_receipt})
        except Exception:
            errors.append({'document_id': did, 'reason': 'archive_document_verification_failed'})
    return originals, visible, receipts, errors


def prepare_fixture_identity(archive, public, original_documents, *, include_local=False):
    """Return a bridge to store as ``public['fixture_identity']``."""
    originals, visible, receipts, errors = _prepare_platform(archive, public, original_documents)
    local_receipts = []
    if include_local:
        verified_platform = {entry['document_id'] for entry in receipts}
        local_targets = {did for did, doc in originals.items() if doc.get('content_kind') == 'platform_local_query'} | {
            did for did, doc in visible.items() if doc.get('content_kind') == 'platform_local_query'}
        for did in sorted(local_targets):
            original = originals.get(did)
            if not original or original.get('content_kind') != 'platform_local_query':
                errors.append({'document_id': did, 'reason': 'local_source_missing'})
                continue
            shown = visible.get(did)
            meta = original.get('local_evidence') or {}
            parent_id = meta.get('source_document_id')
            parent = originals.get(parent_id)
            if shown is None or parent is None or not _platform(parent):
                errors.append({'document_id': did, 'reason': 'local_parent_not_visible_or_platform'})
                continue
            if parent_id not in verified_platform:
                errors.append({'document_id': did, 'reason': 'local_parent_not_verified'})
                continue
            try:
                query = meta['query']
                expected = project_query(archive, [parent], query, _local_operation_id(original),
                                         legacy=meta.get('version') == LOCAL_LEGACY_VERSION)
                expected_view = _local_view(expected)
                if any(shown.get(key) != value for key, value in expected_view.items()):
                    raise ValueError('local_public_projection_mismatch')
                if expected['raw_sha256'] != parent['raw_sha256']:
                    raise ValueError('local_parent_hash_mismatch')
                local_receipts.append({'document_id': did,
                    'public_document_digest': digest(shown),
                    'source_document_id': parent_id,
                    'source_observation_id': parent['platform']['observation_id'],
                    'raw_sha256': parent['raw_sha256'],
                    'query_version': meta.get('version')})
            except Exception:
                errors.append({'document_id': did, 'reason': 'local_archive_projection_failed'})
    local_public = {'_verified_local_sources': local_receipts}
    coverage, local_documents, replaced = coverage_views(public, local_receipts) if local_receipts else ([], [], set())
    resolution = _resolution({**public, **local_public}, receipts, errors,
                             local_documents=local_documents, replaced=replaced)
    if include_local:
        return {'version': VERSION, 'input_digest': digest(_input(public)),
                'verified_sources': receipts, 'verified_local_sources': local_receipts,
                'local_coverage': coverage,
                'errors': errors, 'resolution': resolution}
    return {'version': LEGACY_VERSION, 'input_digest': digest(_input(public)),
            'verified_sources': receipts, 'errors': errors,
            'resolution': _resolution(public, receipts, errors)}


def _validate_legacy(public, wrapper):
    if set(wrapper) != _LEGACY_WRAPPER_FIELDS or wrapper.get('version') != LEGACY_VERSION:
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_SCHEMA')
    if wrapper['input_digest'] != digest(_input(public)):
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_INPUT_CHANGED')
    indexed = _index_documents(public)
    _validate_errors(wrapper['errors'])
    seen = _validate_platform_receipts(indexed, wrapper['verified_sources'])
    if not wrapper['errors'] and seen != {did for did, document in indexed.items() if _platform(document)}:
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_SOURCE_COVERAGE')
    expected = _resolution(public, wrapper['verified_sources'], wrapper['errors'])
    if wrapper['resolution'] != expected:
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_RESOLUTION_CHANGED')
    return deepcopy(expected)


def _validate_local_receipts(indexed, receipts):
    if not isinstance(receipts, list):
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_COLLECTIONS')
    local_ids = set()
    for entry in receipts:
        required = {'document_id','public_document_digest','source_document_id',
                    'source_observation_id','raw_sha256','query_version'}
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_LOCAL_RECEIPT_SCHEMA')
        did = entry['document_id']
        if did in local_ids or did not in indexed or indexed[did].get('content_kind') != 'platform_local_query':
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_LOCAL_RECEIPT_ID')
        local_ids.add(did)
        shown, parent = indexed[did], indexed.get(entry['source_document_id'])
        if not parent or not _platform(parent) or entry['public_document_digest'] != digest(shown):
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_LOCAL_RECEIPT_BINDING')
        meta = shown.get('local_evidence')
        if (shown.get('raw_sha256') != entry['raw_sha256']
                or not isinstance(meta, dict)
                or meta.get('source_document_id') != entry['source_document_id']
                or meta.get('source_observation_id') != entry['source_observation_id']
                or meta.get('version') != entry['query_version']
                or parent.get('raw_sha256') != entry['raw_sha256']
                or parent.get('platform', {}).get('observation_id') != entry['source_observation_id']):
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_LOCAL_RAW_BINDING')
        if entry['query_version'] not in {LOCAL_QUERY_VERSION, LOCAL_LEGACY_VERSION}:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_LOCAL_QUERY_VERSION')
    return local_ids


def validate_fixture_identity(public):
    """Check the archived binding without new archive reads or hidden records."""
    wrapper = public.get('fixture_identity')
    if isinstance(wrapper, dict) and wrapper.get('version') == LEGACY_VERSION:
        return _validate_legacy(public, wrapper)
    if not isinstance(wrapper, dict) or set(wrapper) != _V2_WRAPPER_FIELDS or wrapper.get('version') != VERSION:
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_SCHEMA')
    if wrapper['input_digest'] != digest(_input(public)):
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_INPUT_CHANGED')
    indexed = _index_documents(public)
    _validate_errors(wrapper['errors'])
    seen = _validate_platform_receipts(indexed, wrapper['verified_sources'])
    local_ids = _validate_local_receipts(indexed, wrapper['verified_local_sources'])
    if not wrapper['errors']:
        if seen != {did for did, document in indexed.items() if _platform(document)}:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_SOURCE_COVERAGE')
        if local_ids != {did for did, document in indexed.items()
                         if document.get('content_kind') == 'platform_local_query'}:
            raise ValueError('FIXTURE_IDENTITY_BRIDGE_LOCAL_SOURCE_COVERAGE')
    expected_coverage, local_derived, replaced = coverage_views(public, wrapper['verified_local_sources'])
    if wrapper['local_coverage'] != expected_coverage:
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_LOCAL_COVERAGE')
    public_with_local = deepcopy(public)
    public_with_local['_verified_local_sources'] = wrapper['verified_local_sources']
    expected = _resolution(public_with_local, wrapper['verified_sources'], wrapper['errors'],
                           local_documents=local_derived, replaced=replaced)
    if wrapper['resolution'] != expected:
        raise ValueError('FIXTURE_IDENTITY_BRIDGE_RESOLUTION_CHANGED')
    return deepcopy(expected)


__all__ = ['VERSION', 'LEGACY_VERSION', 'prepare_fixture_identity', 'validate_fixture_identity']
