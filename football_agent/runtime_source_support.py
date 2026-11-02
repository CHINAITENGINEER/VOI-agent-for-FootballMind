"""Exact fixture-scoped reports from explicitly registered structured sources.

Source JSON has exactly ``schema``, ``fixture`` and ``reports``. ``fixture`` has
home_team, away_team, kickoff and competition; each report has subject,
predicate, value, valid_from and valid_until. No actor-provided verification,
target identifier, quotation or inferred football consequence is accepted.

This is a source-format adapter, not natural-language entailment or world-truth
verification. Ordinary prose remains unsupported. Pure projection consumes an
already trusted visible snapshot; runtime/replay must use prepare_support to
reconstruct its inputs from archived response bytes before numerical use.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import re
from urllib.parse import urlsplit
from uuid import UUID

from .audit import stored_digest as digest
from .runtime_fixture_bridge import validate_fixture_identity
from .source_context_adapter import build_source_backed_context

VERSION = 'runtime-source-support-v1'
SOURCE_SCHEMA = 'football-fixture-reports-v1'
SEMANTICS = 'fixture_scoped_structured_source_reports_not_world_facts_v1'
CAPABILITY_FIELD = 'source_support_capability'
STATE_FIELD = 'source_support_state'
PREDICATES = ('qualification_state', 'tactical_layout', 'rotation_risk', 'lineup_status')
MAX_SOURCES = 32
MAX_REPORTS = 100


def _object(value, keys, reason):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(reason)
    return value


def _text(value, maximum=160):
    return isinstance(value, str) and 0 < len(value) <= maximum and value == value.strip()


def _time(value):
    if not _text(value, 64):
        raise ValueError('SOURCE_SUPPORT_TIMESTAMP_REQUIRED')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('SOURCE_SUPPORT_TIMESTAMP_INVALID') from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('SOURCE_SUPPORT_TIMESTAMP_TIMEZONE_REQUIRED')
    try:
        return result.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ValueError('SOURCE_SUPPORT_TIMESTAMP_INVALID') from exc


def configuration(contract):
    from .experience_sources import clean_url
    _object(contract, {'version', 'sources'}, 'SOURCE_SUPPORT_CONFIGURATION_FIELDS')
    if contract['version'] != VERSION:
        raise ValueError('SOURCE_SUPPORT_CONFIGURATION_VERSION')
    sources = contract['sources']
    if type(sources) is not list or len(sources) > MAX_SOURCES:
        raise ValueError('SOURCE_SUPPORT_CONFIGURATION_SOURCES')
    ids, urls = set(), set()
    for item in sources:
        _object(item, {'source_id', 'url'}, 'SOURCE_SUPPORT_SOURCE_FIELDS')
        if not _text(item['source_id'], 120) or not _text(item['url'], 2500):
            raise ValueError('SOURCE_SUPPORT_SOURCE_BOUNDED_TEXT')
        try:
            url = urlsplit(item['url'])
            valid = (url.scheme == 'https' and url.hostname and not url.username
                     and not url.password and not url.fragment
                     and not any(c.isspace() or ord(c) < 32 for c in item['url']))
            url.port
        except ValueError as exc:
            raise ValueError('SOURCE_SUPPORT_SOURCE_HTTPS_URL') from exc
        if not valid:
            raise ValueError('SOURCE_SUPPORT_SOURCE_HTTPS_URL')
        try:
            executable = clean_url(item['url'])
        except ValueError as exc:
            raise ValueError('SOURCE_SUPPORT_SOURCE_EXECUTABLE_URL') from exc
        if executable != item['url']:
            raise ValueError('SOURCE_SUPPORT_SOURCE_CANONICAL_URL_REQUIRED')
        if item['source_id'] in ids or item['url'] in urls:
            raise ValueError('SOURCE_SUPPORT_SOURCE_DUPLICATE')
        ids.add(item['source_id'])
        urls.add(item['url'])
    return deepcopy(contract)


def capability(contract):
    return configuration(contract)


def ensure_capability(public):
    return configuration(public.get(CAPABILITY_FIELD))


def policy_contract(identity):
    if type(identity) is not dict or identity.get('version') != 'state-bound-runtime-policy-v6':
        return None
    try:
        config = identity['configuration']
        model = config['model_descriptor']
        contract = configuration(config['source_context_contract'])
        if (model['source_context_contract'] != contract
                or model['adapter_version'] != 'runtime-voi-model-v3'
                or model['context_semantics'] != SEMANTICS
                or config['context_semantics'] != SEMANTICS
                or config['context_projection'] != 'runtime-observed-context-v3'):
            raise ValueError('mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('SOURCE_SUPPORT_POLICY_CONTRACT_MISMATCH') from exc
    return contract


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('SOURCE_SUPPORT_DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('SOURCE_SUPPORT_NONFINITE_JSON')


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('SOURCE_SUPPORT_NONFINITE_JSON')
    return parsed


def _loads(text):
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_constant, parse_float=_float)


def _source_binding(doc):
    return {key: deepcopy(doc.get(key)) for key in ('document_id', 'url', 'raw_sha256',
        'request_id', 'acquisition_id', 'selection_operation_id', 'observed_at',
        'source_published_at', 'return_observation')}


def _document_reason(doc, cutoff):
    if doc.get('status') != 'readable':
        return 'source_document_not_readable'
    if doc.get('excerpt_truncated') is not False or doc.get('text_truncated') is not False:
        return 'source_document_incomplete_visible_json'
    if not isinstance(doc.get('excerpt'), str) or len(doc['excerpt']) > 4500:
        return 'source_document_visible_json_invalid'
    if (not all(_text(doc.get(k), 200) for k in ('document_id', 'request_id',
                                               'acquisition_id', 'selection_operation_id'))
            or not isinstance(doc.get('raw_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', doc['raw_sha256'])):
        return 'source_document_provenance_missing'
    if (type(doc.get('return_observation')) is not dict or doc['return_observation'].get('evidence_surface')
            not in {'body_text', 'structured_records'}):
        return 'source_document_evidence_surface_unsupported'
    try:
        observed = _time(doc.get('observed_at'))
        published = doc.get('source_published_at')
        if observed > cutoff or published is not None and _time(published) > cutoff:
            return 'source_document_after_snapshot'
    except ValueError:
        return 'source_document_clock_invalid'
    return None


def _fixture(payload, identity):
    fixture = _object(payload, {'home_team', 'away_team', 'kickoff', 'competition'},
                      'source_fixture_fields_invalid')
    if not all(_text(fixture[k]) for k in ('home_team', 'away_team', 'competition')):
        raise ValueError('source_fixture_names_invalid')
    kickoff = _time(fixture['kickoff'])
    if identity['status'] != 'resolved':
        raise ValueError('source_fixture_identity_unresolved')
    candidates = [c for c in identity['candidates'] if c['target_key'] == identity['target_key']]
    if not candidates or any(
            c['home']['name'] != fixture['home_team'] or c['away']['name'] != fixture['away_team']
            or c['competition']['name'] != fixture['competition'] or _time(c['kickoff']) != kickoff
            for c in candidates):
        raise ValueError('source_fixture_mismatch')
    return deepcopy(fixture)


def _project_context(public, contract=None, *, invalid_raw_json_ids=()):
    """Projection with quarantine IDs supplied only by the trusted caller."""
    contract = configuration(public.get(CAPABILITY_FIELD) if contract is None else contract)
    if CAPABILITY_FIELD in public and public[CAPABILITY_FIELD] != contract:
        raise ValueError('SOURCE_SUPPORT_CAPABILITY_CHANGED')
    identity = validate_fixture_identity(public)
    cutoff = _time(public.get('snapshot_as_of'))
    documents = public.get('documents')
    if type(documents) is not list:
        raise ValueError('SOURCE_SUPPORT_DOCUMENT_COLLECTION_REQUIRED')
    by_url = {source['url']: source for source in contract['sources']}
    records, excluded, parsed_ids, seen = [], [], [], set()
    rows = {'home': [], 'away': []}
    for doc in documents:
        if type(doc) is not dict or not _text(doc.get('document_id'), 200):
            raise ValueError('SOURCE_SUPPORT_DOCUMENT_ID_REQUIRED')
        did = doc['document_id']
        if did in seen:
            raise ValueError('SOURCE_SUPPORT_DUPLICATE_DOCUMENT_ID')
        seen.add(did)
        source = by_url.get(doc['url']) if isinstance(doc.get('url'), str) else None
        issue = lambda reason, **extra: excluded.append({'document_id': did, 'reason': reason, **extra})
        if source is None:
            issue('source_url_not_registered')
            continue
        if doc.get('content_kind') != 'json':
            issue('source_format_not_supported_prose_unassessed')
            continue
        parsed_ids.append(did)
        if did in invalid_raw_json_ids:
            issue('source_raw_json_ambiguous_or_invalid')
            continue
        reason = _document_reason(doc, cutoff)
        if reason:
            issue(reason)
            continue
        try:
            payload = _loads(doc['excerpt'])
            _object(payload, {'schema', 'fixture', 'reports'}, 'source_payload_fields_invalid')
            if payload['schema'] != SOURCE_SCHEMA:
                raise ValueError('source_schema_unsupported')
            fixture = _fixture(payload['fixture'], identity)
            if type(payload['reports']) is not list or len(payload['reports']) > MAX_REPORTS:
                raise ValueError('source_reports_bounded_list_required')
        except (ValueError, TypeError, RecursionError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else 'source_json_invalid'
            issue(reason)
            continue
        for index, report in enumerate(payload['reports']):
            try:
                _object(report, {'subject', 'predicate', 'value', 'valid_from', 'valid_until'},
                        'source_report_fields_invalid')
                if report['subject'] not in ('home', 'away') or report['predicate'] not in PREDICATES:
                    raise ValueError('source_report_subject_or_predicate_invalid')
                if not _text(report['value']):
                    raise ValueError('source_report_value_invalid')
                start, until = _time(report['valid_from']), _time(report['valid_until'])
                if start >= until:
                    raise ValueError('source_report_interval_invalid')
            except (ValueError, TypeError) as exc:
                issue(str(exc) if isinstance(exc, ValueError) else 'source_report_invalid', report_index=index)
                continue
            reason = 'source_interval_not_started' if cutoff < start else 'source_interval_expired' if cutoff >= until else None
            record = {'adapter_version': VERSION, 'source_schema': SOURCE_SCHEMA,
                'source_id': source['source_id'], 'source': _source_binding(doc),
                'source_record_path': '/reports/' + str(index), 'fixture': fixture,
                'target_key': identity['target_key'], 'subject_name': fixture[report['subject'] + '_team'],
                **deepcopy(report), 'status': 'excluded' if reason else 'source_record_matched',
                'exclusion_reason': reason, 'world_truth_verified': False,
                'numeric_voi_training_eligible': False}
            record['record_id'] = digest(record)
            records.append(record)
            if reason:
                issue(reason, report_index=index, record_id=record['record_id'])
                continue
            rows[report['subject']].append({'source_id': record['record_id'],
                'source_version': VERSION, 'content_digest': doc['raw_sha256'],
                'observed_at': doc['observed_at'], 'available_at': doc['observed_at'],
                'quality': 'SOURCE_REPORTED_NOT_WORLD_TRUTH',
                'values': {report['predicate']: report['value']}})
    sides, values = {}, {}
    for side in rows:
        sides[side] = build_source_backed_context(rows[side],
            target_key=identity['target_key'] or 'unresolved', prediction_cutoff=public['snapshot_as_of'],
            required_fields=PREDICATES)['binding']
        values.update({f'{side}_reported_{key}': sides[side]['context'][key] for key in PREDICATES[:2]})
    return {'version': VERSION, 'semantics': SEMANTICS, 'context': values, 'state': sides,
        'records': records, 'excluded_documents': excluded, 'parsed_document_ids': parsed_ids,
        'invalid_raw_json_document_ids': sorted(invalid_raw_json_ids),
        'snapshot_as_of': public['snapshot_as_of'], 'contract': contract,
        'world_truth_verified': False, 'independent_semantic_validation': False,
        'numeric_voi_training_eligible': False, 'numeric_voi_calibrated': False,
        'cross_field_inference': False, 'actor_assertions_used': False,
        'general_natural_language_support_verified': False}


def project_context(public, contract=None):
    """Rebuild visible content alone, ignoring all saved support-state fields.

    This pure helper has no access to ambiguity erased by transport JSON
    rewriting. Runtime/replay use prepare_support; selectors use
    validate_support on the executor-produced public snapshot.
    """
    return _project_context(public, contract)


def validate_support(public):
    """Recompute every field of an executor-produced selection snapshot.

    A pure selector trusts the executor's raw-quarantine input. Runtime and
    durable replay independently reconstruct those IDs from archived bytes;
    actor responses never supply the public snapshot or quarantine IDs.
    """
    saved = public.get(STATE_FIELD)
    if type(saved) is not dict:
        raise ValueError('SOURCE_SUPPORT_STATE_REQUIRED')
    invalid_ids = saved.get('invalid_raw_json_document_ids')
    if (type(invalid_ids) is not list or any(not isinstance(did, str) for did in invalid_ids)
            or len(invalid_ids) != len(set(invalid_ids))):
        raise ValueError('SOURCE_SUPPORT_QUARANTINE_IDS_INVALID')
    contract = ensure_capability(public)
    urls = {source['url'] for source in contract['sources']}
    documents = public.get('documents')
    if type(documents) is not list:
        raise ValueError('SOURCE_SUPPORT_DOCUMENT_COLLECTION_REQUIRED')
    permitted = {doc.get('document_id') for doc in documents if type(doc) is dict
                 and isinstance(doc.get('url'), str) and doc['url'] in urls
                 and doc.get('content_kind') == 'json' and isinstance(doc.get('document_id'), str)}
    if set(invalid_ids) - permitted:
        raise ValueError('SOURCE_SUPPORT_QUARANTINE_IDS_INVALID')
    expected = _project_context(public, contract, invalid_raw_json_ids=invalid_ids)
    if saved != expected:
        raise ValueError('SOURCE_SUPPORT_STATE_CHANGED')
    return expected


def prepare_support(archive, public, originals):
    """Verify raw JSON, source metadata and the actual public viewport before use."""
    from football_spider.models.contracts import Observation
    from .runtime_source_claims import verify_archived_claims
    projection = project_context(public)
    ids = projection['parsed_document_ids']
    invalid_ids = []
    verify_archived_claims(archive, public, [{'document_id': did} for did in ids], originals)
    for did in ids:
        shown = next(doc for doc in public['documents'] if doc['document_id'] == did)
        original = next(doc for doc in originals if doc['document_id'] == did)
        if (_source_binding(shown) != _source_binding(original)
                or shown.get('content_kind') != original.get('content_kind')
                or shown.get('status') != original.get('status')
                or shown.get('excerpt') != original.get('text', '')[:4500]
                or shown.get('text_truncated') != original.get('text_truncated', False)
                or shown.get('excerpt_truncated') != (len(original.get('text', '')) > 4500
                                                      or original.get('text_truncated', False))):
            raise ValueError('SOURCE_SUPPORT_VISIBLE_PROJECTION_CHANGED')
        oid = str(UUID(did.removeprefix('doc-')))
        paths = list((archive.root / 'raw' / 'experience_web').glob('*/*_' + oid + '.observation.json'))
        if len(paths) != 1:
            raise ValueError('SOURCE_SUPPORT_ARCHIVE_DOCUMENT_NOT_UNIQUE')
        observation = Observation.model_validate_json(paths[0].read_bytes())
        raw_body = archive.read_body(observation)
        try:
            raw = _loads(raw_body.decode('utf-8'))
        except (ValueError, UnicodeError, RecursionError):
            # The legacy archive verifier already reconstructed transport text
            # above. This is malformed/ambiguous source data, not trusted
            # state corruption; preserve the document but isolate its values.
            invalid_ids.append(did)
            continue
        # Normal transport projection sorts/rewrites JSON; compare full text
        # without using hidden values as model-visible evidence.
        projected = json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if original.get('text') != projected[:12000] or original.get('text_truncated') != (len(projected) > 12000):
            raise ValueError('SOURCE_SUPPORT_ARCHIVE_JSON_PROJECTION_CHANGED')
    return _project_context(public, invalid_raw_json_ids=invalid_ids)
