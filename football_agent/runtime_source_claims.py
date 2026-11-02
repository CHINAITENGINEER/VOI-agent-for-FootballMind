"""Actor-extracted, source-quoted state; never world facts or gain labels.

The legacy projection validates identity, visible quotations and operational
clocks, not semantic entailment or an actor's interpretation of source validity.
Explicit structured-support mode instead derives numerical context from the
executor's independently parsed source records; actor assertions remain notes.
"""
from copy import deepcopy
from datetime import datetime, timezone
import re

from jsonschema import validate

from .audit import stored_digest as digest
from .runtime_fixture_bridge import validate_fixture_identity
from .source_context_adapter import build_source_backed_context

VERSION = 'runtime-source-claims-v1'
SEMANTICS = 'actor_source_claims_not_world_facts_v1'
CAPABILITY_FIELD = 'source_claim_capability'
FIELD = 'source_claims'
HISTORY_FIELD = 'source_claim_history'
STATE_FIELD = 'source_claim_state'
PREDICATES = ('qualification_state', 'tactical_layout', 'rotation_risk', 'lineup_status')
NUMERIC_KEYS = tuple(f'{side}_reported_{field}' for side in ('home', 'away')
                     for field in PREDICATES[:2])


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 160}
QUOTE = {'type': 'string', 'minLength': 1, 'maxLength': 900}
CLAIM_SCHEMA = _object({
    'target_key': TEXT, 'subject': {'type': 'string', 'enum': ['home', 'away']},
    'subject_name': TEXT, 'predicate': {'type': 'string', 'enum': list(PREDICATES)},
    'value': TEXT, 'document_id': TEXT, 'quote': QUOTE,
    'validity': {'anyOf': [{'type': 'null'}, _object({
        'from': TEXT, 'until': TEXT, 'quote': QUOTE})]},
})
CLAIMS_SCHEMA = {'type': 'array', 'maxItems': 8, 'items': CLAIM_SCHEMA}


def capability():
    return {'version': VERSION, 'semantics': SEMANTICS,
            'subject_binding': 'resolved_source_fixture_home_or_away_exact_name',
            'validity': 'actor_extracted_explicit_source_interval_half_open_or_unknown',
            'world_truth_verified': False, 'numeric_voi_training_eligible': False,
            'extra_model_calls': 0}


def ensure_capability(public):
    if public.get(CAPABILITY_FIELD) != capability():
        raise ValueError('SOURCE_CLAIM_CAPABILITY_CHANGED')


PROMPT = '''
SOURCE ASSERTION STATE CONTRACT:
Return source_claims ONCE, shared by every candidate in this same planning call.
Use [] when no NEW explicit source assertion can be extracted. There is no
mandatory predicate or query order. Extract only what acquired body text says;
never infer rotation, weakness or tactical intent from qualification status.
Each assertion binds the resolved fixture target_key and its exact source home
or away name. The unique visible quote must contain that name. Until fixture
identity is source-resolved, keep source_claims=[]. Search snippets are discovery
only. Cite body excerpts, never hidden text, user beliefs or model assessments.
Use qualification_state, tactical_layout, rotation_risk or lineup_status only
when explicitly discussed; values are concise source-assertion categories.
Separate each team. Do not combine a home qualification claim with away tactics.
validity is null unless the source explicitly states an applicable interval.
If present, copy an exact visible quote supporting from/until timestamps with
timezones. Never manufacture an interval from retrieval time or kickoff. Missing
intervals remain inspectable statements but are unknown in timed value features.
These are ACTOR_PROVISIONAL_NOT_GOLD extractions: quotes do not prove entailment,
source truth, applicability or independent evaluation. Keep conflicting claims;
do not replace or delete old ones. Submit only new assertions, not entries already
in source_claim_history. Current shared assertions may affect this same selection;
their committed history appears only after the decision is accepted.
'''


def extend_schema(schema):
    result = deepcopy(schema)
    result['properties'][FIELD] = deepcopy(CLAIMS_SCHEMA)
    result['required'].append(FIELD)
    return result


def extend_protocol(protocol, *, source_support=False):
    suffix = '-source-claims-v1'
    prompt = protocol.prompt + PROMPT
    if source_support:
        suffix += '-structured-support-v1'
        prompt += '''\nSTRUCTURED SOURCE CONTEXT MODE:
source_support_state is independently parsed from acquired source records under
the frozen source_support_capability. Its fixture/time-scoped fields are source
reports, not world truth or evidence of positive VoI. Your source_claims remain
provisional notes; changing their predicate, value or validity cannot update the
numeric context. Ordinary prose has unassessed support in this mode. You may
use it to propose further information needs and new queries, without pretending
it has gained numerical eligibility. Do not infer rotation or tactics from
qualification. No predicate or source is mandatory; preserve budgeted choice.
'''
    return type('SourceClaimProtocol', (), {'version': protocol.version + suffix,
                'schema': extend_schema(protocol.schema), 'prompt': prompt})


def _time(value):
    if not isinstance(value, str):
        raise ValueError('SOURCE_CLAIM_TIME_REQUIRED')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('SOURCE_CLAIM_TIMEZONE_REQUIRED')
    return parsed.astimezone(timezone.utc)


def _identity(public):
    return validate_fixture_identity(public)


def _document(public, did):
    rows = [doc for doc in public['documents'] if doc.get('document_id') == did]
    if len(rows) != 1:
        raise ValueError('SOURCE_CLAIM_DOCUMENT_NOT_UNIQUE')
    return rows[0]


def _source_binding(doc):
    return {key: deepcopy(doc.get(key)) for key in ('document_id', 'url', 'raw_sha256',
        'request_id', 'acquisition_id', 'selection_operation_id', 'observed_at',
        'source_published_at', 'return_observation')} | {'visible_document_digest': digest(doc)}


def _quote(doc, text):
    if not text.strip() or doc.get('excerpt', '').count(text) != 1:
        raise ValueError('SOURCE_CLAIM_QUOTE_NOT_VISIBLE_UNIQUE')


def _bind(public, claim, identity):
    if identity['status'] != 'resolved' or claim['target_key'] != identity['target_key']:
        raise ValueError('SOURCE_CLAIM_RESOLVED_TARGET_REQUIRED')
    names = {row[claim['subject']]['name'] for row in identity['candidates']
             if row['target_key'] == identity['target_key']}
    if names != {claim['subject_name']} or claim['subject_name'] not in claim['quote']:
        raise ValueError('SOURCE_CLAIM_SUBJECT_MISMATCH')
    if any(claim[key] != claim[key].strip() for key in ('subject_name', 'value')):
        raise ValueError('SOURCE_CLAIM_PADDED_VALUE')
    doc = _document(public, claim['document_id'])
    if (doc.get('status') != 'readable' or doc.get('content_kind') == 'discovery'
            or (doc.get('return_observation') or {}).get('evidence_surface')
            not in {'body_text', 'structured_records', 'local_structured_records'}):
        raise ValueError('SOURCE_CLAIM_BODY_REQUIRED')
    required = ('acquisition_id', 'selection_operation_id')
    if doc.get('content_kind') != 'platform_local_query':
        required += ('request_id',)
    if (not all(doc.get(k) for k in required)
            or not re.fullmatch(r'[0-9a-f]{64}', doc.get('raw_sha256') or '')):
        raise ValueError('SOURCE_CLAIM_ACQUISITION_BINDING_REQUIRED')
    observed, cutoff = _time(doc.get('observed_at')), _time(public['snapshot_as_of'])
    if observed > cutoff or (doc.get('source_published_at') is not None
                            and _time(doc['source_published_at']) > cutoff):
        raise ValueError('SOURCE_CLAIM_AFTER_SNAPSHOT')
    _quote(doc, claim['quote'])
    interval = claim['validity']
    if interval is not None:
        if _time(interval['from']) >= _time(interval['until']):
            raise ValueError('SOURCE_CLAIM_VALIDITY_ORDER')
        _quote(doc, interval['quote'])
    assertion = {'claim': deepcopy(claim), 'source': _source_binding(doc),
                 'extracted_at': public['snapshot_as_of'],
                 'identity_digest': digest(identity)}
    assertion['assertion_id'] = digest(assertion)
    return assertion


def _history(public):
    history = public.get(HISTORY_FIELD)
    if not isinstance(history, list):
        raise ValueError('SOURCE_CLAIM_HISTORY_REQUIRED')
    assertions = []
    for record in history:
        if (not isinstance(record, dict) or record.get('version') != VERSION
                or record.get('record_digest') != digest({k: v for k, v in record.items() if k != 'record_digest'})
                or record.get('semantics') != SEMANTICS
                or record.get('task_id') != public.get('task_id')
                or record.get('label_status') != 'ACTOR_PROVISIONAL_NOT_GOLD'
                or record.get('world_truth_verified') is not False
                or record.get('numeric_voi_training_eligible') is not False):
            raise ValueError('SOURCE_CLAIM_HISTORY_RECORD_CHANGED')
        for assertion in record['assertions']:
            if assertion['assertion_id'] != digest({k: v for k, v in assertion.items() if k != 'assertion_id'}):
                raise ValueError('SOURCE_CLAIM_ASSERTION_CHANGED')
            claim = assertion['claim']
            validate(claim, CLAIM_SCHEMA)
            doc = _document(public, claim['document_id'])
            if assertion['source'] != _source_binding(doc):
                raise ValueError('SOURCE_CLAIM_HISTORICAL_DOCUMENT_CHANGED')
            if _time(assertion['extracted_at']) > _time(public['snapshot_as_of']):
                raise ValueError('SOURCE_CLAIM_HISTORY_FROM_FUTURE')
            assertions.append(assertion)
    return assertions


def verify_archived_claims(archive, public, claims, originals):
    """Verify quoted web bytes; platform/local archives use the existing bridge.

    Pure selectors consume the resulting trusted public snapshot. Runtime and
    durable replay call this boundary before accepting an actor assertion.
    """
    from uuid import UUID
    from football_spider.models.contracts import Observation
    from .experience_sources import project_body
    for did in sorted({claim['document_id'] for claim in claims}):
        shown = _document(public, did)
        rows = [doc for doc in originals if doc.get('document_id') == did]
        if len(rows) != 1:
            raise ValueError('SOURCE_CLAIM_ORIGINAL_DOCUMENT_REQUIRED')
        doc = rows[0]
        if doc.get('content_kind') in {'platform_structured', 'platform_local_query'}:
            wrapper = public['fixture_identity']
            verified = wrapper['verified_sources'] + wrapper.get('verified_local_sources', [])
            if did not in {item['document_id'] for item in verified}:
                raise ValueError('SOURCE_CLAIM_PLATFORM_ARCHIVE_NOT_VERIFIED')
            continue
        oid = str(UUID(did.removeprefix('doc-')))
        paths = list((archive.root/'raw'/'experience_web').glob('*/*_' + oid + '.observation.json'))
        if len(paths) != 1:
            raise ValueError('SOURCE_CLAIM_ARCHIVE_DOCUMENT_NOT_UNIQUE')
        obs = Observation.model_validate_json(paths[0].read_bytes())
        if (obs.id != oid or obs.context.get('url') != shown['url']
                or obs.context.get('run_id') != public['task_id']
                or obs.request_id != shown['request_id']
                or obs.content_hash != shown['raw_sha256']
                or _time(shown['observed_at']) != obs.observed_at
                or shown.get('source_published_at') is not None):
            raise ValueError('SOURCE_CLAIM_ARCHIVE_METADATA_CHANGED')
        expected = project_body(archive.read_body(obs), shown['url'], obs.http_status,
            obs.response_headers.get('content-type', ''), obs.error_class)
        if any(doc.get(key) != value for key, value in expected.items()):
            raise ValueError('SOURCE_CLAIM_ARCHIVE_BODY_CHANGED')


def validate_claims(response, public):
    ensure_capability(public)
    if response.get('task_id') != public.get('task_id'):
        raise ValueError('SOURCE_CLAIM_TASK_MISMATCH')
    claims = response[FIELD]
    validate(claims, CLAIMS_SCHEMA)
    history = _history(public)
    seen = {digest(item['claim']) for item in history}
    identity = _identity(public)
    bound = []
    for claim in claims:
        key = digest(claim)
        if key in seen:
            raise ValueError('SOURCE_CLAIM_DUPLICATE')
        seen.add(key)
        bound.append(_bind(public, claim, identity))
    return bound


def claim_record(public, response, operation_id):
    assertions = validate_claims(response, public)
    if not assertions:
        return None
    record = {'version': VERSION, 'semantics': SEMANTICS,
              'operation_id': operation_id, 'task_id': public['task_id'],
              'decision_digest': digest(response), 'snapshot_digest': digest(public),
              'assertions': assertions, 'label_status': 'ACTOR_PROVISIONAL_NOT_GOLD',
              'world_truth_verified': False, 'independent_semantic_validation': False,
              'numeric_voi_training_eligible': False, 'realized_decision_gain': None}
    record['record_digest'] = digest(record)
    return record


def project_context(public, response=None):
    """Rebuild current state; expiration/conflict never retain stale old values."""
    ensure_capability(public)
    assertions = _history(public)
    if response is not None:
        assertions += validate_claims(response, public)
    if 'source_support_capability' in public:
        from .runtime_source_support import validate_support
        result = validate_support(public)
        result.update(assertion_count=len(assertions), actor_assertions_used=False,
                      subject_and_interval_entailment_verified=False,
                      ignored_assertions=[{'assertion_id': row['assertion_id'],
                          'reason': 'actor_note_not_numeric_source_record'} for row in assertions])
        return result
    identity = _identity(public)
    cutoff = _time(public['snapshot_as_of'])
    rows = {'home': [], 'away': []}
    ignored = []
    for assertion in assertions:
        claim, source = assertion['claim'], assertion['source']
        interval = claim['validity']
        reason = None
        if identity['status'] != 'resolved' or identity['target_key'] != claim['target_key']:
            reason = 'current_identity_not_resolved_to_claim_target'
        elif {row[claim['subject']]['name'] for row in identity['candidates']} != {claim['subject_name']}:
            reason = 'current_subject_changed'
        elif interval is None:
            reason = 'source_applicability_unknown'
        elif cutoff < _time(interval['from']):
            reason = 'source_interval_not_started'
        elif cutoff >= _time(interval['until']):
            reason = 'source_interval_expired'
        if reason:
            ignored.append({'assertion_id': assertion['assertion_id'], 'reason': reason})
            continue
        rows[claim['subject']].append({'source_id': assertion['assertion_id'],
            'source_version': VERSION, 'content_digest': source['raw_sha256'],
            'observed_at': source['observed_at'], 'available_at': assertion['extracted_at'],
            'quality': 'ACTOR_PROVISIONAL_NOT_GOLD',
            'values': {claim['predicate']: claim['value']}})
    sides, values = {}, {}
    for side in rows:
        sides[side] = build_source_backed_context(rows[side],
            target_key=identity['target_key'] or 'unresolved',
            prediction_cutoff=public['snapshot_as_of'], required_fields=PREDICATES)['binding']
        values.update({f'{side}_reported_{key}': sides[side]['context'][key]
                       for key in PREDICATES[:2]})
    return {'version': VERSION, 'semantics': SEMANTICS, 'context': values,
            'state': sides, 'ignored_assertions': ignored,
            'assertion_count': len(assertions), 'snapshot_as_of': public['snapshot_as_of'],
            'subject_and_interval_entailment_verified': False,
            'world_truth_verified': False, 'numeric_voi_training_eligible': False,
            'cross_field_inference': False}
