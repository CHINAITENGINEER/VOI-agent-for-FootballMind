"""Quoted source assertions retain identity, time and provisional boundaries."""
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import runpy

import pytest

from football_agent.audit import stored_digest as digest
from football_agent.runtime_fixture_bridge import prepare_fixture_identity
from football_agent import runtime_source_claims as claims


FIXTURES = runpy.run_path(str(Path(__file__).with_name('test_runtime_fixture_bridge.py')))
QUOTES = {
    ('home', 'qualification_state', 'already_qualified'): '构造甲队已经出线。',
    ('home', 'qualification_state', 'must_win'): '构造甲队仍须获胜。',
    ('away', 'qualification_state', 'must_win'): '构造乙队仍须获胜。',
    ('home', 'tactical_layout', 'defensive'): '构造甲队将使用防守阵型。',
    ('away', 'tactical_layout', 'attacking'): '构造乙队将使用进攻阵型。',
}


def source_public(tmp_path):
    archive, public, originals = FIXTURES['setup'](tmp_path)
    cutoff = datetime.fromisoformat(public['snapshot_as_of'])
    start, until = (cutoff - timedelta(hours=1)).isoformat(), (cutoff + timedelta(hours=1)).isoformat()
    interval_quote = f'这些安排适用于 {start} 至 {until}。'
    text = '\n'.join([*QUOTES.values(), interval_quote])
    body = {'document_id': 'source-body', 'url': 'https://example.test/source-report',
            'title': 'Constructed source statements', 'content_kind': 'html', 'status': 'readable',
            'excerpt': text, 'text_truncated': False, 'excerpt_truncated': False,
            'observed_at': (cutoff - timedelta(seconds=1)).isoformat(),
            'source_published_at': None, 'raw_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'request_id': 'body-request', 'acquisition_id': 'body-acquisition',
            'selection_operation_id': 'body-selection',
            'return_observation': {'evidence_surface': 'body_text'}}
    public['documents'].append(body)
    public[claims.CAPABILITY_FIELD] = claims.capability()
    public[claims.HISTORY_FIELD] = []
    originals.append(deepcopy(body))
    rebind(archive, public, originals)
    assert public['fixture_identity']['resolution']['status'] == 'resolved'
    return archive, public, originals


def rebind(archive, public, originals):
    public['fixture_identity'] = prepare_fixture_identity(archive, public, originals)


def claim(public, *, subject='home', predicate='qualification_state', value='already_qualified',
          validity=True):
    doc = next(d for d in public['documents'] if d['document_id'] == 'source-body')
    interval_quote = doc['excerpt'].splitlines()[-1]
    start, until = interval_quote.removeprefix('这些安排适用于 ').removesuffix('。').split(' 至 ')
    return {'target_key': 'spftll:1001', 'subject': subject,
            'subject_name': '构造甲队' if subject == 'home' else '构造乙队',
            'predicate': predicate, 'value': value, 'document_id': doc['document_id'],
            'quote': QUOTES[(subject, predicate, value)],
            'validity': {'from': start, 'until': until, 'quote': interval_quote} if validity else None}


def response(public, *items):
    return {'task_id': public['task_id'], claims.FIELD: list(items)}


def test_current_source_assertions_are_subject_bound_and_never_gold(tmp_path):
    _, public, _ = source_public(tmp_path)
    before = deepcopy(public)
    submitted = response(public, claim(public), claim(public, subject='away', value='must_win'),
                         claim(public, predicate='tactical_layout', value='defensive'))
    projection = claims.project_context(public, submitted)
    assert projection['context']['home_reported_qualification_state'] == 'already_qualified'
    assert projection['context']['away_reported_qualification_state'] == 'must_win'
    assert projection['context']['home_reported_tactical_layout'] == 'defensive'
    assert projection['context']['away_reported_tactical_layout'] == 'unknown'
    assert projection['state']['home']['context']['rotation_risk'] == 'unknown'
    assert projection['state']['home']['context']['lineup_status'] == 'unknown'
    assert projection['cross_field_inference'] is False
    assert projection['subject_and_interval_entailment_verified'] is False
    assert projection['world_truth_verified'] is False
    assert projection['numeric_voi_training_eligible'] is False
    record = claims.claim_record(public, submitted, 'review-operation')
    assert record['label_status'] == 'ACTOR_PROVISIONAL_NOT_GOLD'
    assert record['independent_semantic_validation'] is False
    assert record['realized_decision_gain'] is None
    assert public == before


def test_missing_applicability_stays_inspectable_but_unknown_for_timed_features(tmp_path):
    _, public, _ = source_public(tmp_path)
    submitted = response(public, claim(public, validity=False))
    projection = claims.project_context(public, submitted)
    assert set(projection['context'].values()) == {'unknown'}
    assert projection['assertion_count'] == 1
    assert projection['ignored_assertions'][0]['reason'] == 'source_applicability_unknown'
    assert claims.claim_record(public, submitted, 'review-operation')['assertions']


def test_conflicting_current_and_historical_claims_become_unknown(tmp_path):
    _, public, _ = source_public(tmp_path)
    record = claims.claim_record(public, response(public, claim(public)), 'earlier-review')
    public[claims.HISTORY_FIELD] = [record]
    projected = claims.project_context(public, response(public, claim(public, value='must_win')))
    assert projected['context']['home_reported_qualification_state'] == 'unknown'
    assert projected['state']['home']['conflicting_fields'] == ['qualification_state']
    assert len(projected['state']['home']['supporting_source_ids']['qualification_state']) == 2


@pytest.mark.parametrize('where', ['current', 'history'])
def test_duplicate_assertions_are_rejected(tmp_path, where):
    _, public, _ = source_public(tmp_path)
    item = claim(public)
    if where == 'history':
        public[claims.HISTORY_FIELD] = [claims.claim_record(public, response(public, item), 'review')]
        submitted = response(public, item)
    else:
        submitted = response(public, item, deepcopy(item))
    with pytest.raises(ValueError, match='SOURCE_CLAIM_DUPLICATE'):
        claims.project_context(public, submitted)


@pytest.mark.parametrize('location,key,value,error', [
    ('claim', 'target_key', 'spftll:777', 'RESOLVED_TARGET'),
    ('claim', 'subject', 'away', 'SUBJECT_MISMATCH'),
    ('claim', 'subject_name', '其他甲队', 'SUBJECT_MISMATCH'),
    ('claim', 'quote', '构造甲队隐藏的句子', 'QUOTE_NOT_VISIBLE_UNIQUE'),
    ('claim', 'value', ' already_qualified', 'PADDED_VALUE'),
    ('document', 'content_kind', 'discovery', 'BODY_REQUIRED'),
    ('document', 'status', 'blocked', 'BODY_REQUIRED'),
    ('document', 'return_observation', {'evidence_surface': 'discovery_metadata'}, 'BODY_REQUIRED'),
    ('document', 'raw_sha256', 'not-a-digest', 'ACQUISITION_BINDING'),
    ('document', 'request_id', None, 'ACQUISITION_BINDING'),
    ('document', 'acquisition_id', None, 'ACQUISITION_BINDING'),
    ('document', 'selection_operation_id', None, 'ACQUISITION_BINDING'),
    ('document', 'observed_at', '2099-01-01T00:00:00Z', 'AFTER_SNAPSHOT'),
    ('document', 'source_published_at', '2099-01-01T00:00:00Z', 'AFTER_SNAPSHOT'),
    ('document', 'observed_at', '2026-10-01T00:00:00', 'TIMEZONE_REQUIRED'),
])
def test_source_claim_provenance_boundaries(tmp_path, location, key, value, error):
    archive, public, originals = source_public(tmp_path)
    item = claim(public)
    target = item if location == 'claim' else public['documents'][-1]
    target[key] = value
    rebind(archive, public, originals)
    with pytest.raises(ValueError, match=error):
        claims.project_context(public, response(public, item))


def test_duplicate_visible_quotation_is_rejected(tmp_path):
    archive, public, originals = source_public(tmp_path)
    item = claim(public)
    public['documents'][-1]['excerpt'] += '\n' + item['quote']
    rebind(archive, public, originals)
    with pytest.raises(ValueError, match='QUOTE_NOT_VISIBLE_UNIQUE'):
        claims.project_context(public, response(public, item))


def test_unconfirmed_canonical_input_does_not_bind_new_source_claim(tmp_path):
    archive, public, originals = source_public(tmp_path)
    item = claim(public)
    public['match_input'] = 'spftll:1001'
    public['documents'] = public['documents'][-1:]
    originals = originals[-1:]
    rebind(archive, public, originals)
    assert public['fixture_identity']['resolution']['status'] == 'canonical_input'
    with pytest.raises(ValueError, match='RESOLVED_TARGET'):
        claims.project_context(public, response(public, item))


@pytest.mark.parametrize('position,expected', [('from', 'already_qualified'), ('until', 'unknown')])
def test_validity_interval_is_half_open_and_history_never_retains_expired_value(tmp_path, position, expected):
    archive, public, originals = source_public(tmp_path)
    item = claim(public)
    # The assertion was extracted before either tested boundary.
    original_clock = datetime.fromisoformat(public['snapshot_as_of'])
    shift = timedelta(hours=2)
    item['validity']['from'] = (original_clock + shift).isoformat()
    item['validity']['until'] = (original_clock + shift + timedelta(hours=1)).isoformat()
    prior_quote = item['validity']['quote']
    item['validity']['quote'] = f"这些安排适用于 {item['validity']['from']} 至 {item['validity']['until']}。"
    doc = public['documents'][-1]
    doc['excerpt'] = doc['excerpt'].replace(prior_quote, item['validity']['quote'])
    doc['raw_sha256'] = hashlib.sha256(doc['excerpt'].encode()).hexdigest()
    originals[-1] = deepcopy(doc)
    rebind(archive, public, originals)
    record = claims.claim_record(public, response(public, item), 'earlier-review')
    public[claims.HISTORY_FIELD] = [record]
    early = claims.project_context(public)
    assert early['ignored_assertions'][0]['reason'] == 'source_interval_not_started'
    public['snapshot_as_of'] = item['validity'][position]
    rebind(archive, public, originals)
    projected = claims.project_context(public)
    assert projected['context']['home_reported_qualification_state'] == expected
    if position == 'until':
        assert projected['ignored_assertions'][0]['reason'] == 'source_interval_expired'


@pytest.mark.parametrize('mutation', ['record', 'assertion', 'document', 'future'])
def test_history_tampering_or_future_extraction_is_rejected(tmp_path, mutation):
    archive, public, originals = source_public(tmp_path)
    record = claims.claim_record(public, response(public, claim(public)), 'review')
    public[claims.HISTORY_FIELD] = [record]
    if mutation == 'record':
        record['world_truth_verified'] = True
        expected = 'HISTORY_RECORD_CHANGED'
    elif mutation == 'assertion':
        record['assertions'][0]['claim']['value'] = 'must_win'
        record['record_digest'] = digest({k: v for k, v in record.items() if k != 'record_digest'})
        expected = 'ASSERTION_CHANGED'
    elif mutation == 'document':
        public['documents'][-1]['excerpt'] += '\nchanged source body'
        rebind(archive, public, originals)
        expected = 'HISTORICAL_DOCUMENT_CHANGED'
    else:
        assertion = record['assertions'][0]
        assertion['extracted_at'] = '2099-01-01T00:00:00Z'
        assertion['assertion_id'] = digest({k: v for k, v in assertion.items() if k != 'assertion_id'})
        record['record_digest'] = digest({k: v for k, v in record.items() if k != 'record_digest'})
        expected = 'HISTORY_FROM_FUTURE'
    with pytest.raises(ValueError, match=expected):
        claims.project_context(public)


def test_cross_task_history_is_rejected_even_with_identical_visible_documents(tmp_path):
    _, public, _ = source_public(tmp_path)
    public[claims.HISTORY_FIELD] = [claims.claim_record(public, response(public, claim(public)), 'review')]
    public['task_id'] = 'another-task-with-copied-documents'
    with pytest.raises(ValueError, match='SOURCE_CLAIM_HISTORY_RECORD_CHANGED'):
        claims.project_context(public)
