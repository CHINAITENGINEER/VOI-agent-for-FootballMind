"""Real immutable archive/replay with fake HTTP transport, never real sources."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import runpy

import pytest

from football_agent.audit import stored_digest as digest
from football_agent.experience import ExperienceService, public_state, now
from football_agent.runtime_fixture_bridge import prepare_fixture_identity, validate_fixture_identity
from football_agent.structured_sources import source_catalog

H = runpy.run_path(str(Path(__file__).with_name('test_structured_sources.py')))
fixture_row, recent_payload = H['fixture_row'], H['recent_payload']


def setup(tmp_path, rows=None, *, kind='recent', match='2026-10-01 构造甲队 vs 构造乙队'):
    payload = recent_payload(rows) if kind == 'recent' else rows
    if kind == 'lottery':
        tools = H['make_tools'](tmp_path / 'archive', [], {kind: payload})
        result = tools.execute({'tool': 'read_url', 'url': source_catalog()['entries'][1]['url']},
            run_id='fixture', reserve_request=lambda url: 'request-1')
        archive = tools.archive
    else:
        result, _, archive = H['fetch_once'](tmp_path / 'archive', payload, kind=kind)
    document = result['document']
    service = ExperienceService(tmp_path / 'service', mode='test_fixture')
    try:
        state = service.create({'match': match})
        state['documents'] = [document]
        public = public_state(state)
        public['snapshot_as_of'] = (now() + timedelta(seconds=1)).isoformat()
        # Identity-related public flag added with this bridge integration.
        public['documents'][0]['text_truncated'] = document.get('text_truncated', False)
    finally:
        service.close()
    return archive, public, [document]


def bind(archive, public, originals):
    public['fixture_identity'] = prepare_fixture_identity(archive, public, originals)
    return validate_fixture_identity(public)


def test_real_archive_verification_and_identity_resolution(tmp_path):
    archive, public, originals = setup(tmp_path)
    before = deepcopy(public)
    wrapper = prepare_fixture_identity(archive, public, originals)
    assert public == before and not wrapper['errors']
    assert wrapper['resolution']['target_key'] == 'spftll:1001'
    assert len(wrapper['verified_sources']) == 1
    public['fixture_identity'] = wrapper
    assert validate_fixture_identity(public) == wrapper['resolution']
    assert len(archive.observations()) == 1


@pytest.mark.parametrize('field,change', [
    ('structured_records', []), ('excerpt', 'invented'), ('excerpt_truncated', True),
    ('text_truncated', True), ('observed_at', '2020-01-01T00:00:00Z'),
    ('raw_sha256', 'b' * 64), ('request_id', 'forged-request'),
    ('url', 'https://example.com/forged'), ('status', 'empty_structured'),
    ('content_kind', 'html'), ('source_published_at', '2020-01-01T00:00:00Z'),
])
def test_visible_identity_mutation_blocks_canonical_and_text_targets(tmp_path, field, change):
    archive, public, originals = setup(tmp_path)
    public['documents'][0][field] = change
    for match in (public['match_input'], 'spftll:1001'):
        public['match_input'] = match
        result = bind(archive, public, originals)
        assert result['status'] == 'incomplete' and result['target_key'] is None
        assert result['reason'] == 'archive_identity_verification_failed'


def test_platform_metadata_mutation_is_detected(tmp_path):
    archive, public, originals = setup(tmp_path)
    public['documents'][0]['platform']['quality'] = 'partial'
    assert bind(archive, public, originals)['status'] == 'incomplete'


def test_hidden_records_never_enter_public_identity(tmp_path):
    rows = [fixture_row(matchId=1000+i, home='其他甲队'+str(i)) for i in range(80)]
    rows[-1]['home'] = '构造甲队'
    archive, public, originals = setup(tmp_path, rows)
    assert '构造甲队' not in public['documents'][0]['excerpt']
    wrapper = prepare_fixture_identity(archive, public, originals)
    assert not wrapper['errors'] and wrapper['resolution']['status'] == 'incomplete'
    assert wrapper['resolution']['target_key'] is None
    assert '构造甲队' not in json.dumps(wrapper['verified_sources'], ensure_ascii=False)
    public['documents'][0]['structured_records'].append({'kind': 'fixture', 'source_id': '1079'})
    assert prepare_fixture_identity(archive, public, originals)['errors']


@pytest.mark.parametrize('kind', ['recent', 'lottery', 'detail'])
def test_duplicate_raw_ids_detected_before_normalizer_hides_them(tmp_path, kind):
    row = fixture_row()
    if kind == 'recent':
        payload = [row, deepcopy(row)]
    elif kind == 'lottery':
        payload = {'code': 0, 'data': [{'list': [row, deepcopy(row)]}]}
    else:
        payload = {'code': 0, 'data': {**row, 'homeBattleList': [deepcopy(row)]}}
    archive, public, originals = setup(tmp_path, payload, kind=kind)
    wrapper = prepare_fixture_identity(archive, public, originals)
    assert any(error['reason'] == 'duplicate_raw_source_ids' for error in wrapper['errors'])
    assert wrapper['resolution']['target_key'] is None


@pytest.mark.parametrize('where', ['public', 'originals'])
def test_duplicate_document_ids_are_verification_errors(tmp_path, where):
    archive, public, originals = setup(tmp_path)
    collection = public['documents'] if where == 'public' else originals
    collection.append(deepcopy(collection[0]))
    wrapper = prepare_fixture_identity(archive, public, originals)
    assert any(e['reason'] == 'duplicate_document_id' for e in wrapper['errors'])
    assert wrapper['resolution']['target_key'] is None


def test_omitted_public_source_document_cannot_hide_conflicting_source(tmp_path):
    archive, public, originals = setup(tmp_path)
    public['documents'] = []
    assert bind(archive, public, originals)['target_key'] is None
    assert public['fixture_identity']['errors']


def test_missing_or_corrupt_archive_is_bounded_error_without_raw_content(tmp_path):
    archive, public, originals = setup(tmp_path)
    observation = archive.observations()[0]
    (archive.root / observation.payload_path).write_bytes(b'raw-private-content-that-must-not-leak')
    wrapper = prepare_fixture_identity(archive, public, originals)
    assert wrapper['errors'][0]['reason'] == 'archive_document_verification_failed'
    assert 'raw-private-content' not in json.dumps(wrapper)
    assert wrapper['resolution']['target_key'] is None


@pytest.mark.parametrize('mutation', ['digest', 'resolution', 'receipt_hash', 'receipt_id', 'receipt_duplicate',
                                    'receipt_missing', 'error_shape', 'version', 'extra_field'])
def test_saved_wrapper_tampering_is_rejected(tmp_path, mutation):
    archive, public, originals = setup(tmp_path)
    bind(archive, public, originals)
    wrapper = public['fixture_identity']
    if mutation == 'digest': wrapper['input_digest'] = '0' * 64
    elif mutation == 'resolution': wrapper['resolution']['target_key'] = 'spftll:900'
    elif mutation == 'receipt_hash': wrapper['verified_sources'][0]['raw_receipt']['raw_sha256'] = '0' * 64
    elif mutation == 'receipt_id': wrapper['verified_sources'][0]['document_id'] = 'other'
    elif mutation == 'receipt_duplicate': wrapper['verified_sources'].append(deepcopy(wrapper['verified_sources'][0]))
    elif mutation == 'receipt_missing': wrapper['verified_sources'] = []
    elif mutation == 'error_shape': wrapper['errors'] = ['bad']
    elif mutation == 'version': wrapper['version'] = 'future'
    else: wrapper['secret_field'] = 'unexpected'
    with pytest.raises(ValueError, match='FIXTURE_IDENTITY_BRIDGE'):
        validate_fixture_identity(public)


def test_public_document_mutation_after_binding_is_rejected_even_if_input_digest_updated(tmp_path):
    archive, public, originals = setup(tmp_path)
    bind(archive, public, originals)
    public['documents'][0]['title'] = 'changed'
    public['fixture_identity']['input_digest'] = digest({k: public.get(k) for k in
                                                      ('match_input', 'snapshot_as_of', 'documents')})
    with pytest.raises(ValueError, match='RECEIPT_BINDING'):
        validate_fixture_identity(public)


def test_nonplatform_documents_ignored_and_no_document_canonical_input_preserved(tmp_path):
    public = {'match_input': 'spftll:42', 'snapshot_as_of': now().isoformat(),
              'documents': [{'document_id': 'web-1', 'content_kind': 'html', 'text': 'arbitrary'}]}
    wrapper = prepare_fixture_identity(None, public, deepcopy(public['documents']))
    assert not wrapper['errors'] and not wrapper['verified_sources']
    public['fixture_identity'] = wrapper
    assert validate_fixture_identity(public)['status'] == 'canonical_input'


def test_missing_wrapper_is_not_an_unverified_identity_fallback():
    with pytest.raises(ValueError, match='SCHEMA'):
        validate_fixture_identity({'match_input': 'spftll:42'})
