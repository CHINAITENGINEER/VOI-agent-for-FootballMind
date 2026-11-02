"""Synthetic structured reports and immutable local archives; no live sources."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import runpy

import pytest

from football_agent import runtime_source_support as support
from football_agent.experience_sources import project_body
from football_agent.runtime_fixture_bridge import prepare_fixture_identity

H = runpy.run_path(str(Path(__file__).with_name('test_runtime_fixture_bridge.py')))
URL = 'https://example.test/synthetic-match-report'
CONTRACT = {'version': support.VERSION, 'sources': [{'source_id': 'synthetic-report', 'url': URL}]}


def setup(tmp_path, *, change=None, raw_transform=None, second=None):
    archive, public, originals = H['setup'](tmp_path)
    public['fixture_identity'] = prepare_fixture_identity(archive, public, originals)
    candidate = public['fixture_identity']['resolution']['candidates'][0]
    cutoff = datetime.fromisoformat(public['snapshot_as_of'])
    payload = {'schema': support.SOURCE_SCHEMA,
        'fixture': {'home_team': candidate['home']['name'], 'away_team': candidate['away']['name'],
                    'kickoff': candidate['kickoff'], 'competition': candidate['competition']['name']},
        'reports': [{'subject': 'home', 'predicate': 'qualification_state', 'value': 'already_qualified',
                     'valid_from': (cutoff - timedelta(hours=1)).isoformat(),
                     'valid_until': (cutoff + timedelta(hours=1)).isoformat()}]}
    if change:
        change(payload)
    if second:
        payload['reports'].append({**payload['reports'][0], **second})
    raw = json.dumps(payload, ensure_ascii=False).encode()
    if raw_transform:
        raw = raw_transform(raw)
    observation = archive.save('experience_web', URL, {}, raw, status=200,
        headers={'content-type': 'application/json'}, request_id='synthetic-request',
        context={'url': URL, 'run_id': public['task_id']}, observed_at=cutoff - timedelta(seconds=1))
    original = project_body(raw, URL, 200, 'application/json')
    original.update(document_id='doc-' + observation.id, raw_sha256=observation.content_hash,
        observed_at=observation.observed_at.isoformat(), source_published_at=None,
        request_id=observation.request_id, acquisition_id='synthetic-acquisition',
        selection_operation_id='synthetic-selection', return_observation={'evidence_surface': 'body_text'})
    shown = {k: deepcopy(v) for k, v in original.items() if k != 'text'}
    shown.update(excerpt=original['text'][:4500],
                 excerpt_truncated=len(original['text']) > 4500 or original['text_truncated'])
    originals.append(original)
    public['documents'].append(shown)
    public[support.CAPABILITY_FIELD] = deepcopy(CONTRACT)
    rebind(archive, public, originals)
    return archive, public, originals, payload


def rebind(archive, public, originals):
    public['fixture_identity'] = prepare_fixture_identity(archive, public, originals)


def test_archive_bound_report_supplies_only_its_explicit_field(tmp_path):
    archive, public, originals, _ = setup(tmp_path)
    before = deepcopy(public)
    projected = support.prepare_support(archive, public, originals)
    assert projected['context']['home_reported_qualification_state'] == 'already_qualified'
    assert projected['context']['home_reported_tactical_layout'] == 'unknown'
    assert projected['state']['home']['context']['rotation_risk'] == 'unknown'
    assert projected['state']['home']['context']['lineup_status'] == 'unknown'
    assert len(projected['records']) == 1
    record = projected['records'][0]
    assert record['source_record_path'] == '/reports/0'
    assert record['target_key'] == 'spftll:1001'
    assert record['subject_name'] == '构造甲队'
    assert projected['parsed_document_ids'] == [public['documents'][-1]['document_id']]
    for key in ('world_truth_verified', 'independent_semantic_validation', 'numeric_voi_training_eligible',
                'numeric_voi_calibrated', 'cross_field_inference', 'actor_assertions_used',
                'general_natural_language_support_verified'):
        assert projected[key] is False
    assert public == before


@pytest.mark.parametrize('value', ['defensive', 'attacking', 'must_win'])
def test_actor_and_forged_support_metadata_do_not_change_projection(tmp_path, value):
    archive, public, originals, _ = setup(tmp_path)
    baseline = support.prepare_support(archive, public, originals)
    public['source_claims'] = [{'predicate': 'tactical_layout', 'value': value, 'verified': True}]
    public[support.STATE_FIELD] = {'context': {'home_reported_tactical_layout': value}, 'verified': True}
    assert support.prepare_support(archive, public, originals) == baseline


@pytest.mark.parametrize('field,value', [('home_team', '另一甲队'), ('away_team', '另一乙队'),
    ('competition', '另一赛事'), ('kickoff', '2030-01-01T00:00:00Z')])
def test_same_named_or_previous_fixture_never_cross_binds(tmp_path, field, value):
    archive, public, originals, _ = setup(tmp_path, change=lambda p: p['fixture'].update({field: value}))
    result = support.prepare_support(archive, public, originals)
    assert set(result['context'].values()) == {'unknown'}
    assert any(e['reason'] == 'source_fixture_mismatch' for e in result['excluded_documents'])
    assert len(result['parsed_document_ids']) == 1


def test_kickoff_compares_actual_utc_instant(tmp_path):
    def change(p):
        when = datetime.fromisoformat(p['fixture']['kickoff'])
        from datetime import timezone
        p['fixture']['kickoff'] = when.astimezone(timezone(timedelta(hours=8))).isoformat()
    archive, public, originals, _ = setup(tmp_path, change=change)
    assert support.prepare_support(archive, public, originals)['records'][0]['status'] == 'source_record_matched'


@pytest.mark.parametrize('mode,reason', [('future', 'source_interval_not_started'),
    ('expired', 'source_interval_expired'), ('reversed', 'source_report_interval_invalid'),
    ('naive', 'SOURCE_SUPPORT_TIMESTAMP_TIMEZONE_REQUIRED')])
def test_source_interval_not_actor_time_controls_projection(tmp_path, mode, reason):
    def change(p):
        row = p['reports'][0]
        if mode == 'future':
            row['valid_from'], row['valid_until'] = '2099-01-01T00:00:00Z', '2099-01-02T00:00:00Z'
        elif mode == 'expired':
            row['valid_from'], row['valid_until'] = '2001-01-01T00:00:00Z', '2001-01-02T00:00:00Z'
        elif mode == 'reversed':
            row['valid_until'] = row['valid_from']
        else:
            row['valid_from'] = '2026-01-01T00:00:00'
    archive, public, originals, _ = setup(tmp_path, change=change)
    result = support.prepare_support(archive, public, originals)
    assert set(result['context'].values()) == {'unknown'}
    assert any(e['reason'] == reason for e in result['excluded_documents'])
    if mode in {'future', 'expired'}:
        assert result['records'][0]['exclusion_reason'] == reason


def test_half_open_interval_and_conflicts_keep_all_source_records(tmp_path):
    archive, public, originals, payload = setup(tmp_path, second={'value': 'must_win'})
    result = support.prepare_support(archive, public, originals)
    assert result['context']['home_reported_qualification_state'] == 'unknown'
    assert result['state']['home']['conflicting_fields'] == ['qualification_state']
    assert len(result['records']) == 2
    public['snapshot_as_of'] = payload['reports'][0]['valid_until']
    rebind(archive, public, originals)
    later = support.prepare_support(archive, public, originals)
    assert len(later['records']) == 2
    assert all(r['exclusion_reason'] == 'source_interval_expired' for r in later['records'])


def test_subjects_remain_separate(tmp_path):
    archive, public, originals, _ = setup(tmp_path, second={'subject': 'away', 'value': 'must_win'})
    result = support.prepare_support(archive, public, originals)
    assert result['context']['home_reported_qualification_state'] == 'already_qualified'
    assert result['context']['away_reported_qualification_state'] == 'must_win'


@pytest.mark.parametrize('change,reason', [
    (lambda p: p.update(verified=True), 'source_payload_fields_invalid'),
    (lambda p: p.update(schema='future-version'), 'source_schema_unsupported'),
    (lambda p: p['fixture'].update(target_key='spftll:1001'), 'source_fixture_fields_invalid'),
    (lambda p: p['reports'][0].update(verified=True), 'source_report_fields_invalid'),
    (lambda p: p['reports'][0].update(predicate='qualification_to_tactics'), 'source_report_subject_or_predicate_invalid'),
    (lambda p: p['reports'][0].update(subject='another_team'), 'source_report_subject_or_predicate_invalid'),
    (lambda p: p['reports'][0].update(value=' padded '), 'source_report_value_invalid'),
])
def test_explicit_schema_rejects_asserted_verification_or_inferred_fields(tmp_path, change, reason):
    archive, public, originals, _ = setup(tmp_path, change=change)
    result = support.prepare_support(archive, public, originals)
    assert set(result['context'].values()) == {'unknown'}
    assert any(e['reason'] == reason for e in result['excluded_documents'])
    assert len(result['parsed_document_ids']) == 1


def test_registered_prose_and_unregistered_json_remain_unassessed(tmp_path):
    archive, public, originals, _ = setup(tmp_path,
        raw_transform=lambda raw: '构造甲队已经出线；据此判定全力防守。'.encode())
    result = support.prepare_support(archive, public, originals)
    assert set(result['context'].values()) == {'unknown'}
    assert any(e['reason'] == 'source_format_not_supported_prose_unassessed' for e in result['excluded_documents'])
    public[support.CAPABILITY_FIELD]['sources'] = []
    assert support.prepare_support(archive, public, originals)['records'] == []


@pytest.mark.parametrize('field,value,reason', [
    ('raw_sha256', 55, 'source_document_provenance_missing'),
    ('request_id', None, 'source_document_provenance_missing'),
    ('acquisition_id', None, 'source_document_provenance_missing'),
    ('return_observation', [], 'source_document_evidence_surface_unsupported'),
    ('observed_at', '2099-01-01T00:00:00Z', 'source_document_after_snapshot'),
    ('source_published_at', '2099-01-01T00:00:00Z', 'source_document_after_snapshot'),
    ('observed_at', '2026-01-01T00:00:00', 'source_document_clock_invalid'),
    ('excerpt_truncated', True, 'source_document_incomplete_visible_json'),
])
def test_provenance_errors_exclude_with_explicit_reason(tmp_path, field, value, reason):
    archive, public, originals, _ = setup(tmp_path)
    public['documents'][-1][field] = value
    rebind(archive, public, originals)
    result = support.project_context(public)
    assert not result['records']
    assert any(e['reason'] == reason for e in result['excluded_documents'])


@pytest.mark.parametrize('mutation', ['visible', 'original', 'bytes', 'acquisition'])
def test_archive_and_visible_projection_tampering_rejected(tmp_path, mutation):
    archive, public, originals, _ = setup(tmp_path)
    if mutation == 'visible':
        public['documents'][-1]['excerpt'] = public['documents'][-1]['excerpt'].replace('already_qualified', 'must_win')
    elif mutation == 'original':
        originals[-1]['text'] = originals[-1]['text'].replace('already_qualified', 'must_win')
    elif mutation == 'acquisition':
        public['documents'][-1]['acquisition_id'] = 'forged-acquisition'
    else:
        observation = archive.observations('experience_web')[0]
        (archive.root / observation.payload_path).write_bytes(b'changed')
    rebind(archive, public, originals)
    with pytest.raises((ValueError, OSError)):
        support.prepare_support(archive, public, originals)


@pytest.mark.parametrize('prefix', [b'"schema":"shadow",', b'"unused":NaN,',
    b'"unused":Infinity,', b'"unused":1e999,'])
def test_original_json_ambiguity_rejected_after_transport_has_rewritten_it(tmp_path, prefix):
    archive, public, originals, _ = setup(tmp_path, raw_transform=lambda raw: b'{' + prefix + raw[1:])
    result = support.prepare_support(archive, public, originals)
    assert not result['records']
    assert set(result['context'].values()) == {'unknown'}
    assert result['invalid_raw_json_document_ids'] == [public['documents'][-1]['document_id']]
    assert any(row['reason'] == 'source_raw_json_ambiguous_or_invalid' for row in result['excluded_documents'])
    public[support.STATE_FIELD] = result
    assert support.validate_support(public) == result


@pytest.mark.parametrize('change', [
    lambda c: c.update(version='future'), lambda c: c.update(verified=True),
    lambda c: c['sources'].append(deepcopy(c['sources'][0])),
    lambda c: c['sources'][0].update(url='http://example.test/report'),
    lambda c: c['sources'][0].update(url='https://user:password@example.test/report'),
    lambda c: c['sources'][0].update(url='https://example.test/report#fragment'),
    lambda c: c['sources'][0].update(url='https://example.test/has space'),
    lambda c: c['sources'][0].update(url='https://EXAMPLE.test/report'),
    lambda c: c['sources'][0].update(url='https://example.test'),
    lambda c: c['sources'][0].update(url='https://example.test:8000/report'),
    lambda c: c['sources'][0].update(source_id=''),
])
def test_frozen_source_configuration_rejects_ambiguous_or_unbounded_entries(change):
    contract = deepcopy(CONTRACT)
    change(contract)
    with pytest.raises(ValueError, match='SOURCE_SUPPORT_'):
        support.configuration(contract)


def test_configuration_is_detached_and_policy_contract_requires_v3_semantics():
    frozen = support.configuration(CONTRACT)
    frozen['sources'][0]['source_id'] = 'modified'
    assert CONTRACT['sources'][0]['source_id'] == 'synthetic-report'
    identity = {'version': 'state-bound-runtime-policy-v6', 'configuration': {
        'source_context_contract': CONTRACT, 'context_semantics': support.SEMANTICS,
        'context_projection': 'runtime-observed-context-v3', 'model_descriptor': {
            'source_context_contract': CONTRACT, 'context_semantics': support.SEMANTICS,
            'adapter_version': 'runtime-voi-model-v3'}}}
    assert support.policy_contract(identity) == CONTRACT
    identity['configuration']['model_descriptor']['adapter_version'] = 'runtime-voi-model-v2'
    with pytest.raises(ValueError, match='POLICY_CONTRACT_MISMATCH'):
        support.policy_contract(identity)
    identity['version'] = 'state-bound-runtime-policy-v5'
    assert support.policy_contract(identity) is None


@pytest.mark.parametrize('field,value', [('valid_from', '0001-01-01T00:00:00+23:59'),
                                      ('valid_until', '9999-12-31T23:59:59-23:59')])
def test_extreme_source_time_is_excluded_without_interrupting_projection(tmp_path, field, value):
    archive, public, originals, _ = setup(tmp_path,
        change=lambda p: p['reports'][0].update({field: value}))
    result = support.prepare_support(archive, public, originals)
    assert result['records'] == []
    assert any(e['reason'] == 'SOURCE_SUPPORT_TIMESTAMP_INVALID' for e in result['excluded_documents'])


@pytest.mark.parametrize('mutation', ['context', 'unknown_id', 'unregistered_id', 'duplicate', 'missing'])
def test_saved_quarantine_and_context_are_rebuilt_not_trusted(tmp_path, mutation):
    archive, public, originals, _ = setup(tmp_path)
    saved = support.prepare_support(archive, public, originals)
    public[support.STATE_FIELD] = saved
    assert support.validate_support(public) == saved
    if mutation == 'context':
        saved['context']['home_reported_tactical_layout'] = 'defensive'
    elif mutation == 'unknown_id':
        saved['invalid_raw_json_document_ids'] = ['unknown-document']
    elif mutation == 'unregistered_id':
        saved['invalid_raw_json_document_ids'] = [public['documents'][0]['document_id']]
    elif mutation == 'duplicate':
        saved['invalid_raw_json_document_ids'] = [public['documents'][-1]['document_id']] * 2
    else:
        saved.pop('invalid_raw_json_document_ids')
    with pytest.raises(ValueError, match='SOURCE_SUPPORT_'):
        support.validate_support(public)


def test_raw_reconstruction_detects_self_consistent_removed_quarantine(tmp_path):
    archive, public, originals, _ = setup(tmp_path,
        raw_transform=lambda raw: b'{"schema":"shadow",' + raw[1:])
    verified = support.prepare_support(archive, public, originals)
    assert verified['context']['home_reported_qualification_state'] == 'unknown'
    # A pure selector validates an executor-produced snapshot, not archive
    # bytes. Replacing the whole snapshot can be internally self-consistent;
    # runtime/replay must independently reproduce it from the actual archive.
    forged = support.project_context(public)
    assert forged['context']['home_reported_qualification_state'] == 'already_qualified'
    public[support.STATE_FIELD] = forged
    assert support.validate_support(public) == forged
    assert support.prepare_support(archive, public, originals) == verified
    assert support.prepare_support(archive, public, originals) != public[support.STATE_FIELD]


def test_raw_reconstruction_ignores_forged_quarantine_on_valid_source(tmp_path):
    archive, public, originals, _ = setup(tmp_path)
    expected = support.prepare_support(archive, public, originals)
    public[support.STATE_FIELD] = {'invalid_raw_json_document_ids': [public['documents'][-1]['document_id']]}
    assert support.project_context(public) == expected
    assert support.prepare_support(archive, public, originals) == expected


def test_second_archive_read_failure_is_not_source_json_quarantine(tmp_path, monkeypatch):
    archive, public, originals, _ = setup(tmp_path)
    original_read = archive.read_body
    reads = 0

    def read_with_second_failure(observation):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise ValueError('Raw archive checksum mismatch on second read')
        return original_read(observation)

    monkeypatch.setattr(archive, 'read_body', read_with_second_failure)
    with pytest.raises(ValueError, match='checksum mismatch on second read'):
        support.prepare_support(archive, public, originals)
    assert reads == 2
