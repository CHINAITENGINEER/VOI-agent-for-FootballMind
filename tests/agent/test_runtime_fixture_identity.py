"""Synthetic identity fixtures; no source access and no real match assertions."""
from copy import deepcopy
import json

import pytest

from football_agent.runtime_fixture_identity import resolve_match_identity
from football_agent.structured_sources import PROJECTION_VERSION


def fixture(sid='42', *, home='Manchester United', away='Chelsea',
            kickoff='2026-10-01T18:30:00Z', competition='Premier League'):
    return {'kind': 'fixture', 'source': 'spftll', 'source_id': sid,
        'home': {'name': home, 'source_id': 'home-' + home, 'category': 'men_senior'},
        'away': {'name': away, 'source_id': 'away-' + away, 'category': 'men_senior'},
        'competition': {'name': competition, 'source_id': 'league-' + competition},
        'kickoff': kickoff, 'score': {'home': 999, 'away': 123}, 'odds': 'excluded'}


def document(rows=None, **kw):
    return {'document_id': 'doc-1', 'content_kind': 'platform_structured', 'status': 'readable',
        'observed_at': '2026-10-01T10:00:00Z', 'raw_sha256': 'a' * 64,
        'platform': {'source': 'spftll', 'projection_version': PROJECTION_VERSION,
            'quality': 'normalized', 'observation_id': 'observation-1'},
        'structured_records': rows if rows is not None else [fixture()], **kw}


def run(match='2026-10-02 Manchester United vs Chelsea', docs=None, *, verified=('doc-1',)):
    return resolve_match_identity({'match_input': match, 'snapshot_as_of': '2026-10-01T11:00:00Z',
        'documents': [document()] if docs is None else docs}, verified_document_ids=verified)


@pytest.mark.parametrize('text', [
    '2026-10-02 Manchester United vs Chelsea', 'Chelsea VS Manchester United 2026-10-02',
    '[Premier League] 2026-10-02 Manchester United 对阵 Chelsea',
    '２０２６-１０-０２ Manchester   United 對陣 Chelsea [Premier League]',
])
def test_exact_full_input_resolution(text):
    result = run(text)
    assert result['status'] == 'resolved'
    assert result['target_key'] == 'spftll:42'
    assert result['identity_basis'] == 'source_reported'
    assert result['candidates'][0]['home']['name'] == 'Manchester United'
    assert result['input_constraints']['date_utc_offset'] == '+08:00'
    assert result['evidence'][0]['raw_sha256'] == 'a' * 64
    serialized = json.dumps(result)
    assert '999' not in serialized and 'score' not in serialized and 'odds' not in serialized


def test_no_source_verification_no_automatic_identity():
    assert run(verified=())['status'] == 'unresolved'
    assert run(docs=[{'content_kind': 'html', 'text': json.dumps(fixture())}])['status'] == 'unresolved'


@pytest.mark.parametrize('text,status', [
    ('Manchester United vs Chelsea', 'needs_date'),
    ('2026-02-30 Manchester United vs Chelsea', 'unsupported_input'),
    ('2026-10-02 Manchester United vs Chelsea 2026-10-03', 'unsupported_input'),
    ('Manchester United 2026-10-02 vs Chelsea', 'unsupported_input'),
    ('2026-10-02 Manchester United - Chelsea', 'unsupported_input'),
    ('2026-10-02 Manchester United vs Chelsea vs Arsenal', 'unsupported_input'),
    ('2026-10-02 Chelsea vs Chelsea', 'unsupported_input'),
    ('2026-10-02 Manchester United vs Chelsea [PL] [Cup]', 'unsupported_input'),
])
def test_input_grammar_is_explicit(text, status):
    assert run(text)['status'] == status


@pytest.mark.parametrize('match', [
    '2026-10-02 Man Utd vs Chelsea', '2026-10-02 Manchester United U23 vs Chelsea',
    '2026-10-02 Manchester United vs Chelsea [FA Cup]',
    '2026-10-01 Manchester United vs Chelsea',
    'Please analyze 2026-10-02 Manchester United vs Chelsea',
])
def test_alias_category_date_and_unconsumed_instruction_never_guess(match):
    assert run(match)['target_key'] is None


def test_same_visible_match_under_multiple_ids_is_ambiguous():
    result = run(docs=[document([fixture(), fixture('43')])])
    assert result['status'] == 'ambiguous' and result['target_key'] is None


@pytest.mark.parametrize('change', [
    {'kickoff': '2026-10-02T18:30:00Z'}, {'home': 'Chelsea', 'away': 'Manchester United'},
    {'competition': 'FA Cup'}, {'home': 'Manchester United U23'},
])
def test_same_id_rescheduling_or_identity_conflict_is_not_latest_wins(change):
    result = run(docs=[document([fixture(), fixture(**change)])])
    assert result['status'] == 'conflicting' and result['target_key'] is None


def test_repeated_identical_observations_are_one_candidate_identity():
    assert run(docs=[document([fixture(), fixture()])])['status'] == 'resolved'


@pytest.mark.parametrize('updates', [
    {'text_truncated': True}, {'excerpt_truncated': True}, {'structured_records': None},
    {'status': 'unsupported_content'},
])
def test_partial_visibility_does_not_claim_unique_match(updates):
    assert run(docs=[document(**updates)])['status'] == 'incomplete'


def test_partial_quality_and_projection_version_fail_closed():
    for field, value in [('quality', 'partial'), ('projection_version', 'future-v99')]:
        doc = document()
        doc['platform'][field] = value
        assert run(docs=[doc])['status'] == 'incomplete'


@pytest.mark.parametrize('when', ['2026-10-01T12:00:00Z', '2026-10-01T10:00:00', None])
def test_future_and_unknown_observation_clocks_ignored(when):
    result = run(docs=[document(observed_at=when)])
    assert result['target_key'] is None and result['issues']


def test_current_observation_survives_excluded_future_document():
    docs = [document(), document([fixture('99')], document_id='future', observed_at='2026-10-02T10:00:00Z')]
    assert run(docs=docs, verified=('doc-1', 'future'))['target_key'] == 'spftll:42'


def test_canonical_input_remains_explicit_assertion_without_verified_source():
    result = run('spftll:42', docs=[])
    assert result['status'] == 'canonical_input' and result['target_key'] == 'spftll:42'
    assert result['identity_basis'] == 'canonical_user_input_not_source_verified'
    assert run('spftll:42')['status'] == 'resolved'
    assert run('spftll:0042')['target_key'] is None
    truncated = run('spftll:42', docs=[document(text_truncated=True)])
    assert truncated['status'] == 'canonical_input'
    assert truncated['identity_basis'] == 'canonical_user_input_not_source_verified'


def test_no_match_means_not_observed_not_absent_and_model_identity_ignored():
    public = {'match_input': '2026-10-02 Arsenal vs Chelsea',
        'snapshot_as_of': '2026-10-01T11:00:00Z', 'documents': [document()],
        'match_identity': {'status': 'identified', 'label': 'spftll:42'},
        'analysis_plan': [{'target_key': 'spftll:42'}]}
    before = deepcopy(public)
    result = resolve_match_identity(public, verified_document_ids=('doc-1',))
    assert result['reason'] == 'not_observed_in_source_not_proof_of_absence'
    assert result['target_key'] is None and public == before


@pytest.mark.parametrize('field,value', [('source_id', '0042'), ('kickoff', '2026-10-01T18:30:00'),
    ('home', {'name': ''}), ('source', 'other'), ('home', {'name': 'Manchester United', 'source_id': {}}),
    ('competition', {'name': 'Premier League', 'source_id': []}),
    ('away', {'name': 'Chelsea', 'category': {'claimed': 'senior'}})])
def test_malformed_source_fixture_blocks_text_identity(field, value):
    row = fixture()
    row[field] = value
    assert run(docs=[document([row])])['status'] == 'incomplete'
