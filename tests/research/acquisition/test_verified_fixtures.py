from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
from html import escape
import json

import pytest

from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot, json_record
from football_research.data_adapters.verified_fixtures import (
    _digest, SCHEDULE_URL, verify_fixture_identity, verify_identity_binding,
    verify_independent_settlement, main)
from football_spider.storage.raw_archive import RawArchive

NOW = datetime(2026, 10, 1, 8, tzinfo=timezone.utc)
KICKOFF = NOW + timedelta(hours=2)
AFTER = KICKOFF + timedelta(hours=3)
URL = 'https://www.uefa.com/uefanationsleague/match/2047946--croatia-vs-spain/'


def manifest():
    fixture = ResearchMatch('spftll:match:m1', 'spftll:competition:1864', None, 'spftll:team:1',
        'spftll:team:2', KICKOFF, source='spftll', status='scheduled',
        competition_name='欧国联', home_team_name='克罗地亚', away_team_name='西班牙',
        fixture_available_at=NOW-timedelta(minutes=2), input_manifest=('observation:fixture',))
    quote = ResearchOddsSnapshot(fixture.match_id, NOW-timedelta(minutes=1), 2., 3., 4.,
        input_manifest=('observation:quote',))
    return MarketPriorPolicy(json_record(fixture), json_record(quote), frozen_at=NOW,
        prediction_cutoff=KICKOFF-timedelta(minutes=1)).manifest


def source_objects(status='Upcoming', kickoff=KICKOFF):
    event = {'@type': 'SportsEvent', 'url': URL, 'name': 'Croatia vs Spain',
        'description': 'UEFA Nations League 2027 League phase: Croatia-Spain',
        'startDate': kickoff.isoformat(), 'eventStatus': 'EventMovedOnline' if status == 'Finished' else 'EventScheduled',
        'homeTeam': {'@type': 'SportsTeam', 'name': 'Croatia',
                     'sameAs': 'https://www.uefa.com/uefanationsleague/teams/56370--croatia/'},
        'awayTeam': {'@type': 'SportsTeam', 'name': 'Spain',
                     'sameAs': 'https://www.uefa.com/uefanationsleague/teams/122--spain/'}}
    match = {'id': '2047946', 'status': status, 'seasonYear': '2027',
        'homeTeam': {'id': '56370', 'internationalName': 'Croatia'},
        'awayTeam': {'id': '122', 'internationalName': 'Spain'},
        'competition': {'id': '2014', 'displayName': 'UEFA Nations League', 'age': 'Adult',
                        'sex': 'Male', 'sportsType': 'Football', 'teamCategory': 'National'},
        'round': {'displayName': 'League phase', 'mode': 'Group'},
        'kickOffTime': {'dateTime': kickoff.isoformat()}}
    if status == 'Finished':
        match['fullTimeAt'] = (kickoff+timedelta(minutes=110)).isoformat()
    return event, match


def body(objects=None, status='Upcoming'):
    event, match = objects or source_objects(status)
    return ('<html><script type="application/ld+json">' + json.dumps({'mainEntity': event})
            + '</script><div data-options="' + escape(json.dumps({'match': match}), quote=True)
            + '"></div></html>').encode()


def save(archive, content, *, endpoint=URL, at=NOW+timedelta(minutes=1), status=200):
    return archive.save('open_page', endpoint, {}, content, status=status, observed_at=at,
        context={'requested_at': (at-timedelta(seconds=1)).isoformat()})


def identity(tmp_path, content=None, **kwargs):
    archive = RawArchive(tmp_path/'identity')
    obs = save(archive, content or body())
    binding = verify_fixture_identity(manifest(), archive_root=archive.root, observation_ids=[obs.id],
        aliases=kwargs.pop('aliases', {'home': ['克罗地亚', 'Croatia'], 'away': ['西班牙', 'Spain']}),
        as_of=NOW+timedelta(minutes=2), **kwargs)
    return archive, binding


def result(archive, *, text='Croatia 1-0 Spain', at=AFTER):
    return save(archive, ('<html><a href="'+URL+'">'+text+'</a></html>').encode(),
                endpoint=SCHEDULE_URL, at=at)


def verify(binding, identity_archive, result_archive, **kwargs):
    return verify_independent_settlement(binding, identity_archive_root=identity_archive.root,
        result_archive_root=result_archive.root,
        observation_ids=[obs.id for obs in result_archive.observations()],
        as_of=kwargs.get('as_of', AFTER+timedelta(minutes=1)))


def test_identity_binds_exact_canonical_aliases_and_source_manifest(tmp_path):
    archive, binding = identity(tmp_path)
    assert binding['status'] == 'CONFIRMED'
    assert binding['canonical']['category'] == 'senior_men_national'
    assert binding['initialization_manifest'] == manifest()
    assert binding['initialization_manifest']['market_initialization']['fixture']['score_period'] == 'unknown'
    assert verify_identity_binding(binding, archive_root=archive.root, as_of=AFTER) == binding


@pytest.mark.parametrize('kind', ['category', 'team_order', 'kickoff', 'competition'])
def test_source_identity_conflict_has_no_canonical_identity(tmp_path, kind):
    event, match = source_objects()
    if kind == 'category': match['competition']['sex'] = 'Female'
    elif kind == 'competition': match['competition']['id'] = '999'
    elif kind == 'team_order': match['homeTeam'], match['awayTeam'] = match['awayTeam'], match['homeTeam']
    else:
        event['startDate'] = (KICKOFF+timedelta(hours=1)).isoformat()
        match['kickOffTime']['dateTime'] = event['startDate']
    _, binding = identity(tmp_path, body((event, match)))
    assert binding['status'] == 'DISPUTED'
    assert binding['canonical'] is None


def test_alias_declaration_cannot_silently_fuzzy_match_another_country(tmp_path):
    _, binding = identity(tmp_path, aliases={'home': ['克罗地亚', 'Croatia'], 'away': ['西班牙', 'Sweden']})
    assert binding['status'] == 'DISPUTED'
    assert binding['canonical'] is None


def test_binding_hash_rewrite_does_not_authorize_changed_identity(tmp_path):
    archive, binding = identity(tmp_path)
    binding['canonical']['home_name'] = 'France'
    binding['binding_sha256'] = _digest({k: v for k, v in binding.items() if k != 'binding_sha256'})
    with pytest.raises(ValueError, match='does not reproduce'):
        verify_identity_binding(binding, archive_root=archive.root, as_of=AFTER)


def test_raw_identity_mutation_detected_before_settlement(tmp_path):
    archive, binding = identity(tmp_path)
    obs = archive.observations()[0]
    (archive.root/obs.payload_path).write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError, match='checksum'):
        verify(binding, archive, RawArchive(tmp_path/'results'))


def test_actual_pending_status_and_numeric_live_score_do_not_create_labels(tmp_path):
    archive, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    save(results, body(), at=AFTER)
    result(results)
    check = verify(binding, archive, results)
    assert check['status'] == 'PENDING'
    assert check['settlement_rows'] == []


def test_empty_and_unrelated_archives_are_pending(tmp_path):
    archive, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    assert verify(binding, archive, results)['status'] == 'PENDING'
    save(results, body(status='Finished'), endpoint=URL.replace('2047946', '2047948'), at=AFTER)
    save(results, b'<html>not a result</html>', endpoint='https://www.uefa.com/news/', at=AFTER)
    check = verify(binding, archive, results)
    assert check['status'] == 'PENDING'
    assert len(check['ignored']) == 2


def test_complete_raw_independent_finished_result_with_real_receipt_clocks(tmp_path):
    archive, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    save(results, body(status='Finished'), at=AFTER)
    result(results)
    # An unrelated event in the same result capture cannot poison this target.
    save(results, b'<html>other</html>', endpoint=URL.replace('2047946', '2047948'), at=AFTER)
    check = verify(binding, archive, results)
    assert check['status'] == 'VERIFIED'
    row, = check['settlement_rows']
    assert (row['home_goals'], row['away_goals'], row['score_period']) == (1, 0, 'regulation')
    assert row['available_at'] == (AFTER+timedelta(minutes=1)).isoformat()
    assert row['observed_at'] == AFTER.isoformat()
    assert len(row['input_manifest']) == 2
    assert row['identity_binding_sha256'] == binding['binding_sha256']


def test_finished_without_score_is_unconfirmed_not_verified_empty(tmp_path):
    archive, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    save(results, body(status='Finished'), at=AFTER)
    check = verify(binding, archive, results)
    assert check['status'] == 'UNCONFIRMED'
    assert check['settlement_rows'] == []


@pytest.mark.parametrize('state', ['Postponed', 'Cancelled'])
def test_explicit_inactive_status_remains_separate_and_unlabelled(tmp_path, state):
    archive, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    event, match = source_objects(state, KICKOFF+timedelta(days=1))
    save(results, body((event, match)), at=AFTER)
    check = verify(binding, archive, results)
    assert check['status'] == state.upper()
    assert check['settlement_rows'] == []


@pytest.mark.parametrize('kind', ['reversed_score', 'conflicting_revisions', 'wrong_final_team', 'future_whistle'])
def test_result_conflicts_do_not_enter_training(tmp_path, kind):
    archive, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    event, match = source_objects('Finished')
    if kind == 'wrong_final_team': match['awayTeam']['id'] = '999'
    if kind == 'future_whistle': match['fullTimeAt'] = (AFTER+timedelta(days=1)).isoformat()
    save(results, body((event, match)), at=AFTER)
    result(results, text='Spain 0-1 Croatia' if kind == 'reversed_score' else 'Croatia 1-0 Spain')
    if kind == 'conflicting_revisions': result(results, text='Croatia 2-0 Spain', at=AFTER+timedelta(seconds=1))
    check = verify(binding, archive, results)
    assert check['status'] == 'DISPUTED'
    assert check['settlement_rows'] == []


def test_live_score_before_final_whistle_cannot_settle_finished_status(tmp_path):
    archive, binding = identity(tmp_path)
    results = RawArchive(tmp_path/'results')
    save(results, body(status='Finished'), at=AFTER)
    result(results, at=KICKOFF+timedelta(minutes=20))
    check = verify(binding, archive, results)
    assert check['status'] == 'UNCONFIRMED'
    assert check['settlement_rows'] == []


def test_future_observation_or_binding_fails_raw_reverification(tmp_path):
    archive, binding = identity(tmp_path)
    with pytest.raises(ValueError, match='not yet available'):
        verify_identity_binding(binding, archive_root=archive.root, as_of=NOW)
    results = RawArchive(tmp_path/'results')
    save(results, body(), at=AFTER+timedelta(days=1))
    with pytest.raises(ValueError, match='unavailable'):
        verify(binding, archive, results)


def test_offline_cli_never_treats_nonempty_sidecar_as_verified_evidence(tmp_path):
    archive, binding = identity(tmp_path)
    bundle = tmp_path/'bundle.json'
    bundle.write_text(json.dumps({'bindings': {binding['match_id']: binding}}), encoding='utf-8')
    output = tmp_path/'audit.json'
    main(['verify-results', '--identity-bundle', str(bundle), '--identity-archive', str(archive.root),
          '--result-archive', str(tmp_path/'empty'), '--output', str(output)])
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['settlement_rows'] == []
    assert report['reports'][binding['match_id']]['status'] == 'PENDING'
    with pytest.raises(ValueError, match='fresh immutable'):
        main(['verify-results', '--identity-bundle', str(bundle), '--identity-archive', str(archive.root),
              '--result-archive', str(tmp_path/'empty'), '--output', str(output)])
