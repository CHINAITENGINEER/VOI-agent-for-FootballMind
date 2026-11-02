from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip

import pytest

from football_research.data_adapters.premier_league_fixture import (
    LEEDS_URL, SCHEDULE_URL, TIME_RULE, TITLE, VERSION, bind_premier_league_fixture,
)
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot, json_record
from football_research.data_adapters.verified_fixtures import (
    _digest, verify_identity_binding, verify_independent_settlement,
)
from football_research.data_adapters.fixture_role_identity import resolve_fixture_role_identity, verify_fixture_role_identity
from football_spider.storage.raw_archive import RawArchive

NOW = datetime(2026, 10, 9, 8, 30, tzinfo=timezone.utc)
OBSERVED = NOW + timedelta(hours=3)
AUDIT = OBSERVED + timedelta(minutes=1)
KICKOFF = datetime(2026, 10, 10, 11, 30, tzinfo=timezone.utc)


def manifest(**changes):
    fixture = ResearchMatch('spftll:match:3003899', 'spftll:competition:36', None,
        'spftll:team:19', 'spftll:team:56', KICKOFF, source='spftll', status='scheduled',
        competition_name='英超', home_team_name='阿森纳', away_team_name='利兹联',
        fixture_available_at=NOW-timedelta(minutes=20), input_manifest=('observation:fixture',))
    row = json_record(fixture)
    row.update(changes)
    quote = ResearchOddsSnapshot(row['match_id'], NOW-timedelta(minutes=1), 1.49, 4., 4.8,
        input_manifest=('observation:quote',))
    return MarketPriorPolicy(row, json_record(quote), frozen_at=NOW,
        prediction_cutoff=KICKOFF-timedelta(minutes=1)).manifest


def body(row='12:30 Arsenal v Leeds United (TNT Sports)', heading='Saturday 10 October'):
    return (f'<html><article data-article-id="4675097"><h1>{TITLE}</h1><time>19 Jun 2026</time>'
            f'<div class="article__content"><p>{TIME_RULE}</p><p><strong>{heading}</strong><br/>'
            f'{row}</p></div></article></html>').encode()


def club_body(clock='12:30pm KO', category='Mens'):
    return (f'<html><span>{category}</span><article class="richTextRenderer"><h4>Arsenal vs Leeds United</h4>'
            f'<ul><li>Saturday 10th October 2026</li><li>{clock}</li></ul></article></html>').encode()


def identity(tmp_path, content=None, *, initial=None, club=None, at=OBSERVED,
             requested=None, status=200, endpoint=SCHEDULE_URL, as_of=AUDIT):
    archive = RawArchive(tmp_path/'identity')
    obs = archive.save('open_page', endpoint, {}, content if content is not None else body(),
        status=status, observed_at=at, context={'requested_at': (requested or at-timedelta(seconds=1)).isoformat()})
    ids = [obs.id]
    if club is not None:
        ids.append(archive.save('open_page', LEEDS_URL, {}, club, status=200,
            observed_at=at, context={'requested_at': (at-timedelta(seconds=1)).isoformat()}).id)
    binding = bind_premier_league_fixture(initial or manifest(), archive_root=archive.root,
        observation_ids=ids, as_of=as_of)
    return archive, binding


def test_observed_identity_roundtrip_preserves_honest_clocks_names_and_non_native_ids(tmp_path):
    archive, binding = identity(tmp_path, club=club_body())
    assert binding['schema'] == VERSION and binding['status'] == 'CONFIRMED'
    c = binding['canonical']
    assert (c['home_name'], c['away_name']) == ('Arsenal', 'Leeds United')
    assert c['home_id'] == 'footballmind:england:senior-men-club:arsenal'
    assert c['season'] == '2026/27' and c['category'] == 'senior_men_club'
    assert c['kickoff'] == KICKOFF.isoformat() and c['source_local_kickoff'].endswith('+01:00')
    assert c['source_timezone'] == 'Europe/London'
    assert c['source_native_event_id'] is c['source_native_home_team_id'] is c['source_native_away_team_id'] is None
    assert binding['aliases'] == {'home': ['阿森纳', 'Arsenal'], 'away': ['利兹联', 'Leeds United']}
    assert binding['verified_at'] == AUDIT.isoformat()
    assert binding['retroactive_actor_evidence'] is False
    assert binding['initialization_manifest'] == manifest()
    assert binding['source_projections'][1]['source_timezone'] is None
    assert binding['source_projections'][0]['source_published_at'] is None
    assert verify_identity_binding(binding, archive_root=archive.root, as_of=AUDIT) == binding
    capsule = resolve_fixture_role_identity(binding, archive_root=archive.root, as_of=AUDIT)
    assert capsule['scope']['available_at'] == AUDIT.isoformat()
    assert verify_fixture_role_identity(capsule, as_of=AUDIT)['identity_sha256'] == capsule['identity_sha256']


def test_explicit_weekend_default_is_parsed_in_uk_timezone(tmp_path):
    initial = manifest(home_team_id='spftll:team:20', away_team_id='spftll:team:365',
                       home_team_name='维拉', away_team_name='布伦特', kickoff='2026-10-10T14:00:00+00:00')
    _, binding = identity(tmp_path, body('Aston Villa v Brentford'), initial=initial)
    assert binding['status'] == 'CONFIRMED'
    assert binding['canonical']['clock_basis'] == 'explicit_article_default_weekend_or_midweek_clock'


@pytest.mark.parametrize('content', [
    body().replace(b'Premier League season', b'Premier League 2 season'),
    body().replace(b'UK time', b'local time'),
    body('12:30 Leeds United v Arsenal'),
    body().replace(b'article__content', b'unrelated'),
])
def test_missing_scope_timezone_or_order_is_unconfirmed(tmp_path, content):
    _, binding = identity(tmp_path, content)
    assert binding['status'] == 'UNCONFIRMED' and binding['canonical'] is None


@pytest.mark.parametrize('content', [
    body('13:30 Arsenal v Leeds United'),
    body(heading='Sunday 10 October'),
    body('12:30 GMT Arsenal v Leeds United'),
    body('12:30 Arsenal v Leeds United<br/>13:30 Arsenal v Leeds United'),
])
def test_official_clock_and_revisions_conflicts_are_quarantined(tmp_path, content):
    _, binding = identity(tmp_path, content)
    assert binding['status'] == 'DISPUTED' and binding['canonical'] is None


@pytest.mark.parametrize('changes', [
    {'home_team_name': '阿森纳女足'}, {'home_team_id': 'spftll:team:999'},
    {'competition_id': 'spftll:competition:999'}, {'competition_name': '英超青年'},
    {'season': '2025/26'},
])
def test_undeclared_provider_name_id_competition_or_season_cannot_bind(tmp_path, changes):
    _, binding = identity(tmp_path, initial=manifest(**changes))
    assert binding['status'] == 'DISPUTED' and binding['canonical'] is None


def test_club_clock_disagreement_and_missing_mens_scope_do_not_confirm(tmp_path):
    _, binding = identity(tmp_path/'clock', club=club_body('1:30pm KO'))
    assert binding['status'] == 'DISPUTED' and binding['canonical'] is None
    _, binding = identity(tmp_path/'category', club=club_body(category='Womens'))
    assert binding['status'] == 'UNCONFIRMED' and binding['canonical'] is None


def test_real_observation_cannot_be_backdated_or_inserted_after_cutoff(tmp_path):
    with pytest.raises(ValueError, match='unavailable'):
        identity(tmp_path/'future', as_of=NOW)
    with pytest.raises(ValueError, match='receipt'):
        identity(tmp_path/'request', requested=OBSERVED+timedelta(seconds=1))
    _, binding = identity(tmp_path/'post', as_of=KICKOFF)
    assert binding['status'] == 'UNCONFIRMED' and binding['canonical'] is None


def test_failures_are_retained_and_unsupported_urls_rejected(tmp_path):
    _, binding = identity(tmp_path/'failed', status=503)
    assert binding['status'] == 'UNCONFIRMED' and len(binding['observations']) == 1
    with pytest.raises(ValueError, match='URL'):
        identity(tmp_path/'foreign', endpoint=SCHEDULE_URL.replace('www.', 'untrusted.'))


def test_rehashed_alias_mutation_and_corrupt_raw_fail_reverification(tmp_path):
    archive, binding = identity(tmp_path)
    changed = deepcopy(binding)
    changed['aliases']['home'].append('Unproven alternative')
    changed['binding_sha256'] = _digest({k: v for k, v in changed.items() if k != 'binding_sha256'})
    with pytest.raises(ValueError, match='reproduce'):
        verify_identity_binding(changed, archive_root=archive.root, as_of=AUDIT)
    payload = archive.root/archive.observations()[0].payload_path
    payload.write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError, match='checksum'):
        verify_identity_binding(binding, archive_root=archive.root, as_of=AUDIT)


def test_identity_does_not_admit_outcomes_or_uefa_regulation_rule(tmp_path):
    archive, binding = identity(tmp_path)
    result = verify_independent_settlement(binding, identity_archive_root=archive.root,
        result_archive_root=tmp_path/'results', observation_ids=[], as_of=KICKOFF+timedelta(hours=3))
    assert result['status'] == 'PENDING' and result['regulation_rule'] is None
    assert result['settlement_rows'] == []
