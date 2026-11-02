from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip

import pytest

from football_research.data_adapters.canada_fixture import (
    CAMP_NAME, CAMP_URL, MATCH_URL, VERSION, _digest,
    bind_canada_fixture, verify_canada_identity,
)
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot, json_record
from football_spider.storage.raw_archive import RawArchive

NOW = datetime(2026, 10, 6, 19, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 10, 7, tzinfo=timezone.utc)
OBSERVED = NOW + timedelta(minutes=1)
AUDIT = NOW + timedelta(minutes=2)
IMAGE = 'https://canadasoccer.com/wp-content/uploads/canada-soccer-files/TEAM_LOGO_{}.png'


def manifest(**changes):
    fixture = ResearchMatch('spftll:match:3097883', 'spftll:competition:1366', None,
        'spftll:team:797', 'spftll:team:795', KICKOFF, source='spftll', status='scheduled',
        competition_name='国际赛', home_team_name='美国', away_team_name='加拿大',
        fixture_available_at=NOW-timedelta(minutes=2), input_manifest=('observation:fixture',))
    row = json_record(fixture)
    row.update(changes)
    quote = ResearchOddsSnapshot(row['match_id'], NOW-timedelta(minutes=1), 1.49, 4., 4.8,
        input_manifest=('observation:quote',))
    return MarketPriorPolicy(row, json_record(quote), frozen_at=NOW,
        prediction_cutoff=KICKOFF-timedelta(minutes=1)).manifest


def body():
    return f'''<html><div class="featured-match-card upcoming-match" id="match-6630">
        <div class="team-home"><img src="{IMAGE.format('USA')}"/><h5>USA</h5></div>
        <p class="match-date">06 Oct 2026</p><p class="match-time">20:00 EDT</p>
        <div class="team-visitor"><img src="{IMAGE.format('CAN')}"/><h5>#CANMNT</h5></div>
        <a href="{MATCH_URL}">Match Preview</a></div>
        <main><div class="match-banner upcoming-match" id="match-banner-6630">
        <a class="back-button" href="{CAMP_URL}">Back to {CAMP_NAME}</a>
        <h1>USA vs. Men's National Team</h1>
        <div class="team-home"><img src="{IMAGE.format('USA')}" alt="USA"/><h2>USA</h2></div>
        <p class="camp-name">{CAMP_NAME}</p><p class="match-date">06 Oct 2026</p>
        <p class="match-time">20:00 EDT</p><div class="team-visitor">
        <img src="{IMAGE.format('CAN')}" alt="Men's National Team"/>
        <h2>Men's National Team</h2></div></div>
        <div id="match-panel-info"><ul class="details-box"><li><span class="label">Kickoff</span>
        <span class="value">06 Oct 2026  | 20:00 EDT</span></li></ul></div>
        <div class="past-match" id="match-6619">Canada 2 Peru 0 Full Time</div></main></html>'''.encode()


def identity(tmp_path, content=None, *, initial=None, endpoint=MATCH_URL,
             at=OBSERVED, requested=None, status=200, source='open_page'):
    archive = RawArchive(tmp_path/'identity')
    obs = archive.save(source, endpoint, {}, content if content is not None else body(), status=status,
        observed_at=at, context={'requested_at': (requested or at-timedelta(seconds=1)).isoformat()})
    binding = bind_canada_fixture(initial or manifest(), archive_root=archive.root,
        observation_ids=[obs.id], as_of=AUDIT)
    return archive, binding


def rehash(binding):
    binding['binding_sha256'] = _digest({k:v for k,v in binding.items() if k!='binding_sha256'})
    return binding


def test_exact_official_identity_roundtrips_without_settlement_inference(tmp_path):
    archive, binding = identity(tmp_path)
    assert binding['schema'] == VERSION and binding['status'] == 'CONFIRMED'
    assert binding['initialization_manifest'] == manifest()
    assert binding['canonical']['source_event_id'] == 'canadasoccer:match:6630'
    assert binding['canonical']['home_name'] == 'United States'
    assert binding['canonical']['away_name'] == 'Canada'
    assert binding['canonical']['kickoff'] == KICKOFF.isoformat()
    assert binding['canonical']['category'] == 'senior_men_national'
    assert binding['canonical']['score_period'] == binding['canonical']['settlement_rule'] == 'unknown'
    assert binding['regulation_rule'] is None
    assert binding['settlement_rows'] == [] and binding['settlement_status'] == 'PENDING'
    assert binding['aliases']['home'][0] == '美国' and binding['aliases']['away'][0] == '加拿大'
    assert binding['aliases']['away'] == ['加拿大', 'Canada', '#CANMNT']
    assert "Men's National Team" not in binding['aliases']['away']
    assert verify_canada_identity(binding, archive_root=archive.root, as_of=AUDIT) == binding


@pytest.mark.parametrize('old,new', [
    ('match-banner-6630', 'match-banner-9999'),
    ('<h2>USA</h2>', '<h2>Mexico</h2>'),
    ("<h2>Men's National Team</h2>", "<h2>Women's National Team</h2>"),
    ("USA vs. Men's National Team", "Men's National Team vs. USA"),
    ('TEAM_LOGO_USA.png', 'TEAM_LOGO_MEX.png'),
    ('#CANMNT', '#CANWNT'),
    ('#CANMNT', '#CANU17'),
    (CAMP_NAME, 'Men’s U-17 international series'),
    (CAMP_URL, CAMP_URL.replace('/1957/', '/2999/')),
    ('06 Oct 2026  | 20:00 EDT', '06 Oct 2026  | 21:00 EDT'),
])
def test_identity_or_category_conflict_never_has_canonical_identity(tmp_path, old, new):
    _, binding = identity(tmp_path, body().replace(old.encode(), new.encode()))
    assert binding['status'] == 'DISPUTED'
    assert binding['canonical'] is None and binding['settlement_rows'] == []


@pytest.mark.parametrize('old,new', [
    ('20:00 EDT', '20:00 ET'),
    ('20:00 EDT', '20:00 EST'),
    ('06 Oct 2026', '32 Oct 2026'),
    ('id="match-6630"', 'id="match-6620"'),
    ('class="match-banner upcoming-match"', 'class="match-banner past-match"'),
    ('<main>', '<section>'),
])
def test_incomplete_ambiguous_or_non_upcoming_dom_is_unconfirmed(tmp_path, old, new):
    _, binding = identity(tmp_path, body().replace(old.encode(), new.encode()))
    assert binding['status'] == 'UNCONFIRMED' and binding['canonical'] is None


@pytest.mark.parametrize('change', [
    {'match_id':'spftll:match:999'}, {'home_team_name':'美洲'},
    {'home_team_id':'spftll:team:795','away_team_id':'spftll:team:797'},
    {'competition_name':'国际友谊女子'}, {'competition_id':'spftll:competition:999'},
    {'kickoff':(KICKOFF+timedelta(hours=1)).isoformat()},
])
def test_explicit_mapping_does_not_generalize_to_another_fixture(tmp_path, change):
    _, binding = identity(tmp_path, initial=manifest(**change))
    assert binding['status'] == 'DISPUTED' and binding['canonical'] is None


@pytest.mark.parametrize('endpoint', [
    'https://canadasoccer.com/match/6631/', 'https://news.canadasoccer.com/match/6630/',
    'https://canadasoccer.com/', 'http://canadasoccer.com/match/6630/',
])
def test_only_exact_native_official_match_url_is_supported(tmp_path, endpoint):
    with pytest.raises(ValueError, match='exact acquired'):
        identity(tmp_path, endpoint=endpoint)


def test_future_receipt_and_request_clock_fail(tmp_path):
    with pytest.raises(ValueError, match='unavailable'):
        identity(tmp_path/'future', at=AUDIT+timedelta(seconds=1))
    with pytest.raises(ValueError, match='recorded request'):
        identity(tmp_path/'reverse', requested=OBSERVED+timedelta(seconds=1))


def test_failure_preserved_without_promoting_embedded_identity(tmp_path):
    _, binding = identity(tmp_path, status=403)
    assert binding['status']=='UNCONFIRMED' and binding['canonical'] is None


@pytest.mark.parametrize('changed', ['canonical','alias','prior','observation'])
def test_binding_rehash_cannot_authorize_changed_identity_or_initialization(tmp_path, changed):
    archive, binding = identity(tmp_path)
    if changed=='canonical': binding['canonical']['away_name']='Mexico'
    elif changed=='alias': binding['aliases']['away'].append('Mexico')
    elif changed=='prior': binding['initialization_manifest']['market_initialization']['probabilities'][0]=0.9
    else: binding['observations'][0]['raw_sha256']='0'*64
    with pytest.raises(ValueError):
        verify_canada_identity(rehash(binding), archive_root=archive.root, as_of=AUDIT)


def test_raw_bytes_are_reverified(tmp_path):
    archive, binding = identity(tmp_path)
    obs=archive.observations()[0]
    (archive.root/obs.payload_path).write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError,match='checksum'):
        verify_canada_identity(binding,archive_root=archive.root,as_of=AUDIT)


def test_distinct_observations_required_and_future_binding_ineligible(tmp_path):
    archive,binding=identity(tmp_path)
    obs=archive.observations()[0]
    with pytest.raises(ValueError,match='Distinct'):
        bind_canada_fixture(manifest(),archive_root=archive.root,observation_ids=[obs.id,obs.id],as_of=AUDIT)
    with pytest.raises(ValueError,match='not yet available'):
        verify_canada_identity(binding,archive_root=archive.root,as_of=NOW)


def test_existing_gateway_dispatch_and_result_rule_remain_explicitly_pending(tmp_path):
    from football_research.data_adapters.verified_fixtures import (
        verify_identity_binding, verify_independent_settlement)
    archive,binding=identity(tmp_path)
    assert verify_identity_binding(binding,archive_root=archive.root,as_of=AUDIT)==binding
    result=verify_independent_settlement(binding,identity_archive_root=archive.root,
        result_archive_root=tmp_path/'empty-results',observation_ids=[],as_of=AUDIT)
    assert result['status']=='PENDING' and result['regulation_rule'] is None
    assert result['settlement_rows']==[]
