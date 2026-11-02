from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip

import pytest

from football_research.data_adapters.finland_fixture import (
    CALENDAR_LINK, CALENDAR_URL, FIXTURES_URL, HOME_URL, VERSION, _digest,
    bind_finland_fixture, verify_finland_identity,
)
from football_research.data_adapters.prospective_text import MarketPriorPolicy
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot, json_record
from football_spider.storage.raw_archive import RawArchive

NOW = datetime(2026, 10, 7, 5, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 10, 7, 16, tzinfo=timezone.utc)
OBSERVED = NOW + timedelta(minutes=1)
AUDIT = NOW + timedelta(minutes=2)


def manifest(**changes):
    fixture = ResearchMatch('spftll:match:3085210', 'spftll:competition:13', None,
        'spftll:team:2429', 'spftll:team:386', KICKOFF, source='spftll', status='scheduled',
        competition_name='芬超', home_team_name='赫尔火花', away_team_name='国际图尔',
        fixture_available_at=NOW-timedelta(minutes=2), input_manifest=('observation:test-fixture',))
    row = json_record(fixture)
    row.update(changes)
    quote = ResearchOddsSnapshot(row['match_id'], NOW-timedelta(minutes=1), 3.58, 3.55, 1.76,
                                input_manifest=('observation:test-quote',))
    return MarketPriorPolicy(row, json_record(quote), frozen_at=NOW,
        prediction_cutoff=datetime.fromisoformat(row['kickoff'])-timedelta(minutes=1)).manifest


def schedule():
    return ('''<html><h1 class="entry-title">Miesten ottelut</h1>
    <h2>Mestaruussarja 2026</h2><figure><table><tbody>
    <tr><td>Päivä</td><td>PVM.</td><td>Aika</td><td>Paikka</td><td>Koti</td><td>Vieras</td>
    <td>Tulos</td><td>Ennakko</td><td>Raportti</td></tr>
    <tr><td>Ke</td><td>7.10</td><td>19.00</td><td>Markku.fi Areena</td><td>IF Gnistan</td>
    <td>FC Inter</td><td></td><td></td><td></td></tr>
    <tr><td>Ma</td><td>2.11</td><td>18.00</td><td>Veritas Stadion</td><td>FC Inter</td>
    <td>IF Gnistan</td><td></td><td></td><td></td></tr></tbody></table></figure>
    <h2>Veikkausliiga 2026</h2><table><tr><td>Historical 2-0 is not a target result</td></tr></table>
    <a href="''' + CALENDAR_LINK + '''">tästä.</a></html>''').encode()


def calendar():
    return ('''BEGIN:VCALENDAR
VERSION:2.0
NAME:IF Gnistan – Veikkausliiga
X-WR-CALNAME:IF Gnistan – Veikkausliiga
X-WR-TIMEZONE:Europe/Helsinki
BEGIN:VEVENT
DTSTAMP:20261007T160000Z
UID:4353542@spl.torneopal.fi
CATEGORIES:Veikkausliiga
SUMMARY:IF Gnistan – FC Inter, Veikkausliiga
URL:https://tulospalvelu.palloliitto.fi/match/4353542
STATUS:CONFIRMED
DESCRIPTION:https://tulospalvelu.palloliitto.fi/match/4353542
DTSTART:20261007T160000Z
DTEND:20261007T180000Z
END:VEVENT
BEGIN:VEVENT
UID:4353552@spl.torneopal.fi
CATEGORIES:Veikkausliiga
SUMMARY:FC Inter – IF Gnistan, Veikkausliiga
URL:https://tulospalvelu.palloliitto.fi/match/4353552
STATUS:CONFIRMED
DESCRIPTION:https://tulospalvelu.palloliitto.fi/match/4353552
DTSTART:20261102T160000Z
END:VEVENT
END:VCALENDAR
''').replace('\n', '\r\r\n').encode()


def identity(tmp_path, *, initial=None, schedule_body=None, calendar_body=None,
             missing=None, at=OBSERVED, requested=None, calendar_params=None, calendar_status=200):
    archive = RawArchive(tmp_path/'identity')
    ids = []
    for role, url, body in [('schedule', FIXTURES_URL, schedule() if schedule_body is None else schedule_body),
                            ('calendar', CALENDAR_URL, calendar() if calendar_body is None else calendar_body)]:
        if role == missing:
            continue
        obs = archive.save('open_page', url, (calendar_params or {}) if role == 'calendar' else {}, body,
            status=calendar_status if role == 'calendar' else 200, observed_at=at,
            context={'requested_at': (requested or at-timedelta(seconds=1)).isoformat()})
        ids.append(obs.id)
    binding = bind_finland_fixture(initial or manifest(), archive_root=archive.root,
                                   observation_ids=ids, as_of=AUDIT)
    return archive, binding


def rehash(binding):
    binding['binding_sha256'] = _digest({k:v for k,v in binding.items() if k!='binding_sha256'})
    return binding


def test_raw_pair_roundtrip_uses_native_event_but_declares_derived_team_ids(tmp_path):
    archive, binding = identity(tmp_path)
    assert binding['status']=='CONFIRMED' and binding['schema']==VERSION
    canonical = binding['canonical']
    assert canonical['source_event_id']=='palloliitto:match:4353542'
    assert canonical['home_id']=='finland:veikkausliiga:2026:men:if-gnistan'
    assert canonical['away_id']=='finland:veikkausliiga:2026:men:fc-inter'
    assert canonical['kickoff']==KICKOFF.isoformat()
    assert canonical['team_id_basis']=='scoped_registered_official_name_derived_not_native_numeric'
    assert binding['native_team_ids']=={'home':'60534','away':None}
    assert binding['native_numeric_team_ids_complete'] is False
    assert binding['regulation_rule'] is None and binding['settlement_rows']==[]
    assert binding['settlement_status']=='PENDING'
    assert canonical['score_period']==canonical['settlement_rule']=='unknown'
    projection = next(f for f in binding['source_projections'] if f.get('role')=='federation_calendar')
    assert projection['publication_time'] is None
    assert projection['dtstamp_and_dtend_not_used_as_evidence_clocks'] is True
    assert verify_finland_identity(binding,archive_root=archive.root,as_of=AUDIT)==binding


def test_reverse_fixture_new_native_uid_is_selected_without_per_match_patch(tmp_path):
    initial = manifest(match_id='spftll:match:998877', home_team_id='spftll:team:386',
        away_team_id='spftll:team:2429', home_team_name='图尔库国际', away_team_name='格尼斯坦',
        kickoff='2026-11-02T16:00:00+00:00')
    _, binding = identity(tmp_path, initial=initial)
    assert binding['status']=='CONFIRMED'
    assert binding['canonical']['source_event_id']=='palloliitto:match:4353552'
    assert binding['canonical']['home_name']=='FC Inter'
    assert binding['canonical']['away_name']=='IF Gnistan'
    assert binding['native_team_ids']=={'home':None,'away':'60534'}


@pytest.mark.parametrize('old,new', [
    ('<td>19.00</td>', '<td>20.00</td>'),
    ('<td>Ke</td>', '<td>Ti</td>'),
    ('<td>IF Gnistan</td>', '<td>Gnistan U17</td>'),
    ('<td>FC Inter</td>', '<td>FC Inter Women</td>'),
    ('<td>Koti</td><td>Vieras</td>', '<td>Vieras</td><td>Koti</td>'),
    ('Mestaruussarja 2026', 'Mestaruussarja 2025'),
    ('Miesten ottelut', 'Naisten ottelut'),
    (CALENDAR_LINK, CALENDAR_LINK.replace('/60534','/99999')),
    ('<td>FC Inter</td><td></td>', '<td>FC Inter</td><td>1-0</td>'),
])
def test_schedule_identity_stage_header_or_live_score_changes_never_confirm(tmp_path, old, new):
    assert old.encode() in schedule()
    _, binding = identity(tmp_path, schedule_body=schedule().replace(old.encode(),new.encode()))
    assert binding['status']!='CONFIRMED' and binding['canonical'] is None
    assert binding['settlement_rows']==[]


@pytest.mark.parametrize('old,new', [
    ('4353542@spl.torneopal.fi','999999@spl.torneopal.fi'),
    ('match/4353542','match/999999'),
    ('IF Gnistan – FC Inter, Veikkausliiga','FC Inter – IF Gnistan, Veikkausliiga'),
    ('X-WR-TIMEZONE:Europe/Helsinki','X-WR-TIMEZONE:UTC'),
    ('CATEGORIES:Veikkausliiga','CATEGORIES:Liigacup'),
    ('STATUS:CONFIRMED','STATUS:CANCELLED'),
    ('DTSTART:20261007T160000Z','DTSTART:20261007T170000Z'),
    ('URL:https://tulospalvelu.palloliitto.fi','URL:https://example.com'),
])
def test_calendar_identity_and_time_changes_never_confirm(tmp_path, old, new):
    _, binding = identity(tmp_path,calendar_body=calendar().replace(old.encode(),new.encode()))
    assert binding['status']!='CONFIRMED' and binding['canonical'] is None


def test_duplicate_matching_calendar_event_is_ambiguous(tmp_path):
    content = calendar()
    event = content.split(b'BEGIN:VEVENT')[1].split(b'END:VEVENT')[0]
    content = content.replace(b'END:VCALENDAR',b'BEGIN:VEVENT'+event+b'END:VEVENT\r\r\nEND:VCALENDAR')
    _, binding = identity(tmp_path,calendar_body=content)
    assert binding['status']!='CONFIRMED'


@pytest.mark.parametrize('changes', [
    {'home_team_name':'阿森纳'}, {'home_team_id':'spftll:team:999'},
    {'competition_id':'spftll:competition:99'}, {'competition_name':'芬超女子'}, {'season':'2025'},
])
def test_registry_scope_never_fuzzy_matches_new_provider_entities(tmp_path, changes):
    _, binding = identity(tmp_path,initial=manifest(**changes))
    assert binding['status']=='DISPUTED' and binding['canonical'] is None


@pytest.mark.parametrize('missing',['schedule','calendar'])
def test_both_independent_components_are_required(tmp_path, missing):
    _, binding = identity(tmp_path,missing=missing)
    assert binding['status']=='UNCONFIRMED' and binding['canonical'] is None


def test_failed_response_cannot_use_its_body(tmp_path):
    _, binding = identity(tmp_path,calendar_status=403)
    assert binding['status']=='UNCONFIRMED'


def test_only_exact_observed_cache_parameter_is_accepted(tmp_path):
    _, binding = identity(tmp_path/'ok',calendar_params={'t':'1741853856737'})
    assert binding['status']=='CONFIRMED'
    with pytest.raises(ValueError,match='query parameters'):
        identity(tmp_path/'bad',calendar_params={'team':'999','t':'1741853856737'})


def test_actual_availability_is_required_not_future_calendar_dtstamp(tmp_path):
    with pytest.raises(ValueError,match='unavailable'):
        identity(tmp_path/'future',at=AUDIT+timedelta(seconds=1))
    with pytest.raises(ValueError,match='recorded request'):
        identity(tmp_path/'reverse',requested=OBSERVED+timedelta(seconds=1))
    archive, _ = identity(tmp_path/'cutoff')
    binding = bind_finland_fixture(manifest(),archive_root=archive.root,
        observation_ids=[o.id for o in archive.observations()],as_of=AUDIT)
    assert binding['comparison']['status']=='CONFIRMED'


@pytest.mark.parametrize('field',['canonical','native','alias','mapping','observation','prior'])
def test_rehash_cannot_authorize_claims_not_reproduced_from_raw(tmp_path,field):
    archive,binding=identity(tmp_path)
    if field=='canonical':binding['canonical']['away_id']='palloliitto:team:999'
    elif field=='native':binding['native_team_ids']['away']='999'
    elif field=='alias':binding['aliases']['home'].append('Arsenal')
    elif field=='mapping':binding['mapping_provenance']['provider_raw_sha256']='0'*64
    elif field=='observation':binding['observations'][0]['raw_sha256']='0'*64
    else:binding['initialization_manifest']['market_initialization']['probabilities'][0]=0.9
    with pytest.raises(ValueError):
        verify_finland_identity(rehash(binding),archive_root=archive.root,as_of=AUDIT)


def test_mutated_raw_is_rejected_and_gateway_keeps_results_pending(tmp_path):
    from football_research.data_adapters.verified_fixtures import verify_identity_binding,verify_independent_settlement
    archive,binding=identity(tmp_path)
    assert verify_identity_binding(binding,archive_root=archive.root,as_of=AUDIT)==binding
    result=verify_independent_settlement(binding,identity_archive_root=archive.root,
        result_archive_root=tmp_path/'no-results',observation_ids=[],as_of=AUDIT)
    assert result['status']=='PENDING' and result['regulation_rule'] is None and result['settlement_rows']==[]
    obs=archive.observations()[0]
    (archive.root/obs.payload_path).write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError,match='checksum'):
        verify_finland_identity(binding,archive_root=archive.root,as_of=AUDIT)
