"""Versioned identity proof for the observed Gnistan men's league fixture.

Club schedule, its actual federation calendar link and a native match response
must agree. Calendar DTSTAMP/DTEND are not publication or final-whistle clocks.
No result or regulation rule is inferred from confirming this fixture.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from football_platform.information.fixture_corroboration import compare_assertion
from football_research.acquisition.state import parse_time
from football_research.acquisition.value import _canonical
from football_spider.storage.raw_archive import RawArchive
from .prospective_text import MarketPriorPolicy

VERSION = 'finland-club-fixture-identity-v1'
PARSER_VERSION = 'gnistan-palloliitto-observed-men-fixture-v1'
HOME_URL = 'https://gnistan.fi/'
FIXTURES_URL = 'https://gnistan.fi/miehet/ottelut/'
CALENDAR_URL = 'https://spl.torneopal.fi/calendar/team/60534'
CALENDAR_LINK = CALENDAR_URL + '?t=1741853856737'
REGISTERED_TEAMS = {
    'spftll:team:2429': {'name': 'IF Gnistan', 'slug': 'if-gnistan',
        'provider_names': ('赫尔火花', '格尼斯坦'), 'native_team_id': '60534'},
    'spftll:team:386': {'name': 'FC Inter', 'slug': 'fc-inter',
        'provider_names': ('国际图尔', '图尔库国际'), 'native_team_id': None},
}
_MATCH = re.compile(r'https://tulospalvelu\.palloliitto\.fi/match/([1-9][0-9]*)')
_MAPPING_PROVENANCE = {
    'basis': 'reviewed_provider_native_ids_with_two_ordered_historical_fixture_corroborations',
    'provider_observation_id': '7d47259e-0f05-41dc-8163-10cdd22934a4',
    'provider_raw_sha256': '085d79be980b0cd099e1f1b2136db17e3a16e42ab47a4c0745fef327b41b1449',
    'official_schedule_raw_sha256': '41c0c15cbc288ce27e990cdcc68ac985a72573930d841556b846596d17a8ab7f',
    'corroborating_provider_match_ids': ['2913603', '2913674'],
    'corroboration_is_identity_mapping_not_training_labels': True,
}


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _text(node):
    return ' '.join(node.get_text(' ', strip=True).split())


def _failure(reason, *, disputed=False):
    return {'status': 'DISPUTED' if disputed else 'UNCONFIRMED', 'reason': reason}


def _home(body):
    soup = BeautifulSoup(body, 'html.parser')
    if soup.title is None or _text(soup.title) != 'Etusivu - IF Gnistan':
        return _failure('club_home_identity_missing')
    if not any(a.get('href') == FIXTURES_URL and _text(a) == 'Miesten ottelut'
               for a in soup.find_all('a', href=True)):
        return _failure('official_home_has_no_exact_mens_fixture_entry')
    return {'status': 'EXTRACTED', 'role': 'club_home', 'club': 'IF Gnistan',
            'observed_fixture_url': FIXTURES_URL}


def _schedule(body, fixture):
    soup = BeautifulSoup(body, 'html.parser')
    titles = soup.find_all('h1', class_='entry-title')
    if len(titles) != 1 or _text(titles[0]) != 'Miesten ottelut':
        return _failure('official_mens_fixture_category_missing')
    headings = [h for h in soup.find_all('h2') if _text(h) == 'Mestaruussarja 2026']
    if len(headings) != 1:
        return _failure('official_stage_or_season_missing_or_ambiguous')
    tables = []
    for sibling in headings[0].next_siblings:
        if getattr(sibling, 'name', None) == 'h2':
            break
        if getattr(sibling, 'name', None):
            tables.extend(sibling.find_all('table'))
    if len(tables) != 1:
        return _failure('official_stage_table_missing_or_ambiguous')
    rows = [[_text(cell) for cell in row.find_all('td', recursive=False)]
            for row in tables[0].find_all('tr')]
    expected = ['Päivä', 'PVM.', 'Aika', 'Paikka', 'Koti', 'Vieras', 'Tulos', 'Ennakko', 'Raportti']
    if not rows or rows[0] != expected:
        return _failure('official_table_order_not_recognised')
    local = parse_time(fixture['kickoff']).astimezone(ZoneInfo('Europe/Helsinki'))
    date = f'{local.day}.{local.month}'
    targets = [row for row in rows[1:] if len(row) == 9 and row[1] == date]
    if len(targets) != 1:
        return _failure('official_target_date_missing_or_ambiguous')
    row = targets[0]
    expected_names = [REGISTERED_TEAMS[fixture[side+'_team_id']]['name'] for side in ('home', 'away')]
    weekday = ('Ma', 'Ti', 'Ke', 'To', 'Pe', 'La', 'Su')[local.weekday()]
    if row[:3] != [weekday, date, local.strftime('%H.%M')] or row[4:6] != expected_names:
        return _failure('official_ordered_teams_or_local_clock_conflict', disputed=True)
    if row[6]:
        return _failure('official_target_has_a_result_not_a_prematch_fixture')
    if not any(a.get('href') == CALENDAR_LINK for a in soup.find_all('a', href=True)):
        return _failure('official_schedule_calendar_link_missing')
    return {'status': 'EXTRACTED', 'role': 'mens_schedule', 'season': '2026',
            'stage': 'Mestaruussarja', 'home_name': row[4], 'away_name': row[5],
            'local_kickoff': local.replace(tzinfo=None).isoformat(), 'observed_calendar_url': CALENDAR_LINK,
            'score_present': False, 'table_headers': expected}


def _calendar(body, fixture):
    try:
        lines = []
        for line in body.decode('utf-8-sig').splitlines():
            if not line:
                continue
            if line.startswith((' ', '\t')):
                if not lines:
                    raise ValueError('orphan calendar continuation')
                lines[-1] += line[1:]
            else:
                lines.append(line)
        if not lines or lines[0] != 'BEGIN:VCALENDAR' or lines[-1] != 'END:VCALENDAR':
            raise ValueError('calendar framing missing')
        headers, events, current = {}, [], None
        for line in lines[1:-1]:
            if line == 'BEGIN:VEVENT':
                if current is not None:
                    raise ValueError('nested event')
                current = {}
            elif line == 'END:VEVENT':
                if current is None:
                    raise ValueError('orphan event end')
                events.append(current)
                current = None
            else:
                key, separator, value = line.partition(':')
                target = headers if current is None else current
                if not separator or key in target or key in {'BEGIN', 'END'}:
                    raise ValueError('ambiguous calendar property')
                target[key] = value
        if current is not None:
            raise ValueError('incomplete event')
        if any(headers.get(key) != value for key, value in {
                'VERSION': '2.0', 'NAME': 'IF Gnistan – Veikkausliiga',
                'X-WR-CALNAME': 'IF Gnistan – Veikkausliiga',
                'X-WR-TIMEZONE': 'Europe/Helsinki'}.items()):
            return _failure('calendar_team_competition_or_timezone_conflict', disputed=True)
        start_expected = parse_time(fixture['kickoff']).astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        matches = [e for e in events if e.get('DTSTART') == start_expected]
        if len(matches) != 1:
            return _failure('native_calendar_event_missing_or_ambiguous')
        event = matches[0]
        native = _MATCH.fullmatch(event.get('URL', ''))
        if (native is None or event.get('UID') != native.group(1)+'@spl.torneopal.fi'
                or len([e for e in events if e.get('UID') == event.get('UID')]) != 1):
            return _failure('calendar_native_event_or_url_conflict', disputed=True)
        home, away = [REGISTERED_TEAMS[fixture[side+'_team_id']]['name'] for side in ('home', 'away')]
        required = {'CATEGORIES': 'Veikkausliiga', 'SUMMARY': home+' – '+away+', Veikkausliiga',
                    'DESCRIPTION': event['URL']}
        if any(event.get(key) != value for key, value in required.items()):
            return _failure('native_calendar_identity_conflict', disputed=True)
        if event.get('STATUS') != 'CONFIRMED':
            return _failure('native_calendar_event_not_confirmed')
        start = event.get('DTSTART', '')
        if not re.fullmatch(r'\d{8}T\d{6}Z', start):
            return _failure('explicit_utc_calendar_kickoff_missing')
        kickoff = datetime.strptime(start, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
        if kickoff.strftime('%Y%m%dT%H%M%SZ') != start:
            raise ValueError('noncanonical timestamp')
    except (UnicodeError, ValueError):
        return _failure('invalid_or_ambiguous_calendar')
    return {'status': 'EXTRACTED', 'role': 'federation_calendar',
            'source_event_id': 'palloliitto:match:'+native.group(1), 'source_url': event['URL'],
            'home_name': home, 'away_name': away, 'competition': 'Veikkausliiga',
            'calendar_owner_native_team_id': '60534',
            'kickoff': kickoff.isoformat(), 'timezone': headers['X-WR-TIMEZONE'],
            'local_kickoff': kickoff.astimezone(ZoneInfo(headers['X-WR-TIMEZONE'])).replace(tzinfo=None).isoformat(),
            'calendar_status': event['STATUS'], 'publication_time': None,
            'dtstamp_and_dtend_not_used_as_evidence_clocks': True}


def _registered(fixture):
    return (isinstance(fixture.get('match_id'), str)
        and re.fullmatch(r'spftll:match:[1-9][0-9]*', fixture['match_id']) is not None
        and fixture.get('competition_id') == 'spftll:competition:13'
        and fixture.get('competition_name') == '芬超'
        and fixture.get('season') in (None, '2026')
        and parse_time(fixture['kickoff']).year == 2026
        and {fixture.get('home_team_id'), fixture.get('away_team_id')} == set(REGISTERED_TEAMS)
        and all(fixture.get(side+'_team_name') in REGISTERED_TEAMS[fixture[side+'_team_id']]['provider_names']
                for side in ('home', 'away')))


def _project(observation, body, fixture, *, as_of):
    if observation.observed_at > as_of:
        raise ValueError('Official observation was unavailable at audit time')
    requested = observation.context.get('requested_at')
    if requested is None or parse_time(requested) > observation.observed_at:
        raise ValueError('Official receipt must follow its recorded request')
    functions = {HOME_URL: _home, FIXTURES_URL: _schedule, CALENDAR_URL: _calendar}
    if observation.source != 'open_page' or observation.endpoint not in functions:
        raise ValueError('Expected a registered club schedule or its federation calendar')
    allowed_params = ({}, {'t': '1741853856737'}) if observation.endpoint == CALENDAR_URL else ({},)
    if observation.params not in allowed_params:
        raise ValueError('Unexpected official identity query parameters')
    if observation.http_status != 200 or observation.error_class:
        return _failure('official_response_unavailable')
    if len(body) > 2_000_000 or observation.context.get('response_complete') is False:
        return _failure('official_identity_response_incomplete_or_oversized')
    if not _registered(fixture):
        return _failure('provider_fixture_not_in_registered_2026_mens_pair')
    function = functions[observation.endpoint]
    return function(body) if function is _home else function(body, fixture)


def bind_finland_fixture(initialization_manifest, *, archive_root, observation_ids, as_of):
    """Bind registered men's identities using actual schedule and calendar bytes.

    Source-native event IDs are parsed from the unique matching calendar event.
    Canonical team IDs are explicit scoped name-derived registry identities;
    they do not claim that the missing FC Inter native numeric ID was observed.
    """
    as_of = parse_time(as_of)
    policy = MarketPriorPolicy.from_manifest(initialization_manifest)
    fixture = policy.model.fixture
    if parse_time(policy.model.frozen_at) > as_of:
        raise ValueError('Market initialization was unavailable at identity verification')
    if (not isinstance(observation_ids, (list, tuple)) or not 1 <= len(observation_ids) <= 8
            or any(not isinstance(i, str) or not i for i in observation_ids)
            or len(set(observation_ids)) != len(observation_ids)):
        raise ValueError('Distinct explicit source observations are required')
    archive = RawArchive(Path(archive_root))
    all_observations = archive.observations()
    by_id = {o.id: o for o in all_observations}
    if len(by_id) != len(all_observations) or any(i not in by_id for i in observation_ids):
        raise ValueError('Source observations are missing or ambiguous')
    facts, evidence = [], []
    for ident in sorted(observation_ids):
        observation = by_id[ident]
        facts.append(_project(observation, archive.read_body(observation), fixture, as_of=as_of))
        evidence.append({'observation_id': observation.id, 'raw_sha256': observation.content_hash,
                         'url': observation.endpoint, 'observed_at': observation.observed_at.isoformat()})
    comparison, canonical = None, None
    status, reason = 'UNCONFIRMED', 'required_official_identity_components_missing'
    roles = {role: [f for f in facts if f.get('role') == role and f['status'] == 'EXTRACTED']
             for role in ('mens_schedule', 'federation_calendar')}
    if not _registered(fixture):
        status, reason = 'DISPUTED', 'provider_fixture_not_in_registered_2026_mens_pair'
    elif any(f['status'] == 'DISPUTED' for f in facts):
        status, reason = 'DISPUTED', 'official_order_identity_or_clock_conflict'
    elif any(f['status'] != 'EXTRACTED' for f in facts):
        reason = next(f['reason'] for f in facts if f['status'] != 'EXTRACTED')
    elif any(len(items) > 1 for items in roles.values()):
        reason = 'multiple_official_identity_components_are_ambiguous'
    elif all(len(items) == 1 for items in roles.values()):
        schedule, calendar = roles['mens_schedule'][0], roles['federation_calendar'][0]
        if (any(schedule[key] != calendar[key] for key in ('home_name', 'away_name', 'local_kickoff'))
                or parse_time(calendar['kickoff']) != parse_time(fixture['kickoff'])):
            status, reason = 'DISPUTED', 'provider_schedule_and_calendar_disagree'
        else:
            comparison = compare_assertion({key: fixture[key] for key in (
                'match_id', 'competition_id', 'season', 'home_team_id', 'away_team_id', 'kickoff',
                'home_team_name', 'away_team_name')}, source_event_id=calendar['source_event_id'],
                explicitly_mapped_match_id=fixture['match_id'], fields={
                    'home_team_id': fixture['home_team_id'], 'away_team_id': fixture['away_team_id'],
                    'competition_id': fixture['competition_id'], 'kickoff': calendar['kickoff']},
                evidence_basis='acquired_structured',
                available_at=max(parse_time(e['observed_at']) for e in evidence),
                cutoff=min(as_of, parse_time(policy.model.cutoff)))
            status, reason = comparison['status'], comparison['reason']
            if status == 'CONFIRMED':
                canonical = {key: calendar[key] for key in
                             ('home_name', 'away_name', 'kickoff', 'competition', 'source_url', 'source_event_id')}
                canonical.update(category='senior_men_club', stage=schedule['stage'], season='2026',
                    event_status='EventScheduled', parser_version=PARSER_VERSION,
                    team_id_basis='scoped_registered_official_name_derived_not_native_numeric',
                    category_basis='official_mens_schedule_and_explicit_veikkausliiga_stage',
                    score_period='unknown', settlement_rule='unknown')
                for side in ('home', 'away'):
                    canonical[side+'_id'] = 'finland:veikkausliiga:2026:men:' + REGISTERED_TEAMS[fixture[side+'_team_id']]['slug']
    aliases = {side: list(REGISTERED_TEAMS[fixture[side+'_team_id']]['provider_names'])
                    + [REGISTERED_TEAMS[fixture[side+'_team_id']]['name']]
               if fixture.get(side+'_team_id') in REGISTERED_TEAMS else [] for side in ('home', 'away')}
    native_ids = {side: REGISTERED_TEAMS[fixture[side+'_team_id']]['native_team_id']
                  if fixture.get(side+'_team_id') in REGISTERED_TEAMS else None for side in ('home', 'away')}
    result = {'schema': VERSION, 'status': status, 'reason': reason, 'match_id': fixture['match_id'],
        'initialization_manifest': deepcopy(initialization_manifest), 'initialization_sha256': _digest(initialization_manifest),
        'canonical': canonical, 'aliases': aliases, 'observations': evidence, 'verified_at': as_of.isoformat(),
        'comparison': comparison, 'source_projections': facts,
        'mapping_basis': 'reviewed_stable_provider_ids_ordered_club_schedule_and_native_calendar_event',
        'mapping_provenance': deepcopy(_MAPPING_PROVENANCE), 'native_team_ids': native_ids,
        'native_numeric_team_ids_complete': False,
        'independent_from_initial_quote_provider': True, 'retroactive_actor_evidence': False,
        'settlement_status': 'PENDING', 'settlement_rows': [], 'regulation_rule': None}
    result['binding_sha256'] = _digest(result)
    return result


def verify_finland_identity(binding, *, archive_root, as_of):
    """Recompute the same version's complete proof from immutable raw receipts."""
    if (not isinstance(binding, dict) or binding.get('schema') != VERSION
            or parse_time(binding['verified_at']) > parse_time(as_of)
            or binding.get('binding_sha256') != _digest({k: v for k, v in binding.items() if k != 'binding_sha256'})):
        raise ValueError('Finland identity binding is changed or not yet available')
    rebuilt = bind_finland_fixture(binding['initialization_manifest'], archive_root=archive_root,
        observation_ids=[e['observation_id'] for e in binding['observations']], as_of=binding['verified_at'])
    if rebuilt != binding:
        raise ValueError('Finland identity binding does not reproduce official raw evidence')
    return rebuilt


__all__ = ['VERSION', 'PARSER_VERSION', 'REGISTERED_TEAMS', 'HOME_URL', 'FIXTURES_URL',
           'CALENDAR_URL', 'CALENDAR_LINK', 'bind_finland_fixture', 'verify_finland_identity']
