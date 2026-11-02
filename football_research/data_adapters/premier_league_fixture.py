"""Raw-backed identity for the observed October 2026 Premier League schedule.

This adapter performs no HTTP and emits no results. Official article text gives
names and clocks, not native team or match IDs. Its name-based identifiers are
explicit local identities; the exact spftll ID/name aliases are reviewed inputs.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import re
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from football_research.acquisition.state import parse_time
from .prospective_text import MarketPriorPolicy
from .verified_fixtures import _digest, _observations

VERSION = 'premier-league-fixture-identity-v1'
PARSER_VERSION = 'pl-2026-27-october-official-article-dom-v1'
SCHEDULE_URL = 'https://www.premierleague.com/en/news/4675097'
LEEDS_URL = 'https://www.leedsunited.com/en/news/television-picks-made-for-october-fixtures'
TITLE = 'All 380 fixtures for 2026/27 Premier League season'
TIME_RULE = ('The kick-off times of weekend and Bank Holiday matches are 15:00 UK time, '
             'while for midweek matches it is 20:00 unless otherwise stated.')
# Explicit cross-provider declarations. Every English name must also be found
# in the acquired official row; these declarations alone never confirm a match.
TEAMS = {
    '19': ('阿森纳', 'Arsenal'), '56': ('利兹联', 'Leeds United'),
    '20': ('维拉', 'Aston Villa'), '365': ('布伦特', 'Brentford'),
    '24': ('切尔西', 'Chelsea'), '348': ('伯恩茅斯', 'AFC Bournemouth'),
    '68': ('伊普斯', 'Ipswich Town'), '29': ('富勒姆', 'Fulham'),
    '35': ('水晶宫', 'Crystal Palace'), '49': ('诺丁汉', 'Nottingham Forest'),
    '27': ('曼联', 'Manchester United'), '33': ('热刺', 'Tottenham Hotspur'),
    '384': ('赫尔城', 'Hull City'), '31': ('埃弗顿', 'Everton'),
    '25': ('利物浦', 'Liverpool'), '26': ('曼城', 'Manchester City'),
}
_DATE = re.compile(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) (\d{1,2}) October(?: (2026))?')
_ROW = re.compile(r'(?:(\d{2}:\d{2})(?: (GMT))? )?(.+?) v (.+?)(?: \([^)]*\))?\*?')


def _failure(status, reason):
    return {'status': status, 'reason': reason, 'canonical': None}


def _slug(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def _checked_soup(observation, body, as_of):
    if observation.observed_at > parse_time(as_of):
        raise ValueError('Official observation was unavailable at verification')
    if observation.endpoint not in {SCHEDULE_URL, LEEDS_URL} or observation.source != 'open_page' or observation.params:
        raise ValueError('Expected an explicitly supported official acquired HTML URL')
    requested = observation.context.get('requested_at')
    if requested is None or parse_time(requested) > observation.observed_at:
        raise ValueError('Official receipt must follow its recorded request')
    if len(body) > 2_000_000:
        raise ValueError('Official response exceeds bounded parser input')
    if observation.http_status != 200 or observation.error_class:
        return None
    return BeautifulSoup(body, 'html.parser')


def _schedule(soup, names):
    articles = soup.select('article[data-article-id="4675097"]')
    if len(articles) != 1:
        return _failure('UNCONFIRMED', 'official_schedule_article_missing_or_ambiguous')
    article = articles[0]
    title, content = article.select('h1'), article.select('.article__content')
    if len(title) != 1 or title[0].get_text(' ', strip=True) != TITLE or len(content) != 1:
        return _failure('UNCONFIRMED', 'official_competition_season_scope_missing')
    if TIME_RULE not in content[0].get_text(' ', strip=True):
        return _failure('UNCONFIRMED', 'explicit_uk_timezone_and_default_clock_rule_missing')
    matches = []
    for paragraph in content[0].find_all('p', recursive=False):
        lines = [' '.join(line.split()) for line in paragraph.get_text('\n', strip=True).splitlines()]
        if not lines or not (heading := _DATE.fullmatch(lines[0])):
            continue
        try:
            date = datetime(2026, 10, int(heading.group(2)))
        except ValueError:
            return _failure('DISPUTED', 'invalid_official_october_date')
        if date.strftime('%A') != heading.group(1):
            return _failure('DISPUTED', 'official_date_and_weekday_disagree')
        for line in lines[1:]:
            row = _ROW.fullmatch(line)
            if not row or (row.group(3), row.group(4)) != names:
                continue
            clock = row.group(1) or ('15:00' if date.weekday() >= 5 else '20:00')
            try:
                local = datetime.fromisoformat(date.date().isoformat()+'T'+clock).replace(tzinfo=ZoneInfo('Europe/London'))
            except ValueError:
                return _failure('DISPUTED', 'invalid_official_kickoff_clock')
            if row.group(2) and local.utcoffset().total_seconds() != 0:
                return _failure('DISPUTED', 'official_gmt_and_uk_dst_clock_conflict')
            matches.append({'home_name': names[0], 'away_name': names[1],
                'kickoff': local.astimezone(timezone.utc).isoformat(), 'source_local_kickoff': local.isoformat(),
                'original_time_text': lines[0]+' | '+line,
                'clock_basis': 'explicit_row_clock' if row.group(1) else 'explicit_article_default_weekend_or_midweek_clock'})
    if not matches:
        return _failure('UNCONFIRMED', 'ordered_official_fixture_not_found_in_supported_october_scope')
    if len({m['kickoff'] for m in matches}) != 1:
        return _failure('DISPUTED', 'conflicting_official_schedule_rows')
    match = matches[0]
    canonical = {**match,
        'home_id': 'footballmind:england:senior-men-club:'+_slug(names[0]),
        'away_id': 'footballmind:england:senior-men-club:'+_slug(names[1]),
        'competition': 'Premier League', 'category': 'senior_men_club', 'stage': 'League', 'season': '2026/27',
        'category_basis': 'explicit_official_Premier_League_first_team_competition_scope_registry',
        'source_url': SCHEDULE_URL,
        'source_event_id': 'footballmind:premier-league:2026-27:'+match['source_local_kickoff'][:10]+':'+_slug(names[0])+':'+_slug(names[1]),
        'source_native_event_id': None, 'source_native_home_team_id': None, 'source_native_away_team_id': None,
        'identity_id_basis': 'local_name_based_namespace_from_exact_official_names_not_source_native_ids',
        'source_timezone': 'Europe/London', 'source_timezone_text': 'UK time',
        'timing_basis': 'official_article_UK_time;IANA_Europe/London;season_2026/27_October_is_2026',
        'event_status': 'schedule_listing_subject_to_change', 'parser_version': PARSER_VERSION,
        'score_period': 'unknown', 'settlement_rule': 'unknown'}
    return {'status': 'EXTRACTED', 'reason': 'official_ordered_fixture_date_and_uk_clock',
            'canonical': canonical, 'matched_rows': matches, 'source_published_at': None,
            'original_publication_text': article.select_one('time').get_text(' ', strip=True) if article.select_one('time') else None}


def _leeds(soup, names, canonical):
    # The article does not spell out a timezone. Retain its local clock and
    # corroborate it against the league's explicit UK clock; do not invent one.
    articles = [a for a in soup.find_all('article') if any('richText' in c for c in a.get('class', []))]
    if len(articles) != 1 or 'Mens' not in list(soup.stripped_strings):
        return _failure('UNCONFIRMED', 'official_mens_club_article_missing')
    rows = []
    for heading in articles[0].find_all('h4'):
        if heading.get_text(' ', strip=True) != names[0]+' vs '+names[1]:
            continue
        details = heading.find_next_sibling('ul')
        if details is None:
            continue
        rows.append([li.get_text(' ', strip=True) for li in details.find_all('li')])
    if len(rows) != 1 or len(rows[0]) < 2:
        return _failure('UNCONFIRMED', 'official_club_fixture_missing_or_ambiguous')
    date_text = re.sub(r'(\d)(st|nd|rd|th)', r'\1', rows[0][0])
    try:
        local = datetime.strptime(date_text+' '+rows[0][1].replace(' KO', ''), '%A %d %B %Y %I:%M%p')
    except ValueError:
        return _failure('UNCONFIRMED', 'club_date_or_clock_incomplete')
    expected = datetime.fromisoformat(canonical['source_local_kickoff']).replace(tzinfo=None)
    if local != expected or local.strftime('%A') != date_text.split()[0]:
        return _failure('DISPUTED', 'club_and_league_local_fixture_clocks_disagree')
    return {'status': 'CORROBORATED', 'reason': 'ordered_mens_club_fixture_agrees_with_explicit_league_uk_clock',
            'canonical': None, 'original_time_text': rows[0][:2], 'source_timezone': None,
            'timing_basis': 'club_local_clock_only;timezone_not_stated;compared_to_league_UK_clock',
            'category_text': 'Mens'}


def bind_premier_league_fixture(initialization_manifest, *, archive_root, observation_ids, as_of):
    """Bind a current official observation; never revise the old actor state."""
    as_of = parse_time(as_of)
    policy = MarketPriorPolicy.from_manifest(initialization_manifest)
    fixture = policy.model.fixture
    if parse_time(policy.model.frozen_at) > as_of:
        raise ValueError('Market initialization was unavailable at verification')
    if not isinstance(observation_ids, (list, tuple)) or not 1 <= len(observation_ids) <= 6:
        raise ValueError('One to six explicit identity observations are required')
    pairs, aliases, mapping_ok = [], {}, True
    for side in ('home', 'away'):
        native = fixture.get(side+'_team_id', '')
        mapped = TEAMS.get(native.removeprefix('spftll:team:')) if native.startswith('spftll:team:') else None
        if mapped is None or mapped[0] != fixture.get(side+'_team_name'):
            mapping_ok = False
        pairs.append(mapped[1] if mapped else '')
        aliases[side] = list(mapped) if mapped else [fixture.get(side+'_team_name')]
    mapping_ok = (mapping_ok and pairs[0] != pairs[1] and fixture.get('source') == 'spftll'
                  and fixture.get('competition_id') == 'spftll:competition:36'
                  and fixture.get('competition_name') in {'英超', '英格兰超级联赛', 'Premier League', 'PL'}
                  and fixture.get('season') in {None, '2026/27', '2026-27'})
    evidence, pages, facts = [], [], []
    for obs, body in _observations(archive_root, observation_ids):
        pages.append((obs, _checked_soup(obs, body, as_of)))
        evidence.append({'observation_id': obs.id, 'raw_sha256': obs.content_hash, 'url': obs.endpoint,
                         'observed_at': obs.observed_at.isoformat()})
    for obs, soup in pages:
        if obs.endpoint == SCHEDULE_URL:
            facts.append(_schedule(soup, tuple(pairs)) if soup is not None else _failure('UNCONFIRMED', 'official_response_unavailable'))
    candidates = [f['canonical'] for f in facts if f['status'] == 'EXTRACTED']
    if candidates:
        for obs, soup in pages:
            if obs.endpoint == LEEDS_URL:
                facts.append(_leeds(soup, tuple(pairs), candidates[0]) if soup is not None else _failure('UNCONFIRMED', 'official_response_unavailable'))
    status, reason, canonical = 'UNCONFIRMED', 'official_league_schedule_missing', None
    if not mapping_ok:
        status, reason = 'DISPUTED', 'provider_identity_not_in_explicit_reviewed_mapping'
    elif any(f['status'] == 'DISPUTED' for f in facts):
        status, reason = 'DISPUTED', 'official_identity_or_clock_conflict'
    elif len({_digest(c) for c in candidates}) > 1:
        status, reason = 'DISPUTED', 'conflicting_official_identity_revisions'
    elif candidates:
        candidate = candidates[0]
        if parse_time(candidate['kickoff']) != parse_time(fixture['kickoff']):
            status, reason = 'DISPUTED', 'provider_and_official_kickoff_disagree'
        elif any(f['status'] not in {'EXTRACTED', 'CORROBORATED'} for f in facts):
            reason = 'incomplete_declared_identity_evidence'
        elif not as_of <= parse_time(policy.model.cutoff) < parse_time(candidate['kickoff']):
            reason = 'outside_declared_prematch_prediction_window'
        else:
            status, reason, canonical = 'CONFIRMED', 'exact_declared_aliases_order_and_official_clock_agree', candidate
    elif facts:
        reason = facts[0]['reason']
    result = {'schema': VERSION, 'status': status, 'reason': reason, 'match_id': fixture['match_id'],
        'initialization_manifest': deepcopy(initialization_manifest), 'initialization_sha256': _digest(initialization_manifest),
        'canonical': canonical, 'aliases': aliases, 'observations': evidence, 'verified_at': as_of.isoformat(),
        'source_projections': facts, 'comparison': {'basis': 'explicit_provider_ID_name_mapping_and_official_ordered_schedule',
            'native_source_event_mapping_available': False, 'verified_at': as_of.isoformat(), 'status': status},
        'mapping_basis': 'reviewed_exact_provider_ID_Chinese_alias_to_observed_official_English_name',
        'independent_from_initial_quote_provider': True, 'retroactive_actor_evidence': False,
        'settlement_status': 'PENDING', 'settlement_rows': [], 'regulation_rule': None}
    result['binding_sha256'] = _digest(result)
    return result


def verify_premier_league_identity(binding, *, archive_root, as_of):
    if (not isinstance(binding, dict) or binding.get('schema') != VERSION
            or parse_time(binding['verified_at']) > parse_time(as_of)
            or binding.get('binding_sha256') != _digest({k: v for k, v in binding.items() if k != 'binding_sha256'})):
        raise ValueError('Premier League identity binding is changed or unavailable')
    rebuilt = bind_premier_league_fixture(binding['initialization_manifest'], archive_root=archive_root,
        observation_ids=[e['observation_id'] for e in binding['observations']], as_of=binding['verified_at'])
    if rebuilt != binding:
        raise ValueError('Premier League identity does not reproduce original official bytes')
    return rebuilt
