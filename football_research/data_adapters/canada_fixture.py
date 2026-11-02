"""Narrow, opt-in identity binding for the observed Canada Soccer men's fixture.

This version recognises official event 6630 in camp 1957 only. The country image
codes plus the exact CANMNT designation are explicit entity mappings, not native
numeric team IDs. Identity confirmation provides no regulation settlement rule.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

from bs4 import BeautifulSoup

from football_platform.information.fixture_corroboration import compare_assertion
from football_research.acquisition.state import parse_time
from football_research.acquisition.value import _canonical
from football_spider.storage.raw_archive import RawArchive
from .prospective_text import MarketPriorPolicy

VERSION = 'canada-senior-fixture-identity-v1'
PARSER_VERSION = 'canadasoccer-6630-canmnt-official-dom-v1'
MATCH_URL = 'https://canadasoccer.com/match/6630/'
CAMP_URL = 'https://canadasoccer.com/camp/1957/mens-international-series-in-canada-and-usa/'
CAMP_NAME = 'Men’s international series in Canada and USA'
_IMAGES = {code: 'https://canadasoccer.com/wp-content/uploads/canada-soccer-files/'
                + 'TEAM_LOGO_' + code + '.png' for code in ('USA', 'CAN')}
_ALIASES = {'home': ['美国', 'USA', 'United States'],
            'away': ['加拿大', 'Canada', '#CANMNT']}
_PROVIDER = {'match_id': 'spftll:match:3097883', 'competition_id': 'spftll:competition:1366',
             'competition_name': '国际赛', 'home_team_id': 'spftll:team:797',
             'away_team_id': 'spftll:team:795', 'home_team_name': '美国', 'away_team_name': '加拿大'}


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _one(node, selector):
    matches = node.select(selector)
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous official identity field: ' + selector)
    return matches[0]


def _text(node, selector):
    return ' '.join(_one(node, selector).get_text(' ', strip=True).split())


def _clock(date, time):
    # Explicit abbreviation only: never infer the venue's timezone or use the
    # platform's local clock. This case's source spells EDT, not generic ET.
    if not time.endswith(' EDT'):
        raise ValueError('This identity version requires explicit EDT')
    try:
        stamp = datetime.strptime(date + ' ' + time[:-4], '%d %b %Y %H:%M')
    except ValueError as exc:
        raise ValueError('Invalid official date and kickoff clock') from exc
    if stamp.strftime('%d %b %Y') != date or stamp.strftime('%H:%M') != time[:-4]:
        raise ValueError('Ambiguous official clock spelling')
    return stamp.replace(tzinfo=timezone(timedelta(hours=-4))).astimezone(timezone.utc)


def _failure(status, reason):
    return {'status': status, 'reason': reason, 'canonical': None}


def _extract(observation, body, *, as_of):
    if observation.observed_at > parse_time(as_of):
        raise ValueError('Official observation was unavailable at audit time')
    if observation.endpoint != MATCH_URL or observation.source != 'open_page' or observation.params:
        raise ValueError('Expected the exact acquired Canada Soccer native match URL')
    requested = observation.context.get('requested_at')
    if requested is None or parse_time(requested) > observation.observed_at:
        raise ValueError('Official receipt must follow its recorded request')
    if observation.http_status != 200 or observation.error_class:
        return _failure('UNCONFIRMED', 'official_response_unavailable')
    if len(body) > 2_000_000:
        raise ValueError('Official page exceeds bounded parser input')
    soup = BeautifulSoup(body, 'html.parser')
    try:
        main = _one(soup, 'main')
        banner = _one(main, '.match-banner')
        if banner.get('id') != 'match-banner-6630':
            return _failure('DISPUTED', 'endpoint_and_native_banner_disagree')
        if 'upcoming-match' not in banner.get('class', []):
            return _failure('UNCONFIRMED', 'target_not_explicitly_upcoming')
        if (_text(banner, '.team-home h2') != 'USA'
                or _text(banner, '.team-visitor h2') != "Men's National Team"
                or _text(banner, 'h1') != "USA vs. Men's National Team"):
            return _failure('DISPUTED', 'ordered_official_teams_or_mens_category_conflict')
        for selector, code, name in (('.team-home img', 'USA', 'USA'),
                                     ('.team-visitor img', 'CAN', "Men's National Team")):
            image = _one(banner, selector)
            if image.get('src') != _IMAGES[code] or image.get('alt') != name:
                return _failure('DISPUTED', 'official_country_code_or_category_conflict')
        if (_text(banner, '.camp-name') != CAMP_NAME
                or _one(banner, 'a.back-button').get('href') != CAMP_URL):
            return _failure('DISPUTED', 'unsupported_official_camp')
        date, time = _text(banner, '.match-date'), _text(banner, '.match-time')
        kickoff = _clock(date, time)
        details = _one(main, '#match-panel-info .details-box')
        kickoff_rows = [li for li in details.select('li') if _text(li, '.label') == 'Kickoff']
        if len(kickoff_rows) != 1 or _text(kickoff_rows[0], '.value') != date + ' | ' + time:
            return _failure('DISPUTED', 'official_banner_and_detail_clock_conflict')
        # The site's generic header can describe a different match. Require its
        # exact native ID and ordered entities before using its CANMNT category.
        card = _one(soup, '.featured-match-card#match-6630')
        if ('upcoming-match' not in card.get('class', [])
                or _text(card, '.team-home h5') != 'USA'
                or _text(card, '.team-visitor h5') != '#CANMNT'
                or _text(card, '.match-date') != date or _text(card, '.match-time') != time
                or _one(card, 'a').get('href') != MATCH_URL):
            return _failure('DISPUTED', 'same_event_featured_card_or_canmnt_category_conflict')
        for selector, code in (('.team-home img', 'USA'), ('.team-visitor img', 'CAN')):
            if _one(card, selector).get('src') != _IMAGES[code]:
                return _failure('DISPUTED', 'featured_card_country_order_conflict')
    except ValueError as exc:
        return _failure('UNCONFIRMED', str(exc))
    canonical = {'home_name': 'United States', 'away_name': 'Canada',
        'home_id': 'canadasoccer:senior-men:USA', 'away_id': 'canadasoccer:senior-men:CAN',
        'kickoff': kickoff.isoformat(), 'competition': CAMP_NAME, 'category': 'senior_men_national',
        'stage': 'International series', 'season': str(kickoff.year), 'source_url': MATCH_URL,
        'source_event_id': 'canadasoccer:match:6630', 'event_status': 'EventScheduled',
        'parser_version': PARSER_VERSION, 'score_period': 'unknown', 'settlement_rule': 'unknown'}
    return {'status': 'EXTRACTED', 'reason': 'official_native_match_dom_identity', 'canonical': canonical,
        'identity_basis': 'exact_mens_national_designation_country_image_codes_order_native_match_and_camp',
        'native_numeric_team_ids_available': False, 'country_codes': ['USA', 'CAN']}


def bind_canada_fixture(initialization_manifest, *, archive_root, observation_ids, as_of):
    """Bind one explicit observed friendly identity; never emits outcome labels."""
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
    observed = archive.observations()
    by_id = {o.id: o for o in observed}
    if len(by_id) != len(observed) or any(i not in by_id for i in observation_ids):
        raise ValueError('Source observations are missing or ambiguous')
    facts, evidence = [], []
    for ident in sorted(observation_ids):
        obs = by_id[ident]
        facts.append(_extract(obs, archive.read_body(obs), as_of=as_of))
        evidence.append({'observation_id': obs.id, 'raw_sha256': obs.content_hash,
                         'url': obs.endpoint, 'observed_at': obs.observed_at.isoformat()})
    status, reason, canonical, comparison = 'UNCONFIRMED', facts[0]['reason'], None, None
    extracted = [f['canonical'] for f in facts if f['status'] == 'EXTRACTED']
    if any(fixture.get(k) != v for k, v in _PROVIDER.items()):
        status, reason = 'DISPUTED', 'provider_fixture_not_in_explicit_single_match_mapping'
    elif any(f['status'] == 'DISPUTED' for f in facts):
        status, reason = 'DISPUTED', 'official_identity_conflict'
    elif len({_digest(row) for row in extracted}) > 1:
        status, reason = 'DISPUTED', 'conflicting_official_identity_revisions'
    elif extracted:
        candidate = extracted[0]
        if parse_time(candidate['kickoff']) != parse_time(fixture['kickoff']):
            status, reason = 'DISPUTED', 'provider_and_official_kickoff_disagree'
        elif any(f['status'] != 'EXTRACTED' for f in facts):
            status, reason = 'UNCONFIRMED', 'incomplete_or_unusable_declared_identity_observations'
        else:
            comparison = compare_assertion({k: fixture[k] for k in (
                'match_id', 'competition_id', 'season', 'home_team_id', 'away_team_id', 'kickoff',
                'home_team_name', 'away_team_name')}, source_event_id=candidate['source_event_id'],
                explicitly_mapped_match_id=fixture['match_id'],
                fields={'home_team_id': fixture['home_team_id'], 'away_team_id': fixture['away_team_id'],
                        'competition_id': fixture['competition_id'], 'kickoff': candidate['kickoff']},
                evidence_basis='acquired_structured',
                available_at=max(parse_time(e['observed_at']) for e in evidence),
                cutoff=min(as_of, parse_time(policy.model.cutoff)))
            status, reason = comparison['status'], comparison['reason']
            if status == 'CONFIRMED':
                canonical = candidate
    result = {'schema': VERSION, 'status': status, 'reason': reason, 'match_id': fixture['match_id'],
        'initialization_manifest': deepcopy(initialization_manifest),
        'initialization_sha256': _digest(initialization_manifest), 'canonical': canonical,
        'aliases': deepcopy(_ALIASES), 'observations': evidence, 'verified_at': as_of.isoformat(),
        'comparison': comparison, 'source_projections': facts,
        'mapping_basis': 'explicit_single_match_provider_ids_and_official_mens_country_codes',
        'independent_from_initial_quote_provider': True, 'retroactive_actor_evidence': False,
        'settlement_status': 'PENDING', 'settlement_rows': [], 'regulation_rule': None}
    result['binding_sha256'] = _digest(result)
    return result


def verify_canada_identity(binding, *, archive_root, as_of):
    """Reproduce the entire binding from original archived bytes and clocks."""
    if (not isinstance(binding, dict) or binding.get('schema') != VERSION
            or parse_time(binding['verified_at']) > parse_time(as_of)
            or binding.get('binding_sha256') != _digest({k: v for k, v in binding.items()
                                                       if k != 'binding_sha256'})):
        raise ValueError('Canada identity binding is changed or not yet available')
    rebuilt = bind_canada_fixture(binding['initialization_manifest'], archive_root=archive_root,
        observation_ids=[e['observation_id'] for e in binding['observations']], as_of=binding['verified_at'])
    if rebuilt != binding:
        raise ValueError('Canada identity binding does not reproduce official raw evidence')
    return rebuilt
