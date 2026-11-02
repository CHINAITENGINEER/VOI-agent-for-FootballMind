"""Offline projection of publicly served UEFA match API results.

Only adult men's 2026/27 Nations League group matches are supported. Numeric
scores alone never create a final result. No editorial score inference is used.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re
from urllib.parse import urlsplit

from football_research.acquisition.state import parse_time

PARSER_VERSION = 'uefa-public-match-api-v5-regulation-v1'
SETTLEMENT_VERSION = 'raw-backed-independent-settlement-v2'
_PATH = re.compile(r'/v5/matches/([1-9][0-9]*)')


def official_api_event_id(endpoint):
    """Recognize only the exact unauthenticated official match resource."""
    try:
        url = urlsplit(endpoint)
        found = _PATH.fullmatch(url.path)
        if (url.scheme == 'https' and url.netloc == 'match.uefa.com' and
                not url.query and not url.fragment and not url.username and found):
            return found.group(1)
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def _strict_json(body):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate result field')
            value[key] = item
        return value
    return json.loads(body, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite result value')))


def _pair(value):
    if not isinstance(value, dict) or set(value) != {'home', 'away'}:
        return None
    if any(type(value[k]) is not int or not 0 <= value[k] <= 99 for k in ('home', 'away')):
        return None
    return value['home'], value['away']


def project_official_api_result(observation, body, canonical, *, as_of):
    """Bind archived API bytes to an already raw-verified canonical fixture."""
    def answer(status, reason, **fields):
        return {'status':status, 'reason':reason, 'parser_version':PARSER_VERSION, **fields}

    native = official_api_event_id(observation.endpoint)
    if native is None or 'uefa:match:'+native != canonical['source_event_id']:
        return answer('IGNORED', 'not_target_official_match_api')
    if sha256(body).hexdigest() != observation.content_hash:
        raise ValueError('Official API raw checksum mismatch')
    if observation.observed_at > parse_time(as_of):
        raise ValueError('Official API result was unavailable at verification time')
    requested = observation.context.get('requested_at')
    if requested is None or parse_time(requested) > observation.observed_at:
        raise ValueError('Official API requires a valid actual request/receipt clock')
    if observation.http_status != 200 or observation.error_class:
        return answer('UNCONFIRMED', 'official_api_response_unavailable')
    if (observation.source != 'open_page' or observation.params or
            len(body) > 2_000_000 or observation.context.get('response_complete') is False):
        return answer('UNCONFIRMED', 'unsupported_or_incomplete_official_api_receipt')
    try:
        match = _strict_json(body)
    except (ValueError, UnicodeError, RecursionError):
        return answer('UNCONFIRMED', 'invalid_or_ambiguous_official_api_json')
    if not isinstance(match, dict):
        return answer('UNCONFIRMED', 'official_api_match_object_missing')
    home, away = match.get('homeTeam'), match.get('awayTeam')
    competition, stage, group = match.get('competition'), match.get('round'), match.get('group')
    if not all(isinstance(x, dict) for x in (home, away, competition, stage, group)):
        return answer('UNCONFIRMED', 'official_api_structured_identity_missing')
    if (str(match.get('id')) != native or
            'uefa:team:'+str(home.get('id')) != canonical['home_id'] or
            'uefa:team:'+str(away.get('id')) != canonical['away_id'] or
            home.get('internationalName') != canonical['home_name'] or
            away.get('internationalName') != canonical['away_name']):
        return answer('DISPUTED', 'official_api_event_or_ordered_team_identity_conflict')
    required = {'id':'2014', 'code':'UNL', 'age':'ADULT', 'sex':'MALE',
                'sportsType':'FOOTBALL', 'teamCategory':'NATIONAL'}
    meta = stage.get('metaData')
    if (any(competition.get(k) != v for k,v in required.items()) or
            canonical['competition'] != 'UEFA Nations League' or
            canonical['category'] != 'senior_men_national' or canonical['stage'] != 'League phase' or
            canonical['season'] != '2027' or str(match.get('seasonYear')) != '2027' or
            match.get('type') != 'GROUP_STAGE' or match.get('competitionPhase') != 'TOURNAMENT' or
            stage.get('mode') != 'GROUP' or stage.get('competitionId') != '2014' or
            str(stage.get('seasonYear')) != '2027' or not isinstance(meta, dict) or
            meta.get('name') != 'League phase' or meta.get('type') != 'GROUP_STANDINGS' or
            group.get('competitionId') != '2014' or str(group.get('seasonYear')) != '2027' or
            not isinstance(group.get('teams'), list) or
            not {str(home['id']), str(away['id'])}.issubset({str(x) for x in group['teams']})):
        return answer('DISPUTED', 'official_api_competition_category_or_regulation_stage_conflict')
    try:
        kickoff = parse_time(match['kickOffTime']['dateTime'])
    except (KeyError, TypeError, ValueError):
        return answer('UNCONFIRMED', 'official_api_kickoff_clock_missing')
    if kickoff != parse_time(canonical['kickoff']):
        return answer('DISPUTED', 'official_api_and_bound_fixture_kickoff_conflict')
    status = match.get('status')
    if not isinstance(status, str):
        return answer('UNCONFIRMED', 'unsupported_official_api_result_status')
    if status in {'POSTPONED', 'CANCELLED', 'CANCELED'}:
        return answer('POSTPONED' if status == 'POSTPONED' else 'CANCELLED',
                      'official_api_explicit_inactive_result', source_status=status)
    if status in {'SUSPENDED', 'ABANDONED', 'AWARDED', 'WALKOVER'}:
        return answer('DISPUTED', 'official_api_administrative_or_unfinished_result', source_status=status)
    if status in {'UPCOMING', 'SCHEDULED', 'LIVE', 'IN_PROGRESS', 'INPROGRESS', 'PLAYING'}:
        return answer('PENDING', 'official_api_has_no_final_result', source_status=status)
    if status != 'FINISHED':
        return answer('UNCONFIRMED', 'unsupported_official_api_result_status', source_status=status)
    try:
        final = parse_time(match['fullTimeAt'])
    except (KeyError, TypeError, ValueError):
        return answer('UNCONFIRMED', 'official_api_final_whistle_clock_missing')
    if final > observation.observed_at or (final-kickoff).total_seconds() < 90*60:
        return answer('DISPUTED', 'official_api_finished_state_has_impossible_or_future_clock')
    score = match.get('score')
    regular = _pair(score.get('regular')) if isinstance(score, dict) else None
    total = _pair(score.get('total')) if isinstance(score, dict) else None
    if regular is None or total is None:
        return answer('UNCONFIRMED', 'official_api_explicit_integer_regulation_score_missing')
    if regular != total or set(score)-{'regular', 'total'}:
        return answer('DISPUTED', 'official_api_total_or_score_period_conflict')
    return answer('FINISHED_SCORE', 'official_api_bound_finished_regulation_score',
        source_status=status, full_time_at=final.isoformat(),
        home_goals=regular[0], away_goals=regular[1], score_period='regulation')
