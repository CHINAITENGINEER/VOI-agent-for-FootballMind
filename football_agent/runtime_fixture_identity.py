"""Resolve one user-supplied fixture from verified, already visible source rows.

This is source-scoped identity, not proof of a fixture's existence or a team
alias registry. Callers verify the original archived documents before passing
their IDs. This module performs no network, archive, or database reads.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import re
import unicodedata

from .structured_sources import PROJECTION_VERSION

VERSION = 'runtime-fixture-identity-v1'
TARGET = re.compile(r'spftll:[1-9][0-9]*\Z')
SOURCE_ID = re.compile(r'[1-9][0-9]*\Z')
SHANGHAI = timezone(timedelta(hours=8))


def _normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _time(value):
    if not isinstance(value, str):
        raise ValueError('identity_timestamp_missing')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('identity_timestamp_timezone_missing')
    return parsed.astimezone(timezone.utc)


def _constraints(value):
    """Full-string grammar: optional [competition], full date, team separator."""
    if not isinstance(value, str) or not value.strip():
        return None, 'unsupported_input', 'match_input_required'
    if TARGET.fullmatch(value):
        return {'kind': 'canonical', 'target_key': value}, None, None
    text = unicodedata.normalize('NFKC', value).strip()
    competition = None
    brackets = list(re.finditer(r'\[([^\[\]]+)\]', text))
    if brackets:
        if len(brackets) != 1:
            return None, 'unsupported_input', 'one_optional_competition_required'
        item = brackets[0]
        competition = _normalized(item.group(1))
        if not competition:
            return None, 'unsupported_input', 'competition_empty'
        text = (text[:item.start()] + ' ' + text[item.end():]).strip()
    if '[' in text or ']' in text:
        return None, 'unsupported_input', 'competition_brackets_invalid'
    dates = list(re.finditer(r'(?<![0-9])\d{4}-\d{2}-\d{2}(?![0-9])', text))
    day = None
    if dates:
        if len(dates) != 1:
            return None, 'unsupported_input', 'one_full_date_required'
        item = dates[0]
        if text[:item.start()].strip() and text[item.end():].strip():
            return None, 'unsupported_input', 'date_must_precede_or_follow_teams'
        try:
            day = date.fromisoformat(item.group())
        except ValueError:
            return None, 'unsupported_input', 'invalid_calendar_date'
        text = (text[:item.start()] + ' ' + text[item.end():]).strip()
    sides = re.split(r'\s+vs\s+|对阵|對陣', text, flags=re.IGNORECASE)
    if len(sides) != 2 or not all(part.strip() for part in sides):
        return None, 'unsupported_input', 'two_teams_and_explicit_separator_required'
    teams = [_normalized(side) for side in sides]
    if teams[0] == teams[1] or any(len(side) > 120 for side in teams):
        return None, 'unsupported_input', 'distinct_bounded_team_names_required'
    constraints = {'kind': 'team_date', 'teams': teams,
        'competition': competition, 'date': day.isoformat() if day else None,
        'date_timezone': 'Asia/Shanghai', 'date_utc_offset': '+08:00',
        'team_order': 'unordered_user_input_source_home_away_preserved',
        'name_matching': 'nfkc_casefold_whitespace_exact_no_alias_inference'}
    if day is None:
        return constraints, 'needs_date', 'full_calendar_date_required_no_nearest_fixture_guess'
    return constraints, None, None


def _fixture(row, document):
    """Explicit whitelist: no scores, odds, source narratives or model plans."""
    sid = row.get('source_id')
    if not isinstance(sid, str) or SOURCE_ID.fullmatch(sid) is None:
        raise ValueError('source_match_id_invalid')
    teams = []
    for side in ('home', 'away'):
        entity = row.get(side)
        if not isinstance(entity, dict) or not isinstance(entity.get('name'), str) or not entity['name'].strip():
            raise ValueError('source_team_name_invalid')
        if entity.get('source_id') is not None and not isinstance(entity['source_id'], str):
            raise ValueError('source_team_id_invalid')
        if not isinstance(entity.get('category', 'unknown'), str):
            raise ValueError('source_team_category_invalid')
        teams.append({'name': entity['name'], 'source_id': entity.get('source_id'),
                      'category': entity.get('category', 'unknown')})
    competition = row.get('competition')
    if not isinstance(competition, dict) or not isinstance(competition.get('name'), str):
        raise ValueError('source_competition_invalid')
    if competition.get('source_id') is not None and not isinstance(competition['source_id'], str):
        raise ValueError('source_competition_id_invalid')
    kickoff = _time(row.get('kickoff'))
    return {'target_key': 'spftll:' + sid, 'source': 'spftll', 'source_id': sid,
        'home': teams[0], 'away': teams[1],
        'competition': {'name': competition['name'], 'source_id': competition.get('source_id')},
        'kickoff': kickoff.isoformat(), 'local_date': kickoff.astimezone(SHANGHAI).date().isoformat(),
        'evidence': {'document_id': document['document_id'], 'raw_sha256': document.get('raw_sha256'),
                     'observed_at': document['observed_at'],
                     'observation_id': document['platform'].get('observation_id'),
                     'projection_version': PROJECTION_VERSION}}


def _signature(row):
    return (_normalized(row['home']['name']), _normalized(row['away']['name']),
            row['home']['source_id'], row['away']['source_id'],
            row['home']['category'], row['away']['category'],
            _normalized(row['competition']['name']), row['competition']['source_id'], row['kickoff'])


def resolve_match_identity(public, *, verified_document_ids=()):
    """Pure, reproducible resolution bounded by actor-visible observations.

    ``verified_document_ids`` is supplied by the archive-verifying caller. A
    candidate model's ``match_identity`` or arbitrary document text is ignored.
    A canonical input is an explicit user assertion; only source-backed rows
    can upgrade it to ``resolved`` / ``source_reported``.
    """
    constraints, status, reason = _constraints(public.get('match_input'))
    result = {'version': VERSION, 'status': status or 'unresolved', 'target_key': None,
              'identity_basis': 'unresolved', 'evidence': [], 'candidates': [],
              'reason': reason, 'input_constraints': constraints, 'issues': [],
              'scope': 'verified_visible_spftll_records_not_complete_world_schedule',
              'independently_verified_world_fact': False}
    if status:
        return result
    canonical = constraints['kind'] == 'canonical'
    if canonical:
        result.update(status='canonical_input', target_key=constraints['target_key'],
                      identity_basis='canonical_user_input_not_source_verified',
                      reason='explicit_canonical_input')
    try:
        cutoff = _time(public.get('snapshot_as_of'))
    except (ValueError, TypeError):
        result['issues'].append({'reason': 'snapshot_clock_invalid'})
        if not canonical:
            result.update(status='incomplete', reason='snapshot_clock_invalid')
        return result
    verified = set(verified_document_ids)
    documents = public.get('documents', [])
    if not isinstance(documents, list):
        result.update(status='incomplete', target_key=None, reason='documents_invalid')
        return result
    records, incomplete = [], False
    for doc in documents:
        if not isinstance(doc, dict) or doc.get('content_kind') != 'platform_structured':
            continue
        meta = doc.get('platform') or {}
        if not isinstance(meta, dict) or meta.get('source') != 'spftll':
            continue
        did = doc.get('document_id')
        issue = lambda why: result['issues'].append({'document_id': did, 'reason': why})
        if did not in verified:
            issue('document_not_archive_verified')
            continue
        try:
            observed = _time(doc.get('observed_at'))
        except (ValueError, TypeError):
            issue('observation_clock_invalid')
            continue
        if observed > cutoff:
            issue('observation_after_snapshot')
            continue
        if meta.get('projection_version') != PROJECTION_VERSION:
            issue('projection_version_unsupported')
            incomplete = True
            continue
        if doc.get('status') != 'readable' or meta.get('quality') != 'normalized':
            issue('source_projection_not_complete_normalized')
            incomplete = True
            continue
        if doc.get('text_truncated') or doc.get('excerpt_truncated'):
            issue('visible_source_projection_truncated')
            incomplete = True
        rows = doc.get('structured_records')
        if not isinstance(rows, list):
            issue('visible_structured_records_missing')
            incomplete = True
            continue
        for row in rows:
            if not isinstance(row, dict):
                issue('source_record_invalid')
                incomplete = True
                continue
            if row.get('kind') != 'fixture':
                continue
            if row.get('source') != 'spftll':
                issue('fixture_source_mismatch')
                incomplete = True
                continue
            try:
                record = _fixture(row, doc)
            except (ValueError, TypeError):
                issue('fixture_identity_fields_invalid')
                incomplete = True
                continue
            records.append(record)
    related = [row for row in records if
        row['target_key'] == constraints['target_key']] if canonical else [row for row in records if
        sorted((_normalized(row['home']['name']), _normalized(row['away']['name']))) == sorted(constraints['teams'])]
    matching = related if canonical else [row for row in related if row['local_date'] == constraints['date']
        and (constraints['competition'] is None or
             _normalized(row['competition']['name']) == constraints['competition'])]
    matched_ids = sorted({row['target_key'] for row in matching})
    conflicts = [target for target in matched_ids if len({
        _signature(row) for row in records if row['target_key'] == target}) > 1]
    result['candidates'] = matching
    if conflicts:
        result.update(status='conflicting', target_key=None, reason='same_source_id_has_conflicting_identity')
        result['conflicting_target_keys'] = conflicts
    elif len(matched_ids) > 1:
        result.update(status='ambiguous', target_key=None, reason='multiple_matching_source_ids')
    elif incomplete:
        if not canonical:
            result.update(status='incomplete', reason='visible_source_coverage_incomplete')
        else:
            result['reason'] = 'explicit_canonical_input_source_confirmation_incomplete'
    elif len(matched_ids) == 1:
        result.update(status='resolved', target_key=matched_ids[0], identity_basis='source_reported',
                      evidence=[row['evidence'] for row in matching], reason='one_exact_source_fixture_identity')
    elif not canonical:
        result.update(reason='not_observed_in_source_not_proof_of_absence')
    return result


__all__ = ['VERSION', 'resolve_match_identity']
