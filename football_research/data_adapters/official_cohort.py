"""Plan an ordered, unexposed official cohort from already verified raw bytes.

Planning makes no requests, registers no exposure and constructs no identities.
The live entry point must revalidate every selected query against fresh official
evidence before acquisition. News availability and outcomes do not rank rows.
"""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

from bs4 import BeautifulSoup

from football_research.acquisition.state import parse_time
from football_spider.storage.raw_archive import RawArchive, json_bytes
from .official_fixture import (
    _LEGACY_EVENT, _PATH, _digest, _horizon, _group, _read, _registry, _resolve,
    verify_official_fixture,
)
from .prospective_learning import SCOPE, _previous_inventory

VERSION = 'official-cohort-plan-v1'


def _pins(identity, archive, plans):
    paths = {identity, *plans, *(p for p in archive.rglob('*') if p.is_file())}
    return {str(p): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def plan_official_cohort(identity_path, *, identity_archive, previous_plan_paths,
                         output, horizon_days=7, max_matches=20, as_of=None, schedule_group='2'):
    """Write one immutable plan JSON and return its contents (zero network I/O).

    An explicit fourteen-day window is opt-in; the default remains seven days.
    ``as_of`` must be an already reached aware timestamp. Supplying the same
    inputs and timestamp produces identical bytes at any new output location.
    Previous exposure means physical identity, irrespective of cohort purpose.
    """
    horizon_days = _horizon(horizon_days)
    schedule_group = _group(schedule_group)
    if type(max_matches) is not int or not 1 <= max_matches <= 20:
        raise ValueError('max_matches must be an integer from 1 through 20')
    if not isinstance(previous_plan_paths, (list, tuple)):
        raise ValueError('previous_plan_paths must be an explicit sequence')
    now = datetime.now(timezone.utc)
    at = parse_time(now if as_of is None else as_of)
    if at > now:
        raise ValueError('A cohort cannot be planned at a future as_of')
    identity, archive_root, target = (Path(p).resolve() for p in
                                    (identity_path, identity_archive, output))
    plans = [Path(p).resolve() for p in previous_plan_paths]
    if not identity.is_file() or not archive_root.is_dir() or any(not p.is_file() for p in plans):
        raise ValueError('Existing identity, raw archive and registration files are required')
    if target.exists():
        raise FileExistsError('Use a new official cohort plan output file')
    if target.is_relative_to(identity.parent) or target.is_relative_to(archive_root):
        raise ValueError('Plan output must be outside the original identity and raw archive')

    original_pins = _pins(identity, archive_root, plans)
    binding = verify_official_fixture(_read(identity), archive_root=archive_root, as_of=at)
    if binding.get('schedule_group','2') != schedule_group:
        raise ValueError('Cohort schedule group differs from its verified original identity')
    inventory = _previous_inventory([_read(p) for p in plans], at)
    registered = {row['source_event_id'] for row in inventory}
    archive = RawArchive(archive_root)
    # Verification above enforces one original response and its bound receipt.
    observation, = archive.observations()
    body = archive.read_body(observation)
    observed_ids = {}
    for index, row in enumerate(BeautifulSoup(body, 'html.parser').select('tr[data-time]')):
        for link in row.find_all('a', href=True):
            if match := _PATH.fullmatch(link['href']):
                observed_ids.setdefault(match.group(1), set()).add(index)

    candidates, eligible = [], []
    for native in sorted(observed_ids, key=int):
        event = ('palloliitto:match:' if native == _LEGACY_EVENT else
                 'veikkausliiga:2026:match:') + native
        resolved = _resolve(body, {'native_event_id': native}, at, horizon_days=horizon_days,
                            schedule_group=schedule_group)
        row = {'native_event_id': native, 'source_event_id': event,
               'raw_schedule_row_indices': sorted(observed_ids[native]),
               'resolver_status': resolved['status'], 'decision': 'rejected'}
        if resolved['status'] == 'ready':
            fact = resolved['fact']
            row.update({key: fact[key] for key in ('home_name', 'away_name', 'kickoff',
                'competition', 'season', 'stage', 'source_url')})
        if event in registered:
            row['reason'] = 'already_registered_physical_fixture'
            if resolved['status'] != 'ready':
                row['resolver_reason'] = resolved['reason']
        elif resolved['status'] != 'ready':
            row['reason'] = resolved['reason']
        else:
            row['reason'] = 'cohort_limit_reached'
            eligible.append(row)
        candidates.append(row)

    eligible.sort(key=lambda row: (parse_time(row['kickoff']), int(row['native_event_id'])))
    selected = eligible[:max_matches]
    for row in selected:
        row.update(decision='selected', reason='unexposed_official_fixture_in_frozen_window')
    # Keep resolvable fixtures in chronological order; rejected unresolved rows
    # remain explicitly accounted for without inventing a canonical kickoff.
    candidates.sort(key=lambda row: (row.get('kickoff', '9999'), int(row['native_event_id'])))
    result = {'schema': VERSION, 'as_of': at.isoformat(), 'horizon_days': horizon_days,
        'minimum_lead_minutes': 30, 'window_end': (at + timedelta(days=horizon_days)).isoformat(),
        'max_matches': max_matches, 'queries': [row['source_event_id'] for row in selected],
        'selected': selected, 'candidates': candidates,
        'counts': {'observed_native_ids': len(observed_ids), 'registered_physical_matches': len(inventory),
            'eligible_unexposed': len(eligible), 'selected': len(selected),
            'rejected': len(candidates)-len(selected)},
        'source': {'identity_path': str(identity), 'identity_archive': str(archive_root),
            'binding_sha256': binding['binding_sha256'], 'observation_id': observation.id,
            'raw_sha256': observation.content_hash, 'url': observation.endpoint,
            'params': observation.params, 'observed_at': observation.observed_at.isoformat()},
        'previous_plan_paths': [str(p) for p in plans], 'inventory_scope': SCOPE,
        'registry_sha256': _digest(_registry(schedule_group)), 'registered_inventory_sha256': _digest(inventory),
        'input_sha256': original_pins,
        'selection_rule': 'kickoff_then_numeric_native_id_without_outcome_or_news_ranking',
        'http_requests': 0, 'new_registrations': 0,
        'fresh_identity_revalidation_required_before_acquisition': True}
    if schedule_group != '2':
        result['schedule_group'] = schedule_group
    if _pins(identity, archive_root, plans) != original_pins:
        raise ValueError('Original official evidence or registrations changed during planning')
    result['plan_sha256'] = sha256(json_bytes(result)).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return result


__all__ = ['VERSION', 'plan_official_cohort']
