"""Implemented endpoints are documented in docs/data-source-capabilities.md."""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import random
from football_spider.config import settings as legacy_settings
from football_spider.collectors.http import ArchiveHttp, FetchError
from football_spider.collectors.aio_transport import AioSession
from football_spider.collectors.quota import reserve
from football_spider.models.contracts import CollectionResult, RunStatus, now


def collect(source, config, archive, *, start: date, end: date, season=None,
            match_ids=None, historical_at=None):
    if source == 'understat':
        from football_spider.collectors.understat import collect_understat
        return collect_understat(config,archive,season=season if season is not None else 2025)
    result = CollectionResult(source=source, scope={'start': str(start), 'end': str(end), 'season': season})
    credentials = {'football_data': config.football_data_token, 'odds_api': config.odds_api_key}
    if source in credentials and not credentials[source]:
        result.status = RunStatus.NEEDS_CREDENTIALS
        return result
    if source not in {'spftll', 'football_data', 'odds_api'}:
        result.status = RunStatus.SKIPPED_DISABLED
        result.errors.append({'category': 'NOT_IMPLEMENTED'})
        return result
    session = AioSession(archive.root / 'cookies' / 'spftll-aio.json') if source == 'spftll' else None
    client = ArchiveHttp(archive, source, max_requests=config.max_requests,
                         interval=6.1 if source == 'football_data' else 1.0, session=session)

    def get(url, params=None, **kwargs):
        try:
            return client.get(url, params, **kwargs)
        except FetchError as exc:
            result.errors.append({'category': exc.category, 'observation': exc.observation.id if exc.observation else None})
            return None, None

    try:
        if source == 'spftll':
            headers = legacy_settings.browser_headers(random.choice(legacy_settings.user_agents))
            ids = list(match_ids or [])
            if not ids:
                _, payload = get(legacy_settings.list_api_url,
                                 {'dateFormat': '', 'notSingle': 'false', 'mKind': 0, 'reqtype': 0}, headers=headers)
                if isinstance(payload, dict) and payload.get('code') == 0:
                    for group in payload.get('data') or []:
                        for row in group.get('list') or []:
                            if str(start) <= str(row.get('matchTime', ''))[:10] <= str(end):
                                ids.append(str(row['matchId']))
                # Actual generic recent schedule endpoint; not an invented date API.
                get(legacy_settings.generic_match_list_api_url, {}, headers=headers)
            for mid in dict.fromkeys(ids[:config.detail_limit] if not match_ids else ids):
                get(legacy_settings.detail_api_url, {'gameKind': 0, 'matchId': mid}, headers=headers)
                get(legacy_settings.more_odds_api_url, {'mKind': 0, 'matchId': mid}, headers=headers)
            result.scope['coverage'] = 'lottery_selection_and_recent_history_not_complete_league'
            result.scope['requested_details'] = len(ids if match_ids else ids[:config.detail_limit])
            if season is not None or end < now().astimezone(ZoneInfo(config.timezone)).date() - timedelta(days=7):
                result.errors.append({'category': 'COVERAGE_UNSUPPORTED', 'reason': 'spftll has recent responses only, not date/season historical lookup'})
        elif source == 'football_data':
            headers = {'X-Auth-Token': config.football_data_token}
            if match_ids:
                for mid in match_ids:
                    get(f'https://api.football-data.org/v4/matches/{mid}', headers=headers)
            else:
                for competition in config.competitions:
                    # This endpoint returns the entire filtered result, no page parameter.
                    windows = [(start, end)]
                    if season is None:
                        windows = []
                        # API v4 dateTo is EXCLUSIVE and dates are UTC. Expand
                        # local day edges to UTC whole days rather than omit
                        # early-morning Asia/Shanghai fixtures.
                        zone = ZoneInfo(config.timezone)
                        cursor = datetime.combine(start, time.min, zone).astimezone(timezone.utc).date()
                        right_edge = datetime.combine(end + timedelta(days=1), time.min, zone).astimezone(timezone.utc)
                        stop_exclusive = right_edge.date() + (timedelta(days=1) if right_edge.time() != time.min else timedelta())
                        while cursor < stop_exclusive:
                            stop = min(stop_exclusive, cursor + timedelta(days=10))
                            windows.append((cursor, stop))
                            cursor = stop
                    for left, right in windows:
                        params = {'season': season} if season is not None else {'dateFrom': str(left), 'dateTo': str(right)}
                        get(f'https://api.football-data.org/v4/competitions/{competition}/matches', params,
                            headers=headers, context={'competition': competition, 'requested_season': season,
                                                      'requested_local_start': str(start), 'requested_local_end': str(end), 'timezone': config.timezone})
            result.scope['competitions'] = config.competitions
        else:
            credits = 0
            for sport in config.odds_sports:
                if not sport.startswith('soccer_'):
                    result.errors.append({'category': 'UNSUPPORTED_SPORT', 'sport': sport})
                    continue
                cost = len(config.odds_markets.split(',')) * len(config.odds_regions.split(',')) * (10 if historical_at else 1)
                if credits + cost > config.odds_credit_budget or not reserve(archive.root, 'odds_api', cost, config.odds_credit_budget):
                    result.errors.append({'category': 'RATE_LIMITED', 'reason': 'local_credit_budget'})
                    break
                # Account for all retry attempts, not just successful operations.
                client.retries = 1
                credits += cost
                params = {'apiKey': config.odds_api_key, 'regions': config.odds_regions,
                          'markets': config.odds_markets, 'oddsFormat': 'decimal', 'dateFormat': 'iso'}
                prefix = 'historical/' if historical_at else ''
                if historical_at:
                    params['date'] = historical_at.isoformat().replace('+00:00', 'Z')
                if match_ids and not historical_at:
                    params['eventIds'] = ','.join(match_ids)
                get(f'https://api.the-odds-api.com/v4/{prefix}sports/{sport}/odds', params,
                    context={'historical': bool(historical_at), 'requested_snapshot': str(historical_at) if historical_at else None})
            result.scope.update({'sports': config.odds_sports, 'estimated_credits_requested': credits})
            result.scope['date_filter_semantics'] = 'provider_current_events_or_explicit_historical_snapshot_not_backfill_date_window'
    finally:
        client.close()
        result.observations = [x.id for x in client.observations]
        result.requests = len(client.observations)
    if result.errors:
        result.status = RunStatus.PARTIAL if any(x.http_status == 200 for x in client.observations) else (
            RunStatus.RATE_LIMITED if any(x['category'] == 'RATE_LIMITED' for x in result.errors) else RunStatus.FAILED)
    elif not result.observations:
        result.status = RunStatus.NO_DATA
    return result
