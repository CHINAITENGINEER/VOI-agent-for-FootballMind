from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pydantic import ValidationError
from football_spider.models.contracts import EntityRef, Issue, MatchRecord, NormalizedBatch, Quote, Score


def stamp(value, zone='UTC'):
    if value in (None, ''):
        return None
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        result = result.replace(tzinfo=ZoneInfo(zone))
    return result.astimezone(timezone.utc)


def number(value):
    if value in (None, ''):
        return None
    # Int conversion must not silently truncate a fractional score or accept bool.
    if isinstance(value, bool) or float(value) != int(float(value)):
        raise ValueError('Non-integral score')
    return int(value)


def team(sid, name, mid, side, category='unknown', country=None):
    return EntityRef(source_id=str(sid) if sid is not None else f'match:{mid}:{side}',
                     name=str(name or ''), category=category, country=country,
                     id_quality='stable' if sid is not None else 'match_scoped')


def spftll_match(row):
    mid = str(row['matchId'])
    league = row.get('leagueChs') or row.get('league') or ''
    league_id = row.get('leagueId')
    raw_time = row.get('matchTime')
    title = str(row.get('finishedTitle') or '')
    state = row.get('footMatchState')
    status = {'未开': 'scheduled', '完场': 'finished', '推迟': 'postponed', '延期': 'postponed', '取消': 'cancelled', '中断': 'suspended'}.get(title)
    if status is None:
        status = {-1: 'finished', 0: 'scheduled'}.get(state, 'unknown')
    # Only the pilot domestic leagues have regulation scoring accepted here.
    # Cup/unknown phase remains unknown rather than treating a shootout as 1x2.
    category = 'men_senior' if league in {'英超', '德甲', '西甲', '意甲', '法甲'} else 'unknown'
    if '女' in league:
        category = 'women_unspecified_age'
    period = 'regulation' if category == 'men_senior' else 'unknown'
    missing = {'season': 'source_does_not_supply_season', 'source_published_at': 'not_verified', 'xg': 'not_integrated'}
    if period == 'unknown':
        missing['regulation_score'] = 'source_phase_not_verified'
    score = Score(home=number(row.get('homeScore')), away=number(row.get('awayScore')))
    if score.home is None or score.away is None:
        missing['score'] = 'not_yet_reported' if status != 'finished' else 'source_missing'
    return MatchRecord(source_id=mid,
        competition=EntityRef(source_id=str(league_id) if league_id is not None else f'name:{league}', name=league,
                              id_quality='stable' if league_id is not None else 'source_name'),
        home=team(row.get('homeId'), row.get('homeChs') or row.get('home'), mid, 'home', category),
        away=team(row.get('awayId'), row.get('awayChs') or row.get('away'), mid, 'away', category),
        kickoff=stamp(raw_time, 'Asia/Shanghai'), status=status, score=score, score_period=period,
        half_score=Score(home=number(row.get('homeHalfScore')), away=number(row.get('awayHalfScore'))),
        original_time=str(raw_time), source_timezone='Asia/Shanghai', missing=missing,
        coverage='selected_matches_incomplete',
        evidence={'source_status': state, 'source_status_title': title,
                  'startTime_uninterpreted': row.get('startTime'), 'leagueId': league_id,
                  'home_rank_raw': row.get('homeRank'), 'away_rank_raw': row.get('awayRank')})


def issue(batch, key, exc, rule='SCHEMA_VALIDATION'):
    batch.issues.append(Issue(rule=rule, severity='error', record_key=str(key),
                              message=type(exc).__name__, evidence={'detail': str(exc)[:1000]}))


def add_quote(batch, **data):
    value = data.get('decimal_price')
    if value in (None, ''):
        batch.issues.append(Issue(rule='MISSING_ODDS', severity='warning',
                                  record_key=str(data['source_match_id']), message='No decimal price',
                                  evidence={'market': data['market_type'], 'selection': data['selection']}))
        return
    try:
        batch.quotes.append(Quote(**data))
    except (ValidationError, ValueError, TypeError) as exc:
        issue(batch, data.get('source_match_id'), exc, 'INVALID_ODDS')


def spftll_quotes(batch, row, more=False):
    mid = str(row['matchId'])
    names = [('home', 'spf_win' if more else 'spfWinFoot'), ('draw', 'spf_draw' if more else 'spfEqualFoot'), ('away', 'spf_lost' if more else 'spfLoseFoot')]
    for selection, key in names:
        add_quote(batch, source_match_id=mid, provider='spftll', market_type='1x2', period='regulation',
                  selection=selection, decimal_price=row.get(key), settlement_rule='90_minutes_plus_stoppage_excludes_extra_time_penalties')
    line = row.get('rfspf_goal' if more else 'goalFoot')
    for selection, key in [('home', 'rfspf_win' if more else 'rspfWinFoot'), ('draw', 'rfspf_draw' if more else 'rspfEqualFoot'), ('away', 'rfspf_lost' if more else 'rspfLoseFoot')]:
        add_quote(batch, source_match_id=mid, provider='spftll', market_type='jc_handicap_3way', period='regulation',
                  selection=selection, line=line, decimal_price=row.get(key), settlement_rule='home_score_plus_line_three_way_regulation')
    if more:
        for i in range(8):
            add_quote(batch, source_match_id=mid, provider='spftll', market_type='exact_total_goals', period='regulation',
                      selection=str(i) if i < 7 else '7+', decimal_price=row.get(f't{i}'), settlement_rule='regulation_exact_goals_7_is_7_or_more')


def normalize_spftll(obs, payload):
    if payload.get('code') != 0:
        raise ValueError(f"spftll business code {payload.get('code')}")
    batch = NormalizedBatch()
    data = payload.get('data')
    if 'data' in payload and data is None and obs.endpoint.endswith(('GetInfo','GetMoreSpInfo','GetList','GetSimpleMatchsAll')):
        # Observed provider code=0,data=null means no data, not malformed facts.
        return batch
    rows = []
    if obs.endpoint.endswith('GetSimpleMatchsAll'):
        if not isinstance(data, list):
            raise ValueError('Expected data array')
        rows = [r for g in data for r in g.get('list', [])]
        for row in rows:
            spftll_quotes(batch, row)
    elif obs.endpoint.endswith('GetInfo'):
        if not isinstance(data, dict):
            raise ValueError('Expected detail object')
        # Historical rows keep TODAY's observation time, never their kickoff as availability.
        rows = [data] + [r for key in ('battleList', 'homeBattleList', 'awayBattleList') for r in data.get(key) or []]
    elif obs.endpoint.endswith('GetMoreSpInfo'):
        more = (data or {}).get('footballMoreSpInfo')
        if more:
            spftll_quotes(batch, more, more=True)
    elif obs.endpoint.endswith('GetList'):
        rows = [r for group in data or [] for r in group.get('matchList') or []]
        # messageList preserved raw only: dynamicKind / team attribution unverified.
    else:
        raise ValueError('Unsupported spftll endpoint')
    seen = set()
    for row in rows:
        key = row.get('matchId')
        if key in seen:
            continue
        try:
            batch.matches.append(spftll_match(row))
            seen.add(key)
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            issue(batch, key, exc)
    return batch


def normalize_football_data(obs, payload):
    if 'matches' not in payload and 'id' not in payload:
        raise ValueError('Expected football-data match collection/object')
    batch = NormalizedBatch()
    for row in payload.get('matches', [payload]):
        try:
            competition = row.get('competition') or payload.get('competition') or {}
            season = row.get('season') or payload.get('season') or {}
            score = row.get('score') or {}
            category = 'men_senior' if competition.get('code') in {'PL', 'BL1', 'PD', 'SA', 'FL1'} else 'unknown'
            duration = score.get('duration')
            regular = score.get('fullTime', {}) if duration == 'REGULAR' else score.get('regularTime', {})
            half = score.get('halfTime') or {}
            missing = {}
            if not regular or regular.get('home') is None:
                missing['score'] = 'regulation_not_reported'
            batch.matches.append(MatchRecord(source_id=str(row['id']),
                competition=EntityRef(source_id=str(competition['id']), name=competition['name']),
                season=str(season['id']) if season.get('id') is not None else None,
                home=team(row['homeTeam'].get('id'), row['homeTeam'].get('name'), row['id'], 'home', category),
                away=team(row['awayTeam'].get('id'), row['awayTeam'].get('name'), row['id'], 'away', category),
                kickoff=stamp(row['utcDate']), original_time=row['utcDate'], source_timezone='UTC',
                status={'SCHEDULED': 'scheduled', 'TIMED': 'scheduled', 'IN_PLAY': 'live', 'PAUSED': 'live',
                        'FINISHED': 'finished', 'POSTPONED': 'postponed', 'SUSPENDED': 'suspended',
                        'CANCELLED': 'cancelled'}.get(row.get('status'), 'unknown'),
                stage=row.get('stage'), score_period='regulation' if regular else 'unknown',
                score=Score(home=number(regular.get('home')), away=number(regular.get('away'))),
                half_score=Score(home=number(half.get('home')), away=number(half.get('away'))),
                # lastUpdated is revision time, NOT evidence the original fact was public then.
                missing=missing, evidence={'lastUpdated': row.get('lastUpdated'), 'duration': duration,
                    'raw_score': score, 'competition_code': competition.get('code'), 'season': season},
                coverage='competition_endpoint_scope_not_yet_audited'))
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            issue(batch, row.get('id'), exc)
    return batch


def normalize_odds_api(obs, payload):
    historical = obs.context.get('historical', False)
    snapshot = stamp(payload.get('timestamp')) if historical and isinstance(payload, dict) else None
    if historical and snapshot is None:
        raise ValueError('Historical response missing snapshot timestamp')
    if snapshot and snapshot > stamp(obs.context['requested_snapshot']):
        raise ValueError('Snapshot later than requested')
    rows = payload.get('data') if historical else payload
    if not isinstance(rows, list):
        raise ValueError('Expected odds event array')
    batch = NormalizedBatch()
    for row in rows:
        mid = row.get('id')
        try:
            competition = row['sport_key']
            category = 'men_senior' if competition in {'soccer_epl', 'soccer_germany_bundesliga', 'soccer_italy_serie_a', 'soccer_spain_la_liga', 'soccer_france_ligue_one'} else 'unknown'
            match = MatchRecord(source_id=mid,
                competition=EntityRef(source_id=competition, name=row.get('sport_title') or competition),
                home=EntityRef(source_id=f"{competition}:{row['home_team']}", name=row['home_team'], category=category, id_quality='source_name'),
                away=EntityRef(source_id=f"{competition}:{row['away_team']}", name=row['away_team'], category=category, id_quality='source_name'),
                kickoff=stamp(row['commence_time']), status='unknown', original_time=row['commence_time'], source_timezone='UTC',
                trusted_snapshot_at=snapshot, missing={'score': 'odds_endpoint', 'season': 'not_supplied'}, coverage='bookmaker_selected_events')
            batch.matches.append(match)
            for book in row.get('bookmakers', []):
                for market in book.get('markets', []):
                    market_type = {'h2h': '1x2', 'spreads': 'asian_handicap', 'totals': 'over_under'}.get(market['key'])
                    if market_type is None:
                        continue
                    for outcome in market.get('outcomes', []):
                        selection = {row['home_team']: 'home', row['away_team']: 'away', 'Draw': 'draw', 'Over': 'over', 'Under': 'under'}.get(outcome['name'], outcome['name'])
                        add_quote(batch, source_match_id=mid, provider='odds_api', bookmaker=book['key'], market_type=market_type,
                                  period='regulation', selection=selection, line=outcome.get('point'), decimal_price=outcome.get('price'),
                                  source_timestamp=stamp(market.get('last_update') or book.get('last_update')),
                                  trusted_snapshot_at=snapshot,
                                  settlement_rule='regulation_selection_line' if market_type == 'asian_handicap' else '90_minutes_plus_stoppage_excludes_extra_time_penalties')
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            issue(batch, mid, exc)
    return batch


def normalize(obs, payload):
    if obs.source == 'understat':
        from football_spider.normalization.understat import normalize_understat
        return normalize_understat(obs, payload)
    if obs.context.get('legacy_import'):
        batch = NormalizedBatch()
        for row in payload['records']:
            try:
                record = spftll_match(row['source_record'])
                record.evidence['legacy_time_uncertainty'] = {'created_time': row['created_time'], 'updated_time': row['updated_time'], 'policy': 'available_only_at_import_observation'}
                batch.matches.append(record)
                spftll_quotes(batch, row['source_record'])
            except (ValueError, KeyError, TypeError) as exc:
                issue(batch, row.get('id'), exc)
        return batch
    return {'spftll': normalize_spftll, 'football_data': normalize_football_data, 'odds_api': normalize_odds_api}[obs.source](obs, payload)
