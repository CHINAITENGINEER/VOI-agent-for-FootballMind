from collections import defaultdict
from football_spider.models.contracts import Issue


def validate_batch(batch, observed_at, max_score=15):
    issues = list(batch.issues)
    for m in batch.matches:
        def add(rule, severity, message):
            issues.append(Issue(rule=rule, severity=severity, record_key=m.source_id, message=message))
        if not m.home.name.strip() or not m.away.name.strip() or not m.competition.name.strip():
            add('REQUIRED_ENTITY', 'error', 'Missing team or competition name')
        if m.home.source_id == m.away.source_id:
            add('SAME_TEAM', 'error', 'Home and away have the same source ID')
        if m.status == 'finished':
            if m.kickoff > observed_at:
                add('FINISHED_IN_FUTURE', 'error', 'Finished match has future kickoff')
            if m.score.home is None or m.score.away is None:
                add('FINISHED_MISSING_SCORE', 'warning', 'Finished result missing; excluded from labels/features')
        if m.status in {'scheduled', 'cancelled', 'postponed'} and any(x is not None for x in (m.score.home, m.score.away)):
            add('STATUS_SCORE_CONFLICT', 'warning', 'Score retained but unavailable as finished label')
        for side in ('home', 'away'):
            full, half = getattr(m.score, side), getattr(m.half_score, side)
            if m.score_period == 'regulation' and full is not None and half is not None and half > full:
                add('HALF_EXCEEDS_FULL', 'error', 'Half-time score exceeds regulation score')
            if full is not None and full > max_score:
                add('EXTREME_SCORE', 'warning', 'Unusual but not automatically rejected score')
        if m.home.id_quality != 'stable' or m.away.id_quality != 'stable':
            add('PROVISIONAL_TEAM_ID', 'warning', 'Source has no stable team ID; manual resolution required')
        if m.coverage != 'verified_complete':
            add('INCOMPLETE_HISTORY', 'warning', 'League history coverage is not established')
    markets = defaultdict(list)
    for q in batch.quotes:
        if q.source_timestamp and q.source_timestamp > observed_at:
            issues.append(Issue(rule='FUTURE_QUOTE_TIMESTAMP', severity='warning', record_key=q.source_match_id, message='Source timestamp later than observed; PIT query must gate it'))
        # Asian handicap line is selection-relative; group opposing signs together.
        line = abs(q.line) if q.market_type == 'asian_handicap' and q.line is not None else q.line
        markets[(q.source_match_id, q.bookmaker, q.market_type, q.period, line)].append(q)
        if q.decimal_price > 1000:
            issues.append(Issue(rule='EXTREME_ODDS', severity='warning', record_key=q.source_match_id, message='Large valid decimal quote'))
    for (mid, _, market, _, _), quotes in markets.items():
        selections = {q.selection for q in quotes}
        expected = {'1x2': {'home', 'draw', 'away'}, 'jc_handicap_3way': {'home', 'draw', 'away'},
                    'asian_handicap': {'home', 'away'}, 'over_under': {'over', 'under'},
                    'exact_total_goals': {str(x) for x in range(7)} | {'7+'}}[market]
        if selections != expected or len(selections) != len(quotes):
            issues.append(Issue(rule='INCOMPLETE_MARKET', severity='warning', record_key=mid, message=f'Expected selections for {market}', evidence={'actual': sorted(selections)}))
        if market in {'asian_handicap', 'jc_handicap_3way', 'over_under'} and any(q.line is None for q in quotes):
            issues.append(Issue(rule='MISSING_LINE', severity='error', record_key=mid, message='Line market without a line'))
        if market == 'jc_handicap_3way' and any(q.line is not None and not float(q.line).is_integer() for q in quotes):
            issues.append(Issue(rule='JC_NON_INTEGER_LINE', severity='error', record_key=mid, message='JC three-way handicap requires integer goal line'))
        if market == 'asian_handicap' and len(quotes) == 2 and all(q.line is not None for q in quotes) and abs(sum(q.line for q in quotes)) > 1e-9:
            issues.append(Issue(rule='HANDICAP_LINE_MISMATCH', severity='error', record_key=mid, message='Opposite selection lines must sum to zero'))
    return issues
