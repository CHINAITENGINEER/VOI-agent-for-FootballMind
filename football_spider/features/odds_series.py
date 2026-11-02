"""Observed quote sequence summaries; no fabricated true opening/closing price."""
from collections import defaultdict
from football_spider.features.store import select_matches


def summarize(repo, match_id, cutoff):
    versions = repo.conn.execute('''SELECT *,greatest(observed_at,ingested_at) AS available_at FROM fm.match_versions
        WHERE match_id=%s AND observed_at<=%s AND ingested_at<=%s''', (match_id, cutoff, cutoff)).fetchall()
    target = select_matches(versions).get(str(match_id))
    if target is None:
        raise ValueError('Match unknown at cutoff')
    rows = repo.conn.execute('''SELECT * FROM fm.odds_history WHERE match_id=%s AND observed_at<=%s AND ingested_at<=%s
        AND observed_at<%s AND (source_timestamp IS NULL OR source_timestamp<=%s) ORDER BY observed_at,id''',
        (match_id, cutoff, cutoff, target['event_time'], cutoff)).fetchall()
    groups = defaultdict(list)
    for q in rows:
        groups[(q['provider'], q['bookmaker'], q['market_type'], q['period'], q['selection'], q['line'], q['settlement_rule'])].append(q)
    result = []
    for key, series in groups.items():
        first, last = series[0], series[-1]
        result.append({'provider': key[0], 'bookmaker': key[1], 'market_type': key[2], 'period': key[3], 'selection': key[4], 'line': key[5], 'settlement_rule': key[6],
                       'observations': len(series), 'first_observed_price': first['decimal_price'], 'latest_observed_price': last['decimal_price'],
                       'price_change': last['decimal_price'] - first['decimal_price'], 'true_opening_price': None,
                       'last_captured_pre_kickoff_price': last['decimal_price'] if cutoff >= target['event_time'] else None,
                       'true_closing_price': None, 'first_quote_id': str(first['id']), 'last_quote_id': str(last['id']),
                       'definition': 'first/latest captured before known kickoff, not guaranteed bookmaker open/close'})
    return {'status': 'OK' if result else 'NO_DATA', 'match_id': match_id, 'as_of': cutoff.isoformat(), 'series': result}
