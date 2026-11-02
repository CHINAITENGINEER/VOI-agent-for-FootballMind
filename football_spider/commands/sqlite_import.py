"""Read-only SQLite import; deterministic snapshot identity makes reruns safe."""
import json
import sqlite3
from pathlib import Path
from football_spider.storage.raw_archive import digest, json_bytes, atomic_write


def import_sqlite(repository, archive, path):
    path = Path(path).resolve()
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        records = [dict(r) for r in conn.execute('SELECT * FROM matches ORDER BY id')]
    finally:
        conn.close()
    items = []
    for row in records:
        detail = json.loads(row.get('details_json') or '{}')
        extras = detail.get('extras', {})
        source_record = {'matchId': row['id'], 'matchTime': row['match_time'], 'league': row['league'],
                         'home': row['home_team'], 'away': row['away_team'],
                         'homeId': extras.get('home_id'), 'awayId': extras.get('away_id'),
                         'homeScore': row['home_score'], 'awayScore': row['away_score'],
                         'homeHalfScore': row.get('home_half_score'), 'awayHalfScore': row.get('away_half_score'),
                         'finishedTitle': row.get('status'), 'spfWinFoot': row['odds_home'],
                         'spfEqualFoot': row['odds_draw'], 'spfLoseFoot': row['odds_away'],
                         'goalFoot': (detail.get('handicap') or {}).get('line'),
                         'rspfWinFoot': (detail.get('handicap') or {}).get('home'),
                         'rspfEqualFoot': (detail.get('handicap') or {}).get('draw'),
                         'rspfLoseFoot': (detail.get('handicap') or {}).get('away')}
        items.append({'id': row['id'], 'source_record': source_record, 'created_time': row['created_time'],
                      'updated_time': row['updated_time'], 'legacy_row': row})
    payload = {'records': items, 'legacy_source_path': str(path)}
    snapshot_hash = digest(payload)
    existing = next((o for o in archive.observations('spftll') if o.context.get('legacy_snapshot_hash') == snapshot_hash), None)
    obs = existing or archive.save('spftll', 'sqlite:///matches', {}, json_bytes(payload), status=200,
                                  context={'legacy_import': True, 'legacy_snapshot_hash': snapshot_hash, 'time_basis': 'import_observed_at_only'})
    result = repository.process(obs, archive)
    imported = repository.conn.execute('''SELECT source_id,record,match_id FROM fm.match_versions
        WHERE observation_id=%s ORDER BY source_id''', (obs.id,)).fetchall()
    by_id = {r['source_id']: r for r in imported}
    differences = []
    mappings = []
    for row in records:
        actual = by_id.get(row['id'])
        if not actual:
            differences.append({'id': row['id'], 'field': 'missing_import'})
            continue
        mappings.append({'legacy_id': row['id'], 'internal_id': str(actual['match_id'])})
        for column, value in [('home_team', actual['record']['home']['name']), ('away_team', actual['record']['away']['name']),
                              ('league', actual['record']['competition']['name']), ('home_score', actual['record']['score']['home']), ('away_score', actual['record']['score']['away'])]:
            if row[column] != value:
                differences.append({'id': row['id'], 'field': column})
    report = {'source_rows': len(records), 'imported_versions': len(imported), 'unique_source_ids': len(by_id),
              'differences': differences, 'id_map': mappings, 'observation_id': obs.id,
              'processing': result, 'source_untouched': True, 'timestamp_policy': 'no_legacy_timestamp_backdating',
              'status': 'OK' if not differences and len(records) == len(imported) else 'PARTIAL'}
    atomic_write(archive.root / 'reports' / ('sqlite-import-' + obs.id + '.json'), json_bytes(report))
    return report
