from __future__ import annotations
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.types.json import Jsonb
from football_spider.models.contracts import now
from football_spider.storage.raw_archive import digest, json_bytes
from football_spider.identity.resolver import stable_id

VERSION = 'prematch-v1.1'
PARAMETERS = {'initial_elo': 1500.0, 'k': 20.0, 'home_advantage': 65.0, 'recent_n': 5,
              'provider_priority': ['football_data', 'spftll', 'odds_api', 'spftll_legacy'], 'market_max_age_seconds': 86400}


def select_matches(rows):
    """Per-source latest known version, then explicit provider precedence."""
    latest = {}
    for row in rows:
        key = (str(row['match_id']), row['source'])
        previous = latest.get(key)
        if previous is None or (row.get('version_time', row['observed_at']), row['observed_at'], str(row['id'])) > (previous.get('version_time', previous['observed_at']), previous['observed_at'], str(previous['id'])):
            latest[key] = row
    selected = {}
    priority = PARAMETERS['provider_priority']
    for row in latest.values():
        mid = str(row['match_id'])
        rank = priority.index(row['source']) if row['source'] in priority else 99
        if mid not in selected or rank < selected[mid][0]:
            selected[mid] = (rank, row)
    return {k: v[1] for k, v in selected.items()}


def compute(target, history, quotes, cutoff):
    target_id = str(target['match_id'])
    home, away = str(target['home_id']), str(target['away_id'])
    eligible = [r for r in history if str(r['match_id']) != target_id and r['event_time'] < cutoff
                and r['record']['status'] == 'finished' and r['record']['score_period'] == 'regulation'
                and None not in r['record']['score'].values()]
    eligible.sort(key=lambda r: (r['event_time'], str(r['match_id'])))
    ratings = defaultdict(lambda: PARAMETERS['initial_elo'])
    games = defaultdict(list)
    lineage = []
    # All already-known results of the target competition are used. Sparse
    # sampling is explicitly flagged; it is NOT interpreted as a full schedule.
    for row in eligible:
        hid, aid = str(row['home_id']), str(row['away_id'])
        hs, aws = row['record']['score']['home'], row['record']['score']['away']
        expected = 1 / (1 + 10 ** ((ratings[aid] - ratings[hid] - PARAMETERS['home_advantage']) / 400))
        actual = 1.0 if hs > aws else 0.0 if hs < aws else 0.5
        change = PARAMETERS['k'] * (actual - expected)
        ratings[hid] += change
        ratings[aid] -= change
        games[hid].append((row, hs, aws, 'home'))
        games[aid].append((row, aws, hs, 'away'))
        lineage.append(str(row['id']))

    def form(tid, venue):
        recent = games[tid][-PARAMETERS['recent_n']:]
        by_venue = [x for x in games[tid] if x[3] == venue][-PARAMETERS['recent_n']:]
        def summary(values):
            return {'matches': len(values), 'points': sum(3 if gf > ga else 1 if gf == ga else 0 for _, gf, ga, _ in values) if values else None,
                    'goals_for': sum(gf for _, gf, _, _ in values) if values else None,
                    'goals_against': sum(ga for _, _, ga, _ in values) if values else None,
                    'version_ids': [str(r['id']) for r, *_ in values]}
        return {**summary(recent), 'venue': summary(by_venue), 'history_matches_observed': len(games[tid]),
                'rest_days': (target['event_time'] - recent[-1][0]['event_time']).total_seconds() / 86400 if recent else None,
                'rest_days_definition': 'since_last_observed_finished_regulation_match_in_this_competition_not_all_competitions',
                'coverage': 'unverified_incomplete', 'missing_reason': 'incomplete_league_history' if recent else 'no_known_eligible_history'}

    values = {'elo': {'home': ratings[home] if games[home] else None, 'away': ratings[away] if games[away] else None,
                      'difference': ratings[home] - ratings[away] if games[home] and games[away] else None,
                      'input_version_ids': lineage, 'coverage': 'unverified_incomplete', 'parameters': PARAMETERS},
              'home_form': form(home, 'home'), 'away_form': form(away, 'away'),
              'xg': {'home': None, 'away': None, 'reason': 'xg_provider_not_integrated'},
              'market': {'probabilities': None, 'reason': 'no_complete_semantically_consistent_market'}}
    groups = defaultdict(list)
    for q in quotes:
        if q['market_type'] == '1x2' and q['period'] == 'regulation' and q['line'] is None:
            groups[(str(q['raw_observation_id']), q['provider'], q['bookmaker'], q['settlement_rule'])].append(q)
    valid = []
    for key, items in groups.items():
        if len(items) == 3 and {q['selection'] for q in items} == {'home', 'draw', 'away'}:
            if len({q['source_timestamp'] for q in items}) > 1:
                continue
            valid.append((max(q.get('version_time', q['observed_at']) for q in items), key, items))
    quote_ids = []
    if valid:
        _, key, chosen = max(valid, key=lambda g: (g[0], str(g[1])))
        age_base = min(q['source_timestamp'] or q.get('version_time', q['observed_at']) for q in chosen)
        age = max(0, (cutoff - age_base).total_seconds())
        weights = {q['selection']: 1 / q['decimal_price'] for q in chosen}
        overround = sum(weights.values())
        quote_ids = sorted(str(q['id']) for q in chosen)
        values['market'] = {'probabilities': {k: v / overround for k, v in weights.items()} if age <= PARAMETERS['market_max_age_seconds'] else None,
                            'overround': overround, 'age_seconds': age, 'provider': key[1], 'bookmaker': key[2],
                            'quote_ids': quote_ids, 'reason': None if age <= PARAMETERS['market_max_age_seconds'] else 'stale_quotes',
                            'definition': 'normalized_inverse_decimal_prices_not_a_forecast'}
    return values, lineage, quote_ids


class FeatureStore:
    def __init__(self, repository, output_dir):
        self.repo = repository
        self.output_dir = Path(output_dir)

    def get_features(self, match_id, as_of, feature_version=VERSION, information_mode='operational', assumed_delay_seconds=None, *,version_ids=None,assumptions=None):
        if feature_version == 'prematch-season-v2.0':
            from football_spider.features.season import SeasonFeatureStore
            return SeasonFeatureStore(self).get_features(match_id,as_of,information_mode,version_ids=version_ids,assumptions=assumptions)
        if feature_version != VERSION:
            raise ValueError('Unknown feature version')
        if as_of.tzinfo is None:
            raise ValueError('Prediction cutoff requires timezone')
        if information_mode not in {'operational', 'historical_reconstruction'}:
            raise ValueError('Unknown information mode')
        if information_mode == 'historical_reconstruction' and (assumed_delay_seconds is None or assumed_delay_seconds < 0):
            raise ValueError('Historical reconstruction requires an explicit nonnegative assumed delay')
        with self.repo.conn.transaction():
            self.repo.conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            if information_mode == 'operational':
                # Processing availability is also gated; late parser changes and
                # imports cannot rewrite what the system actually knew before.
                matches = self.repo.conn.execute('''SELECT *, greatest(observed_at,ingested_at) AS available_at FROM fm.match_versions
                    WHERE observed_at<=%s AND ingested_at<=%s''', (as_of, as_of)).fetchall()
                quotes = self.repo.conn.execute('''SELECT *, greatest(observed_at,ingested_at) AS available_at FROM fm.odds_history
                    WHERE match_id=%s AND observed_at<=%s AND ingested_at<=%s AND (source_timestamp IS NULL OR source_timestamp<=%s)''', (match_id, as_of, as_of, as_of)).fetchall()
            else:
                limit = as_of - timedelta(seconds=assumed_delay_seconds)
                matches = self.repo.conn.execute('''SELECT *, trusted_snapshot_at+(%s * interval '1 second') AS available_at FROM fm.match_versions
                    WHERE trusted_snapshot_at IS NOT NULL AND trusted_snapshot_at<=%s''', (assumed_delay_seconds, limit)).fetchall()
                quotes = self.repo.conn.execute('''SELECT *, trusted_snapshot_at+(%s * interval '1 second') AS available_at FROM fm.odds_history
                    WHERE match_id=%s AND trusted_snapshot_at IS NOT NULL AND trusted_snapshot_at<=%s
                    AND (source_timestamp IS NULL OR source_timestamp<=%s)''', (assumed_delay_seconds, match_id, limit, as_of)).fetchall()
                for record in matches + quotes:
                    record['version_time'] = record['trusted_snapshot_at']
            selected = select_matches(matches)
            if str(match_id) not in selected:
                raise ValueError('No target match known at requested cutoff/mode')
            target = selected[str(match_id)]
            if target['status'] not in ('scheduled', 'unknown'):
                raise ValueError('Target is not an active prematch fixture at cutoff')
            if as_of >= target['event_time']:
                raise ValueError('Prematch feature cutoff must precede known kickoff')
            history = [r for r in selected.values() if r['competition_id'] == target['competition_id']]
            values, inputs, quote_ids = compute(target, history, quotes, as_of)
            manifest = {'match_id': str(match_id), 'prediction_cutoff': as_of.isoformat(), 'feature_version': feature_version,
                        'information_mode': information_mode, 'assumed_delay_seconds': assumed_delay_seconds,
                        'availability_basis': 'max(observed_at,ingested_at)' if information_mode == 'operational' else 'trusted_provider_snapshot_plus_assumed_delay',
                        'parameters': PARAMETERS, 'target_version': str(target['id']), 'history_version_ids': sorted(inputs),
                        'quote_ids': quote_ids,
                        'observation_ids': sorted({str(r['observation_id']) for r in history if str(r['id']) in set(inputs)} | {str(target['observation_id'])} |
                                                 {str(q['raw_observation_id']) for q in quotes if str(q['id']) in set(quote_ids)})}
            manifest_hash = digest(manifest)
            row = self.repo.conn.execute('''INSERT INTO fm.feature_snapshots(id,match_id,prediction_cutoff,feature_version,information_mode,input_manifest_hash,manifest,features)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(match_id,prediction_cutoff,feature_version,information_mode,input_manifest_hash)
                DO UPDATE SET input_manifest_hash=EXCLUDED.input_manifest_hash RETURNING *''',
                (str(uuid4()), match_id, as_of, feature_version, information_mode, manifest_hash, Jsonb(manifest), Jsonb(values))).fetchone()
        self.export(row)
        for team_id, side in ((target['home_id'], 'home'), (target['away_id'], 'away')):
            state_key = digest([str(team_id), as_of.isoformat(), feature_version, manifest_hash])
            self.repo.conn.execute('''INSERT INTO fm.team_state_history(id,team_id,calculation_cutoff,method_version,state,input_manifest_hash)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
                (stable_id('features', 'team_state', state_key), team_id, as_of, feature_version,
                 Jsonb({'elo': row['features']['elo'][side], 'form': row['features'][side + '_form'], 'feature_snapshot_id': str(row['id']), 'coverage': 'unverified_incomplete'}), manifest_hash))
        return {'snapshot_id': str(row['id']), 'manifest_hash': manifest_hash, 'manifest': manifest, 'features': row['features'], 'parquet_path': str(self.output_dir / f'{row["id"]}.parquet')}

    def export(self, row):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dest = self.output_dir / f'{row["id"]}.parquet'
        fd, temp = tempfile.mkstemp(suffix='.parquet', dir=self.output_dir)
        os.close(fd)
        try:
            table = pa.Table.from_pylist([{'match_id': str(row['match_id']), 'prediction_cutoff': row['prediction_cutoff'],
                                          'feature_version': row['feature_version'], 'information_mode': row['information_mode'],
                                          'input_manifest_hash': row['input_manifest_hash'],
                                          'features_json': json_bytes(row['features']).decode(), 'manifest_json': json_bytes(row['manifest']).decode()}])
            pq.write_table(table, temp)
            os.replace(temp, dest)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.repo.conn.execute('UPDATE fm.feature_snapshots SET parquet_path=%s WHERE id=%s', (str(dest), row['id']))

    def replay_snapshot(self, snapshot_id):
        row = self.repo.conn.execute('SELECT * FROM fm.feature_snapshots WHERE id=%s', (snapshot_id,)).fetchone()
        if not row:
            raise ValueError('Unknown snapshot')
        if row['feature_version']=='prematch-season-v2.0':
            from football_spider.features.season import SeasonFeatureStore
            return SeasonFeatureStore(self).replay(row)
        if row['feature_version'] != VERSION:
            raise ValueError('Snapshot requires its original feature-version implementation')
        manifest = row['manifest']
        if digest(manifest) != row['input_manifest_hash']:
            raise ValueError('Manifest hash mismatch')
        target = self.repo.conn.execute('SELECT * FROM fm.match_versions WHERE id=%s', (manifest['target_version'],)).fetchone()
        history = self.repo.conn.execute('SELECT * FROM fm.match_versions WHERE id=ANY(%s::uuid[])', (manifest['history_version_ids'],)).fetchall()
        quotes = self.repo.conn.execute('SELECT * FROM fm.odds_history WHERE id=ANY(%s::uuid[])', (manifest['quote_ids'],)).fetchall()
        for q in quotes:
            q['available_at'] = max(q['observed_at'], q['ingested_at']) if row['information_mode'] == 'operational' else q['trusted_snapshot_at'] + timedelta(seconds=manifest['assumed_delay_seconds'])
            q['version_time'] = q['observed_at'] if row['information_mode'] == 'operational' else q['trusted_snapshot_at']
        if len(history) != len(manifest['history_version_ids']) or len(quotes) != len(manifest['quote_ids']):
            raise ValueError('Missing immutable input records')
        values, _, _ = compute(target, history, quotes, row['prediction_cutoff'])
        if values != row['features']:
            raise ValueError('Feature replay mismatch')
        self.export(row)
        return {'snapshot_id': str(row['id']), 'consistent': True, 'manifest_hash': row['input_manifest_hash']}
