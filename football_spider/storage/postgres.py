from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from football_spider.models.contracts import EntityRef, Issue, PARSER_VERSION, SCHEMA_VERSION, now
from football_spider.identity.resolver import IdentityResolver, stable_id
from football_spider.normalization.sources import normalize
from football_spider.quality.rules import validate_batch
from football_spider.quality.entity_mapping import entity_mapping_state
from football_spider.storage.raw_archive import digest


def parser_version(obs):
    return 'understat-1.0' if obs.source == 'understat' else PARSER_VERSION


class Repository:
    def __init__(self, url):
        if not url:
            raise ValueError('DATABASE_URL_MISSING')
        self.conn = psycopg.connect(url, connect_timeout=5, autocommit=True, row_factory=dict_row)
        self.conn.execute("SET TIME ZONE 'UTC'")
        self.identity = IdentityResolver(self.conn)

    def close(self):
        self.conn.close()

    def migrate(self):
        with self.conn.transaction():
            self.conn.execute('SELECT pg_advisory_xact_lock(718643219)')
            self.conn.execute('CREATE SCHEMA IF NOT EXISTS fm')
            self.conn.execute('CREATE TABLE IF NOT EXISTS fm.schema_migrations(version text PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT clock_timestamp())')
            for path in sorted((Path(__file__).parent / 'migrations').glob('*.sql')):
                raw = path.read_bytes()
                checksum = hashlib.sha256(raw).hexdigest()
                previous = self.conn.execute('SELECT checksum FROM fm.schema_migrations WHERE version=%s', (path.name,)).fetchone()
                if previous:
                    if previous['checksum'] != checksum:
                        raise ValueError('Applied migration checksum changed: ' + path.name)
                    continue
                self.conn.execute(raw.decode('utf-8'))
                self.conn.execute('INSERT INTO fm.schema_migrations(version,checksum) VALUES (%s,%s)', (path.name, checksum))

    def register_observation(self, obs):
        previous = self.conn.execute('SELECT source,content_hash,observed_at FROM fm.raw_observations WHERE id=%s',(obs.id,)).fetchone()
        if previous and (previous['source'] != obs.source or previous['content_hash'] != obs.content_hash or previous['observed_at'] != obs.observed_at):
            raise ValueError('Immutable raw observation collision')
        self.conn.execute('''INSERT INTO fm.raw_observations(id,source,observed_at,content_hash,metadata)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
            (obs.id, obs.source, obs.observed_at, obs.content_hash, Jsonb(obs.model_dump(mode='json'))))

    def save_issue(self, obs, issue):
        data = issue.model_dump(mode='json')
        self.conn.execute('''INSERT INTO fm.quality_issues(observation_id,rule,rule_version,severity,record_key,issue_key,detail)
            VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
            (obs.id, issue.rule, issue.rule_version, issue.severity, issue.record_key, digest([obs.id, parser_version(obs), data]), Jsonb(data)))

    def process(self, obs, archive, force=False):
        self.register_observation(obs)
        version = parser_version(obs)
        previous = self.conn.execute('SELECT * FROM fm.processing WHERE observation_id=%s AND parser_version=%s', (obs.id, version)).fetchone()
        if previous and previous['status'] in ('OK', 'NO_DATA', 'QUARANTINED') and not force:
            return {'valid': 0, 'quarantined': 0, 'skipped': 1, 'failed': 0, 'quotes': 0, 'statistics':0, 'status': 'SKIPPED'}
        counts = {'valid': 0, 'quarantined': 0, 'skipped': 0, 'failed': 0, 'quotes': 0, 'statistics':0, 'status': 'OK'}
        try:
            if obs.http_status != 200:
                raise ValueError('HTTP_RESPONSE_NOT_SUCCESSFUL')
            batch = normalize(obs, json.loads(archive.read_body(obs)))
            issues = validate_batch(batch, obs.observed_at)
            counts['entity_mapping_state'] = entity_mapping_state(batch, issues)
            counts['parser_version'] = version
            hard = {x.record_key for x in issues if x.severity == 'error'}
            with self.conn.transaction():
                # Serializes ingestion/mapping identity creation across platform workers.
                self.conn.execute('SELECT pg_advisory_xact_lock(718643220)')
                for issue in issues:
                    self.save_issue(obs, issue)
                for key in hard:
                    record = next((m.model_dump(mode='json') for m in batch.matches if m.source_id == key), {'raw_observation': obs.id})
                    self.conn.execute('''INSERT INTO fm.quarantined_records(observation_id,parser_version,record_key,record)
                        VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING''', (obs.id, version, key, Jsonb(record)))
                counts['quarantined'] = len(hard)
                for match in batch.matches:
                    if match.source_id not in hard:
                        counts['valid'] += self.save_match(obs, match)
                for statistic in batch.statistics:
                    if statistic.source_match_id not in hard:
                        counts['statistics'] += self.save_statistic(obs,statistic)
                for quote in batch.quotes:
                    if quote.source_match_id in hard:
                        continue
                    mapping = self.conn.execute("SELECT entity_id FROM fm.entity_source_map WHERE source=%s AND entity_type='match' AND source_id=%s AND valid_to IS NULL", (obs.source, quote.source_match_id)).fetchone()
                    if not mapping:
                        self.save_issue(obs, Issue(rule='UNRESOLVED_QUOTE_MATCH', severity='recoverable', record_key=quote.source_match_id, message='Replay after match identity is ingested'))
                        counts['failed'] += 1
                        continue
                    q = quote.model_dump(mode='json')
                    key = digest([obs.id, version, q])
                    result = self.conn.execute('''INSERT INTO fm.odds_history(id,match_id,provider,bookmaker,market_type,period,selection,line,decimal_price,
                        source_timestamp,observed_at,trusted_snapshot_at,raw_observation_id,parser_version,settlement_rule,quote_key)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(quote_key) DO NOTHING''',
                        (stable_id('quote', 'quote', key), mapping['entity_id'], quote.provider, quote.bookmaker, quote.market_type, quote.period,
                         quote.selection, quote.line, quote.decimal_price, quote.source_timestamp, obs.observed_at,
                         quote.trusted_snapshot_at, obs.id, version, quote.settlement_rule, key))
                    counts['quotes'] += result.rowcount
                counts['status'] = 'PARTIAL' if counts['failed'] else 'QUARANTINED' if hard else 'OK' if batch.matches or batch.quotes else 'NO_DATA'
                self._processed(obs, counts)
        except Exception as exc:
            counts.update({'valid': 0, 'quotes': 0, 'statistics':0, 'failed': 1, 'status': 'FAILED', 'error_class': type(exc).__name__,
                           'entity_mapping_state': 'not_evaluated', 'parser_version': version})
            # Rolled back facts, retained raw index. If the DB itself is gone the
            # outer pipeline retains the file-based run report and raw payloads.
            self.save_issue(obs, Issue(rule='PROCESSING_FAILED', severity='recoverable', record_key=obs.id, message=type(exc).__name__))
            self._processed(obs, counts, type(exc).__name__)
        return counts

    def _processed(self, obs, counts, error=None):
        self.conn.execute('''INSERT INTO fm.processing(observation_id,parser_version,status,counts,error_class)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT(observation_id,parser_version) DO UPDATE
            SET status=EXCLUDED.status,counts=EXCLUDED.counts,error_class=EXCLUDED.error_class,
                attempts=fm.processing.attempts+1,updated_at=clock_timestamp()''', (obs.id, parser_version(obs), counts['status'], Jsonb(counts), error))

    def save_match(self, obs, match):
        version = parser_version(obs)
        fact_source = 'spftll_legacy' if obs.context.get('legacy_import') else obs.source
        duplicate = self.conn.execute('SELECT 1 FROM fm.match_versions WHERE observation_id=%s AND parser_version=%s AND source_id=%s', (obs.id, version, match.source_id)).fetchone()
        if duplicate:
            return 0
        home = self.identity.resolve(obs.source, 'team', match.home, obs.observed_at)
        away = self.identity.resolve(obs.source, 'team', match.away, obs.observed_at)
        if home == away:
            raise ValueError('MAPPED_HOME_AWAY_COLLISION')
        comp = self.identity.resolve(obs.source, 'competition', match.competition, obs.observed_at)
        season = None
        if match.season:
            season = self.identity.resolve(obs.source, 'season', EntityRef(source_id=f'{match.competition.source_id}:{match.season}', name=match.season), obs.observed_at)
        mid = self.identity.resolve(obs.source, 'match', EntityRef(source_id=match.source_id, name=f'{match.home.name} - {match.away.name}'), obs.observed_at)
        candidates = self.conn.execute('''SELECT DISTINCT match_id FROM fm.latest_matches
            WHERE ((home_id=%s AND away_id=%s) OR (home_id=%s AND away_id=%s))
            AND match_id<>%s AND abs(extract(epoch FROM (event_time-%s::timestamptz)))<172800''', (home, away, away, home, mid, match.kickoff)).fetchall()
        for c in candidates:
            self.identity.candidate(obs.source, 'match', match.source_id, c['match_id'], 'same_teams_near_kickoff_review_competition', {'kickoff': match.kickoff.isoformat(), 'competition': comp})
        previous = self.conn.execute('''SELECT * FROM fm.match_versions WHERE match_id=%s AND source=%s AND observed_at<=%s
            ORDER BY observed_at DESC,ingested_at DESC LIMIT 1''', (mid, fact_source, obs.observed_at)).fetchone()
        record = match.model_dump(mode='json')
        if previous and (str(previous['home_id']), str(previous['away_id'])) != (home, away):
            self.save_issue(obs, Issue(rule='SOURCE_ID_RELATION_CONFLICT', severity='warning', record_key=match.source_id,
                                       message='Same source match ID changed teams or home-away relation; versions retained',
                                       evidence={'previous_version': str(previous['id'])}))
            self.identity.candidate(obs.source, 'match', match.source_id, mid, 'source_id_relation_change', {'previous_version': str(previous['id']), 'observation_id': obs.id})
        if previous and previous['record']['status'] == 'finished' and previous['record']['score'] != record['score']:
            self.save_issue(obs, Issue(rule='RESULT_REVISION', severity='warning', record_key=match.source_id, message='Prior result retained', evidence={'previous_version': str(previous['id'])}))
        conflicts = self.conn.execute("SELECT id,record FROM fm.latest_matches WHERE match_id=%s AND source<>%s AND status='finished'", (mid, obs.source)).fetchall()
        for conflict in conflicts:
            if match.status == 'finished' and conflict['record']['score_period'] == match.score_period and conflict['record']['score'] != record['score']:
                self.save_issue(obs, Issue(rule='CROSS_SOURCE_SCORE_CONFLICT', severity='warning', record_key=match.source_id, message='Result disagreement', evidence={'other_version': str(conflict['id'])}))
        vid = stable_id(obs.source, 'version', f'{obs.id}:{version}:{match.source_id}')
        first = self.conn.execute('SELECT min(observed_at) AS value FROM fm.match_versions WHERE match_id=%s AND source=%s', (mid, fact_source)).fetchone()['value']
        self.conn.execute('''INSERT INTO fm.match_versions(id,match_id,source,source_id,observation_id,parser_version,schema_version,
            home_id,away_id,competition_id,season_id,event_time,source_published_at,first_observed_at,observed_at,valid_from,supersedes_id,
            trusted_snapshot_at,content_hash,status,record) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            (vid, mid, fact_source, match.source_id, obs.id, version, SCHEMA_VERSION, home, away, comp, season,
             match.kickoff, match.source_published_at, min(first, obs.observed_at) if first else obs.observed_at,
             obs.observed_at, obs.observed_at, previous['id'] if previous else None, match.trusted_snapshot_at, digest(record), match.status, Jsonb(record)))
        if match.status == 'finished' and match.score_period == 'regulation' and match.score.home is not None and match.score.away is not None:
            label = 'home' if match.score.home > match.score.away else 'away' if match.score.home < match.score.away else 'draw'
            self.conn.execute('''INSERT INTO fm.match_labels(version_id,match_id,label,period,revealed_by,time_basis)
                VALUES (%s,%s,%s,'regulation',%s,'first_observation_of_this_result_version_not_exact_final_whistle')''', (vid, mid, label, obs.observed_at))
        return 1

    def save_statistic(self,obs,statistic):
        version=parser_version(obs)
        match=self.conn.execute('SELECT id,match_id,event_time FROM fm.match_versions WHERE observation_id=%s AND parser_version=%s AND source_id=%s',
                                (obs.id,version,statistic.source_match_id)).fetchone()
        team=self.conn.execute("SELECT entity_id FROM fm.entity_source_map WHERE source=%s AND entity_type='team' AND source_id=%s AND valid_to IS NULL",
                               (obs.source,statistic.source_team_id)).fetchone()
        if not match or not team:
            raise ValueError('UNRESOLVED_STATISTIC_RELATION')
        key=digest([obs.id,version,statistic.model_dump(mode='json')])
        result=self.conn.execute('''INSERT INTO fm.team_statistics(id,team_id,match_id,window_end,metric,value,unit,provider,definition_version,
            observed_at,observation_id,match_version_id,parser_version,evidence,source_published_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
            (stable_id(obs.source,'statistic',key),team['entity_id'],match['match_id'],match['event_time'],statistic.metric,statistic.value,
             statistic.unit,statistic.provider,statistic.definition_version,obs.observed_at,obs.id,match['id'],version,
             Jsonb(statistic.evidence),statistic.source_published_at))
        return result.rowcount

    def save_run(self, manifest):
        with self.conn.transaction():
            self.conn.execute('''INSERT INTO fm.pipeline_runs(id,started_at,finished_at,status,manifest) VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT(id) DO UPDATE SET finished_at=EXCLUDED.finished_at,status=EXCLUDED.status,manifest=EXCLUDED.manifest''',
                (manifest['run_id'], manifest['started_at'], manifest.get('finished_at'), manifest['status'], Jsonb(manifest)))
            for source, result in manifest.get('sources', {}).items():
                self.conn.execute('''INSERT INTO fm.collector_runs(run_id,source,status,report) VALUES (%s,%s,%s,%s)
                    ON CONFLICT(run_id,source) DO UPDATE SET status=EXCLUDED.status,report=EXCLUDED.report''', (manifest['run_id'], source, result['status'], Jsonb(result)))

    def status(self, timezone='Asia/Shanghai'):
        counts = {}
        for table in ('raw_observations', 'match_versions', 'matches', 'odds_history', 'quality_issues', 'quarantined_records', 'feature_snapshots', 'match_labels'):
            counts[table] = self.conn.execute(f'SELECT count(*) AS n FROM fm.{table}').fetchone()['n']
        counts['pending_entities'] = self.conn.execute("SELECT count(*) AS n FROM fm.mapping_candidates WHERE status='pending'").fetchone()['n']
        counts['today_matches'] = self.conn.execute('''SELECT count(DISTINCT match_id) AS n FROM fm.latest_matches
            WHERE (event_time AT TIME ZONE %s)::date=(clock_timestamp() AT TIME ZONE %s)::date''', (timezone, timezone)).fetchone()['n']
        counts['source_freshness'] = self.conn.execute('''SELECT source,max(observed_at) AS last_observation,
            extract(epoch FROM(clock_timestamp()-max(observed_at))) AS age_seconds FROM fm.raw_observations GROUP BY source''').fetchall()
        for row in counts['source_freshness']:
            row['age_seconds'] = float(row['age_seconds'])
            row['status'] = 'STALE' if row['age_seconds'] > 86400 else 'RECENT_OBSERVATION_NOT_A_COVERAGE_GUARANTEE'
        counts['field_coverage'] = self.conn.execute('''SELECT source,count(*) AS records,
            count(*) FILTER (WHERE status='finished') AS finished,
            count(*) FILTER (WHERE record->'score'->>'home' IS NOT NULL AND record->'score'->>'away' IS NOT NULL) AS with_score,
            count(*) FILTER (WHERE season_id IS NOT NULL) AS with_season,
            count(*) FILTER (WHERE record->>'score_period'='regulation') AS regulation_period
            FROM fm.latest_matches GROUP BY source''').fetchall()
        counts['quality_by_rule'] = self.conn.execute('SELECT rule,severity,count(*) AS count FROM fm.quality_issues GROUP BY rule,severity ORDER BY rule').fetchall()
        return counts
