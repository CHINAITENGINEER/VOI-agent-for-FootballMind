"""REAL-data acceptance in a dedicated PostgreSQL validation cluster.

No fixtures and no production database edits. Archives/manifests and a pg_dump
are kept; validation server is stopped on exit. Report says validation, not prod.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import os
import subprocess
from datetime import timedelta
from football_spider.platform_config import PlatformSettings, ROOT
from football_spider.models.contracts import now
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes
from football_spider.storage.postgres import Repository
from football_spider.commands.pipeline import run_pipeline
from football_spider.commands.sqlite_import import import_sqlite
from football_spider.features.store import FeatureStore
from football_spider.scheduler.jobs import plan, tick
from football_spider.identity.resolver import stable_id
from football_spider.tools.pg_test_cluster import local_cluster


def main():
    config = PlatformSettings()
    archive = RawArchive(config.data_dir)
    report_path = config.data_dir / 'reports' / 'real-acceptance.json'
    report = {'environment': 'isolated_validation_postgresql_not_production', 'started_at': now().isoformat(),
              'production_database_configured': bool(config.database_url), 'fixtures_used': False}
    with local_cluster() as (url, cluster):
        report['cluster_path'] = str(cluster)
        test_config = config.model_copy(update={'database_url': url, 'detail_limit': 2, 'environment': 'validation'})
        repo = Repository(url)
        try:
            repo.migrate()
            repo.migrate()
            report['migrations_rerun'] = True
            report['import_first'] = import_sqlite(repo, archive, ROOT / 'data' / 'matches.db')
            report['import_repeat'] = import_sqlite(repo, archive, ROOT / 'data' / 'matches.db')
            print('SQLite reconciliation:', report['import_first']['status'], report['import_first']['source_rows'], flush=True)
            # Replay successful prior real HTTP responses, no repeat network calls.
            ids = [o.id for o in archive.observations('spftll') if o.http_status == 200]
            report['initial_replay'] = run_pipeline(test_config, sources=['spftll'], replay=True, observation_ids=ids, build_features=False)
            print('Replay:', report['initial_replay']['status'], report['initial_replay']['totals'], flush=True)
            target = repo.conn.execute('''SELECT * FROM fm.latest_matches WHERE source='spftll' AND event_time>%s
                AND record->'competition'->>'name' IN ('英超','德甲') ORDER BY event_time LIMIT 1''', (now() + timedelta(minutes=10),)).fetchone()
            if target is None:
                report['status'] = 'PARTIAL'
                report['reason'] = 'No real pilot match with future kickoff; run normal schedule sync first'
            else:
                mid, source_mid = str(target['match_id']), target['source_id']
                report['target'] = {'match_id': mid, 'source_id': source_mid, 'home': target['record']['home']['name'],
                                    'away': target['record']['away']['name'], 'kickoff': target['event_time'].isoformat()}
                first = run_pipeline(test_config, sources=['spftll'], match_ids=[source_mid], build_features=False)
                report['first_run'] = first
                cutoff = now()
                feature_store = FeatureStore(repo, config.data_dir / 'features')
                first_feature = feature_store.get_features(mid, cutoff)
                second = run_pipeline(test_config, sources=['spftll'], match_ids=[source_mid], build_features=False)
                report['second_run'] = second
                second_feature = feature_store.get_features(mid, cutoff)
                report['point_in_time_unchanged'] = first_feature['manifest_hash'] == second_feature['manifest_hash'] and first_feature['features'] == second_feature['features']
                report['feature'] = first_feature
                report['manifest_replay'] = feature_store.replay_snapshot(first_feature['snapshot_id'])
                report['new_cutoff_feature'] = feature_store.get_features(mid, now())
                all_ids = first['sources']['spftll']['observations'] + second['sources']['spftll']['observations']
                report['repeat_replay'] = run_pipeline(test_config, sources=['spftll'], replay=True, observation_ids=all_ids, build_features=False)
                report['same_match_observations'] = repo.conn.execute('''SELECT v.observation_id,v.observed_at,v.content_hash,o.content_hash AS raw_content_hash,
                    v.status,v.record->'score' AS score FROM fm.match_versions v JOIN fm.raw_observations o ON o.id=v.observation_id
                    WHERE v.match_id=%s AND v.observation_id=ANY(%s::uuid[]) ORDER BY observed_at''', (mid, all_ids)).fetchall()
                report['odds_observations'] = repo.conn.execute('''SELECT raw_observation_id,observed_at,market_type,selection,line,decimal_price
                    FROM fm.odds_history WHERE match_id=%s AND raw_observation_id=ANY(%s::uuid[]) ORDER BY observed_at,market_type,selection''', (mid, all_ids)).fetchall()
                report['schedule_plan'] = plan(repo, test_config)
                # Offline wake-up test with no jobs due: validates entry and locks,
                # not a claim that a permanent Windows task is enabled.
                repo.conn.execute("UPDATE fm.scheduled_jobs SET status='done' WHERE kind<>'prematch'")
                repo.conn.execute("UPDATE fm.scheduled_jobs SET status='done' WHERE due_at<=clock_timestamp()")
                report['manual_scheduler_tick'] = tick(test_config)
                report['status'] = 'OK' if first['status'] == second['status'] == 'OK' and report['point_in_time_unchanged'] else 'PARTIAL'
                print('Real two-observation acceptance:', report['status'], report['target'], flush=True)
                # Exercise the full default source flow and automatic feature stage.
                report['full_run'] = run_pipeline(test_config)
                print('Full run:', report['full_run']['status'], report['full_run']['totals'], flush=True)
                atomic_write(ROOT / 'docs' / 'evidence' / 'platform-real-run.json', json_bytes(report['full_run']))
            # Missing providers reported by actual collector logic, no requests
            # made for missing credentials. Existing accounts are not purchased.
            missing = [s for s, key in [('football_data', config.football_data_token), ('odds_api', config.odds_api_key)] if not key]
            if missing:
                report['missing_provider_run'] = run_pipeline(test_config, sources=missing, build_features=False)
            report['database_counts'] = repo.status(config.timezone)
            binary = Path(os.getenv('PG_BIN', 'D:/PostgreSQL/bin'))
            dump = config.data_dir / 'validation' / 'footballmind-real-validation.dump'
            dump.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(binary / 'pg_dump'), '-d', url, '-Fc', '-f', str(dump)], check=True, timeout=60,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            report['database_dump'] = str(dump)
            # Verify recoverability by restoring to ANOTHER DB in this dedicated
            # validation cluster, never to an existing user database.
            repo.conn.execute('CREATE DATABASE footballmind_restore')
            restore_url = url.rsplit('/', 1)[0] + '/footballmind_restore'
            subprocess.run([str(binary / 'pg_restore'), '-d', restore_url, '--no-owner', '--no-privileges', str(dump)],
                           check=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            restored = Repository(restore_url)
            try:
                restored.migrate()
                restored_counts = restored.status(config.timezone)
                compared = ['raw_observations', 'matches', 'match_versions', 'odds_history', 'feature_snapshots', 'match_labels']
                report['restore_verification'] = {'counts_equal': all(restored_counts[k] == report['database_counts'][k] for k in compared),
                                                  'compared': compared}
                if report.get('feature'):
                    report['restore_verification']['feature_replay'] = FeatureStore(restored, config.data_dir / 'features').replay_snapshot(report['feature']['snapshot_id'])
            finally:
                restored.close()
        finally:
            repo.close()
            report['finished_at'] = now().isoformat()
            atomic_write(report_path, json_bytes(report))
    report['validation_server_stopped'] = True
    report['automatic_scheduler_registered'] = False
    report['single_source_acceptance'] = report['status']
    report['production_database_status'] = 'CONFIGURED_NOT_VALIDATED_BY_THIS_SCRIPT' if config.database_url else 'NEEDS_DATABASE'
    if not config.database_url or not config.football_data_token or not config.odds_api_key:
        report['status'] = 'PARTIAL'
    atomic_write(report_path, json_bytes(report))
    print('Report:', report_path, flush=True)


if __name__ == '__main__':
    main()
