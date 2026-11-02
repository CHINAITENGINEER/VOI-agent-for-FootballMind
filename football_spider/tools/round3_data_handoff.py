"""Offline round-three season freeze and explicit result-time handoff.

Uses retained round-two PostgreSQL backup in fresh loopback-only validation
clusters. No collector or refresh service is invoked; no HTTP request is made.
All new outputs, Parquet files, databases and backups stay under round3/data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_spider.features.store import FeatureStore
from football_spider.services.export import export_season
from football_spider.services.reader import utc
from football_spider.storage.postgres import Repository
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes
from football_spider.tools.pg_test_cluster import local_cluster
from football_spider.tools.round2_verify import pg, counts

OUT = ROOT / 'var/parallel/collectors/round3/data'
DATA = ROOT / 'football_spider/platform_data'
BACKUP = ROOT / 'var/parallel/collectors/round2/footballmind-round2-followup-validation.dump'
RESULTS = ROOT / ('var/parallel/collectors/round2/results/'
    'results-export-v1-31797adea32d326cd39e76f55be063b56d601d9519e6afff14a9e1b75a1e0711')
OBSERVATION_ID = '843eefe4-1c04-45c4-a3ae-b319bf6096e9'


class ReadOnlyRawView:
    """Read actual production raw, but export new features only in role outputs."""
    def __init__(self, archive, output_root):
        self.archive, self.root = archive, Path(output_root)

    def observations(self, source=None):
        return self.archive.observations(source)

    def read_body(self, obs):
        return self.archive.read_body(obs)


def verify_complete(directory):
    directory = Path(directory).resolve()
    raw = (directory / 'manifest.json').read_bytes()
    marker = json.loads((directory / 'COMPLETE.json').read_bytes())
    if hashlib.sha256(raw).hexdigest() != marker['manifest_sha256']:
        raise ValueError('COMPLETE manifest hash differs')
    manifest = json.loads(raw)
    rows = {}
    for name, spec in manifest['files'].items():
        path = (directory / spec['path']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Manifest path escape')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != spec['sha256']:
            raise ValueError('Manifest file hash differs')
        rows[name] = [json.loads(line) for line in content.splitlines() if line.strip()]
        if len(rows[name]) != spec['row_count']:
            raise ValueError('Manifest row count differs')
    return manifest, rows


def time_handoff(repo, destination):
    manifest, source_rows = verify_complete(RESULTS)
    ids = [row['version_id'] for row in source_rows['result_versions']]
    database_rows = repo.conn.execute(
        'SELECT id,source_published_at,observed_at,ingested_at FROM fm.match_versions WHERE id=ANY(%s::uuid[])',
        (ids,)).fetchall()
    times = {str(row['id']): row for row in database_rows}
    if len(times) != len(set(ids)):
        raise ValueError('Frozen result version missing from restored warehouse')

    def projected(row):
        stored = times[row['version_id']]
        if utc(row['observed_at']) != stored['observed_at'] or utc(row['ingested_at']) != stored['ingested_at']:
            raise ValueError('Frozen and database version timestamps differ')
        available = max(stored['observed_at'], stored['ingested_at'])
        return {**row, 'source_published_at': stored['source_published_at'],
                'source_published_at_missing_reason': 'not_supplied' if stored['source_published_at'] is None else None,
                'operational_available_at': available,
                'label_available_at': available if row['settlement_status'] == 'final' else None,
                'time_contract': 'observed_at=response receipt; ingested_at=database commit; operational_available_at=max(both)',
                'original_result_export_version': manifest['dataset_version']}

    matches = [projected(row) for row in source_rows['matches']]
    versions = [projected(row) for row in source_rows['result_versions']]
    issues = []
    for row in source_rows['quality']:
        raw = repo.conn.execute('SELECT observed_at,ingested_at FROM fm.raw_observations WHERE id=%s',
                               (row['observation_id'],)).fetchone()
        if not raw or raw['observed_at'] != utc(row['observed_at']):
            raise ValueError('Quarantine evidence timestamp missing or changed')
        issues.append({**row, 'source_published_at': None,
                       'source_published_at_missing_reason': 'not_supplied',
                       'ingested_at': raw['ingested_at'],
                       'operational_available_at': max(raw['observed_at'], raw['ingested_at']),
                       'disposition': 'QUARANTINED_RETAINED_NOT_A_SETTLEMENT_LABEL'})
    status_counts = dict(Counter(row['settlement_status'] for row in matches))
    if status_counts != {'pending': 6, 'final': 12} or len(versions) != 187 or len(issues) != 5:
        raise ValueError('Specified frozen result scope differs from expected source manifest')
    if any(row['rule'] != 'HALF_EXCEEDS_FULL' for row in issues):
        raise ValueError('Unexpected quality conflict semantics')
    files = {}
    for name, records in [('matches', matches), ('result_versions', versions), ('quality', issues),
                          ('id_mappings', source_rows['id_mappings'])]:
        content = b''.join(json_bytes(row) + b'\n' for row in records)
        atomic_write(destination / (name + '.jsonl'), content)
        files[name] = {'path': name + '.jsonl', 'sha256': hashlib.sha256(content).hexdigest(), 'row_count': len(records)}
    output_manifest = {
        'schema': 'footballmind.platform.result-time-handoff.v1',
        'source_export_directory': str(RESULTS),
        'source_manifest_sha256': hashlib.sha256((RESULTS / 'manifest.json').read_bytes()).hexdigest(),
        'source_dataset_version': manifest['dataset_version'], 'information_mode': 'operational',
        'files': files, 'counts': status_counts, 'http_requests': 0,
        'score_or_status_revisions_introduced': 0,
        'contract': 'Preserve receipt and commit timestamps separately; availability may be later than observed_at',
        'ownership': 'Research owns settlement/ledger integration; platform does not modify predictions or consumers',
    }
    content = json_bytes(output_manifest)
    atomic_write(destination / 'manifest.json', content)
    atomic_write(destination / 'COMPLETE.json', json_bytes({'manifest_sha256': hashlib.sha256(content).hexdigest()}))
    verify_complete(destination)
    return {'directory': str(destination), 'manifest': output_manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing', action='store_true', help='Check completed outputs without starting a database')
    parser.add_argument('--consumer-check', action='store_true', help='Read completed export using existing research loaders; no training')
    args = parser.parse_args()
    if args.consumer_check:
        report = json.loads((OUT / 'report.json').read_bytes())
        if report['status'] != 'OK':
            raise ValueError('Last handoff is not complete')
        directory = report['export']['directory']
        verify_complete(directory)
        # Acceptance-only invocation of the consumer. No production dependency
        # or modification of the research package is introduced here.
        from football_research.data_adapters.platform_history import load_completed_history_export
        from football_research.data_adapters.batch_features import load_feature_index
        dataset = load_completed_history_export(directory)
        index = load_feature_index(directory, strict=True)
        joined, join_report = index.assemble_training_rows(dataset.matches,
            feature_version='prematch-season-v2.0', lead_hours=24,
            required_features=('home_elo', 'away_elo', 'home_rolling_xg_for', 'away_rolling_xg_for'),
            information_mode='historical_reconstruction', allow_missing_values=False)
        accepted = index.report()
        result = {'status': 'OK', 'run_id': report['run_id'], 'http_requests': 0,
                  'database_started': False, 'training_performed': False,
                  'directory': directory, 'matches': len(dataset.matches),
                  'accepted_snapshots': accepted['accepted_snapshots'],
                  'rejected_snapshots': accepted['rejected_rows'],
                  'join_report': join_report, 'joined_rows': len(joined)}
        atomic_write(OUT / 'consumer-check.json', json_bytes(result))
        print(json.dumps({k: v for k, v in result.items() if k != 'join_report'}))
        return
    if args.verify_existing:
        report = json.loads((OUT / 'report.json').read_bytes())
        if report['status'] != 'OK':
            raise ValueError('Last handoff is not complete')
        verify_complete(report['export']['directory'])
        verify_complete(report['results']['directory'])
        if hashlib.sha256(Path(report['backup']).read_bytes()).hexdigest() != report['backup_sha256']:
            raise ValueError('Backup hash differs')
        print(json.dumps({'status': 'OK', 'http_requests': 0, 'database_started': False,
                          'run_id': report['run_id'], 'export': report['export']['directory']}))
        return

    run_id = str(uuid4())
    run_dir = OUT / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = RawArchive(DATA)
    working_archive = ReadOnlyRawView(archive, run_dir / 'feature-workspace')
    observation = next(row for row in archive.observations('understat') if row.id == OBSERVATION_ID)
    archive.read_body(observation)
    report = {'run_id': run_id, 'status': 'RUNNING', 'started_at': datetime.now(timezone.utc).isoformat(),
              'environment': 'isolated_validation_not_production', 'http_requests': 0,
              'source_backup': str(BACKUP), 'source_backup_sha256': hashlib.sha256(BACKUP.read_bytes()).hexdigest(),
              'observation': observation.model_dump(mode='json'), 'actual_raw_archive_root': str(DATA),
              'season_role': '2024/25 earlier than seen 2025/26; training/internal forward validation, not future confirmation'}
    started = time.monotonic()

    def save(stage):
        report['stage'] = stage
        atomic_write(run_dir / 'report.json', json_bytes(report))
        atomic_write(OUT / 'report.json', json_bytes(report))
        print(stage, flush=True)

    try:
        with local_cluster(run_dir / 'pg') as (url, cluster):
            report['cluster'] = str(cluster)
            pg('pg_restore', '--no-owner', '--no-privileges', '-d', url, BACKUP)
            repo = Repository(url)
            try:
                repo.migrate()
                repo.migrate()
                report['before'] = counts(repo)
                control = repo.conn.execute("SELECT id FROM fm.feature_snapshots WHERE feature_version='prematch-season-v2.0' ORDER BY prediction_cutoff LIMIT 1").fetchone()
                store = FeatureStore(repo, working_archive.root / 'features')
                report['old_feature_before'] = store.replay_snapshot(str(control['id']))
                report['processing'] = repo.process(observation, archive)
                if report['processing']['status'] != 'OK' or report['processing']['valid'] != 380 or report['processing']['statistics'] != 1520:
                    raise ValueError('Unexpected new-season processing result')
                report['after_ingest'] = counts(repo)
                report['replay'] = repo.process(observation, archive)
                report['after_replay'] = counts(repo)
                if report['after_replay'] != report['after_ingest'] or report['replay']['status'] != 'SKIPPED':
                    raise ValueError('Observation replay was not idempotent')
                report['old_feature_after'] = store.replay_snapshot(str(control['id']))
                if report['old_feature_before'] != report['old_feature_after']:
                    raise ValueError('Older frozen feature changed after new observation')
                save('restored_ingested_and_idempotence_verified')
                report['results'] = time_handoff(repo, run_dir / 'result-time-handoff')
                save('zero_network_result_time_handoff_frozen')
                report['export'] = export_season(repo, working_archive, run_dir / 'exports',
                    source='understat', season='2024', observation_id=OBSERVATION_ID,
                    information_mode='historical_reconstruction', lead_hours=(24, 1))
                manifest, rows = verify_complete(report['export']['directory'])
                if len(rows['matches']) != 380 or len(rows['features']) != 760 or rows['feature_exclusions']:
                    raise ValueError('Unexpected normalized frozen season coverage')
                repeated = export_season(repo, working_archive, run_dir / 'exports',
                    source='understat', season='2024', observation_id=OBSERVATION_ID,
                    information_mode='historical_reconstruction', lead_hours=(24, 1))
                if not repeated['reused'] or repeated['manifest'] != manifest:
                    raise ValueError('Frozen export replay differs')
                report['export_reused_on_repeat'] = True
                report['feature_replays'] = [store.replay_snapshot(rows['features'][i]['snapshot_id']) for i in (0, 380, 759)]
                report['final_counts'] = counts(repo)
                backup = run_dir / 'footballmind-round3-adjacent-validation.dump'
                pg('pg_dump', '--format=custom', '--file', backup, url)
                report['backup'] = str(backup)
                report['backup_sha256'] = hashlib.sha256(backup.read_bytes()).hexdigest()
                save('season_export_complete_and_backup_written')
            finally:
                repo.close()
        report['validation_database_state'] = 'STOPPED_RETAINED'
        with local_cluster(run_dir / 'pg') as (url, cluster):
            report['restore_cluster'] = str(cluster)
            pg('pg_restore', '--no-owner', '--no-privileges', '-d', url, report['backup'])
            recovered = Repository(url)
            try:
                recovered.migrate()
                report['restored_counts'] = counts(recovered)
                if report['restored_counts'] != report['final_counts']:
                    raise ValueError('Restored backup counts differ')
                report['restored_feature_replay'] = FeatureStore(recovered, working_archive.root / 'features').replay_snapshot(rows['features'][-1]['snapshot_id'])
            finally:
                recovered.close()
        report['restore_database_state'] = 'STOPPED_RETAINED'
        report['status'] = 'OK'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        save('handoff_complete')
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = {'category': type(exc).__name__, 'message': str(exc)[:600]}
        save('handoff_failed')
        raise
    print(json.dumps({'status': report['status'], 'http_requests': 0,
                      'export_directory': report['export']['directory'],
                      'result_time_handoff': report['results']['directory'],
                      'report': str(run_dir / 'report.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
