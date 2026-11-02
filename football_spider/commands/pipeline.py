from __future__ import annotations
from datetime import timedelta
from uuid import uuid4
import time
from zoneinfo import ZoneInfo
from filelock import FileLock, Timeout
from loguru import logger
from football_spider.collectors.sources import collect
from football_spider.models.contracts import now, RunStatus
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes
from football_spider.storage.postgres import Repository, parser_version
from football_spider.features.store import FeatureStore
from football_spider.commands.evidence import implementation_evidence


def open_repository(config):
    if not config.database_url:
        return None, {'status': 'NEEDS_DATABASE', 'reason': 'DATABASE_URL_MISSING'}
    repo = None
    try:
        repo = Repository(config.database_url)
        repo.migrate()
        return repo, {'status': 'OK'}
    except Exception as exc:
        if repo:
            repo.close()
        return None, {'status': 'NEEDS_DATABASE', 'reason': type(exc).__name__}


def run_pipeline(config, *, start=None, end=None, season=None, match_ids=None, sources=None,
                 historical_at=None, replay=False, observation_ids=None, build_features=True):
    config.data_dir.mkdir(parents=True, exist_ok=True)
    sink = logger.add(config.data_dir / 'logs' / 'platform.log', rotation='10 MB', retention=10, encoding='utf-8')
    lock = FileLock(str(config.data_dir / 'pipeline.lock'))
    try:
        with lock.acquire(timeout=0):
            return _run(config, start=start, end=end, season=season, match_ids=match_ids, sources=sources,
                        historical_at=historical_at, replay=replay, observation_ids=observation_ids, build_features=build_features)
    except Timeout:
        return {'status': 'PARTIAL', 'reason': 'CONCURRENT_RUN_LOCKED', 'run_id': None}
    finally:
        logger.remove(sink)


def _run(config, *, start, end, season, match_ids, sources, historical_at, replay, observation_ids, build_features):
    began = time.monotonic()
    started = now()
    start = start or started.astimezone(ZoneInfo(config.timezone)).date()
    end = end or start + timedelta(days=7)
    if end < start:
        raise ValueError('End date precedes start date')
    archive = RawArchive(config.data_dir)
    repo, database = open_repository(config)
    report = {'run_id': str(uuid4()), 'started_at': started.isoformat(), 'status': 'RUNNING', 'timezone': config.timezone,
              'config': config.public(), 'database': database, 'sources': {}, 'features': [], 'feature_errors': [],
              'operation': 'replay' if replay else 'collect', 'scope': {'start': str(start), 'end': str(end), 'season': season},
              'totals': {'requests': 0, 'raw_observations': 0, 'valid': 0, 'quarantined': 0, 'skipped': 0, 'failed': 0, 'quotes': 0, 'features': 0}}
    path = config.data_dir / 'reports' / (report['run_id'] + '.json')
    atomic_write(path, json_bytes(report))
    source_list = sources or config.sources
    try:
        for source in source_list:
            if replay:
                observed = [o for o in archive.observations(source) if not observation_ids or o.id in observation_ids]
                entry = {'source': source, 'status': 'OK', 'requests': 0, 'observations': [o.id for o in observed], 'errors': []}
            else:
                try:
                    entry = collect(source, config, archive, start=start, end=end, season=season,
                                    match_ids=match_ids, historical_at=historical_at).model_dump(mode='json')
                except Exception as exc:
                    entry = {'source': source, 'status': 'FAILED', 'requests': 0, 'observations': [], 'errors': [{'category': type(exc).__name__}]}
                ids = set(entry['observations'])
                observed = [o for o in archive.observations(source) if o.id in ids]
            counts = {key: 0 for key in ('valid', 'quarantined', 'skipped', 'failed', 'quotes')}
            entry.update(implementation_evidence(source))
            entry['processing'] = []
            # Every match-bearing response precedes quote-only responses on replay.
            ordered = sorted(observed, key=lambda o: (o.endpoint.endswith('GetMoreSpInfo'), o.observed_at))
            if repo:
                for obs in ordered:
                    try:
                        result = repo.process(obs, archive)
                    except Exception as exc:
                        result = {'valid': 0, 'quarantined': 0, 'skipped': 0, 'failed': 1, 'quotes': 0,
                                  'status': 'FAILED', 'error_class': type(exc).__name__,
                                  'entity_mapping_state': 'not_evaluated',
                                  'parser_version': parser_version(obs)}
                        report['database'] = {'status': 'FAILED', 'reason': type(exc).__name__}
                    entry['processing'].append({'observation_id': obs.id, **result})
                    for key in counts:
                        counts[key] += result[key]
                if counts['failed'] or counts['quarantined']:
                    entry['status'] = 'PARTIAL' if counts['valid'] or counts['quotes'] else 'FAILED'
                elif entry['status'] == 'OK' and not counts['valid'] and not counts['quotes'] and not counts['skipped']:
                    entry['status'] = 'NO_DATA'
                try:
                    entry['verified_records_scope'] = repo.conn.execute('''SELECT count(*) AS match_versions,
                        min(event_time) AS earliest_kickoff,max(event_time) AS latest_kickoff,
                        array_agg(DISTINCT record->'competition'->>'name') AS competition_names,
                        array_agg(DISTINCT record->>'season') FILTER (WHERE record->>'season' IS NOT NULL) AS seasons,
                        count(*) FILTER (WHERE record->'score'->>'home' IS NOT NULL AND record->'score'->>'away' IS NOT NULL) AS with_score,
                        count(*) FILTER (WHERE record->'half_score'->>'home' IS NOT NULL AND record->'half_score'->>'away' IS NOT NULL) AS with_half_score
                        FROM fm.match_versions WHERE observation_id=ANY(%s::uuid[])''', ([o.id for o in observed],)).fetchone()
                except Exception as exc:
                    entry['verified_records_scope'] = {'status': 'NEEDS_DATABASE', 'error_class': type(exc).__name__}
                    report['database'] = {'status': 'FAILED', 'reason': type(exc).__name__, 'affected_stages': ['normalization_persistence', 'features', 'database_run_report']}
            elif observed:
                entry['network_status'] = entry['status']
                entry['status'] = 'NEEDS_DATABASE'
            # Keep source-return metadata in the run manifest. The timestamp
            # below is the collector run close, not a prediction cutoff; an
            # Agent decision supplies its own cutoff before Ledger materialization.
            try:
                from football_agent.collector_ledger_adapter import collection_events
                versions = {item.get('parser_version') for item in entry['processing'] if item.get('parser_version')}
                version = next(iter(versions)) if len(versions) == 1 else (
                    f'{source}-mixed-parser' if versions else
                    (parser_version(observed[0]) if observed else f'{source}-unknown'))
                mapping = {item['observation_id']: item.get('entity_mapping_state')
                           for item in entry['processing'] if item.get('observation_id')}
                entry['collector_events'] = collection_events(
                    entry, observed, source_version=version, prediction_cutoff=now().isoformat(),
                    entity_mapping_by_observation=mapping, captured_at=now().isoformat())
                entry['collector_events_cutoff_kind'] = 'collector_run_close_not_prediction_cutoff'
            except Exception as exc:
                entry['collector_events'] = []
                entry['collector_events_error'] = type(exc).__name__
            entry['counts'] = counts
            entry['raw_observations'] = len(observed)
            entry['network_verified'] = any(o.http_status == 200 for o in observed) if not replay else None
            entry['network_verification_date'] = started.date().isoformat() if entry['network_verified'] else None
            entry['feature_snapshots_using_current_observations'] = 0
            report['sources'][source] = entry
            for key, value in counts.items():
                report['totals'][key] += value
            report['totals']['requests'] += entry['requests']
            report['totals']['raw_observations'] += len(observed)
            atomic_write(path, json_bytes(report))
            logger.info('source={} status={} requests={} observations={} valid={} quarantined={} quotes={}', source, entry['status'], entry['requests'], len(observed), counts['valid'], counts['quarantined'], counts['quotes'])
        if repo and report['database']['status'] == 'OK':
            if build_features:
                cutoff = now()
                targets = repo.conn.execute('''SELECT DISTINCT match_id FROM fm.latest_matches
                    WHERE event_time>%s AND event_time<%s AND status IN ('scheduled','unknown')''', (cutoff, cutoff + timedelta(days=8))).fetchall()
                report['feature_targets'] = len(targets)
                store = FeatureStore(repo, config.data_dir / 'features')
                for row in targets:
                    try:
                        feature = store.get_features(str(row['match_id']), cutoff)
                        report['features'].append({k: feature[k] for k in ('snapshot_id', 'manifest_hash', 'parquet_path')})
                        used = set(feature['manifest']['observation_ids'])
                        for entry in report['sources'].values():
                            if used.intersection(entry['observations']):
                                entry['feature_snapshots_using_current_observations'] += 1
                    except Exception as exc:
                        report['feature_errors'].append({'match_id': str(row['match_id']), 'error_class': type(exc).__name__})
                report['totals']['features'] = len(report['features'])
            report['database_counts'] = repo.status(config.timezone)
        statuses = [x['status'] for x in report['sources'].values()]
        report['status'] = 'NEEDS_DATABASE' if not repo else 'PARTIAL' if report['database']['status'] != 'OK' or report['feature_errors'] or any(x not in ('OK', 'NO_DATA', 'SKIPPED_DISABLED') for x in statuses) else 'OK'
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error_class'] = type(exc).__name__
    finally:
        report['finished_at'] = now().isoformat()
        report['elapsed_seconds'] = round(time.monotonic() - began, 3)
        if repo:
            try:
                # JSON adaptation needs datetime/Decimal converted in query summaries.
                import json
                report = json.loads(json_bytes(report))
                repo.save_run(report)
            except Exception as exc:
                report['database_report_error'] = type(exc).__name__
                report['status'] = 'PARTIAL'
            repo.close()
        atomic_write(path, json_bytes(report))
    return report
