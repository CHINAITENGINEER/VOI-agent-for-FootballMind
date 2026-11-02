"""FootballMind CLI. Legacy `python main.py` continues to work unchanged."""
from __future__ import annotations
if __package__ in (None, ''):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from datetime import date, datetime, timedelta
import json
import sys
from pathlib import Path
from football_spider.models.contracts import now
from football_spider.platform_config import PlatformSettings
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes
from football_spider.commands.pipeline import open_repository, run_pipeline
from football_spider.commands.sqlite_import import import_sqlite
from football_spider.features.store import FeatureStore, VERSION


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command')
    sub.add_parser('health')
    sub.add_parser('migrate')
    status = sub.add_parser('status')
    status.add_argument('--run-id')
    status.add_argument('--source')
    for command in ('run-once', 'backfill', 'refresh', 'replay'):
        s = sub.add_parser(command)
        s.add_argument('--sources', help='spftll,football_data,odds_api,understat')
        s.add_argument('--start', type=date.fromisoformat)
        s.add_argument('--end', type=date.fromisoformat)
        s.add_argument('--season', type=int)
        s.add_argument('--match-ids', help='Source match IDs, comma separated')
        s.add_argument('--historical-at', type=datetime.fromisoformat)
        s.add_argument('--observation-ids')
        s.add_argument('--no-features', action='store_true')
    imp = sub.add_parser('import-sqlite')
    imp.add_argument('--path', type=Path, default=Path(__file__).parent / 'data' / 'matches.db')
    f = sub.add_parser('features')
    f.add_argument('--match-id')
    f.add_argument('--as-of', type=datetime.fromisoformat)
    f.add_argument('--competition-id')
    f.add_argument('--season-id')
    f.add_argument('--lead-hours', type=float, default=24)
    f.add_argument('--mode', choices=['operational', 'historical_reconstruction'], default='operational')
    f.add_argument('--assumed-delay-seconds', type=int)
    f.add_argument('--snapshot-id', help='Replay an immutable stored manifest')
    f.add_argument('--version', default=VERSION)
    f.add_argument('--input-bundle',type=Path,help='Frozen season manifest for prematch-season-v2.0 reconstruction')
    research=sub.add_parser('register-research')
    research.add_argument('--manifest',type=Path,required=True)
    season_export=sub.add_parser('export-season')
    season_export.add_argument('--source',default='understat')
    season_export.add_argument('--season',default='2025')
    season_export.add_argument('--observation-id',required=True)
    season_export.add_argument('--output',type=Path,required=True)
    season_export.add_argument('--mode',choices=['operational','historical_reconstruction'],default='historical_reconstruction')
    season_export.add_argument('--lead-hours',type=float,nargs='+',default=[24,1])
    bounded=sub.add_parser('refresh-service')
    bounded.add_argument('--match-id',required=True)
    bounded.add_argument('--topics',default='result')
    bounded.add_argument('--request-id',required=True)
    bounded.add_argument('--max-requests',type=int,default=2)
    bounded.add_argument('--timeout',type=float,default=60)
    for name in ('refresh-results','export-results'):
        result=sub.add_parser(name);result.add_argument('--predictions',type=Path,required=True)
        if name=='refresh-results':
            result.add_argument('--operation-id',required=True);result.add_argument('--max-requests',type=int,default=18)
        else:result.add_argument('--output',type=Path,required=True)
    sub.add_parser('mapping-pending')
    odds = sub.add_parser('odds-history')
    odds.add_argument('--match-id', required=True)
    odds.add_argument('--as-of', type=datetime.fromisoformat, required=True)
    m = sub.add_parser('mapping-confirm')
    for name in ('source', 'entity-type', 'source-id', 'target-id', 'reason'):
        m.add_argument('--' + name, required=True)
    m.add_argument('--reviewed-context-override', action='store_true')
    q = sub.add_parser('quality-resolve')
    q.add_argument('--issue-id', type=int, required=True)
    q.add_argument('--note', required=True)
    sub.add_parser('schedule-plan')
    sub.add_parser('schedule-tick')
    return p


def execute(args, config):
    command = args.command or 'run-once'
    if command=='refresh-service':
        from football_spider.services.refresh import RefreshService
        return RefreshService(config).refresh(args.match_id,args.topics.split(','),args.request_id,args.max_requests,args.timeout)
    if command=='refresh-results':
        from football_spider.services.results import refresh_results,prediction_match_ids
        return refresh_results(config,prediction_match_ids(args.predictions),operation_id=args.operation_id,max_requests=args.max_requests)
    research_observation=None
    if command=='register-research':
        from football_spider.collectors.understat import register_research
        research_observation=register_research(RawArchive(config.data_dir),args.manifest,workspace_root=Path(__file__).resolve().parents[1])
    if command in ('run-once', 'backfill', 'refresh', 'replay'):
        sources = args.sources.split(',') if getattr(args, 'sources', None) else None
        historical = getattr(args, 'historical_at', None)
        if historical and historical.tzinfo is None:
            raise ValueError('--historical-at requires timezone')
        if command == 'backfill' and getattr(args, 'season', None) is None and getattr(args, 'start', None) is None:
            raise ValueError('backfill requires --start or --season')
        if command == 'refresh' and not getattr(args, 'match_ids', None):
            raise ValueError('refresh requires source --match-ids')
        if getattr(args, 'match_ids', None) and (sources is None or len(sources) != 1):
            raise ValueError('Source match IDs require exactly one --sources value')
        return run_pipeline(config, sources=sources, start=getattr(args, 'start', None), end=getattr(args, 'end', None),
                            season=getattr(args, 'season', None), match_ids=args.match_ids.split(',') if getattr(args, 'match_ids', None) else None,
                            historical_at=historical, replay=command == 'replay',
                            observation_ids=args.observation_ids.split(',') if getattr(args, 'observation_ids', None) else None,
                            build_features=not getattr(args, 'no_features', False))
    if command == 'schedule-tick':
        from football_spider.scheduler.jobs import tick
        return tick(config)
    if command == 'status' and args.run_id:
        # File reports remain accessible when the DB is down.
        from uuid import UUID
        run_id = str(UUID(args.run_id))
        report = json.loads((config.data_dir / 'reports' / f'{run_id}.json').read_bytes())
        return report['sources'].get(args.source, {}) if args.source else report
    repo, state = open_repository(config)
    if command == 'health':
        if repo:
            repo.close()
        return {'status': state['status'], 'database': state, 'configuration': config.public()}
    if not repo:
        return {**state,**({'observation':research_observation.model_dump(mode='json')} if research_observation else {})}
    try:
        if command == 'migrate':
            return {'status': 'OK', 'migrations': repo.conn.execute('SELECT * FROM fm.schema_migrations ORDER BY version').fetchall()}
        if command=='register-research':
            return {'observation_id':research_observation.id,**repo.process(research_observation,RawArchive(config.data_dir))}
        if command=='export-season':
            from football_spider.services.export import export_season
            return {'status':'OK',**export_season(repo,RawArchive(config.data_dir),args.output,source=args.source,season=args.season,
                observation_id=args.observation_id,information_mode=args.mode,lead_hours=args.lead_hours)}
        if command=='export-results':
            from football_spider.services.results import export_results,prediction_match_ids
            return {'status':'OK',**export_results(repo,RawArchive(config.data_dir),prediction_match_ids(args.predictions),args.output)}
        if command == 'status':
            result = {'status': 'OK', 'database': repo.status(config.timezone)}
            if args.source:
                result['source_runs'] = repo.conn.execute('''SELECT c.* FROM fm.collector_runs c JOIN fm.pipeline_runs p ON p.id=c.run_id
                    WHERE source=%s ORDER BY p.started_at DESC LIMIT 10''', (args.source,)).fetchall()
            return result
        if command == 'import-sqlite':
            return import_sqlite(repo, RawArchive(config.data_dir), args.path)
        if command == 'mapping-pending':
            return {'status': 'OK', 'candidates': repo.conn.execute("SELECT * FROM fm.mapping_candidates WHERE status='pending' ORDER BY id").fetchall()}
        if command == 'odds-history':
            from football_spider.features.odds_series import summarize
            if args.as_of.tzinfo is None:
                raise ValueError('--as-of requires timezone')
            return summarize(repo, args.match_id, args.as_of)
        if command == 'mapping-confirm':
            return {'status': repo.identity.confirm(args.source, args.entity_type, args.source_id, args.target_id, args.reason, args.reviewed_context_override)}
        if command == 'quality-resolve':
            with repo.conn.transaction():
                if not repo.conn.execute('SELECT id FROM fm.quality_issues WHERE id=%s', (args.issue_id,)).fetchone():
                    raise ValueError('Unknown quality issue')
                repo.conn.execute('INSERT INTO fm.quality_resolutions(issue_id,action,note) VALUES (%s,%s,%s)', (args.issue_id, 'reviewed', args.note))
                repo.conn.execute("UPDATE fm.quality_issues SET status='reviewed' WHERE id=%s", (args.issue_id,))
            return {'status': 'OK', 'note': 'Review recorded; raw data and fact versions are unchanged'}
        if command == 'schedule-plan':
            from football_spider.scheduler.jobs import plan
            return {'status': 'OK', **plan(repo, config)}
        if command == 'features':
            store = FeatureStore(repo, config.data_dir / 'features')
            frozen={}
            if args.input_bundle:
                manifest=json.loads((args.input_bundle/'manifest.json').read_bytes())
                marker=json.loads((args.input_bundle/'COMPLETE.json').read_bytes())
                import hashlib
                if marker['manifest_sha256']!=hashlib.sha256((args.input_bundle/'manifest.json').read_bytes()).hexdigest():
                    raise ValueError('Input bundle manifest mismatch')
                frozen={'version_ids':manifest['version_ids'],'assumptions':manifest.get('assumptions')}
            if args.snapshot_id:
                return {'status': 'OK', **store.replay_snapshot(args.snapshot_id)}
            if args.match_id:
                if args.as_of is None:
                    raise ValueError('--as-of required with --match-id')
                return {'status': 'OK', **store.get_features(args.match_id, args.as_of, args.version, args.mode, args.assumed_delay_seconds,**frozen)}
            if not args.competition_id:
                raise ValueError('Batch features require --competition-id')
            rows = repo.conn.execute('''SELECT DISTINCT match_id,event_time FROM fm.latest_matches WHERE competition_id=%s
                AND (%s::uuid IS NULL OR season_id=%s::uuid)''', (args.competition_id, args.season_id, args.season_id)).fetchall()
            outputs, errors = [], []
            for row in rows:
                try:
                    cutoff = args.as_of or row['event_time'] - timedelta(hours=args.lead_hours)
                    outputs.append(store.get_features(str(row['match_id']), cutoff, args.version, args.mode, args.assumed_delay_seconds,**frozen))
                except ValueError as exc:
                    errors.append({'match_id': str(row['match_id']), 'reason': str(exc)})
            return {'status': 'PARTIAL' if errors else 'OK' if outputs else 'NO_DATA', 'snapshots': outputs, 'errors': errors}
    finally:
        repo.close()
    raise ValueError('Unknown command')


def main():
    args = parser().parse_args()
    try:
        result = execute(args, PlatformSettings())
    except Exception as exc:
        # No connection URLs or HTTP exception strings (which can contain keys).
        result = {'status': 'FAILED', 'error_class': type(exc).__name__}
        if isinstance(exc, ValueError):
            result['reason'] = str(exc) if not any(x in str(exc).lower() for x in ('password', 'postgres://', 'postgresql://', 'apikey=')) else 'INVALID_CONFIGURATION'
    print(json_bytes(result).decode(), flush=True)
    return 0 if result.get('status') in ('OK', 'NO_DATA') else 2


if __name__ == '__main__':
    sys.exit(main())
