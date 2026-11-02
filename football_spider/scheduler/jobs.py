from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from filelock import FileLock, Timeout
from psycopg.types.json import Jsonb
from football_spider.models.contracts import now
from football_spider.storage.raw_archive import digest
from football_spider.commands.pipeline import open_repository, run_pipeline


def plan(repo, config, instant=None):
    instant = instant or now()
    zone = ZoneInfo(config.timezone)
    local = instant.astimezone(zone)
    due_time = time.fromisoformat(config.daily_time)
    jobs = []
    # Daily tasks remain catch-up eligible for 48h; odds catch-up stops at kickoff.
    for offset in (-1, 0, 1):
        day = local.date() + timedelta(days=offset)
        due = datetime.combine(day, due_time, tzinfo=zone)
        for kind in ('future_schedule', 'recent_results'):
            jobs.append((f'{kind}:{day}', kind, due, {'date': str(day)}))
    matches = repo.conn.execute("SELECT * FROM fm.latest_matches WHERE source=ANY(%s::text[]) AND event_time>%s AND event_time<%s AND status IN ('scheduled','unknown')", (config.sources, instant, instant + timedelta(days=8))).fetchall()
    active_quote_jobs = []
    for m in matches:
        for lead in config.refresh_minutes:
            # Schedule version depends on kickoff value, not observation ID.
            payload = {'source': m['source'], 'source_match_id': m['source_id'], 'match_id': str(m['match_id']), 'kickoff': m['event_time'].isoformat(), 'lead_minutes': lead}
            key = 'prematch:' + digest(payload)
            active_quote_jobs.append(key)
            jobs.append((key, 'prematch', m['event_time'] - timedelta(minutes=lead), payload))
    with repo.conn.transaction():
        repo.conn.execute("UPDATE fm.scheduled_jobs SET status='superseded',updated_at=clock_timestamp() WHERE kind='prematch' AND status IN ('pending','retry') AND NOT(job_key=ANY(%s::text[]))", (active_quote_jobs,))
        for key, kind, due, payload in jobs:
            repo.conn.execute('''INSERT INTO fm.scheduled_jobs(job_key,kind,due_at,status,payload)
                VALUES (%s,%s,%s,'pending',%s) ON CONFLICT(job_key) DO NOTHING''', (key, kind, due, Jsonb(payload)))
    return {'generated': len(jobs), 'jobs': repo.conn.execute('SELECT status,count(*) AS count FROM fm.scheduled_jobs GROUP BY status').fetchall()}


def tick(config):
    config.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(config.data_dir / 'scheduler.lock')).acquire(timeout=0):
            repo, status = open_repository(config)
            if not repo:
                return status
            try:
                if not repo.conn.execute('SELECT pg_try_advisory_lock(718643221) AS acquired').fetchone()['acquired']:
                    return {'status': 'PARTIAL', 'reason': 'SCHEDULER_LOCKED'}
                summary = plan(repo, config)
                # Restart recovery, no permanently running jobs after a crash.
                repo.conn.execute("UPDATE fm.scheduled_jobs SET status='retry' WHERE status='running'")
                repo.conn.execute("UPDATE fm.scheduled_jobs SET status='failed' WHERE status='retry' AND attempts>=5")
                instant = now()
                jobs = repo.conn.execute("SELECT * FROM fm.scheduled_jobs WHERE due_at<=%s AND status IN ('pending','retry') AND attempts<5 ORDER BY due_at LIMIT 20", (instant,)).fetchall()
                executions = []
                for job in jobs:
                    payload = job['payload']
                    late = (instant - job['due_at']).total_seconds()
                    expired = late > 172800 if job['kind'] != 'prematch' else (
                        instant >= datetime.fromisoformat(payload['kickoff']) or late > config.job_grace_minutes * 60)
                    if expired:
                        repo.conn.execute("UPDATE fm.scheduled_jobs SET status='expired' WHERE job_key=%s", (job['job_key'],))
                        continue
                    repo.conn.execute("UPDATE fm.scheduled_jobs SET status='running',attempts=attempts+1,updated_at=clock_timestamp() WHERE job_key=%s", (job['job_key'],))
                    day = instant.astimezone(ZoneInfo(config.timezone)).date()
                    kwargs = {}
                    if job['kind'] == 'recent_results':
                        # spftll has only recent rows; full historical backfill uses football-data.org.
                        kwargs = {'start': day - timedelta(days=2), 'end': day}
                    elif job['kind'] == 'prematch':
                        kwargs = {'sources': [payload['source']], 'match_ids': [payload['source_match_id']]}
                    result = run_pipeline(config, **kwargs)
                    success = result['status'] == 'OK'
                    repo.conn.execute('''UPDATE fm.scheduled_jobs SET status=%s,last_run=%s,updated_at=clock_timestamp() WHERE job_key=%s''', ('done' if success else 'retry', result.get('run_id'), job['job_key']))
                    executions.append({'job_key': job['job_key'], 'status': result['status'], 'run_id': result.get('run_id')})
                return {'status': 'OK' if all(x['status'] == 'OK' for x in executions) else 'PARTIAL', 'plan': summary, 'executions': executions}
            finally:
                repo.close()
    except Timeout:
        return {'status': 'PARTIAL', 'reason': 'SCHEDULER_LOCKED'}
