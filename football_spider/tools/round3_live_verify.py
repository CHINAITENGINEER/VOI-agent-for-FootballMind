"""At most one live detail request; independent PG validation and public cutoff check."""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from football_platform.information import LiveEvidenceEnvironment
from football_platform.information.contracts import Fixture
from football_spider.normalization.sources import normalize_spftll
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes, digest
from football_spider.services.refresh import RefreshService
from football_spider.platform_config import PlatformSettings
from football_spider.tools.pg_test_cluster import local_cluster
from football_spider.tools.round2_verify import pg, counts
from football_spider.storage.postgres import Repository

OUT = ROOT / 'var/parallel/collectors/round3/live'
DATA = ROOT / 'football_spider/platform_data'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Explicitly permit one bounded actual HTTP attempt')
    args = parser.parse_args()
    if not args.refresh:
        report = json.loads((OUT / 'report.json').read_bytes())
        assert report['status'] == 'OK'
        archive = RawArchive(DATA)
        observations = {o.id: o for o in archive.observations('spftll')}
        for oid in report['observation_ids']:
            archive.read_body(observations[oid])
        print(json.dumps({'status': 'OK', 'mode': 'verify_existing_zero_HTTP', 'observation_ids': report['observation_ids']}))
        return
    archive = RawArchive(DATA)
    source = next(o for o in archive.observations('spftll') if o.id == '3b3ff4de-5361-4b6a-92c3-8109b18be48b')
    previous = next(m for m in normalize_spftll(source, json.loads(archive.read_body(source))).matches if m.source_id == '3012999')
    now = datetime.now(timezone.utc)
    if previous.kickoff <= now:
        raise ValueError('Pinned valid fixture is no longer prematch; select a newly evidenced ID before another live run')
    fixture = Fixture(match_id='spftll:' + previous.source_id, competition_id='spftll:' + previous.competition.source_id,
        home_team_id='spftll:' + previous.home.source_id, away_team_id='spftll:' + previous.away.source_id,
        home_team_name=previous.home.name, away_team_name=previous.away.name, kickoff=previous.kickoff)
    report = {'run_id': str(uuid4()), 'status': 'RUNNING', 'fixture': fixture.model_dump(mode='json'),
              'source_selection_observation': source.id, 'environment': 'isolated_validation'}
    dest = OUT / report['run_id']
    dest.mkdir(parents=True, exist_ok=True)
    backup = ROOT / 'var/parallel/collectors/round2/footballmind-round2-followup-validation.dump'
    with local_cluster(dest / 'pg') as (url, cluster):
        pg('pg_restore', '--no-owner', '--no-privileges', '-d', url, backup)
        repo = Repository(url)
        repo.migrate()
        before = counts(repo)
        service = RefreshService(PlatformSettings(database_url=url, data_dir=DATA, environment='validation'))
        env = LiveEvidenceEnvironment(service, fixture=fixture, fixture_observed_at=source.observed_at, max_requests=1)
        initial = env.start(now)
        result = env.acquire('fixture_history', request_id=report['run_id'])
        report['result'] = result.model_dump(mode='json')
        report['fixed_cutoff_after_refresh'] = env.visible_state().model_dump(mode='json')
        assert env.visible_state().prediction_cutoff == initial.prediction_cutoff
        assert env.visible_state().groups == initial.groups == {}
        report['fixed_cutoff_unchanged'] = True
        duplicate = env.acquire('fixture_history', request_id=report['run_id'])
        assert duplicate.repeated_request and duplicate.charged_cost.requests.amount == 0
        report['repeat'] = duplicate.model_dump(mode='json')
        if result.status != 'NEEDS_NEW_CUTOFF':
            report['status'] = 'PARTIAL'
        else:
            advanced = env.advance_cutoff(datetime.now(timezone.utc))
            assert advanced.mask['fixture_refresh'] == 'OBSERVED'
            assert all(v.times.feature_available_at <= advanced.prediction_cutoff for v in advanced.groups.values())
            assert initial.groups == {}
            report['new_cutoff_state'] = advanced.model_dump(mode='json')
            old_values = {'kickoff': previous.kickoff.isoformat(), 'home_team_name': previous.home.name,
                'away_team_name': previous.away.name, 'competition_name': previous.competition.name,
                'status': previous.status, 'score_period': previous.score_period}
            report['content_comparison'] = {'fixture_content_changed': advanced.groups['fixture_refresh'].values != old_values,
                'previous_content_hash': digest(old_values), 'new_content_hash': digest(advanced.groups['fixture_refresh'].values),
                'new_observation_retained_even_when_content_unchanged': True}
            report['status'] = 'OK'
        report['before_counts'], report['after_counts'] = before, counts(repo)
        report['observation_ids'] = [r.split(':', 1)[1] for r in result.evidence_refs if r.startswith('observation:')]
        report['backup'] = str(dest / 'footballmind-round3-live-validation.dump')
        pg('pg_dump', '--format=custom', '--file', report['backup'], url)
        repo.close()
    report['database_state'] = 'STOPPED_RETAINED_NOT_PRODUCTION'
    report['actual_requests'] = result.charged_cost.requests.amount
    report['actual_money'] = None
    report['actual_money_status'] = 'unknown'
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    atomic_write(dest / 'report.json', json_bytes(report))
    atomic_write(OUT / 'report.json', json_bytes(report))
    print(json.dumps({k: report[k] for k in ('status', 'actual_requests', 'observation_ids', 'database_state')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
