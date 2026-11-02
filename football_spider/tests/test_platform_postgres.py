from datetime import timedelta
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
from football_spider.models.contracts import now, EntityRef
from football_spider.storage.raw_archive import RawArchive
from football_spider.identity.resolver import stable_id
from football_spider.features.store import FeatureStore
from football_spider.commands.sqlite_import import import_sqlite
from football_spider.scheduler.jobs import plan
from football_spider.platform_config import PlatformSettings
from test_platform_contracts import fixture_row, save_detail


def test_migrate_idempotence_revision_and_reschedule(repo, tmp_path):
    repo.migrate()
    archive = RawArchive(tmp_path)
    row = fixture_row()
    first = save_detail(archive, row)
    assert repo.process(first, archive)['valid'] == 1
    assert repo.process(first, archive)['skipped'] == 1
    row['matchTime'] = fixture_row(days=2)['matchTime']
    second = save_detail(archive, row)
    assert repo.process(second, archive)['valid'] == 1
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.matches').fetchone()['n'] == 1
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.match_versions').fetchone()['n'] == 2
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.match_versions WHERE supersedes_id IS NOT NULL').fetchone()['n'] == 1


def test_quarantine_and_transaction_failure_recovery(repo, tmp_path):
    a = RawArchive(tmp_path)
    row = fixture_row(home=10, away=10)
    obs = save_detail(a, row)
    assert repo.process(obs, a)['quarantined'] == 1
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.match_versions').fetchone()['n'] == 0
    obs = save_detail(a, fixture_row())
    original = repo.save_match
    def fail(*args):
        original(*args)
        raise RuntimeError('synthetic transaction fault')
    with patch.object(repo, 'save_match', fail):
        assert repo.process(obs, a)['status'] == 'FAILED'
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.matches').fetchone()['n'] == 0
    assert repo.process(obs, a)['valid'] == 1


def test_point_in_time_immutable_manifest_and_unknown_xg(repo, tmp_path):
    a = RawArchive(tmp_path)
    for row in [fixture_row(mid=1, days=2), fixture_row(mid=2, days=-2, status=-1, hs=1, aws=0)]:
        assert repo.process(save_detail(a, row), a)['valid'] == 1
    mid = stable_id('spftll', 'match', '1')
    cutoff = now()
    store = FeatureStore(repo, tmp_path / 'features')
    initial = store.get_features(mid, cutoff)
    assert initial['features']['home_form']['matches'] == 1
    assert initial['features']['xg']['home'] is None
    # A later correction of old result, and a newly-observed match with an old
    # event time, must not leak into the earlier operational snapshot.
    for row in [fixture_row(mid=2, days=-2, status=-1, hs=0, aws=5), fixture_row(mid=3, days=-1, status=-1, hs=3, aws=0)]:
        repo.process(save_detail(a, row), a)
    repeated = store.get_features(mid, cutoff)
    assert initial['manifest_hash'] == repeated['manifest_hash']
    assert initial['features'] == repeated['features']
    assert store.replay_snapshot(initial['snapshot_id'])['consistent']
    assert store.get_features(mid, now())['features']['home_form']['matches'] == 2
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.match_labels').fetchone()['n'] == 3


def test_odds_append_unchanged_and_replay(repo, tmp_path):
    a = RawArchive(tmp_path)
    repo.process(save_detail(a, fixture_row()), a)
    body = json.dumps({'code': 0, 'data': {'footballMoreSpInfo': {'matchId': 1, 'spf_win': 2, 'spf_draw': 3, 'spf_lost': 4,
                          'rfspf_goal': -1, 'rfspf_win': 3, 'rfspf_draw': 4, 'rfspf_lost': 2, **{f't{i}': 10 for i in range(8)}}}}).encode()
    for _ in range(2):
        obs = a.save('spftll', 'https://fixture.invalid/GetMoreSpInfo', {}, body, status=200, context={'test_fixture': True})
        assert repo.process(obs, a)['quotes'] == 14
        assert repo.process(obs, a)['skipped'] == 1
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.odds_history').fetchone()['n'] == 28
    feature = FeatureStore(repo, tmp_path / 'features').get_features(stable_id('spftll', 'match', '1'), now())
    assert abs(sum(feature['features']['market']['probabilities'].values()) - 1) < 1e-9
    assert feature['features']['market']['bookmaker'] is None


def test_identity_alias_conflict_and_confirm_audit(repo):
    identity = repo.identity
    left = identity.resolve('fixture_a', 'team', EntityRef(source_id='1', name='United', category='men_senior'), now())
    right = identity.resolve('fixture_b', 'team', EntityRef(source_id='2', name='United', category='women_senior'), now())
    assert left != right
    import pytest
    with pytest.raises(ValueError, match='category'):
        identity.confirm('fixture_b', 'team', '2', left, 'test evidence')
    assert repo.conn.execute("SELECT count(*) AS n FROM fm.mapping_candidates WHERE status='pending'").fetchone()['n'] > 0
    third = identity.resolve('fixture_c', 'team', EntityRef(source_id='3', name='United', category='men_senior'), now())
    identity.confirm('fixture_c', 'team', '3', left, 'fixture manually verified')
    assert identity.resolve('fixture_c', 'team', EntityRef(source_id='3', name='United'), now()) == left
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.mapping_audit').fetchone()['n'] == 1


def test_schedule_reschedule_not_new_observation(repo, tmp_path):
    a = RawArchive(tmp_path)
    c = PlatformSettings(data_dir=tmp_path)
    row = fixture_row()
    repo.process(save_detail(a, row), a)
    plan(repo, c)
    before = repo.conn.execute("SELECT count(*) AS n FROM fm.scheduled_jobs WHERE kind='prematch'").fetchone()['n']
    repo.process(save_detail(a, row), a)
    plan(repo, c)
    assert repo.conn.execute("SELECT count(*) AS n FROM fm.scheduled_jobs WHERE kind='prematch'").fetchone()['n'] == before
    row['matchTime'] = fixture_row(days=2)['matchTime']
    repo.process(save_detail(a, row), a)
    plan(repo, c)
    assert repo.conn.execute("SELECT count(*) AS n FROM fm.scheduled_jobs WHERE status='superseded'").fetchone()['n'] == before


def test_sqlite_import_repeat_and_reconcile(repo, tmp_path):
    from football_spider.storage.database import Database
    from football_spider.models.match import Match
    path = tmp_path / 'legacy.db'
    with Database(path) as db:
        db.upsert_match(Match(id='99', time=now().replace(tzinfo=None) + timedelta(days=2), league='英超', home_team='A', away_team='B'))
    before = path.read_bytes()
    archive = RawArchive(tmp_path / 'platform')
    first = import_sqlite(repo, archive, path)
    second = import_sqlite(repo, archive, path)
    assert first['status'] == second['status'] == 'OK'
    assert first['observation_id'] == second['observation_id']
    assert second['processing']['skipped'] == 1
    assert path.read_bytes() == before


def test_source_id_relation_conflict_is_reviewable(repo, tmp_path):
    archive = RawArchive(tmp_path)
    repo.process(save_detail(archive, fixture_row(home=10, away=20)), archive)
    repo.process(save_detail(archive, fixture_row(home=20, away=10)), archive)
    assert repo.conn.execute('SELECT count(*) AS n FROM fm.matches').fetchone()['n'] == 1
    assert repo.conn.execute("SELECT count(*) AS n FROM fm.quality_issues WHERE rule='SOURCE_ID_RELATION_CONFLICT'").fetchone()['n'] == 1
    assert repo.conn.execute("SELECT count(*) AS n FROM fm.mapping_candidates WHERE reason='source_id_relation_change'").fetchone()['n'] == 1
