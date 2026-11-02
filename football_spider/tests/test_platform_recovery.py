from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch, Mock
import json
from filelock import FileLock
import pytest
from football_spider.collectors.sources import collect
from football_spider.models.contracts import now, CollectionResult, RunStatus
from football_spider.platform_config import PlatformSettings
from football_spider.storage.raw_archive import RawArchive
from football_spider.commands.pipeline import run_pipeline
from football_spider.features.store import FeatureStore
from football_spider.features.odds_series import summarize
from football_spider.identity.resolver import stable_id
from test_platform_contracts import save_detail, fixture_row


def test_missing_credentials_make_no_requests(tmp_path):
    config = PlatformSettings(data_dir=tmp_path, football_data_token='', odds_api_key='')
    for source in ('football_data', 'odds_api'):
        result = collect(source, config, RawArchive(tmp_path), start=date(2026, 9, 20), end=date(2026, 9, 27))
        assert result.status == RunStatus.NEEDS_CREDENTIALS and result.requests == 0


def test_football_data_exclusive_date_window_and_header(tmp_path):
    config = PlatformSettings(data_dir=tmp_path, football_data_token='test-only-token', competitions=['PL'])
    with patch('football_spider.collectors.sources.ArchiveHttp') as factory:
        client = factory.return_value
        client.observations = []
        client.get.return_value = (None, {'matches': []})
        collect('football_data', config, RawArchive(tmp_path), start=date(2026, 9, 20), end=date(2026, 9, 27))
        args, kwargs = client.get.call_args
        assert args[0] == 'https://api.football-data.org/v4/competitions/PL/matches'
        assert args[1] == {'dateFrom': '2026-09-19', 'dateTo': '2026-09-28'}
        assert kwargs['headers'] == {'X-Auth-Token': 'test-only-token'}


def test_local_daily_odds_budget_across_runs(tmp_path):
    from football_spider.collectors.quota import reserve
    assert reserve(tmp_path, 'fixture', 3, 5)
    assert not reserve(tmp_path, 'fixture', 3, 5)
    assert reserve(tmp_path, 'fixture', 2, 5)
    assert not reserve(tmp_path, 'fixture', 1, 5)


def test_run_lock_and_source_failure_isolation(repo, pg_url, tmp_path):
    config = PlatformSettings(data_dir=tmp_path, database_url=pg_url)
    archive = RawArchive(tmp_path)
    observation = save_detail(archive, fixture_row())
    def fake_collect(source, *args, **kwargs):
        if source == 'spftll':
            return CollectionResult(source=source, observations=[observation.id], requests=1)
        raise RuntimeError('synthetic provider failure')
    with FileLock(str(tmp_path / 'pipeline.lock')):
        assert run_pipeline(config)['reason'] == 'CONCURRENT_RUN_LOCKED'
    with patch('football_spider.commands.pipeline.collect', fake_collect):
        result = run_pipeline(config, sources=['spftll', 'football_data'], build_features=False)
    assert result['status'] == 'PARTIAL'
    assert result['sources']['spftll']['status'] == 'OK'
    assert result['sources']['football_data']['status'] == 'FAILED'
    assert result['totals']['valid'] == 1


def test_parse_failure_and_archive_only_without_db(repo, tmp_path):
    archive = RawArchive(tmp_path)
    obs = archive.save('spftll', 'https://fixture.invalid/GetInfo', {}, b'<html>not JSON</html>', status=200)
    assert repo.process(obs, archive)['status'] == 'FAILED'
    assert archive.read_body(obs).startswith(b'<html>')
    config = PlatformSettings(database_url='', data_dir=tmp_path)
    result = run_pipeline(config, sources=['spftll'], replay=True, build_features=False)
    assert result['status'] == 'NEEDS_DATABASE'
    assert result['totals']['raw_observations'] == 1
    assert result['totals']['valid'] == 0


def test_historical_snapshot_delay_not_observation_backdating(repo, tmp_path):
    archive = RawArchive(tmp_path)
    payload = {'timestamp': '2025-01-01T12:00:00Z', 'data': [{'id': 'event', 'sport_key': 'soccer_epl', 'sport_title': 'EPL',
        'home_team': 'A', 'away_team': 'B', 'commence_time': '2025-01-02T12:00:00Z',
        'bookmakers': [{'key': 'book', 'last_update': '2025-01-01T11:59:00Z', 'markets': [{'key': 'h2h', 'outcomes': [
            {'name': 'A', 'price': 2}, {'name': 'Draw', 'price': 3}, {'name': 'B', 'price': 4}]}]}]}]}
    obs = archive.save('odds_api', 'https://fixture.invalid/historical/odds', {}, json.dumps(payload).encode(), status=200,
                       context={'historical': True, 'requested_snapshot': '2025-01-01T12:01:00Z', 'test_fixture': True})
    assert repo.process(obs, archive)['status'] == 'OK'
    mid = stable_id('odds_api', 'match', 'event')
    store = FeatureStore(repo, tmp_path / 'features')
    cutoff = datetime(2025, 1, 1, 12, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='No target'):
        store.get_features(mid, cutoff)
    with pytest.raises(ValueError, match='No target'):
        store.get_features(mid, cutoff, information_mode='historical_reconstruction', assumed_delay_seconds=120)
    result = store.get_features(mid, cutoff, information_mode='historical_reconstruction', assumed_delay_seconds=60)
    assert result['features']['market']['probabilities'] is not None
    assert result['features']['elo']['home'] is None
    assert store.replay_snapshot(result['snapshot_id'])['consistent']
    saved = repo.conn.execute('SELECT observed_at FROM fm.match_versions').fetchone()['observed_at']
    assert saved.year >= 2026


def test_out_of_order_replay_keeps_latest_observed_version(repo, tmp_path):
    archive = RawArchive(tmp_path)
    current = fixture_row()
    old = {**current, 'matchTime': fixture_row(days=2)['matchTime']}
    first = save_detail(archive, current)
    older = save_detail(archive, old, observed_at=first.observed_at - timedelta(hours=1))
    repo.process(first, archive)
    repo.process(older, archive)
    result = FeatureStore(repo, tmp_path / 'features').get_features(stable_id('spftll', 'match', '1'), now())
    assert result['manifest']['target_version'] == stable_id('spftll', 'version', f'{first.id}:1.0:1')


def test_synthetic_price_change_does_not_rewrite_prior_market(repo, tmp_path):
    archive = RawArchive(tmp_path)
    repo.process(save_detail(archive, fixture_row()), archive)
    mid = stable_id('spftll', 'match', '1')
    payload = {'code': 0, 'data': {'footballMoreSpInfo': {'matchId': 1, 'spf_win': 2.0, 'spf_draw': 3, 'spf_lost': 4}}}
    def save():
        observation = archive.save('spftll', 'https://fixture.invalid/GetMoreSpInfo', {}, json.dumps(payload).encode(), status=200,
                                   context={'test_fixture': True, 'scenario': 'synthetic_price_change'})
        return repo.process(observation, archive)
    assert save()['quotes'] == 3
    cutoff = now()
    store = FeatureStore(repo, tmp_path / 'features')
    before = store.get_features(mid, cutoff)
    payload['data']['footballMoreSpInfo']['spf_win'] = 2.2
    assert save()['quotes'] == 3
    assert store.get_features(mid, cutoff)['manifest_hash'] == before['manifest_hash']
    after = store.get_features(mid, now())
    assert after['features']['market']['probabilities']['home'] != before['features']['market']['probabilities']['home']
    history = summarize(repo, mid, now())
    home = next(s for s in history['series'] if s['selection'] == 'home')
    assert home['observations'] == 2
    assert abs(home['price_change'] - 0.2) < 1e-9
    assert home['true_opening_price'] is None
