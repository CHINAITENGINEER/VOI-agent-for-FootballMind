"""Synthetic fixtures ONLY; never imported by production commands."""
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock
import pytest
import requests
from pydantic import ValidationError
from football_spider.models.contracts import Score, Quote, now
from football_spider.normalization.sources import normalize, spftll_match
from football_spider.quality.rules import validate_batch
from football_spider.storage.raw_archive import RawArchive
from football_spider.collectors.http import ArchiveHttp, FetchError


def fixture_row(mid=1, home=10, away=20, days=1, status=0, hs=None, aws=None):
    return {'matchId': mid, 'matchTime': (now() + timedelta(days=days)).astimezone(timezone.utc).replace(tzinfo=None).isoformat(),
            'homeId': home, 'awayId': away, 'home': f'Team {home}', 'away': f'Team {away}', 'league': '英超', 'leagueId': 1,
            'footMatchState': status, 'homeScore': hs, 'awayScore': aws, 'homeHalfScore': 0 if hs is not None else None,
            'awayHalfScore': 0 if aws is not None else None}


def save_detail(archive, row, observed_at=None):
    return archive.save('spftll', 'https://fixture.invalid/api/match/GetInfo', {}, json.dumps({'code': 0, 'data': row}).encode(),
                        status=200, observed_at=observed_at, context={'test_fixture': True})


def test_null_zero_and_strict_score():
    assert Score().home is None
    assert Score(home=0, away=0).home == 0
    for value in (-1, 1.2, True, '2'):
        with pytest.raises(ValidationError):
            Score(home=value)


def test_odds_contract():
    values = dict(source_match_id='1', provider='fixture', market_type='asian_handicap', period='regulation', selection='home', line=-1.5, settlement_rule='regulation_selection_line')
    assert Quote(**values, decimal_price=1.8).line == -1.5
    for value in (0, 1, -2, float('nan'), float('inf')):
        with pytest.raises(ValidationError):
            Quote(**values, decimal_price=value)


def test_archive_failure_redaction_and_checksum(tmp_path):
    archive = RawArchive(tmp_path)
    obs = archive.save('fixture', 'https://user:password@fixture.invalid/a?apiKey=secret', {'apiKey': 'secret'}, b'<html>bad</html>', status=429)
    assert 'secret' not in obs.model_dump_json()
    assert 'password' not in obs.endpoint
    assert archive.read_body(obs) == b'<html>bad</html>'
    assert obs.payload_format == 'html'
    assert archive.observations()[0].id == obs.id


def test_state_quality(tmp_path):
    a = RawArchive(tmp_path)
    row = fixture_row(days=-1, status=-1, hs=0, aws=0)
    o = save_detail(a, row)
    batch = normalize(o, json.loads(a.read_body(o)))
    assert batch.matches[0].status == 'finished'
    assert batch.matches[0].score_period == 'regulation'
    assert not [i for i in validate_batch(batch, now()) if i.severity == 'error']
    for title, expected in [('延期', 'postponed'), ('取消', 'cancelled')]:
        row['finishedTitle'] = title
        assert spftll_match(row).status == expected
    row['homeHalfScore'] = 1
    row['finishedTitle'] = '完场'
    o = save_detail(a, row)
    assert any(i.rule == 'HALF_EXCEEDS_FULL' for i in validate_batch(normalize(o, json.loads(a.read_body(o))), now()))


def test_football_data_extra_time_not_labelled_regulation(tmp_path):
    a = RawArchive(tmp_path)
    payload = {'matches': [{'id': 1, 'utcDate': '2025-01-01T12:00:00Z', 'competition': {'id': 1, 'name': 'Cup'},
                          'season': {'id': 2025}, 'homeTeam': {'id': 10, 'name': 'A'}, 'awayTeam': {'id': 20, 'name': 'B'},
                          'status': 'FINISHED', 'score': {'duration': 'PENALTY_SHOOTOUT', 'fullTime': {'home': 5, 'away': 4}, 'halfTime': {'home': 0, 'away': 0}}}]}
    obs = a.save('football_data', 'https://fixture.invalid/matches', {}, json.dumps(payload).encode(), status=200)
    m = normalize(obs, payload).matches[0]
    assert m.score.home is None and m.score_period == 'unknown'
    payload['matches'][0]['score']['regularTime'] = {'home': 0, 'away': 0}
    assert normalize(obs, payload).matches[0].score.home == 0


def test_http_timeout_429_retry_archive_and_retry_after(tmp_path):
    session = Mock()
    response = Mock(status_code=429, content=b'{"message":"quota"}', headers={'Retry-After': '2'})
    success = Mock(status_code=200, content=b'{"ok":true}', headers={})
    session.get.side_effect = [requests.Timeout(), response, success]
    sleep = Mock()
    client = ArchiveHttp(RawArchive(tmp_path), 'fixture', session=session, sleep=sleep)
    _, payload = client.get('https://fixture.invalid', {'apiKey': 'hidden'})
    assert payload['ok']
    assert len(client.observations) == 3
    assert client.observations[0].error_class == 'Timeout'
    assert any(call.args == (2.0,) for call in sleep.call_args_list)
    session.get.side_effect = [Mock(status_code=429, content=b'{}', headers={'Retry-After': '120'})]
    with pytest.raises(FetchError, match='RATE_LIMITED'):
        client.get('https://fixture.invalid')


def test_http_parse_failure_retains_payload(tmp_path):
    session = Mock()
    session.get.return_value = Mock(status_code=200, content=b'<html>invalid</html>', headers={})
    client = ArchiveHttp(RawArchive(tmp_path), 'fixture', session=session, sleep=Mock())
    with pytest.raises(FetchError, match='PARSE_JSON'):
        client.get('https://fixture.invalid')
    assert len(client.observations) == 1


@pytest.mark.parametrize('status', [403, 500])
def test_http_retry_bounded_recoverable_statuses(tmp_path, status):
    session = Mock()
    session.get.side_effect = [Mock(status_code=status, content=b'{}', headers={}), Mock(status_code=200, content=b'{}', headers={})]
    client = ArchiveHttp(RawArchive(tmp_path), 'fixture', session=session, sleep=Mock())
    client.get('https://fixture.invalid')
    assert len(client.observations) == 2


def test_http_permanent_401_not_retried(tmp_path):
    session = Mock()
    session.get.return_value = Mock(status_code=401, content=b'{}', headers={})
    client = ArchiveHttp(RawArchive(tmp_path), 'fixture', session=session, sleep=Mock())
    with pytest.raises(FetchError, match='HTTP_401'):
        client.get('https://fixture.invalid')
    assert len(client.observations) == 1


def test_odds_api_snapshot_and_selection_lines(tmp_path):
    a = RawArchive(tmp_path)
    payload = {'timestamp': '2025-01-01T12:00:00Z', 'data': [{'id': 'event', 'sport_key': 'soccer_epl', 'sport_title': 'EPL',
               'home_team': 'A', 'away_team': 'B', 'commence_time': '2025-01-02T12:00:00Z', 'bookmakers': [{'key': 'book',
               'last_update': '2025-01-01T11:59:00Z', 'markets': [{'key': 'spreads', 'outcomes': [
                   {'name': 'A', 'point': -1.5, 'price': 1.9}, {'name': 'B', 'point': 1.5, 'price': 1.9}]}]}]}]}
    obs = a.save('odds_api', 'https://fixture.invalid/odds', {}, json.dumps(payload).encode(), status=200,
                 context={'historical': True, 'requested_snapshot': '2025-01-01T12:01:00Z', 'test_fixture': True})
    batch = normalize(obs, payload)
    assert batch.quotes[0].line == -1.5 and batch.quotes[0].bookmaker == 'book'
    assert batch.matches[0].trusted_snapshot_at is not None
    assert not [x for x in validate_batch(batch, now()) if x.severity == 'error']
