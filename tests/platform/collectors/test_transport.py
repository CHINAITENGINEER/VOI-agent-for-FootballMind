from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import requests

from football_platform.collectors.http_transport import RequestTransport, TransportError
from football_platform.collectors.envelopes import SourceEnvelopeError, response_count


def response(status=200, body=b'[]', headers=None):
    value = requests.Response()
    value.status_code, value._content = status, body
    value.headers.update(headers or {})
    value._content_consumed = True
    return value


class Session:
    def __init__(self, values):
        self.values, self.calls, self.closed, self.cookies = iter(values), [], False, {}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


def client(values, **kwargs):
    attempts, sleeps = [], []
    session = Session(values)
    transport = RequestTransport('unit_test', on_attempt=lambda **facts: attempts.append(facts),
                                 session=session, sleep=sleeps.append, interval=0, **kwargs)
    return transport, attempts, sleeps, session


def test_identical_responses_are_distinct_real_attempt_facts():
    transport, attempts, _, session = client([response(), response()])
    assert transport.get('https://fixture.invalid/data') == b'[]'
    transport.get('https://fixture.invalid/data')
    assert len({a['observation_id'] for a in attempts}) == 2
    assert len({a['request_id'] for a in attempts}) == 2
    assert attempts[0]['body'] == attempts[1]['body']
    assert all(a['observed_at'].tzinfo for a in attempts)
    assert transport.stats['attempts'] == transport.stats['archived_attempts'] == 2
    assert all(call[1]['timeout'] == (10,35) and not call[1]['allow_redirects'] for call in session.calls)


def test_timeout_retries_are_archived_under_one_logical_request():
    transport, attempts, _, _ = client([requests.Timeout('do-not-log-secrets')] * 3)
    with pytest.raises(TransportError, match='Timeout'):
        transport.get('https://fixture.invalid/data')
    assert len(attempts) == 3
    assert len({a['request_id'] for a in attempts}) == 1
    assert len({a['observation_id'] for a in attempts}) == 3
    assert all(a['body'] == b'' and a['http_status'] is None for a in attempts)
    assert 'do-not-log-secrets' not in str(attempts)
    assert transport.stats['retries'] == 2


@pytest.mark.parametrize('status', [408,429,500,502,503,504])
def test_transient_http_failure_then_success(status):
    transport, attempts, sleeps, _ = client([response(status, b'busy', {'Retry-After':'2'}), response()])
    assert transport.get('https://fixture.invalid/data') == b'[]'
    assert [a['http_status'] for a in attempts] == [status,200]
    assert max(sleeps) >= 2


def test_long_retry_after_defers_without_early_retry():
    transport, attempts, sleeps, _ = client([response(429, b'busy', {'Retry-After':'3600'})])
    with pytest.raises(TransportError, match='RATE_LIMITED'):
        transport.get('https://fixture.invalid/data')
    assert len(attempts) == 1 and not sleeps


@pytest.mark.parametrize('status', [301,401,403,404])
def test_nontransient_failure_no_blind_retry(status):
    transport, attempts, _, _ = client([response(status)])
    with pytest.raises(TransportError, match=f'HTTP_{status}'):
        transport.get('https://fixture.invalid/data')
    assert len(attempts) == 1


def test_every_attempt_including_retry_reserves_credits():
    transport, attempts, _, _ = client([response(500),response(500)], credit_budget=6)
    with pytest.raises(TransportError, match='RATE_LIMITED'):
        transport.get('https://fixture.invalid/data', estimated_credits=3)
    assert len(attempts) == 2
    assert transport.stats['credits_reserved'] == 6


def test_actual_remaining_quota_stops_next_request():
    transport, attempts, _, _ = client([response(headers={'x-requests-remaining':'0'})])
    transport.get('https://fixture.invalid/data')
    with pytest.raises(TransportError, match='RATE_LIMITED'):
        transport.get('https://fixture.invalid/data')
    assert len(attempts) == 1


@pytest.mark.parametrize('quota_header',['x-ratelimit-requests-remaining','X-Ratelimit-Remaining','X-Requests-Available-Minute'])
def test_other_provider_quota_headers_defer_at_zero(quota_header):
    transport,attempts,_,_=client([response(headers={quota_header:'0'})])
    transport.get('https://fixture.invalid/data')
    with pytest.raises(TransportError,match='RATE_LIMITED'):
        transport.get('https://fixture.invalid/data')
    assert len(attempts)==1


def test_cross_process_reservation_hook_denial_sends_nothing():
    transport, attempts, _, session = client([], reserve=lambda *_: False)
    with pytest.raises(TransportError, match='RATE_LIMITED'):
        transport.get('https://fixture.invalid/data')
    assert not attempts and not session.calls


def test_key_echo_redaction_and_response_header_allowlist():
    transport, attempts, _, _ = client([response(body=b'{"echo":"TEST-SECRET"}', headers={'Set-Cookie':'sensitive'})])
    transport.get('https://fixture.invalid/data', params={'apiKey':'TEST-SECRET'}, headers={'X-Auth-Token':'TEST-SECRET'})
    record = attempts[0]
    assert record['body'] == b'{"echo":"[REDACTED]"}'
    assert record['params']['apiKey'] == record['request_headers']['X-Auth-Token'] == '[REDACTED]'
    assert record['credential_echo_redacted'] and not record['response_headers']
    assert 'TEST-SECRET' not in str(record)


def test_non_json_is_archived_before_parse_failure():
    transport, attempts, _, _ = client([response(body=b'<html>maintenance</html>')])
    body = transport.get('https://fixture.invalid/data')
    with pytest.raises(SourceEnvelopeError, match='PARSE_JSON'):
        response_count('football_data','matches',body)
    assert attempts[0]['body'] == body


def test_archive_failure_stops_and_does_not_repeat_http():
    transport, _, _, session = client([response()])
    def fail(**_):
        raise OSError('sensitive filesystem path')
    transport.on_attempt = fail
    with pytest.raises(TransportError, match='ARCHIVE_FAILED'):
        transport.get('https://fixture.invalid/data')
    assert transport.stats['archive_errors'] == 1 and len(session.calls) == 1


def test_real_local_server_cookie_persistence_and_session_reuse(tmp_path):
    """Actual HTTP integration against a local synthetic server, not provider proof."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Set-Cookie','session=fixture; Path=/')
            self.end_headers()
            self.wfile.write((self.headers.get('Cookie') or '').encode())
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread = Thread(target=server.serve_forever,daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/'
    cookie_path = tmp_path/'cookie.jar'
    try:
        with RequestTransport('local_fixture',on_attempt=lambda **_:None,cookie_path=cookie_path,interval=0) as transport:
            assert transport.get(url) == b''
            assert b'session=fixture' in transport.get(url)
        with RequestTransport('local_fixture',on_attempt=lambda **_:None,cookie_path=cookie_path) as transport:
            assert b'session=fixture' in transport.get(url)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
    assert cookie_path.exists()


def test_corrupt_cookie_jar_is_preserved(tmp_path):
    path = tmp_path/'cookie.jar'
    path.write_text('invalid original',encoding='utf-8')
    with RequestTransport('fixture',on_attempt=lambda **_:None,cookie_path=path) as transport:
        assert transport.stats['cookie_errors'] == 1
    assert path.read_text(encoding='utf-8') == 'invalid original'
