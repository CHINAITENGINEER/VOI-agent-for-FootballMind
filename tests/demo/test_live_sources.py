import pytest
import requests
from demo.live_sources import LivePageSession, LiveNewsTools
from football_agent.inquiry_tools import ResponseLimit
from football_spider.storage.raw_archive import RawArchive


class Response:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False
    def iter_content(self, size):
        yield from self.chunks
    def close(self):
        self.closed = True


def test_large_cms_response_preserves_transport_controls(monkeypatch):
    response = Response([b'x' * 3_000_000])
    captured = {}
    def get(session, url, **kwargs):
        captured.update(kwargs)
        assert session.verify is not False and session.auth
        return response
    monkeypatch.setattr(requests.Session, 'get', get)
    result = LivePageSession('environment').get('https://www.ccfc.co.uk/news/', timeout=4, allow_redirects=False)
    assert len(result._content) == 3_000_000 and result._content_consumed
    assert captured['stream'] and captured['allow_redirects'] is False
    assert captured['timeout'] == 4


def test_response_limit_closes_connection_and_records_failure(monkeypatch):
    response = Response([b'x' * 8_000_001])
    monkeypatch.setattr(requests.Session, 'get', lambda *a, **kw: response)
    session = LivePageSession('environment')
    with pytest.raises(ResponseLimit):
        session.get('https://www.ccfc.co.uk/news/')
    assert response.closed and session.failure


def test_search_rejects_changed_endpoint_before_network(tmp_path):
    tools = LiveNewsTools(RawArchive(tmp_path), ['www.ccfc.co.uk'], index_urls=['https://www.ccfc.co.uk/news/'], projection_version=13)
    with pytest.raises(ValueError, match='Unexpected online RSS'):
        tools.execute({'tool':'search_web','endpoint':'https://example.com',
            'params':{'q':'test','format':'rss'},'source':'open_search'}, run_id='test', request_id='test')
