"""Privacy, evidence ordering and bounded attempt behavior of source diagnosis."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from football_agent.transport_probe import DiagnosticSession, environment_flags, exception_facts, run_probe
from football_spider.storage.raw_archive import RawArchive


def test_exception_chain_keeps_root_error_without_proxy_secret():
    inner = ConnectionRefusedError(10061, "proxy credential alice:SECRET@example.invalid")
    outer = requests.exceptions.ProxyError("TOKEN and private endpoint", inner)
    facts = exception_facts(outer)
    assert {"proxy_error", "connection_refused"} <= set(facts["categories"])
    assert "SECRET" not in json.dumps(facts) and "TOKEN" not in json.dumps(facts)
    assert any(e.get("errno") == 10061 for e in facts["chain"])


def test_environment_inspection_exports_presence_only(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:SECRET@127.0.0.1:123")
    flags = environment_flags()
    assert flags["HTTPS_PROXY"] is True
    assert all(type(v) is bool for v in flags.values())
    assert "SECRET" not in json.dumps(flags)


@pytest.mark.parametrize("route", ["environment", "direct"])
def test_sessions_never_consult_netrc_and_preserve_tls(monkeypatch, route):
    def forbidden(*args, **kwargs):
        raise AssertionError("netrc must not be read")
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", forbidden)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "declared-ca.pem")
    with DiagnosticSession(route) as session:
        prepared = session.prepare_request(requests.Request("GET", "https://www.englandfootball.com/"))
        assert "Authorization" not in prepared.headers
        session.rebuild_auth(prepared, SimpleNamespace(request=prepared))
        assert "Authorization" not in prepared.headers
        assert session.trust_env is (route == "environment")
        assert session.verify is not False
        if route == "direct":
            assert session.verify == "declared-ca.pem"


def test_failed_attempt_is_preregistered_archived_and_not_retried(tmp_path):
    calls = []
    class Session:
        failure = None
        def get(self, url, **kwargs):
            status = json.loads((tmp_path / "status.json").read_text())
            intent = tmp_path / status["probes"][-1]["probe_id"] / "intent.json"
            assert intent.exists() and status["probes"][-1]["state"] == "RESERVED_BEFORE_NETWORK"
            assert kwargs["timeout"] == (4, 8) and kwargs["allow_redirects"] is False
            calls.append(url)
            error = requests.ConnectionError("private credentials must not be exported")
            self.failure = exception_facts(error)
            raise error
        def close(self): pass
    result = run_probe(tmp_path, "england_fixtures", "environment", session_factory=lambda route: Session())
    assert len(calls) == 1
    assert result["transport_stats"]["attempts"] == result["transport_stats"]["archived_attempts"] == 1
    assert result["transport_stats"]["retries"] == 0
    assert result["body_bytes"] == 0 and result["document_status"] == "FAILED"
    assert result["failed_acquisition_implies_no_lineup"] is False
    archive = RawArchive(tmp_path / "evidence")
    assert len(archive.observations()) == 1 and archive.read_body(archive.observations()[0]) == b""
    assert "private credentials" not in (tmp_path / "status.json").read_text()


def test_success_archives_original_body_without_promoting_semantic_truth(tmp_path):
    body = b"<html><main><p>The match identity needs independent review.</p></main></html>"
    class Session:
        failure = None
        def get(self, url, **kwargs):
            return SimpleNamespace(status_code=200, content=body, headers={"content-type":"text/html"}, close=lambda: None)
        def close(self): pass
    result = run_probe(tmp_path, "england_fixtures", "direct", session_factory=lambda route: Session())
    assert result["document_status"] == "OK" and result["extracted_characters"] > 0
    assert result["semantic_answer_verified"] is False
    archive = RawArchive(tmp_path / "evidence")
    assert archive.read_body(archive.observations()[0]) == body


def test_four_attempt_budget_is_shared_by_probe_invocations(tmp_path):
    calls = []
    class Session:
        failure = None
        def get(self, url, **kwargs):
            calls.append(url)
            return SimpleNamespace(status_code=503, content=b"Unavailable", headers={}, close=lambda: None)
        def close(self): pass
    for _ in range(4):
        run_probe(tmp_path, "england_fixtures", "environment", session_factory=lambda route: Session())
    with pytest.raises(ValueError, match="budget"):
        run_probe(tmp_path, "england_fixtures", "direct", session_factory=lambda route: Session())
    assert len(calls) == 4


def test_unregistered_target_fails_before_creating_files(tmp_path):
    root = tmp_path / "unused"
    with pytest.raises(ValueError):
        run_probe(root, "guessed_url", "direct")
    assert not root.exists()


def test_header_profile_is_declared_before_request_without_changing_route(tmp_path):
    class Session:
        failure = None
        def get(self, url, **kwargs):
            status = json.loads((tmp_path / "status.json").read_text())
            intent = json.loads((tmp_path / status["probes"][-1]["probe_id"] / "intent.json").read_text())
            assert intent["session_route"] == "direct" and intent["user_agent_profile"] == "browser"
            assert kwargs["headers"] == intent["request_headers"]
            assert kwargs["headers"]["User-Agent"].startswith("Mozilla/")
            return SimpleNamespace(status_code=403, content=b"Blocked", headers={}, close=lambda: None)
        def close(self): pass
    result = run_probe(tmp_path, "england_fixtures", "direct", user_agent_profile="browser",
                       session_factory=lambda route: Session())
    assert result["document_status"] == "FAILED"
