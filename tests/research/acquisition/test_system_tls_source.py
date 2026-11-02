"""Offline subprocess contract checks for the opt-in Schannel identity reader."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_research.data_adapters import system_tls_source as transport
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 10, 7, 6, tzinfo=timezone.utc)
CALENDAR = "https://spl.torneopal.fi/calendar/team/60534"
SCHEDULE = "https://gnistan.fi/miehet/ottelut/"


def spec(url=CALENDAR, params=None, **changes):
    return {
        "url": url, "params": {} if params is None else params,
        "source": "open_page", "requested_at": (NOW - timedelta(seconds=2)).isoformat(),
        "stage": "official_calendar", "request_digest": "registered-request-digest",
        **changes,
    }


@pytest.fixture(autouse=True)
def isolated_client_cache():
    transport._client.cache_clear()
    yield
    transport._client.cache_clear()


class FakeCurl:
    def __init__(self, *, body=b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n", status=b"200",
                 exit_code=0, timeout=False, version=b"curl 8.21.0 Schannel\r\n",
                 version_exit_code=0):
        self.body, self.status, self.exit_code = body, status, exit_code
        self.timeout, self.version, self.version_exit_code = timeout, version, version_exit_code
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), dict(kwargs)))
        if "--version" in command:
            return SimpleNamespace(returncode=self.version_exit_code, stdout=self.version,
                                   stderr=b"")
        Path(command[command.index("--output") + 1]).write_bytes(self.body)
        if self.timeout:
            raise transport.subprocess.TimeoutExpired(command, timeout=30)
        return SimpleNamespace(returncode=self.exit_code, stdout=self.status,
                               stderr=b"not copied to the public receipt")


def install(monkeypatch, fake):
    monkeypatch.setattr(transport.shutil, "which", lambda name: "C:/test/curl.exe")
    monkeypatch.setattr(transport.subprocess, "run", fake)


def test_success_preserves_actual_bytes_request_params_and_observation_clock(tmp_path, monkeypatch):
    fake = FakeCurl(body=b"BEGIN:VCALENDAR\r\nUID:4353542\r\nEND:VCALENDAR\r\n")
    install(monkeypatch, fake)
    request = spec(params={"t": "1741853856737"})
    original = deepcopy(request)
    archive = RawArchive(tmp_path / "archive")
    obs = transport.capture_system_tls_page(request, archive=archive, clock=lambda: NOW)

    assert request == original
    assert obs.endpoint == CALENDAR and obs.params == {"t": "1741853856737"}
    assert obs.observed_at == NOW
    assert obs.context["requested_at"] == request["requested_at"]
    assert obs.context["preparation_stage"] == request["stage"]
    assert obs.context["preparation_request_digest"] == request["request_digest"]
    assert obs.http_status == 200 and obs.error_class is None
    assert archive.read_body(obs) == fake.body
    assert obs.content_hash == sha256(fake.body).hexdigest()
    assert len(archive.observations()) == 1
    assert fake.calls[-1][0][-1] == CALENDAR + "?t=1741853856737"
    policy = obs.context["transport"]
    assert policy["tls_verification"] is True
    assert policy["default_system_certificate_revocation"] is True
    assert policy["primary_http_attempts"] == 1
    assert policy["system_certificate_metadata_requests"] == "not_measured"


def test_actual_command_disables_curlrc_keeps_tls_and_never_follows_redirect(tmp_path, monkeypatch):
    fake = FakeCurl(status=b"302", body=b"redirect response must stay here")
    install(monkeypatch, fake)
    obs = transport.capture_system_tls_page(spec(SCHEDULE), archive=RawArchive(tmp_path),
                                            clock=lambda: NOW)
    assert len(fake.calls) == 2  # One offline version check and one primary GET.
    for command, options in fake.calls:
        assert command[1] == "--disable"
        assert not options.get("shell", False)
        assert not {"-k", "--insecure", "--ssl-no-revoke", "--ssl-revoke-best-effort",
                    "-L", "--location", "--location-trusted", "--config", "-K",
                    "--netrc", "--netrc-optional", "--cacert", "--capath"}.intersection(command)
    command, options = fake.calls[-1]
    assert command[command.index("--noproxy") + 1] == "*"
    assert command[command.index("--proto") + 1] == "=https"
    assert command[command.index("--retry") + 1] == "0"
    assert int(command[command.index("--max-filesize") + 1]) == 2_000_000
    assert options["timeout"] == 30
    assert obs.http_status == 302
    assert obs.context["transport"]["automatic_redirects"] is False


@pytest.mark.parametrize("exit_code,status,body,expected_error", [
    (60, b"000", b"", "CurlExit60"),
    (28, b"200", b"partial response", "CurlExit28"),
    (0, b"403", b"Forbidden", None),
])
def test_transport_failure_and_http_denial_are_archived_without_retry(
        tmp_path, monkeypatch, exit_code, status, body, expected_error):
    fake = FakeCurl(exit_code=exit_code, status=status, body=body)
    install(monkeypatch, fake)
    archive = RawArchive(tmp_path)
    obs = transport.capture_system_tls_page(spec(), archive=archive, clock=lambda: NOW)
    assert obs.error_class == expected_error
    assert obs.http_status == (int(status) or None)
    assert archive.read_body(obs) == body
    assert obs.observed_at == NOW
    assert obs.context["transport"]["curl_exit_code"] == exit_code
    assert len(fake.calls) == 2 and len(archive.observations()) == 1


def test_process_timeout_retains_partial_raw_with_failed_status(tmp_path, monkeypatch):
    fake = FakeCurl(timeout=True, body=b"partial")
    install(monkeypatch, fake)
    archive = RawArchive(tmp_path)
    obs = transport.capture_system_tls_page(spec(), archive=archive, clock=lambda: NOW)
    assert obs.error_class == "CurlProcessTimeout"
    assert obs.http_status is None
    assert obs.context["transport"]["curl_exit_code"] is None
    assert archive.read_body(obs) == b"partial"
    assert len(fake.calls) == 2


@pytest.mark.parametrize("request_spec", [
    spec("https://gnistan.fi.evil.example/miehet/ottelut/"),
    spec("http://gnistan.fi/miehet/ottelut/"),
    spec("https://gnistan.fi/naiset/ottelut/"),
    spec(SCHEDULE, {"t": "123"}),
    spec(CALENDAR, {"t": "123", "redirect": "https://example.test"}),
    spec(CALENDAR, {"t": "123&insecure=true"}),
    spec(source="spftll"),
])
def test_unknown_host_source_and_request_parameters_fail_before_subprocess(
        tmp_path, monkeypatch, request_spec):
    fake = FakeCurl()
    install(monkeypatch, fake)
    archive = RawArchive(tmp_path)
    with pytest.raises(ValueError):
        transport.capture_system_tls_page(request_spec, archive=archive, clock=lambda: NOW)
    assert fake.calls == [] and archive.observations() == []


@pytest.mark.parametrize("version,exit_code", [
    (b"curl 8.21.0 OpenSSL\r\n", 0),
    (b"curl 8.21.0 Schannel\r\n", 1),
])
def test_non_schannel_or_unusable_client_is_rejected_before_http(
        tmp_path, monkeypatch, version, exit_code):
    fake = FakeCurl(version=version, version_exit_code=exit_code)
    install(monkeypatch, fake)
    archive = RawArchive(tmp_path)
    with pytest.raises(ValueError):
        transport.capture_system_tls_page(spec(), archive=archive, clock=lambda: NOW)
    assert len(fake.calls) == 1 and "--version" in fake.calls[0][0]
    assert archive.observations() == []
