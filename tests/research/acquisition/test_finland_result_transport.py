"""Offline result-route contract; format validation is not fixture identity."""
from copy import deepcopy
from hashlib import sha256
import socket

import pytest

from football_research.data_adapters import system_tls_source as transport
from football_spider.storage.raw_archive import RawArchive
from test_system_tls_source import FakeCurl, install, isolated_client_cache, NOW, spec

LIST = "https://www.veikkausliiga.com/tilastot/2026/veikkausliiga/ottelut/"
DETAIL = LIST+"3617344/seuranta/"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Result transport tests must never connect to a real server")
    for obj, name in ((socket, "create_connection"), (socket, "getaddrinfo"),
                      (socket.socket, "connect"), (socket.socket, "connect_ex")):
        monkeypatch.setattr(obj, name, denied)


@pytest.mark.parametrize("url,params", [(LIST, {"group": "2"}), (DETAIL, {})])
def test_registered_result_routes_preserve_raw_bytes_and_use_one_verified_primary_get(tmp_path, monkeypatch, url, params):
    fake = FakeCurl(body=b"<html>constructed official result response</html>")
    install(monkeypatch, fake)
    request = spec(url, params, stage="official_regulation_result")
    original = deepcopy(request)
    archive = RawArchive(tmp_path)
    observed = transport.capture_system_tls_page(request, archive=archive, clock=lambda: NOW)
    assert request == original and observed.endpoint == url and observed.params == params
    assert observed.observed_at == NOW and observed.http_status == 200 and observed.error_class is None
    assert archive.read_body(observed) == fake.body and observed.content_hash == sha256(fake.body).hexdigest()
    assert len(archive.observations()) == 1
    gets = [(args, options) for args, options in fake.calls if "--version" not in args]
    assert len(gets) == 1
    args, options = gets[0]
    assert args[-1] == url+("?group=2" if params else "")
    assert args[1] == "--disable" and args[args.index("--retry")+1] == "0"
    assert args[args.index("--proto")+1] == "=https"
    assert not {"--insecure", "-k", "--ssl-no-revoke", "--ssl-revoke-best-effort", "--location", "-L"}.intersection(args)
    assert not options.get("shell", False)
    assert observed.context["transport"]["tls_verification"] is True
    assert observed.context["transport"]["default_system_certificate_revocation"] is True
    assert observed.context["transport"]["primary_http_attempts"] == 1


@pytest.mark.parametrize("url,params", [
    (LIST, {}), (LIST, {"group": "1"}), (LIST, {"group": 2}),
    (LIST, {"group": "2", "page": "2"}),
    (LIST.replace("/2026/", "/2025/"), {"group": "2"}),
    (LIST+"0/seuranta/", {}), (LIST+"0123/seuranta/", {}),
    (LIST+"-1/seuranta/", {}), (LIST+"palloliitto:4353542/seuranta/", {}),
    (DETAIL, {"group": "2"}), (DETAIL+"?preview=1", {}),
    (LIST.replace("www.veikkausliiga.com", "www.veikkausliiga.com.evil.test"), {"group": "2"}),
])
def test_wrong_group_season_or_malformed_native_id_and_extra_query_fail_before_http(tmp_path, monkeypatch, url, params):
    fake = FakeCurl()
    install(monkeypatch, fake)
    archive = RawArchive(tmp_path)
    with pytest.raises(ValueError):
        transport.capture_system_tls_page(spec(url, params), archive=archive, clock=lambda: NOW)
    assert fake.calls == [] and archive.observations() == []


def test_result_tls_failure_is_archived_once_and_is_not_retried_or_treated_as_score(tmp_path, monkeypatch):
    fake = FakeCurl(exit_code=60, status=b"000", body=b"")
    install(monkeypatch, fake)
    archive = RawArchive(tmp_path)
    observed = transport.capture_system_tls_page(spec(DETAIL), archive=archive, clock=lambda: NOW)
    assert observed.error_class == "CurlExit60" and observed.http_status is None
    assert len([args for args, _ in fake.calls if "--version" not in args]) == 1
    assert len(archive.observations()) == 1 and archive.read_body(observed) == b""
