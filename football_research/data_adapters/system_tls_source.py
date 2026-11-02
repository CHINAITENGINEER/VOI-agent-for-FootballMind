"""One registered public request through the Windows system TLS client.

This optional transport serves Finnish identity and official result endpoints.
It does not change global trust, credentials, proxy settings or legacy readers.
The enclosing workflow owns the durable intent and request budget.
"""
from datetime import datetime, timezone
from functools import lru_cache
import os
import re
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

VERSION = "registered-public-source-windows-schannel-v2"
ENDPOINTS = {
    "https://gnistan.fi/miehet/ottelut/",
    "https://spl.torneopal.fi/calendar/team/60534",
}


@lru_cache(maxsize=1)
def _client():
    executable = shutil.which("curl.exe") if os.name == "nt" else None
    if not executable:
        raise ValueError("This explicit identity transport needs Windows curl with Schannel")
    result = subprocess.run([executable, "--disable", "--version"], capture_output=True, timeout=5,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or b"Schannel" not in result.stdout:
        raise ValueError("The configured client is not the verified system TLS transport")
    return executable, result.stdout.decode("ascii", errors="replace").splitlines()[0]


def capture_system_tls_page(spec, *, archive, clock):
    url, params = spec["url"], spec["params"]
    result_list = url == "https://www.veikkausliiga.com/tilastot/2026/veikkausliiga/ottelut/"
    result_detail = re.fullmatch(r"https://www\.veikkausliiga\.com/tilastot/2026/veikkausliiga/ottelut/[1-9][0-9]*/seuranta/", url)
    if (url not in ENDPOINTS and not result_list and not result_detail) or spec["source"] != "open_page":
        raise ValueError("The system TLS identity transport requires a registered public endpoint")
    valid = (params in ({"group": "2"}, {"group": "3"})) if result_list else (not params if result_detail else
        not params or url == "https://spl.torneopal.fi/calendar/team/60534" and set(params) == {"t"} and str(params["t"]).isdigit())
    if not valid:
        raise ValueError("Unregistered identity URL parameters")
    executable, client = _client()
    request_url = url + ("?" + urlencode(params) if params else "")
    error, status, exit_code = None, None, None
    with TemporaryDirectory(prefix="football-identity-") as temporary:
        body_path = Path(temporary) / "body.bin"
        command = [executable, "--disable", "--noproxy", "*", "--proto", "=https",
            "--retry", "0", "--connect-timeout", "6", "--max-time", "25",
            "--max-filesize", "2000000", "--user-agent", "FootballMind/0.1 research",
            "--silent", "--output", str(body_path), "--write-out", "%{http_code}", request_url]
        try:
            response = subprocess.run(command, capture_output=True, timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            exit_code = response.returncode
            status = int(response.stdout) if response.stdout.strip().isdigit() and int(response.stdout) else None
            error = None if exit_code == 0 else "CurlExit" + str(exit_code)
        except subprocess.TimeoutExpired:
            error = "CurlProcessTimeout"
        body = body_path.read_bytes() if body_path.exists() else b""
        if len(body) > 2000000:
            body, error = b"", "IdentityResponseExceedsLimit"
    return archive.save("open_page", url, params, body, status=status, error_class=error,
        observed_at=clock(), context={"requested_at": spec["requested_at"],
            "preparation_stage": spec["stage"], "preparation_request_digest": spec["request_digest"],
            "transport": {"version": VERSION, "client": client, "route": "direct",
                "tls_verification": True, "default_system_certificate_revocation": True,
                "authentication": "none", "automatic_retries": 0, "automatic_redirects": False,
                "primary_http_attempts": 1, "system_certificate_metadata_requests": "not_measured",
                "curl_exit_code": exit_code}})
