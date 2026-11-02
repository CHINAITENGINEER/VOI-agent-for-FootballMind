"""Explicit, session-scoped transport policy for the existing inquiry tools.

Acquisition, attempt accounting, raw archiving and document projection continue
to use OpenWebTools. Only trusted construction selects a route; candidate JSON
does not supply authentication, request headers or networking configuration.
"""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from urllib.parse import urlsplit

from .audit import digest
from .inquiry_tools import OpenWebTools
from .transport_probe import DiagnosticSession


VERSION = "routed-inquiry-transport-v1"
RESEARCH_USER_AGENT = "FootballMind/0.1 research"
SPFTLL_PRESET_ID = "spftll-public-list-headers-v1"
SPFTLL_ENDPOINT = "https://slb.haoyun999.cn/api/Game/GetSimpleMatchsAll"
# Public fields already verified in the existing site-analysis and Round 25.
# No token/rkey placeholders, credentials, cookies or authentication are added.
SPFTLL_HEADERS = MappingProxyType({
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://xx.spftll.cn/",
    "X-Requested-With": "XMLHttpRequest",
    "XMIME-TYPE": "3",
    "AppType": "0",
    "DeviceKind": "1",
})


def request_header_profile(endpoint):
    parts = urlsplit(endpoint)
    matching = (parts.scheme == "https" and parts.hostname == "slb.haoyun999.cn"
                and parts.port in {None, 443}
                and parts.path == "/api/Game/GetSimpleMatchsAll"
                and not parts.query and not parts.fragment
                and parts.username is None and parts.password is None)
    headers = {"User-Agent": RESEARCH_USER_AGENT}
    if matching:
        headers.update(SPFTLL_HEADERS)
    return {"preset_id": SPFTLL_PRESET_ID if matching else "research-public-page-v1",
            "headers": headers}


class _RoutedSession(DiagnosticSession):
    def get(self, url, **kwargs):
        # The caller's parameters stay untouched; only a registered, public
        # endpoint preset selects headers. Untrusted header overrides are not
        # exposed through the inquiry candidate or prepared-action schema.
        kwargs["headers"] = request_header_profile(url)["headers"]
        return super().get(url, **kwargs)


class _ArchiveWithTransportContext:
    """Per-request view that adds provenance without mutating a shared archive."""
    def __init__(self, archive, policy, session):
        self._archive, self._policy, self._session = archive, policy, session
        self.root = archive.root

    def save(self, source, endpoint, params, body, **kwargs):
        context = deepcopy(kwargs.get("context") or {})
        profile = request_header_profile(endpoint)
        context["transport"] = {
            "version": VERSION, "route": self._policy["route"],
            "policy_sha256": digest(self._policy),
            "tls_verification": True, "authentication": "none_no_netrc",
            "request_header_preset": profile["preset_id"],
            "request_headers": profile["headers"],
            "failure_detail": deepcopy(self._session.failure),
        }
        kwargs["context"] = context
        return self._archive.save(source, endpoint, params, body, **kwargs)

    def read_body(self, observation):
        return self._archive.read_body(observation)


class RoutedOpenWebTools(OpenWebTools):
    """Opt-in routing, with the base prepare/project/archive semantics retained.

    ``route`` is intentionally required. ``environment`` retains environment
    proxy discovery; ``direct`` ignores proxy settings for this session only.
    Both use TLS verification, retain configured CA bundles and skip netrc.
    """
    def __init__(self, archive, allowed_hosts, *, route, clock=None, extraction_version=2):
        if route not in {"environment", "direct"}:
            raise ValueError("The inquiry route must be environment or direct")
        self.route = route
        super().__init__(archive, allowed_hosts, clock=clock, extraction_version=extraction_version)

    def transport_policy(self):
        return {
            "version": VERSION, "route": self.route,
            "session_implementation": "football_agent.transport_probe.DiagnosticSession",
            "authentication": "none_no_netrc", "tls_verification": True,
            "proxy_policy": "inherit_environment" if self.route == "environment" else "ignore_environment_for_this_session",
            "ca_bundle_policy": "honor_requests_environment_else_default_verified_store",
            "default_request_headers": {"User-Agent": RESEARCH_USER_AGENT},
            "header_presets": [{"preset_id": SPFTLL_PRESET_ID, "endpoint": SPFTLL_ENDPOINT,
                                "match_policy": "exact_https_host_path_and_default_port_no_query_or_fragment",
                                "headers": dict(SPFTLL_HEADERS)}],
            "candidate_may_override_policy": False,
            "connect_timeout_seconds": 6, "read_timeout_seconds": 16,
            "response_read_elapsed_limit_seconds": 25,
            "automatic_redirects": False, "request_attempts_per_action": 1,
            "max_response_bytes": 2_000_000,
            "failure_detail_policy": "exception_class_chain_and_integer_codes_without_messages",
        }

    def configuration(self):
        return {**super().configuration(), "transport_policy": self.transport_policy()}

    def execute(self, prepared, *, run_id, request_id):
        if not isinstance(prepared, dict) or set(prepared) != {"tool", "endpoint", "params", "source"}:
            raise ValueError("Prepared actions cannot override the configured transport policy")
        session = _RoutedSession(self.route)
        archive = _ArchiveWithTransportContext(self.archive, self.transport_policy(), session)
        delegate = OpenWebTools(archive, self.allowed_hosts, session_factory=lambda: session,
                                clock=self.clock, extraction_version=self.extraction_version)
        return delegate.execute(prepared, run_id=run_id, request_id=request_id)
