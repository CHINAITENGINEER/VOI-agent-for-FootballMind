"""Explicit per-source routing over the existing single-attempt web transport.

Connection failures and weak search results are different observations. This
adapter can select a configured route before an acquisition, but it does not
retry, infer evidence from failure, choose a news article, or estimate its VoI.
Existing callers of RoutedOpenWebTools keep their original configuration.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from urllib.parse import urlsplit

from .inquiry_tools import OpenWebTools, public_url
from .inquiry_transport import RoutedOpenWebTools


VERSION = "source-route-selection-v1"
SYSTEM_TLS_VERSION = "source-route-selection-v2"
SEARCH_HOST = "www.bing.com"


class SourceRoutedOpenWebTools(OpenWebTools):
    """Use one trusted route per source; each action has exactly one attempt.

    Routes are deployment configuration, independent of candidate text. The
    selected delegate writes its actual route and policy digest to the normal
    observation context, and keeps the same archive/projection contracts.
    """

    def __init__(self, archive, allowed_hosts, *, default_route="environment",
                 route_overrides=None, clock=None, extraction_version=3):
        if default_route not in {"environment", "direct"}:
            raise ValueError("Unknown default source route")
        super().__init__(archive, allowed_hosts, clock=clock,
                         extraction_version=extraction_version)
        if route_overrides is None:
            route_overrides = {}
        if not isinstance(route_overrides, Mapping):
            raise ValueError("Source routes must be an explicit host mapping")
        known = set(self.allowed_hosts) | {SEARCH_HOST}
        for host, route in route_overrides.items():
            if (host not in known or route not in {"environment", "direct", "system_tls"}
                    or route == "system_tls" and host != "www.veikkausliiga.com"):
                raise ValueError("Source route must use a registered exact host and known route")
        self.default_route = default_route
        self.route_overrides = dict(sorted(route_overrides.items()))
        self._system_tls_urls = set()

    def prepare(self, candidate, public):
        prepared = super().prepare(candidate, public)
        if (prepared["tool"] == "read_url" and
                self.route_overrides.get(urlsplit(prepared["endpoint"]).hostname) == "system_tls"):
            from .system_tls_web import authorize_system_tls_prepared
            self._system_tls_urls.add(authorize_system_tls_prepared(prepared, public, self.allowed_hosts))
        return prepared

    def _delegate(self, route):
        if route == "system_tls":
            from .system_tls_web import SystemTLSOpenWebTools
            return SystemTLSOpenWebTools(self.archive, self.allowed_hosts,
                authorized_urls=self._system_tls_urls, clock=self.clock,
                extraction_version=self.extraction_version)
        return RoutedOpenWebTools(self.archive, self.allowed_hosts, route=route,
            clock=self.clock, extraction_version=self.extraction_version)

    def configuration(self):
        result = {**self._delegate(self.default_route).configuration(),
                "source_route_policy": {
                    "version": VERSION, "default_route": self.default_route,
                    "route_overrides": deepcopy(self.route_overrides),
                    "selection_basis": "explicit_configuration_not_information_value",
                    "automatic_failover": False, "request_attempts_per_action": 1,
                    "candidate_may_override_route": False}}
        if "system_tls" in self.route_overrides.values():
            from .system_tls_web import SystemTLSOpenWebTools
            result["source_route_policy"].update(version=SYSTEM_TLS_VERSION,
                system_tls_policy=SystemTLSOpenWebTools.transport_policy())
        return result

    @classmethod
    def from_configuration(cls, archive, configuration, *, clock=None):
        if not isinstance(configuration, Mapping):
            raise ValueError("Complete source transport configuration required")
        policy = configuration.get("source_route_policy")
        if not isinstance(policy, Mapping) or policy.get("version") not in {VERSION, SYSTEM_TLS_VERSION}:
            raise ValueError("Unsupported source route policy")
        try:
            version = configuration["version"]
            if version not in {"open-web-read-v1", "open-web-read-v2", "open-web-read-v3"}:
                raise ValueError("Unsupported archived projection version")
            tools = cls(archive, configuration["allowed_hosts"],
                default_route=policy["default_route"],
                route_overrides=policy["route_overrides"], clock=clock,
                extraction_version=int(version[-1]))
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete source route configuration") from error
        if tools.configuration() != configuration:
            raise ValueError("Source route configuration does not reconstruct exactly")
        return tools

    def execute(self, prepared, *, run_id, request_id):
        if not isinstance(prepared, dict) or set(prepared) != {"tool", "endpoint", "params", "source"}:
            raise ValueError("Prepared action cannot override configured source routes")
        if prepared["tool"] == "search_web":
            if prepared["endpoint"] != "https://www.bing.com/search":
                raise ValueError("Search route is bound to the existing search endpoint")
            host = SEARCH_HOST
        elif prepared["tool"] == "read_url":
            host = urlsplit(public_url(prepared["endpoint"], self.allowed_hosts)).hostname
        else:
            raise ValueError("Unsupported source transport action")
        route = self.route_overrides.get(host, self.default_route)
        return self._delegate(route).execute(prepared, run_id=run_id, request_id=request_id)


__all__ = ["VERSION", "SYSTEM_TLS_VERSION", "SourceRoutedOpenWebTools"]
