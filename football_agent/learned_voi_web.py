"""Existing archived feature tools and open-web transport in one VoI executor.

URL discovery is a transparent lexical proposal mechanism, not a learned value
estimate. The runtime selects from those proposals; page content is evidence,
never executable instructions. No new crawler or prediction model lives here.
"""
from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl

from football_research.acquisition.data import ControlledEpisode
from football_spider.models.contracts import Observation
from .inquiry_tools import public_url
from .learned_voi_runtime import digest
from .g_prior_features import strip_prior_features

VERSION = "learned-voi-existing-web-bridge-v1"


def _web_action(tool, value, *, cost):
    request = {"tool": tool, "query" if tool == "search_web" else "url": value}
    key = "web:"+digest(request)
    return {"action_id": "web-"+digest(request)[:24], "group_ids": [],
            "information_key": key, "available": True, "cost_status": "simulated",
            "cost": deepcopy(cost), "source": "open_web", "version": VERSION, **request}


def _url(value, hosts):
    parts = urlsplit(public_url(value, hosts))
    # Canonical request identity includes query parameters, ignores fragments.
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    if len(dict(pairs)) != len(pairs):
        raise ValueError("Ambiguous duplicate query parameters")
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                       urlencode(sorted(pairs)), ""))


class MatchDocumentCandidates:
    """Generate read/search candidates from the match and admitted page links.

    Fixed source/host capabilities are engineering configuration. The user does
    not choose injury/weather/etc. Failed requests stay attempted. This modest
    generator does not claim semantic completeness or learned relevance.
    """

    version = VERSION

    def __init__(self, *, match_terms, seed_urls, allowed_hosts, numeric_actions,
                 web_cost, max_discovered_links=12):
        if not isinstance(match_terms, (list, tuple)) or len(match_terms) != 2 or any(not isinstance(x, str) or not x.strip() for x in match_terms):
            raise ValueError("Two fixture team names required")
        if type(max_discovered_links) is not int or not 0 <= max_discovered_links <= 32:
            raise ValueError("At most 32 discovered link candidates")
        self.match_terms = tuple(x.strip() for x in match_terms)
        self.allowed_hosts = tuple(allowed_hosts)
        self.seed_urls = sorted({_url(url, allowed_hosts) for url in seed_urls})
        fields = {"action_id", "group_ids", "available", "cost_status", "cost", "source", "tool", "version"}
        self.numeric_actions = [{**{k: deepcopy(v) for k, v in a.items() if k in fields},
                                 "information_key": a["action_id"]} for a in numeric_actions]
        self.web_cost, self.max_links = deepcopy(web_cost), max_discovered_links

    def configuration(self):
        return {"version": self.version, "match_terms": list(self.match_terms),
                "seed_urls": self.seed_urls, "allowed_hosts": list(self.allowed_hosts),
                "numeric_actions": self.numeric_actions, "web_cost": self.web_cost,
                "max_discovered_links": self.max_links,
                "discovery_rule": "observed_link_title_or_url_contains_a_full_fixture_team_name",
                "relevance_status": "lexical_proposal_not_semantic_verification"}

    def __call__(self, public):
        urls = list(self.seed_urls)
        found = set()
        for document in public["documents"]:
            for link in document.get("links", []):
                try:
                    url = _url(link["url"], self.allowed_hosts)
                except (KeyError, TypeError, ValueError):
                    continue
                haystack = (str(link.get("title", ""))+" "+url).casefold().replace("_", " ")
                if any(team.casefold() in haystack for team in self.match_terms):
                    found.add(url)
        urls.extend(sorted(found-set(urls))[:self.max_links])
        kickoff = public["state"]["base"]["kickoff"][:10]
        query = " ".join(self.match_terms)+" "+kickoff+" match preview"
        return deepcopy(self.numeric_actions)+[_web_action("search_web", query, cost=self.web_cost)]+[
            _web_action("read_url", url, cost=self.web_cost) for url in urls]


class ArchivedAndWebExecutor:
    version = VERSION

    def __init__(self, row, numeric_actions, tools, *, seed_urls):
        self.environment = ControlledEpisode(row, numeric_actions)
        self.numeric_ids = {a["action_id"] for a in numeric_actions}
        self.tools, self.seed_urls = tools, list(seed_urls)
        self.run_id = ""
        self.counter = 0

    def __call__(self, action, public):
        self.counter += 1
        if action["action_id"] in self.numeric_ids:
            after = self.environment.reveal(action["action_id"])
            after["action_history"] = deepcopy(public["state"].get("action_history", []))+[
                {"action_id": action["action_id"]}]
            return {"status": "OK", "state_after": after,
                    "actual_cost": {"requests": 0., "money": 0., "latency_ms": None, "tokens": 0.},
                    "receipt": {"provider": "frozen_feature_archive", "action_id": action["action_id"],
                                "scope": "historical_reconstruction_projection_not_network_request"}}
        transport_action = strip_prior_features(action)
        expected = _web_action(transport_action["tool"], transport_action.get("query", transport_action.get("url")), cost=transport_action["cost"])
        if transport_action != expected:
            raise ValueError("Unbound web action identity or payload")
        prepared = self.tools.prepare(transport_action, {"task": {"seed_urls": self.seed_urls},
                                               "documents": public["documents"]})
        returned = self.tools.execute(prepared, run_id=self.run_id, request_id=f"open-{self.counter}")
        observation = Observation.model_validate(returned["observation"])
        # Verify bytes independently of projected page text before admission.
        self.tools.archive.read_body(observation)
        document = returned["document"]
        if document["raw_sha256"] != observation.content_hash:
            raise ValueError("Projected document differs from raw archive")
        return {"status": "OK" if document["status"] == "OK" else "FAILED" if document["status"] == "FAILED" else "NO_DATA",
                "document": document,
                "actual_cost": {"requests": float(returned["http_attempts"]), "money": returned["money"],
                                "latency_ms": 1000*returned["latency_seconds"], "tokens": 0.},
                "receipt": {"prepared": prepared, "observation": returned["observation"],
                            "archive_verified": True, "money_status": returned["money_status"]}}
