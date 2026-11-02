"""Public source capabilities selected from fixture identity, not user questions.

The registry contains generic club entry points observed during source review.
It neither assigns information value nor selects a particular news conclusion.
Transport and content success remain per-response observations, not a promise
made by this catalogue. Unknown clubs do not inherit another club's identity.
"""
from __future__ import annotations

from copy import deepcopy
import unicodedata


VERSION = "match-source-profiles-v1"
_NATIONAL_ENTRIES_V2 = {
    "Estonia": ("https://jalgpall.ee/koondis/1/uudised", r"/koondis/1/uudised/[^/]+/?"),
    "Slovakia": ("https://futbalsfz.sk/?menu=39", r"/muzi-a-[^/]+/"),
}


def _key(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty fixture team name required")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_CLUBS = {
    "arsenal": {
        "canonical_name": "Arsenal",
        "aliases": ("Arsenal", "Arsenal FC", "Arsenal F.C.", "阿森纳"),
        "allowed_hosts": ("www.arsenal.com", "help.arsenal.com"),
        "entries": (
            {"kind": "club_home", "url": "https://www.arsenal.com/"},
            {"kind": "fixture_index", "url": "https://help.arsenal.com/support/solutions/articles/101000594163-fixture-and-event-timings-for-local-residents"},
        ),
    },
    "leeds": {
        "canonical_name": "Leeds United",
        "aliases": ("Leeds", "Leeds United", "Leeds United FC", "Leeds United F.C.", "利兹联"),
        "allowed_hosts": ("www.leedsunited.com",),
        "entries": (
            # This search-observed legacy entry serves the current home page,
            # not the historic article identified by its unused query field.
            # Keep its observed URL verbatim; no invented article paths.
            {"kind": "club_home", "url": "https://www.leedsunited.com/en?article=26856"},
            {"kind": "fixture_index", "url": "https://www.leedsunited.com/en/matches/mens/fixtures/premier-league"},
        ),
    },
}
_ALIASES = {_key(alias): club_id for club_id, club in _CLUBS.items() for alias in club["aliases"]}


def resolve_profile(home_team, away_team):
    """Return exact-match capability bounds for a supplied pair of teams.

    Matching is deliberately limited to known aliases; no fuzzy association or
    league membership is inferred. The common league domain is allowed for
    discovered links only when both clubs are registered in this EPL preset.
    Generic index/home pages are seeds, never case-selected news articles.
    """
    inputs = [home_team, away_team]
    keys = [_key(team) for team in inputs]
    ids = [_ALIASES.get(key) for key in keys]
    if keys[0] == keys[1] or ids[0] is not None and ids[0] == ids[1]:
        raise ValueError("Home and away teams must be distinct")
    entries, hosts, identities = [], set(), []
    for side, team, club_id in zip(("home", "away"), inputs, ids):
        club = _CLUBS.get(club_id)
        identities.append({"side": side, "input_name": team,
                           "canonical_name": club["canonical_name"] if club else None,
                           "registration_status": "registered_alias" if club else "unknown"})
        if club:
            hosts.update(club["allowed_hosts"])
            entries.extend({**entry, "club": club["canonical_name"], "side": side}
                           for entry in deepcopy(club["entries"]))
    if all(ids):
        hosts.add("www.premierleague.com")
    return {"version": VERSION, "teams": identities,
            "status": "registered" if all(ids) else "partial" if any(ids) else "unsupported",
            "allowed_hosts": sorted(hosts),
            "seed_urls": sorted({entry["url"] for entry in entries}),
            "entries": entries,
            "scope": "official_club_entry_points_with_partial_team_coverage",
            "selection_basis": "fixture_identity_to_source_capabilities_not_information_value",
            "current_transport_success_guaranteed": False,
            "unknown_teams": [team for team, club_id in zip(inputs, ids) if club_id is None]}


def resolve_verified_profile(binding, *, version=1):
    """Source entry points for an independently bound fixture, without scores.

    The caller verifies the raw identity archive first. The official match URL
    is observed in that archive, not generated from team names. Generic national
    association entries are capabilities, never case-selected article URLs.
    """
    from urllib.parse import urlsplit
    from .inquiry_tools import public_url
    if type(version) is not int or version not in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
        raise ValueError("Verified source profile version must be from 1 to 9")
    if binding.get("status") != "CONFIRMED" or not binding.get("canonical"):
        raise ValueError("A confirmed fixture identity is required")
    canonical = binding["canonical"]
    if version in {6, 7, 8, 9}:
        if binding.get("schema") != "premier-league-fixture-identity-v1":
            raise ValueError("Profile 6 requires the raw-backed Premier League identity")
        # Generic directories actually fetched in phase3 source diagnostics.
        # This registry contains no article permalinks or target-page content.
        registry = {"Leeds United": "https://www.leedsunited.com/en/news",
                    "Leeds": "https://www.leedsunited.com/en/news",
                    "Arsenal": "https://www.arsenal.com/news"}
        if version >= 7:
            # Observed on the archived official homepage, then fetched as a
            # separate two-request capability diagnostic on 2026-10-09.
            registry["Liverpool"] = "https://www.liverpoolfc.com/news/category/men"
        if version >= 8:
            # Public generic entries observed on 2026-10-10 before freezing
            # the three-match run. No article permalink or body is supplied.
            registry.update({
                "Aston Villa": "https://www.avfc.co.uk/",
                "Brentford": "https://www.brentfordfc.com/en/news/listing/all-news",
                "Ipswich Town": "https://www.itfc.co.uk/",
                "Fulham": "https://www.fulhamfc.com/news/",
                "Chelsea": "https://www.chelseafc.com/en",
            })
        if version >= 9:
            # Exact public directories captured on 2026-10-11. This adds
            # capabilities only; article URLs must come from an Agent read.
            registry.update({
                "Crystal Palace": "https://www.cpfc.co.uk/news/",
                "Nottingham Forest": "https://www.nottinghamforest.co.uk/news/",
            })
        entries = [{"kind": "fixture_index", "url": canonical["source_url"]}]
        teams = []
        for side in ("home", "away"):
            name = canonical[side + "_name"]
            teams.append({"side": side, "canonical_name": name,
                          "registration_status": "raw_verified_fixture"})
            if name in registry:
                entries.append({"kind": "news_index", "url": registry[name],
                                "club": name, "side": side,
                                "source_scope": ("official_generic_directory_archived_20261011_serialized_metadata_partial"
                                                 if version >= 9 and name in {"Crystal Palace", "Nottingham Forest"}
                                                 else "official_generic_entry_observed_before_three_match_run_20261010"
                                                 if version >= 8 and name in {"Aston Villa", "Brentford", "Ipswich Town", "Fulham", "Chelsea"}
                                                 else "official_generic_directory_observed_in_phase3_diagnostics")})
        hosts = sorted({urlsplit(entry["url"]).hostname for entry in entries})
        for entry in entries:
            public_url(entry["url"], hosts)
        return {"version": f"verified-match-source-profiles-v{version}", "teams": teams,
                "status": "verified_fixture_with_available_public_entries",
                "allowed_hosts": hosts, "seed_urls": sorted({entry["url"] for entry in entries}),
                "entries": entries, "identity_binding_sha256": binding["binding_sha256"],
                "selection_basis": "verified_identity_to_capabilities_not_information_value",
                "current_transport_success_guaranteed": False,
                "unknown_teams": [team["canonical_name"] for team in teams if team["canonical_name"] not in registry]}
    canada = binding.get("schema") == "canada-senior-fixture-identity-v1"
    finland = binding.get("schema") == "finland-club-fixture-identity-v1"
    finnish_stage = binding.get("schema") == "finland-stage-fixture-identity-v1"
    official_only = binding.get('schema') == 'official-fixture-evidence-identity-v1'
    if official_only != (version == 5):
        raise ValueError('Market-free official identities require explicit profile version 5')
    if finnish_stage != (version == 4):
        raise ValueError("The Finnish league identity requires explicit profile version 4")
    if finland and version != 3:
        raise ValueError("The Finnish source requires explicit profile version 3")
    if canada and version != 3:
        raise ValueError("The Canada source requires explicit profile version 3")
    url = canonical["source_url"]
    host = urlsplit(url).hostname
    public_url(url, [host])
    entries = [{"kind": "fixture_index", "url": url}]
    hosts, teams = {host}, []
    # Public generic entry actually observed on 2026-10-06. News links are still
    # obtained from its live response and selected by the acquisition policy.
    associations = {"Scotland": "https://www.scottishfa.co.uk/en/scotland"}
    if version >= 2:
        associations.update({name: entry[0] for name, entry in _NATIONAL_ENTRIES_V2.items()})
    if canada:
        # Both domains and the generic entry were actually observed in the
        # official archive. Article URLs remain unread until the actor queries.
        associations["Canada"] = "https://canadasoccer.com/"
        hosts.update({"canadasoccer.com", "news.canadasoccer.com"})
    if finland:
        # Generic first-party entry observed in the official identity discovery;
        # its future article links still require a paid Actor acquisition.
        associations["IF Gnistan"] = "https://gnistan.fi/"
    shared_entries = []
    if finnish_stage or official_only:
        for observed in binding.get('observed_generic_entry_urls', []):
            if observed not in {'https://www.veikkausliiga.com/', 'https://www.veikkausliiga.com/uutiset/'}:
                raise ValueError('Unregistered observed Finnish competition entry')
            if observed not in shared_entries:
                shared_entries.append(observed)
    for side in ("home", "away"):
        name = canonical[side + "_name"]
        teams.append({"side": side, "canonical_name": name,
                      "registration_status": "raw_verified_fixture"})
        if name in associations:
            entry = associations[name]
            hosts.add(urlsplit(entry).hostname)
            entries.append({"kind": "news_index", "url": entry,
                            "club": name, "side": side})
        for entry in shared_entries:
            hosts.add(urlsplit(entry).hostname)
            entries.append({'kind': 'news_index', 'url': entry, 'club': name, 'side': side,
                            'source_scope': 'verified_competition_shared_entry'})
    result = {"version": f"verified-match-source-profiles-v{version}", "teams": teams,
            "status": "verified_fixture_with_available_public_entries",
            "allowed_hosts": sorted(hosts),
            "seed_urls": sorted({item["url"] for item in entries}),
            "entries": entries, "identity_binding_sha256": binding["binding_sha256"],
            "selection_basis": "verified_identity_to_capabilities_not_information_value",
            "current_transport_success_guaranteed": False}
    if finnish_stage or official_only:
        result['identity_schema'] = binding['schema']
        result['shared_entry_basis'] = 'generic_navigation_observed_in_bound_official_schedule'
    return result


__all__ = ["VERSION", "resolve_profile", "resolve_verified_profile"]
