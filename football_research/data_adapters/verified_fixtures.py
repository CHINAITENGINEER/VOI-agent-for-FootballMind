"""Raw-backed UEFA fixture identity and independent result admission.

Fixture identity uses static JSON-LD SportsEvent data on official senior Nations
League pages. Results also support a versioned raw public match API projection.
No JavaScript runs. Source-native IDs, exact aliases and UTC clocks must agree.
Human-written settlement rows are never accepted as the verification input.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from football_platform.information.fixture_corroboration import compare_assertion
from football_research.acquisition.state import parse_time
from football_research.acquisition.value import _canonical
from football_spider.storage.raw_archive import RawArchive
from .prospective_text import MarketPriorPolicy
from .settlement import export_results
from .types import ResearchDataset, ResearchMatch
from .official_settlement import (official_api_event_id, project_official_api_result,
                                 SETTLEMENT_VERSION as API_SETTLEMENT_VERSION)

VERSION = "raw-backed-fixture-identity-v1"
PARSER_VERSION = "uefa-senior-nations-league-jsonld-v1"
SETTLEMENT_VERSION = "raw-backed-independent-settlement-v1"
SCHEDULE_URL = "https://www.uefa.com/uefanationsleague/news/02a2-1fea18abbcbc-456e846509e7-1000--2026-27-uefa-nations-league-all-the-league-phase-fixtures-a/"
# The parser supports only adult men's 2026/27 league-phase matches. Draws are
# final league outcomes, not knockout ties. Articles 12, 14 and 18 define this
# scope; changing competition/season/stage requires a separately reviewed rule.
REGULATION_RULE = {
    "version": "uefa-unl-2026-27-adult-men-league-phase-v1",
    "period": "regulation",
    "meaning": "90_minutes_plus_stoppage_excludes_extra_time_penalties",
    "references": ["https://documents.uefa.com/r/Regulations-of-the-UEFA-Nations-League-2026/27/Article-"
                   + article + "-Online" for article in (
                       "12-Competition-stages", "14-Match-system-league-phase", "18-Extra-time-and-penalty-shoot-outs")],
}
_MATCH = re.compile(r"/uefanationsleague/match/([1-9][0-9]*)--[a-z0-9-]+/(?:matchinfo/)?$")
_TEAM = re.compile(r"/uefanationsleague/teams/([1-9][0-9]*)--[a-z0-9-]+/$")


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _url_id(url, pattern):
    parts = urlsplit(url)
    match = pattern.fullmatch(parts.path)
    if (parts.scheme != "https" or parts.netloc != "www.uefa.com" or parts.query
            or parts.username or not match):
        raise ValueError("Expected an official exact source-native URL")
    return match.group(1)


def _objects(value, depth=0):
    if depth > 20:
        raise ValueError("JSON-LD nesting exceeds limit")
    if isinstance(value, dict):
        yield value
        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from _objects(item, depth+1)
    elif isinstance(value, list):
        for item in value:
                yield from _objects(item, depth+1)


def _match_objects(soup, event_id):
    matches = []
    for tag in soup.find_all(attrs={"data-options": True}):
        try:
            data = _json(tag["data-options"])
        except (ValueError, TypeError):
            continue
        for item in _objects(data):
            if (str(item.get("id")) == event_id and isinstance(item.get("homeTeam"), dict)
                    and isinstance(item.get("awayTeam"), dict) and isinstance(item.get("competition"), dict)):
                matches.append(item)
    return matches


def _extract(observation, body, *, as_of):
    """One deterministic, fail-closed projection from actual bytes."""
    if observation.observed_at > parse_time(as_of):
        raise ValueError("Official observation was unavailable at audit time")
    if observation.http_status != 200 or observation.error_class:
        return {"status": "UNCONFIRMED", "reason": "official_response_unavailable", "canonical": None}
    if observation.source != "open_page":
        raise ValueError("Expected an acquired official HTML page")
    event_id = _url_id(observation.endpoint, _MATCH)
    if len(body) > 2_000_000:
        raise ValueError("Official page exceeds bounded parser input")
    requested = observation.context.get("requested_at")
    if requested is not None and parse_time(requested) > observation.observed_at:
        raise ValueError("Official receipt precedes its request")
    soup = BeautifulSoup(body, "html.parser")
    events = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            value = _json(script.get_text())
        except (ValueError, TypeError):
            return {"status": "UNCONFIRMED", "reason": "invalid_or_ambiguous_jsonld", "canonical": None}
        for obj in _objects(value):
            if obj.get("@type") == "SportsEvent":
                events.append(obj)
    if not events:
        return {"status": "UNCONFIRMED", "reason": "structured_sports_event_missing", "canonical": None}
    normalized = []
    for event in events:
        if _url_id(event["url"], _MATCH) != event_id:
            return {"status": "DISPUTED", "reason": "endpoint_and_embedded_event_disagree", "canonical": None}
        desc = re.fullmatch(r"UEFA Nations League ([0-9]{4}) League phase: .+", event.get("description", ""))
        if not desc:
            return {"status": "UNCONFIRMED", "reason": "competition_category_or_stage_unsupported", "canonical": None}
        teams = {}
        for side in ("home", "away"):
            team = event[side + "Team"]
            if team.get("@type") != "SportsTeam" or not isinstance(team.get("name"), str) or not team["name"].strip():
                raise ValueError("Structured source team identity missing")
            teams[side + "_name"] = team["name"]
            teams[side + "_id"] = "uefa:team:" + _url_id(team["sameAs"], _TEAM)
        if teams["home_id"] == teams["away_id"]:
            raise ValueError("Source event repeats the same team")
        if event.get("name") != teams["home_name"] + " vs " + teams["away_name"]:
            return {"status": "DISPUTED", "reason": "source_team_order_conflict", "canonical": None}
        normalized.append({**teams, "kickoff": parse_time(event["startDate"]).isoformat(),
            "competition": "UEFA Nations League", "category": "senior_men_national", "stage": "League phase",
            "season": desc.group(1), "source_url": event["url"], "source_event_id": "uefa:match:" + event_id,
            "event_status": event.get("eventStatus"), "parser_version": PARSER_VERSION})
    if len({_digest(row) for row in normalized}) != 1:
        return {"status": "DISPUTED", "reason": "conflicting_structured_events", "canonical": None}
    match_objects = _match_objects(soup, event_id)
    if not match_objects:
        return {"status": "UNCONFIRMED", "reason": "structured_competition_category_missing", "canonical": None}
    canonical = normalized[0]
    for match in match_objects:
        competition, round_info = match["competition"], match.get("round", {})
        if any(competition.get(key) != expected for key, expected in {
                "age": "Adult", "sex": "Male", "sportsType": "Football", "teamCategory": "National",
                "displayName": "UEFA Nations League", "id": "2014"}.items()):
            return {"status": "DISPUTED", "reason": "structured_competition_category_conflict", "canonical": None}
        if (round_info.get("displayName") != "League phase" or round_info.get("mode") != "Group"
                or str(match.get("seasonYear")) != canonical["season"]
                or parse_time(match["kickOffTime"]["dateTime"]) != parse_time(canonical["kickoff"])
                or any("uefa:team:" + str(match[side + "Team"].get("id")) != canonical[side + "_id"]
                       or match[side + "Team"].get("internationalName") != canonical[side + "_name"]
                       for side in ("home", "away"))):
            return {"status": "DISPUTED", "reason": "embedded_match_and_jsonld_identity_conflict", "canonical": None}
    return {"status": "EXTRACTED", "reason": "official_structured_identity", "canonical": normalized[0]}


def _observations(archive_root, observation_ids):
    if (not isinstance(observation_ids, (list, tuple)) or not observation_ids
            or len(set(observation_ids)) != len(observation_ids)):
        raise ValueError("Distinct explicit source observations are required")
    archive = RawArchive(Path(archive_root))
    by_id = {obs.id: obs for obs in archive.observations()}
    if any(ident not in by_id for ident in observation_ids):
        raise ValueError("Declared source observation not found in the archive")
    return [(by_id[ident], archive.read_body(by_id[ident])) for ident in sorted(observation_ids)]


def verify_fixture_identity(initialization_manifest, *, archive_root, observation_ids, aliases, as_of):
    """Bind an existing quote initialisation to independently acquired identity.

    Aliases are explicit entity mappings, not fuzzy learned name matching. Both
    the provider's original name and the source's canonical name must occur in
    the same declared side. Failed/ambiguous observations produce no canonical
    identity and cannot enable collection or training.
    """
    as_of = parse_time(as_of)
    policy = MarketPriorPolicy.from_manifest(initialization_manifest)
    fixture = policy.model.fixture
    if not isinstance(aliases, dict) or set(aliases) != {"home", "away"}:
        raise ValueError("Explicit home and away aliases are required")
    aliases = deepcopy(aliases)
    for side in aliases:
        names = aliases[side]
        if (not isinstance(names, list) or not names or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not n.strip() or len(n) > 120 for n in names)
                or fixture[side + "_team_name"] not in names):
            raise ValueError("Aliases must bind exact original provider names")
    if set(aliases["home"]) & set(aliases["away"]):
        raise ValueError("Home and away alias identities must be distinct")
    facts, evidence = [], []
    for obs, body in _observations(archive_root, observation_ids):
        evidence.append({"observation_id": obs.id, "raw_sha256": obs.content_hash,
                         "url": obs.endpoint, "observed_at": obs.observed_at.isoformat()})
        facts.append(_extract(obs, body, as_of=as_of))
    status, reason, canonical, comparison = "UNCONFIRMED", "official_response_unavailable", None, None
    extracted = [f["canonical"] for f in facts if f["status"] == "EXTRACTED"]
    if any(f["status"] == "DISPUTED" for f in facts):
        status, reason = "DISPUTED", "official_structured_identity_conflict"
    elif len({_digest(e) for e in extracted}) > 1:
        status, reason = "DISPUTED", "conflicting_official_identity_revisions"
    elif extracted:
        candidate = extracted[0]
        if any(candidate[side + "_name"] not in aliases[side] for side in ("home", "away")):
            status, reason = "DISPUTED", "official_teams_disagree_with_declared_provider_aliases"
        elif fixture.get("competition_name") != "欧国联":
            status, reason = "DISPUTED", "provider_competition_not_in_explicit_nations_league_mapping"
        elif parse_time(candidate["kickoff"]) != parse_time(fixture["kickoff"]):
            status, reason = "DISPUTED", "provider_and_official_kickoff_disagree"
        elif candidate["event_status"] in {"EventPostponed", "https://schema.org/EventPostponed"}:
            status, reason = "UNCONFIRMED", "official_event_postponed"
        elif candidate["event_status"] in {"EventCancelled", "https://schema.org/EventCancelled"}:
            status, reason = "UNCONFIRMED", "official_event_cancelled"
        else:
            available = max(parse_time(e["observed_at"]) for e in evidence)
            comparison = compare_assertion({k: fixture[k] for k in (
                "match_id", "competition_id", "season", "home_team_id", "away_team_id", "kickoff",
                "home_team_name", "away_team_name")},
                source_event_id=candidate["source_event_id"], explicitly_mapped_match_id=fixture["match_id"],
                fields={"home_team_id": fixture["home_team_id"], "away_team_id": fixture["away_team_id"],
                        "competition_id": fixture["competition_id"], "kickoff": candidate["kickoff"]},
                evidence_basis="acquired_structured", available_at=available,
                cutoff=min(as_of, parse_time(policy.model.cutoff)))
            status, reason = comparison["status"], comparison["reason"]
            if status == "CONFIRMED":
                canonical = candidate
    elif facts:
        reason = facts[0]["reason"]
    result = {"schema": VERSION, "status": status, "reason": reason,
        "match_id": fixture["match_id"], "initialization_manifest": deepcopy(initialization_manifest),
        "initialization_sha256": _digest(initialization_manifest), "canonical": canonical,
        "aliases": aliases, "observations": evidence, "verified_at": as_of.isoformat(),
        "comparison": comparison, "source_projections": facts,
        "mapping_basis": "exact_declared_cross_provider_aliases_not_fuzzy_inference",
        "independent_from_initial_quote_provider": True, "retroactive_actor_evidence": False,
        "settlement_status": "PENDING", "settlement_rows": []}
    result["binding_sha256"] = _digest(result)
    return result


def verify_identity_binding(binding, *, archive_root, as_of):
    """Recompute the exact binding from original bytes; do not trust its flags."""
    if binding.get('schema') == 'premier-league-fixture-identity-v1':
        from .premier_league_fixture import verify_premier_league_identity
        return verify_premier_league_identity(binding, archive_root=archive_root, as_of=as_of)
    if binding.get('schema') == 'official-fixture-evidence-identity-v1':
        from .official_fixture import verify_official_fixture
        return verify_official_fixture(binding, archive_root=archive_root, as_of=as_of)
    if binding.get("schema") == "finland-stage-fixture-identity-v1":
        from .finland_stage_fixture import verify_finland_stage_identity
        return verify_finland_stage_identity(binding, archive_root=archive_root, as_of=as_of)
    if binding.get("schema") == "finland-club-fixture-identity-v1":
        from .finland_fixture import verify_finland_identity
        return verify_finland_identity(binding, archive_root=archive_root, as_of=as_of)
    if binding.get("schema") == "canada-senior-fixture-identity-v1":
        from .canada_fixture import verify_canada_identity
        return verify_canada_identity(binding, archive_root=archive_root, as_of=as_of)
    if (binding.get("schema") != VERSION or parse_time(binding["verified_at"]) > parse_time(as_of)
            or binding.get("binding_sha256") != _digest({k: v for k, v in binding.items() if k != "binding_sha256"})):
        raise ValueError("Fixture identity binding is changed or not yet available")
    rebuilt = verify_fixture_identity(binding["initialization_manifest"], archive_root=archive_root,
        observation_ids=[e["observation_id"] for e in binding["observations"]], aliases=binding["aliases"],
        as_of=binding["verified_at"])
    if rebuilt != binding:
        raise ValueError("Fixture identity binding does not reproduce official raw evidence")
    return rebuilt


def _observation_reference(obs):
    return {"observation_id": obs.id, "raw_sha256": obs.content_hash,
            "url": obs.endpoint, "observed_at": obs.observed_at.isoformat()}


def _result_match_projection(obs, body, canonical, *, as_of):
    fact = _extract(obs, body, as_of=as_of)
    if fact["status"] != "EXTRACTED":
        return {"status": fact["status"], "reason": fact["reason"]}
    parsed = fact["canonical"]
    same_fields = ("home_id", "away_id", "home_name", "away_name", "competition", "category",
                   "stage", "season", "source_event_id")
    if any(parsed[key] != canonical[key] for key in same_fields):
        return {"status": "DISPUTED", "reason": "result_and_bound_fixture_identity_disagree"}
    native = parsed["source_event_id"].rsplit(":", 1)[-1]
    records = _match_objects(BeautifulSoup(body, "html.parser"), native)
    statuses = {record.get("status") for record in records}
    if len(statuses) != 1:
        return {"status": "DISPUTED", "reason": "conflicting_embedded_result_statuses"}
    status = next(iter(statuses))
    if status == "Postponed" or parsed["event_status"] in {"EventPostponed", "https://schema.org/EventPostponed"}:
        return {"status": "POSTPONED", "reason": "source_explicitly_postponed", "source_status": status}
    if status in {"Cancelled", "Canceled"} or parsed["event_status"] in {"EventCancelled", "https://schema.org/EventCancelled"}:
        return {"status": "CANCELLED", "reason": "source_explicitly_cancelled", "source_status": status}
    if parsed["kickoff"] != canonical["kickoff"]:
        return {"status": "DISPUTED", "reason": "result_and_original_kickoff_disagree"}
    if status in {"Suspended", "Abandoned", "Awarded", "Walkover"}:
        return {"status": "DISPUTED", "reason": "administrative_or_unfinished_result", "source_status": status}
    if status in {"Upcoming", "Scheduled", "Live", "InProgress", "Playing"}:
        return {"status": "PENDING", "reason": "source_has_no_final_result", "source_status": status}
    if status != "Finished":
        return {"status": "UNCONFIRMED", "reason": "unsupported_source_result_status", "source_status": status}
    final_times = {record.get("fullTimeAt") for record in records}
    if len(final_times) != 1 or None in final_times:
        return {"status": "UNCONFIRMED", "reason": "unambiguous_full_time_clock_missing"}
    ended = parse_time(next(iter(final_times)))
    kickoff = parse_time(canonical["kickoff"])
    # A numerical table alone could be live. This independent finished-state
    # gate must be observed after the source's final whistle and normal duration.
    if ended < kickoff or (ended-kickoff).total_seconds() < 90*60 or ended > obs.observed_at:
        return {"status": "DISPUTED", "reason": "finished_state_has_impossible_or_future_clock"}
    return {"status": "FINISHED", "reason": "raw_finished_status_and_final_whistle",
            "source_status": status, "full_time_at": ended.isoformat()}


def _schedule_projection(obs, body, canonical):
    if obs.http_status != 200 or obs.error_class:
        return {"status": "UNCONFIRMED", "reason": "official_schedule_response_unavailable"}
    if obs.source != "open_page" or len(body) > 2_000_000:
        raise ValueError("Expected a bounded acquired official schedule page")
    expected = canonical["source_event_id"].rsplit(":", 1)[-1]
    soup = BeautifulSoup(body, "html.parser")
    scores, scheduled = set(), False
    for anchor in soup.find_all("a", href=True):
        try:
            native = _url_id(anchor["href"], _MATCH)
        except (ValueError, TypeError):
            continue
        if native != expected:
            continue
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if text == canonical["home_name"] + " vs " + canonical["away_name"]:
            scheduled = True
            continue
        pattern = (re.escape(canonical["home_name"]) + r" ([0-9]{1,2})[-\u2013]([0-9]{1,2}) "
                   + re.escape(canonical["away_name"]))
        found = re.fullmatch(pattern, text)
        if not found:
            return {"status": "DISPUTED", "reason": "target_schedule_anchor_identity_or_score_ambiguous"}
        scores.add(tuple(int(x) for x in found.groups()))
    if len(scores) > 1 or (scores and scheduled):
        return {"status": "DISPUTED", "reason": "contradictory_target_schedule_entries"}
    if scores:
        home, away = next(iter(scores))
        return {"status": "SCORE", "home_goals": home, "away_goals": away,
                "reason": "exact_event_url_and_ordered_official_score_anchor"}
    return {"status": "PENDING", "reason": "target_schedule_score_missing"}


def verify_independent_settlement(binding, *, identity_archive_root, result_archive_root,
                                  observation_ids, as_of):
    """Re-read raw official identity, finish status and score before label export.

    Unrelated archived matches are ignored explicitly. An observation of a
    numeric live score cannot settle a match. Legacy HTML needs an actual
    Finished/fullTimeAt receipt and an ordered official score anchor. An
    explicitly versioned public API projection can provide both only with full
    identity, final-whistle and regular-period validation. The 2026/27
    league-phase contract is deliberately narrow.
    """
    if binding.get("schema") in {"finland-club-fixture-identity-v1", "finland-stage-fixture-identity-v1", "official-fixture-evidence-identity-v1"}:
        from .finland_results import verify_finland_settlement
        return verify_finland_settlement(binding, identity_archive_root=identity_archive_root,
            result_archive_root=result_archive_root, observation_ids=observation_ids, as_of=as_of)
    as_of = parse_time(as_of)
    verified = verify_identity_binding(binding, archive_root=identity_archive_root, as_of=as_of)
    result = {"schema": SETTLEMENT_VERSION, "match_id": binding["match_id"],
        "identity_binding_sha256": binding["binding_sha256"], "status": "UNCONFIRMED",
        "reason": "fixture_identity_not_confirmed", "observations": [], "projections": [],
        "ignored": [], "settlement_rows": [], "verified_at": as_of.isoformat(),
        "regulation_rule": deepcopy(REGULATION_RULE)}

    def finish(status, reason):
        result.update(status=status, reason=reason)
        result["report_sha256"] = _digest(result)
        return result

    if verified["status"] != "CONFIRMED":
        return finish("UNCONFIRMED", "fixture_identity_not_confirmed")
    canonical = verified["canonical"]
    if binding.get("schema") in {"canada-senior-fixture-identity-v1", "premier-league-fixture-identity-v1"}:
        # Independently confirming who/when does not establish a result's
        # period. The UEFA league-phase rule must not leak into a friendly.
        result["regulation_rule"] = None
        return finish("PENDING", "independent_regulation_settlement_not_yet_supported_for_source")
    if canonical["season"] != "2027":
        return finish("UNCONFIRMED", "regulation_rule_not_validated_for_source_season")
    if not observation_ids:
        return finish("PENDING", "independent_result_observations_missing")
    match_rows, score_rows = [], []
    for obs, body in _observations(result_archive_root, observation_ids):
        if obs.observed_at > as_of:
            raise ValueError("Result observation was unavailable at verification time")
        reference = _observation_reference(obs)
        requested = obs.context.get("requested_at")
        if requested is not None and parse_time(requested) > obs.observed_at:
            raise ValueError("Result receipt precedes its request")
        is_match = False
        try:
            is_match = "uefa:match:" + _url_id(obs.endpoint, _MATCH) == canonical["source_event_id"]
        except (ValueError, TypeError):
            pass
        if is_match:
            projection = _result_match_projection(obs, body, canonical, as_of=as_of)
            match_rows.append({**reference, **projection})
        elif obs.endpoint == SCHEDULE_URL:
            projection = _schedule_projection(obs, body, canonical)
            score_rows.append({**reference, **projection})
        elif ('uefa:match:' + str(official_api_event_id(obs.endpoint)) == canonical['source_event_id']):
            result['schema'] = API_SETTLEMENT_VERSION
            projection = project_official_api_result(obs, body, canonical, as_of=as_of)
            if projection['status'] == 'FINISHED_SCORE':
                match_rows.append({**reference, **projection, 'status':'FINISHED'})
                score_rows.append({**reference, **projection, 'status':'SCORE'})
            else:
                match_rows.append({**reference, **projection})
        else:
            result["ignored"].append({**reference, "reason": "not_target_event_or_supported_official_schedule"})
            continue
        result["observations"].append(reference)
        result["projections"].append({**reference, **projection})
    if any(row["status"] == "DISPUTED" for row in match_rows + score_rows):
        return finish("DISPUTED", "raw_target_identity_status_or_score_conflict")
    identified = [r for r in match_rows if r["status"] != "UNCONFIRMED"]
    if not identified:
        return finish("PENDING", "target_finished_status_not_observed")
    latest_time = max(parse_time(row["observed_at"]) for row in identified)
    latest = [row for row in identified if parse_time(row["observed_at"]) == latest_time]
    if len({row["status"] for row in latest}) != 1:
        return finish("DISPUTED", "simultaneous_target_status_conflict")
    final = latest[0]
    if final["status"] in {"POSTPONED", "CANCELLED"}:
        return finish(final["status"], final["reason"])
    if final["status"] != "FINISHED":
        if any(row["status"] == "FINISHED" for row in identified):
            return finish("DISPUTED", "finished_result_reverted_without_resolution")
        return finish("PENDING", "latest_official_target_status_not_finished")
    final_at = parse_time(final["full_time_at"])
    scores = [r for r in score_rows if r["status"] == "SCORE" and parse_time(r["observed_at"]) >= final_at]
    if not scores:
        return finish("UNCONFIRMED", "finished_target_has_no_independent_post_whistle_score")
    if len({(r["home_goals"], r["away_goals"]) for r in scores}) != 1:
        return finish("DISPUTED", "contradictory_official_result_revisions")
    score = min(scores, key=lambda r: (r["observed_at"], r["observation_id"]))
    used = sorted({r['observation_id']:r for r in (final, score)}.values(),
                  key=lambda r: r['observation_id'])
    receipt_at = max(parse_time(row["observed_at"]) for row in used)
    manifest = tuple("observation:" + r["observation_id"] + "|sha256:" + r["raw_sha256"]
                     + "|at:" + r["observed_at"] for r in used)
    fixture = verified["initialization_manifest"]["market_initialization"]["fixture"]
    match = ResearchMatch(match_id=fixture["match_id"], competition_id=fixture["competition_id"],
        season=fixture["season"], home_team_id=fixture["home_team_id"], away_team_id=fixture["away_team_id"],
        kickoff=parse_time(canonical["kickoff"]), home_goals=score["home_goals"], away_goals=score["away_goals"],
        result_available_at=as_of, fixture_available_at=parse_time(binding["verified_at"]),
        information_mode="operational", source="uefa_official_independent", input_manifest=manifest,
        score_period="regulation", status="finished", time_basis="raw_receipt_and_verification_clocks",
        competition_name=canonical["competition"], home_team_name=canonical["home_name"],
        away_team_name=canonical["away_name"], observed_at=receipt_at, ingested_at=as_of,
        operational_available_at=as_of)
    dataset = ResearchDataset((match,), (), result['schema'] + ":" + _digest({
        "binding": binding["binding_sha256"], "inputs": manifest, "rule": REGULATION_RULE}), manifest, {})
    rows, export_audit = export_results(dataset)
    if len(rows) != 1:
        result["export_audit"] = export_audit
        return finish("UNCONFIRMED", "settlement_export_rejected_clocks_or_semantics")
    rows[0]["identity_binding_sha256"] = binding["binding_sha256"]
    rows[0]["regulation_rule_version"] = REGULATION_RULE["version"]
    rows[0]["verification_schema"] = result['schema']
    result["settlement_rows"] = rows
    return finish("VERIFIED", "raw_independent_identity_finished_status_and_regulation_score_agree")


def main(argv=None):
    """Independent raw result capture, or network-free identity/result audit."""
    import argparse
    from .prospective_text import _save
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-identity", "verify-results", "capture-results"):
        sub = commands.add_parser(name)
        sub.add_argument("--identity-bundle", type=Path, required=True)
        sub.add_argument("--identity-archive", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "verify-results":
            sub.add_argument("--result-archive", type=Path, required=True)
        if name == "capture-results":
            sub.add_argument("--route", choices=("environment", "direct"), default="environment")
            sub.add_argument("--max-http-requests", type=int, default=4)
            sub.add_argument("--result-source", choices=("legacy-html", "official-api"), default="legacy-html")
    args = parser.parse_args(argv)
    bundle = _json(args.identity_bundle.read_text(encoding="utf-8"))
    bindings = bundle.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Identity bundle must contain a nonempty bindings map")
    as_of = datetime.now(timezone.utc)
    for key, binding in bindings.items():
        if key != binding.get("match_id"):
            raise ValueError("Bundle key and bound match disagree")
        verify_identity_binding(binding, archive_root=args.identity_archive, as_of=as_of)
    if args.output.exists():
        raise ValueError("Use a fresh immutable output path")
    if args.command == "verify-identity":
        _save(args.output, {"created_at": as_of.isoformat(), "bindings": bindings})
        return 0
    result_root = args.result_archive if args.command == "verify-results" else args.output
    if args.command == "capture-results":
        from football_agent.inquiry_transport import RoutedOpenWebTools
        if any(b["status"] != "CONFIRMED" for b in bindings.values()):
            raise ValueError("Capture requires confirmed fixture identities")
        official = args.result_source == 'official-api'
        cases = []
        if official:
            for mid, binding in sorted(bindings.items()):
                canonical = binding['canonical']
                event = re.fullmatch(r'uefa:match:([1-9][0-9]*)', canonical['source_event_id'])
                if not event:
                    raise ValueError('Official API capture requires an exact verified source event ID')
                url = 'https://match.uefa.com/v5/matches/' + event.group(1)
                eligible_at = parse_time(canonical['kickoff']) + timedelta(minutes=90)
                due = as_of >= eligible_at
                cases.append({'match_id':mid, 'source_event_id':canonical['source_event_id'],
                    'url':url, 'kickoff':canonical['kickoff'], 'eligible_at':eligible_at.isoformat(),
                    'status':'DUE' if due else 'NOT_DUE',
                    'reason':'minimum_regulation_duration_elapsed' if due else 'kickoff_plus_90_minutes_not_reached'})
            urls = sorted({c['url'] for c in cases if c['status']=='DUE'})
        else:
            urls = [SCHEDULE_URL] + sorted({b["canonical"]["source_url"] for b in bindings.values()})
        if not 1 <= args.max_http_requests <= 32 or len(urls) > args.max_http_requests:
            raise ValueError("Explicit HTTP budget is smaller than the capture plan")
        args.output.mkdir(parents=True, exist_ok=False)
        capture_plan = {"schema": "independent-results-capture-v1",
            "created_at": as_of.isoformat(), "urls": urls, "route": args.route,
            "http_requests_planned": len(urls), "max_http_requests": args.max_http_requests,
            "automatic_retries": 0, "identity_bindings": {k: b["binding_sha256"] for k, b in bindings.items()},
            "purpose": "independent_future_result_verification_not_actor_evidence"}
        if official:
            capture_plan.update(schema='independent-results-capture-v2', result_source='official-api',
                cases=cases, requests_per_due_event=1, polling=False,
                eligibility_is_not_final_status=True)
        _save(args.output / "capture-plan.json", capture_plan)
        tools = RoutedOpenWebTools(RawArchive(args.output),
            ['match.uefa.com'] if official else ["www.uefa.com"],
            route=args.route, extraction_version=3) if urls else None
        receipts = []
        for index, url in enumerate(urls):
            prepared = tools.prepare({"tool": "read_url", "url": url}, {"task": {"seed_urls": urls}, "documents": []})
            started_at = datetime.now(timezone.utc)
            returned = tools.execute(prepared, run_id="independent-settlement", request_id=f"result-{index}")
            _save(args.output / f"response-{index}.json", {"prepared": prepared, **returned})
            if official:
                ended_at = datetime.now(timezone.utc)
                observation = returned['observation']
                requested_at = parse_time(observation['context']['requested_at'])
                observed_at = parse_time(observation['observed_at'])
                if (type(returned.get('http_attempts')) is not int or returned['http_attempts'] != 1 or
                        observation['endpoint'] != url or
                        not started_at <= requested_at <= observed_at <= ended_at):
                    raise ValueError('Official API capture request count, identity or actual receipt clock mismatch')
                receipts.append({'url':url,'observation_id':observation['id'],
                    'raw_sha256':observation['content_hash'],'http_attempts':1,
                    'requested_at':requested_at.isoformat(),'observed_at':observed_at.isoformat()})
        if official:
            if len(RawArchive(args.output).observations()) != len(urls):
                raise ValueError('Official API capture did not archive exactly one response per due event')
            _save(args.output/'capture-receipts.json', {'schema':'independent-results-capture-v2',
                'completed_at':datetime.now(timezone.utc).isoformat(), 'http_requests_actual':len(receipts),
                'max_http_requests':args.max_http_requests,'receipts':receipts})
        as_of = datetime.now(timezone.utc)
    ids = [obs.id for obs in RawArchive(result_root).observations()]
    reports = {mid: verify_independent_settlement(binding, identity_archive_root=args.identity_archive,
        result_archive_root=result_root, observation_ids=ids, as_of=as_of) for mid, binding in bindings.items()}
    _save(args.output / "verification.json" if args.command == "capture-results" else args.output,
        {"schema": API_SETTLEMENT_VERSION if any(r['schema'] == API_SETTLEMENT_VERSION for r in reports.values())
                   else SETTLEMENT_VERSION, "created_at": as_of.isoformat(), "reports": reports,
         "settlement_rows": [row for report in reports.values() for row in report["settlement_rows"]]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
