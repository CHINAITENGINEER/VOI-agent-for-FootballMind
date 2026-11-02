"""Bounded prospective fixture inventory over the existing spftll collector.

This module is deliberately small.  It does not implement HTTP, retries,
quotas, or another response archive.  ``football_spider.collectors.sources``
performs the bounded list request and the existing ``RawArchive`` stores every
attempt, including failures.  This adapter projects future fixtures from
those archived responses, chooses one league under a frozen rule, and writes
an auditable run report.

The projection never fills ``available_at``: the collector knows when it saw a
response, but it does not prove when an Agent could use the fixture.  That
timestamp therefore remains null with an explicit unknown status.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4
from zoneinfo import ZoneInfo

from football_spider.collectors.sources import collect
from football_spider.models.contracts import CollectionResult, Observation
from football_spider.normalization.sources import normalize_spftll
from football_spider.platform_config import PlatformSettings
from football_spider.storage.raw_archive import RawArchive, atomic_write, json_bytes


SCHEMA = "prospective-fixture-inventory-v2"
COMPETITION_ALIASES = frozenset({"Premier League", "English Premier League", "英超", "英格兰超级联赛", "PL"})
# The inventory is deliberately two-stage: inspect the next three weeks first
# and extend to six weeks only when the first window cannot form the minimum
# prospective sample.  ``horizon_days`` remains the public argument for
# backwards compatibility and now means the primary (21 day) window.
DEFAULT_HORIZON_DAYS = 21
MAX_HORIZON_DAYS = 42
MIN_FIXTURES = 3
PREFERRED_FIXTURES = 6
MAX_REQUESTS = 6
MAX_DETAIL_MATCHES = 1


def _utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return parsed.astimezone(timezone.utc)


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _observation_summaries(observations: Iterable[Observation]) -> list[dict[str, Any]]:
    return [
        {
            "observation_id": obs.id,
            "source": obs.source,
            "endpoint": obs.endpoint,
            "request_id": obs.request_id,
            "observed_at": obs.observed_at.isoformat(),
            "http_status": obs.http_status,
            "error_class": obs.error_class,
            "payload_path": obs.payload_path,
            "raw_sha256": obs.content_hash,
            "payload_format": obs.payload_format,
            "context": obs.context,
        }
        for obs in sorted(observations, key=lambda item: (item.observed_at, item.id))
    ]


def _is_premier_league(name: str | None) -> bool:
    return str(name or "").strip().casefold() in {name.casefold() for name in COMPETITION_ALIASES}


def _competition_key(name: str | None) -> str:
    """Return a stable display key without pretending aliases are identities."""
    value = str(name or "").strip()
    return "Premier League" if _is_premier_league(value) else (value or "unknown")


def _fixture_row(
    record: Any,
    observation: Observation,
) -> dict[str, Any]:
    """Project one normalized MatchRecord into the canonical fixture shape."""
    registered_at = observation.observed_at
    return {
        "fixture_id": f"spftll:match:{record.source_id}",
        "physical_match_id": f"spftll:match:{record.source_id}",
        "source_match_id": record.source_id,
        "competition": record.competition.name,
        "competition_key": _competition_key(record.competition.name),
        "competition_source_id": record.competition.source_id,
        "competition_identity_quality": record.competition.id_quality,
        "season": record.season,
        "season_status": "source_not_supplied" if record.season is None else "source_supplied",
        "home_team": record.home.name,
        "away_team": record.away.name,
        "home_team_source_id": record.home.source_id,
        "away_team_source_id": record.away.source_id,
        "home_team_identity_quality": record.home.id_quality,
        "away_team_identity_quality": record.away.id_quality,
        "kickoff_at": record.kickoff.isoformat(),
        "kickoff_timezone": "UTC",
        "source_timezone": record.source_timezone,
        "original_time_text": record.original_time,
        "registered_at": registered_at.isoformat(),
        "fixture_registered_at": registered_at.isoformat(),
        "registration_source": "spftll",
        "registration_observation_id": observation.id,
        "source_observed_at": observation.observed_at.isoformat(),
        "source_published_at": None,
        "source_modified_at": None,
        "captured_at": observation.observed_at.isoformat(),
        "available_at": None,
        "availability_status": "unknown",
        "availability_basis": "collector_observation_does_not_prove_agent_availability",
        "timing_basis": "spftll_matchTime_parsed_as_Asia/Shanghai;first_current_list_observation;publication_and_agent_availability_unknown",
        "information_mode": "prospective_capture",
        "pre_kickoff_frozen": True,
        "status": record.status,
        "score_period": record.score_period,
        "identity_source_url": observation.endpoint,
        "raw_path": observation.payload_path,
        "raw_sha256": observation.content_hash,
        "fixture_binding_basis": "normalized_spftll_source_id_home_away_competition_kickoff",
        "archive_snapshot_at": None,
    }


def _bind_competition_keys(rows: list[dict[str, Any]]) -> None:
    """Bind aliases only by observed names and the source's stable league ID."""
    names_by_source: dict[str, set[str]] = {}
    for row in rows:
        source_id = row.get("competition_source_id")
        if source_id and row.get("competition_identity_quality") == "stable":
            names_by_source.setdefault(source_id, set()).add(str(row.get("competition") or "").strip())
    for row in rows:
        source_id = row.get("competition_source_id")
        names = names_by_source.get(source_id, set())
        if _is_premier_league(row.get("competition")) or any(_is_premier_league(name) for name in names):
            row["competition_key"] = "Premier League"
            row["competition_binding_basis"] = "observed_premier_league_alias_or_shared_stable_source_id"
        elif names:
            row["competition_key"] = sorted(name for name in names if name)[0] if any(names) else "unknown"
            row["competition_binding_basis"] = "shared_stable_source_competition_id"
        else:
            row["competition_key"] = _competition_key(row.get("competition"))
            row["competition_binding_basis"] = "source_name_only"


def project_future_matches(
    archive: RawArchive,
    observations: Iterable[Observation],
    *,
    window_start: datetime,
    window_end: datetime,
    competition_filter: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project scheduled future matches from verified spftll observations.

    ``competition_filter`` is optional for the dynamic inventory.  The legacy
    ``project_future_fixtures`` wrapper below still defaults to Premier League
    so existing callers retain their original contract.
    """
    start, end = _utc(window_start), _utc(window_end)
    allowed = {_competition_key(value) for value in competition_filter} if competition_filter is not None else None
    first_seen: dict[str, tuple[Any, Observation]] = {}
    projection_errors: list[dict[str, str]] = []
    normalized_rows = 0
    successful_observations = 0
    quote_selections: dict[str, set[str]] = {}
    observed_schedules: dict[str, set[str]] = {}
    for observation in sorted(observations, key=lambda item: (item.observed_at, item.id)):
        if observation.source != "spftll" or observation.http_status is None or not 200 <= observation.http_status < 300:
            continue
        if observation.error_class:
            continue
        try:
            payload = json.loads(archive.read_body(observation))
            batch = normalize_spftll(observation, payload)
            successful_observations += 1
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            projection_errors.append({"observation_id": observation.id, "reason": type(exc).__name__})
            continue
        for record in batch.matches:
            normalized_rows += 1
            if (
                record.status != "scheduled"
                or not start <= record.kickoff <= end
            ):
                continue
            observed_schedules.setdefault(record.source_id, set()).add(record.kickoff.isoformat())
            current = first_seen.get(record.source_id)
            if current is None or observation.observed_at < current[1].observed_at:
                first_seen[record.source_id] = (record, observation)
        for quote in batch.quotes:
            if quote.market_type == "1x2" and quote.period == "regulation":
                quote_selections.setdefault(quote.source_match_id, set()).add(quote.selection)

    fixtures: list[dict[str, Any]] = []
    for source_id, (record, observation) in sorted(
        first_seen.items(), key=lambda item: (item[1][0].kickoff, item[0])
    ):
        if not observation.observed_at < record.kickoff:
            projection_errors.append({"match_id": source_id, "reason": "registration_not_before_kickoff"})
            continue
        row = _fixture_row(record, observation)
        row["inventory_quality"] = {
            "complete_1x2_observed": quote_selections.get(source_id, set()) >= {"home", "draw", "away"},
            "schedule_conflict_observed": len(observed_schedules.get(source_id, set())) > 1,
            "schedule_stability": "conflicting_observations" if len(observed_schedules.get(source_id, set())) > 1 else "not_established",
        }
        fixtures.append(row)
    _bind_competition_keys(fixtures)
    if allowed is not None:
        fixtures = [row for row in fixtures if row["competition_key"] in allowed]
    counts: dict[str, int] = {}
    for fixture in fixtures:
        key = fixture["competition_key"]
        counts[key] = counts.get(key, 0) + 1
    return fixtures, {
        "normalized_rows_seen": normalized_rows,
        "successful_observations_projected": successful_observations,
        "future_fixture_count": len(fixtures),
        "future_premier_league_fixtures": sum(_is_premier_league(row["competition"]) for row in fixtures),
        "competition_counts": dict(sorted(counts.items())),
        "projection_errors": projection_errors,
    }


def project_future_fixtures(
    archive: RawArchive,
    observations: Iterable[Observation],
    *,
    window_start: datetime,
    window_end: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project future PL rows from verified archived spftll observations.

    ``window_start`` is the real collection cutoff.  A fixture is registered
    from the first successful observation that contains its source ID.  Later
    duplicate/detail observations are retained in RawArchive but do not move
    the registration timestamp backwards or invent Agent availability.
    """
    fixtures, audit = project_future_matches(
        archive,
        observations,
        window_start=window_start,
        window_end=window_end,
        competition_filter=COMPETITION_ALIASES,
    )
    # Preserve the v1 audit key expected by existing readers/tests.
    audit["future_premier_league_fixtures"] = len(fixtures)
    return fixtures, audit


def select_competition_fixtures(
    fixtures: Iterable[dict[str, Any]],
    *,
    preferred_count: int = PREFERRED_FIXTURES,
    minimum_count: int = MIN_FIXTURES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select PL first, then at most one same-source fallback league.

    The selection is deterministic and only uses pre-kickoff fixture identity
    and completeness fields.  It never looks at scores or outcomes.
    """
    rows = list(fixtures)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("competition_key") or _competition_key(row.get("competition"))
        groups.setdefault(key, []).append(row)
    pl = sorted(groups.get("Premier League", []), key=lambda row: (row.get("kickoff_at", ""), row.get("source_match_id", "")))
    common = {
        "preferred_count": preferred_count,
        "minimum_count": minimum_count,
        "candidate_counts": {key: len(value) for key, value in sorted(groups.items())},
        "premier_league_candidates_retained": [row["physical_match_id"] for row in pl],
        "selection_uses_outcomes": False,
    }
    if len(pl) >= preferred_count:
        return pl, {**common, "selected_competition": "Premier League", "fallback_reason": None,
                    "minimum_reached": len(pl) >= minimum_count}

    fallback_groups = []
    for key, values in groups.items():
        if key in {"Premier League", "unknown"}:
            continue
        ordered = sorted(values, key=lambda row: (row.get("kickoff_at", ""), row.get("source_match_id", "")))
        complete = sum(bool(row.get("source_match_id") and row.get("home_team") and row.get("away_team")
                            and row.get("kickoff_at") and row.get("competition_source_id")) for row in ordered)
        full_odds = sum(row.get("inventory_quality", {}).get("complete_1x2_observed") is True for row in ordered)
        schedule_conflicts = sum(row.get("inventory_quality", {}).get("schedule_conflict_observed") is True for row in ordered)
        fallback_groups.append((len(ordered), complete, full_odds, schedule_conflicts, key, ordered))
    fallback_groups.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    chosen = fallback_groups[0] if fallback_groups else None
    if chosen:
        selected = chosen[5]
        return selected, {
            **common,
            "selected_competition": chosen[4],
            "fallback_competition": chosen[4],
            "fallback_reason": "premier_league_below_preferred_threshold_single_league_fallback",
            "fallback_scores": [{"competition": key, "count": count, "identity_complete": complete,
                                 "complete_1x2_observed": odds, "schedule_conflicts_observed": conflicts}
                                for count, complete, odds, conflicts, key, _ in fallback_groups],
            "minimum_reached": len(selected) >= minimum_count,
        }
    return pl, {**common, "selected_competition": "Premier League" if pl else None,
                 "fallback_reason": "no_non_premier_league_single_league_candidate" if not pl else "premier_league_below_preferred_threshold_no_fallback",
                 "minimum_reached": len(pl) >= minimum_count}


def _run_collection(config: PlatformSettings, *, start: date, end: date, match_ids: list[str] | None = None):
    """Call the existing source collector; this is the only network seam."""
    return collect(
        "spftll",
        config,
        RawArchive(config.data_dir),
        start=start,
        end=end,
        match_ids=match_ids,
    )


def project_inventory_windows(
    archive: RawArchive,
    observations: Iterable[Observation],
    *,
    window_start: datetime,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    max_horizon_days: int = MAX_HORIZON_DAYS,
    minimum_fixtures: int = MIN_FIXTURES,
    preferred_fixtures: int = PREFERRED_FIXTURES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply both bounded windows to one real list response without HTTP.

    spftll's current-list endpoint is not a date-range query. Extending the
    selection window cannot force it to return a longer schedule. Reusing the
    same archived response avoids pretending a repeated request is a future
    schedule endpoint, and lets callers audit an already collected inventory.
    """
    if type(horizon_days) is not int or not 1 <= horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError(f"horizon_days must be between 1 and {MAX_HORIZON_DAYS}")
    if type(max_horizon_days) is not int or not horizon_days <= max_horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError(f"max_horizon_days must be between horizon_days and {MAX_HORIZON_DAYS}")
    start = _utc(window_start)
    observations = list(observations)
    primary_end = start + timedelta(days=horizon_days)
    candidates, primary_projection = project_future_matches(
        archive, observations, window_start=start, window_end=primary_end,
    )
    selected, selection = select_competition_fixtures(
        candidates, minimum_count=minimum_fixtures, preferred_count=preferred_fixtures,
    )
    expanded = len(selected) < minimum_fixtures and max_horizon_days > horizon_days
    end = primary_end
    projection = primary_projection
    if expanded:
        end = start + timedelta(days=max_horizon_days)
        candidates, projection = project_future_matches(
            archive, observations, window_start=start, window_end=end,
        )
        selected, selection = select_competition_fixtures(
            candidates, minimum_count=minimum_fixtures, preferred_count=preferred_fixtures,
        )
    return selected, candidates, {
        "selection": selection,
        "projection": projection,
        "primary_projection": primary_projection,
        "expanded_to_maximum": expanded,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "primary_window_end": primary_end.isoformat(),
        "window_extension_basis": "filter_same_current_list_response_not_date_lookup",
    }


def collect_inventory(
    output: str | Path,
    *,
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    max_horizon_days: int = MAX_HORIZON_DAYS,
    minimum_fixtures: int = MIN_FIXTURES,
    preferred_fixtures: int = PREFERRED_FIXTURES,
) -> dict[str, Any]:
    """Run a dynamic 21-day then (if needed) 42-day inventory.

    The current-list request is filtered to the primary window. Extension is
    a broader filter of the same raw response when single-league selection
    cannot produce ``minimum_fixtures``; spftll has no future-date lookup.
    """
    if type(horizon_days) is not int or not 1 <= horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError(f"horizon_days must be between 1 and {MAX_HORIZON_DAYS}")
    if type(max_horizon_days) is not int or not horizon_days <= max_horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError(f"max_horizon_days must be between horizon_days and {MAX_HORIZON_DAYS}")
    if type(minimum_fixtures) is not int or minimum_fixtures < 1:
        raise ValueError("minimum_fixtures must be a positive integer")
    if type(preferred_fixtures) is not int or preferred_fixtures < minimum_fixtures:
        raise ValueError("preferred_fixtures must be >= minimum_fixtures")
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Inventory output directory is non-empty; choose a new immutable run directory")
    root.mkdir(parents=True, exist_ok=True)
    run_started = _utc(now or datetime.now(timezone.utc))
    local_start = run_started.astimezone(ZoneInfo("Asia/Shanghai")).date()
    primary_end = run_started + timedelta(days=horizon_days)
    local_primary_end = (run_started.astimezone(ZoneInfo("Asia/Shanghai")) + timedelta(days=horizon_days)).date()
    config = PlatformSettings(
        data_dir=root,
        timezone="Asia/Shanghai",
        sources=["spftll"],
        max_requests=MAX_REQUESTS,
        detail_limit=0,
    )
    archive = RawArchive(root)
    # Freeze the scope before the only network seam. This is a run plan, not
    # a second request journal or a generated Agent candidate menu.
    atomic_write(root / "selection-plan.json", json_bytes({
        "schema": SCHEMA,
        "frozen_at": run_started.isoformat(),
        "source": "spftll",
        "window_start": run_started.isoformat(),
        "primary_window_end": primary_end.isoformat(),
        "maximum_window_end": (run_started + timedelta(days=max_horizon_days)).isoformat(),
        "preferred_count": preferred_fixtures,
        "minimum_count": minimum_fixtures,
        "preferred_competition_aliases": sorted(COMPETITION_ALIASES),
        "fallback": "single_competition_by_count_identity_odds_then_schedule_conflicts",
        "max_requests": MAX_REQUESTS,
        "scope_does_not_use_results": True,
    }))
    source_runs: list[dict[str, Any]] = []
    first_error: Exception | None = None
    try:
        first = _run_collection(config, start=local_start, end=local_primary_end)
    except Exception as exc:  # Preserve a report even for an adapter failure.
        first_error = exc
        first = CollectionResult(source="spftll", status="FAILED", errors=[{"category": type(exc).__name__}])
    first_json = first.model_dump(mode="json")
    source_runs.append({"phase": "current_list_21d", "result": first_json})

    first_observations = [obs for obs in archive.observations("spftll") if obs.id in set(first.observations)]
    fixtures, all_candidates, window_projection = project_inventory_windows(
        archive,
        first_observations,
        window_start=run_started,
        horizon_days=horizon_days,
        max_horizon_days=max_horizon_days,
        minimum_fixtures=minimum_fixtures,
        preferred_fixtures=preferred_fixtures,
    )
    selection = window_projection["selection"]
    projection = window_projection["projection"]
    first_projection = window_projection["primary_projection"]
    expanded = window_projection["expanded_to_maximum"]
    max_end = _utc(window_projection["window_end"])
    local_max_end = max_end.astimezone(ZoneInfo("Asia/Shanghai")).date()

    # The list phase intentionally uses no detail limit. If a selected fixture
    # exists, spend at most two remaining requests on its detail and odds; a
    # failed/list retry remains in RawArchive and is included in the report.
    all_list_attempts = len(first.observations) + sum(len(run["result"].get("observations", []))
                                                      for run in source_runs[1:])
    remaining = max(0, MAX_REQUESTS - all_list_attempts)
    detail_target = fixtures[0]["source_match_id"] if fixtures and remaining else None
    if detail_target is not None:
        detail_config = config.model_copy(update={"max_requests": min(2, remaining), "detail_limit": MAX_DETAIL_MATCHES})
        try:
            detail = _run_collection(detail_config, start=local_start, end=local_max_end, match_ids=[detail_target])
        except Exception as exc:
            detail = CollectionResult(source="spftll", status="FAILED", errors=[{"category": type(exc).__name__}])
        source_runs.append({"phase": "optional_detail_and_odds", "target_match_id": detail_target,
                            "result": detail.model_dump(mode="json")})

    all_observations = archive.observations("spftll")
    # Keep the registration projection from the list phase.  Optional detail
    # responses remain in RawArchive and are enumerated in the report.
    atomic_write(root / "fixtures.jsonl", _jsonl(fixtures))
    archive_summary = _observation_summaries(all_observations)
    collector_errors = [error for run in source_runs for error in run["result"].get("errors", [])]
    if fixtures:
        shortage_reason = "selected_competition_below_preferred_six_fixtures" if len(fixtures) < preferred_fixtures else None
    elif first_error is not None:
        shortage_reason = "collector_exception"
    elif collector_errors:
        shortage_reason = "spftll_list_failed_or_empty_before_future_premier_league_projection"
    else:
        shortage_reason = "no_future_spftll_single_league_sample_within_42_days"
    report = {
        "schema": SCHEMA,
        "run_id": str(uuid4()),
        "run_started_at": run_started.isoformat(),
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "competition_scope": {"preferred_league": "Premier League", "aliases": sorted(COMPETITION_ALIASES),
                              "source": "spftll", "selection_rule": "Premier League then one same-source single-league fallback",
                              "preferred_fixture_threshold": preferred_fixtures, "minimum_fixture_threshold": minimum_fixtures,
                              "selected_competition": selection.get("selected_competition"),
                              "fallback_competition": selection.get("fallback_competition"),
                              "fallback_reason": selection.get("fallback_reason")},
        "window": {"start_utc": run_started.isoformat(), "end_utc": max_end.isoformat(),
                    "primary_end_utc": primary_end.isoformat(), "start_local_date": str(local_start),
                    "primary_end_local_date": str(local_primary_end), "end_local_date": str(local_max_end),
                    "primary_horizon_days": horizon_days, "maximum_horizon_days": max_horizon_days,
                    "expanded_to_maximum": expanded, "timezone": "Asia/Shanghai"},
        "request_budget": {"max_requests": MAX_REQUESTS, "detail_limit": MAX_DETAIL_MATCHES,
                           "actual_archived_attempts": len(all_observations),
                           "remaining_after_list": remaining},
        "configuration": config.public(),
        "available_prospective_fixtures": len(fixtures),
        "fixtures_discovered": len(all_candidates),
        "fixtures_discovered_primary_window": first_projection["future_fixture_count"],
        "fixtures_registered": len(fixtures),
        "candidate_registrations": all_candidates,
        "selection": selection,
        "shortage_reason": shortage_reason,
        "fixtures_before_kickoff": sum(_utc(row["registered_at"]) < _utc(row["kickoff_at"]) for row in fixtures),
        "collector_runs": source_runs,
        "observations": archive_summary,
        "projection": projection,
        "primary_projection": first_projection,
        "window_extension_basis": window_projection["window_extension_basis"],
        "raw_archive_root": str((root / "raw").resolve()),
        "available_at_policy": "null_when_agent_availability_is_not_measured",
        "no_agent_menu_or_action_claim": True,
    }
    atomic_write(root / "run-report.json", json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--max-horizon-days", type=int, default=MAX_HORIZON_DAYS)
    parser.add_argument("--minimum-fixtures", type=int, default=MIN_FIXTURES)
    parser.add_argument("--preferred-fixtures", type=int, default=PREFERRED_FIXTURES)
    args = parser.parse_args(argv)
    report = collect_inventory(args.output, horizon_days=args.horizon_days,
                               max_horizon_days=args.max_horizon_days,
                               minimum_fixtures=args.minimum_fixtures,
                               preferred_fixtures=args.preferred_fixtures)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
