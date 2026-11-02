"""Offline acceptance audit for FootballMind's autonomous information mechanism.

This module is intentionally a mechanism audit, not a football-result
evaluator.  It checks that the actor receives only as-of information, that
contextual facts are not silently turned into conclusions, and that source
quality/staleness is handled conservatively.  The scenario package's
``expected_policy`` remains evaluator-side material; it is never passed to an
actor by this module.

The audit is deterministic and requires no network, provider, outcome label,
or empirical ``g`` calibration.  A successful report therefore means that the
research prototype preserves the information-selection contract, not that it
predicts match results accurately.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .autonomous_match_contract import (
    STATE_VERSION,
    bind_source_capabilities,
    build_match_bundle,
)
from .football_context_routing import replay_cases
from .scenario_fixture_pack import (
    ACTOR_FORBIDDEN_KEYS,
    PUBLIC_CANDIDATE_KEYS,
    actor_projection,
    generate_scenario_fixtures,
    validate_actor_projection,
    validate_scenario_fixture,
)


VERSION = "autonomous-mechanism-acceptance-v1"
_VALUE_OR_ORDER_KEYS = {
    "voi", "value", "priority", "rank", "score", "weight", "utility",
    "expected_value", "conditional_value", "selection_probability",
}
_RESULT_KEYS = {
    "result", "results", "score", "scores", "home_score", "away_score",
    "outcome", "winner", "actual_result", "future_result", "gold",
}
_QUALITY_STATES = {"fresh", "stale", "missing", "duplicate", "conflict", "blocked"}


def _check(name: str, passed: bool, *, details: Any = None) -> dict[str, Any]:
    row = {"passed": bool(passed)}
    if details is not None:
        row["details"] = details
    return {name: row}


def _walk_keys(value: Any, path: str = "root") -> list[str]:
    """Return paths for every mapping key, preserving case-insensitive checks."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            text = str(key)
            found.append(f"{path}/{text}")
            found.extend(_walk_keys(child, f"{path}/{text}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, f"{path}/{index}"))
    return found


def _fixture_structural_checks(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one complete fixture and its strict actor projection."""
    checks: dict[str, Any] = {}
    fixture_id = str(fixture.get("fixture_id", "unknown"))
    validation = validate_scenario_fixture(fixture)
    checks.update(_check("fixture_valid", validation.get("valid", False),
                         details=validation.get("reasons", [])))

    actor = None
    if validation.get("valid"):
        try:
            actor = actor_projection(fixture)
            actor_validation = validate_actor_projection(actor)
            checks.update(_check("actor_projection_valid", actor_validation.get("valid", False),
                                 details=actor_validation.get("reasons", [])))
        except Exception as exc:  # pragma: no cover - defensive audit boundary
            checks.update(_check("actor_projection_valid", False, details=str(exc)))
    else:
        checks.update(_check("actor_projection_valid", False,
                             details="fixture invalid; actor projection skipped"))

    # The actor must not receive evaluator labels, outcome labels, or policy
    # expectations.  This is checked recursively because future fixture
    # extensions may nest private material.
    actor_key_paths = _walk_keys(actor) if actor is not None else []
    actor_private = [path for path in actor_key_paths
                     if path.rsplit("/", 1)[-1].casefold() in
                     {str(item).casefold() for item in ACTOR_FORBIDDEN_KEYS}]
    checks.update(_check("actor_has_no_evaluator_or_result_fields", not actor_private,
                         details=actor_private))

    candidates = actor.get("candidate_menu", []) if isinstance(actor, Mapping) else []
    candidate_private: list[str] = []
    candidate_value_fields: list[str] = []
    candidate_ids: set[str] = set()
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, Mapping):
            candidate_private.append("non_object_candidate")
            continue
        candidate_id = str(candidate.get("candidate_id", "?"))
        candidate_ids.add(candidate_id)
        extra = set(candidate) - set(PUBLIC_CANDIDATE_KEYS)
        candidate_private.extend(f"{candidate_id}:{item}" for item in sorted(extra))
        for key in candidate:
            if str(key).casefold() in _VALUE_OR_ORDER_KEYS:
                # declared_cost is allowed; it is a resource contract rather
                # than an estimate of information value.
                if str(key).casefold() != "declared_cost":
                    candidate_value_fields.append(f"{candidate_id}:{key}")
    checks.update(_check("actor_candidate_menu_has_no_private_fields", not candidate_private,
                         details=candidate_private))
    checks.update(_check("actor_candidate_menu_has_no_fixed_priority_or_value",
                         not candidate_value_fields, details=candidate_value_fields))

    family = fixture.get("scenario_family")
    snapshot = fixture.get("public_snapshot", {})
    known = snapshot.get("known_information", {}) if isinstance(snapshot, Mapping) else {}
    candidate_by_id = {item.get("candidate_id"): item for item in candidates
                       if isinstance(item, Mapping)}

    if family == "qualification_signal_low_value":
        # Qualification is a fact; lineup/rotation remains a separate query.
        passed = bool(known.get("home_already_qualified")) and (
            "rotation_check" in candidate_by_id and
            "rotation_check" in {item.get("candidate_id") for item in fixture["candidate_menu"]}
        )
        no_inferred_rotation = not any(
            str(key).casefold() in {"rotation", "rotation_likely", "reserve_strength", "tactical_intent"}
            for key in known
        )
        checks.update(_check("qualification_does_not_infer_rotation_or_tactics",
                             passed and no_inferred_rotation,
                             details={"family": family, "known_keys": sorted(known)}))
    elif family == "competition_constraint_high_value":
        candidate = candidate_by_id.get("tactical_intent", {})
        requires = candidate.get("requires", []) if isinstance(candidate, Mapping) else []
        checks.update(_check("draw_constraint_precedes_tactical_query",
                             "qualification_constraint" in candidate_by_id and
                             "tactical_intent" in candidate_by_id and
                             requires == ["qualification_constraint"],
                             details={"requires": requires}))
    elif family == "injury_high_value":
        injury = candidate_by_id.get("key_forward_availability", {})
        checks.update(_check("injury_candidate_is_explicit_and_actionable",
                             isinstance(injury, Mapping) and
                             injury.get("source_state") == "fresh" and
                             bool(injury.get("source_ids")),
                             details={"candidate": injury}))
    elif family == "weather_low_value":
        weather = candidate_by_id.get("weather_detail", {})
        checks.update(_check("normal_weather_remains_a_candidate_without_forced_query",
                             isinstance(weather, Mapping) and
                             weather.get("source_state") == "fresh" and
                             "weather_detail" in fixture.get("expected_policy", {}).get("low_value_candidates", []),
                             details={"candidate": weather}))

    # For every non-fresh source state, the actor must retain that state.  It
    # cannot be silently rewritten as an absent negative fact or as fresh data.
    quality_violations: list[str] = []
    for candidate in candidates if isinstance(candidates, list) else []:
        if not isinstance(candidate, Mapping):
            continue
        source_state = candidate.get("source_state")
        if source_state in _QUALITY_STATES - {"fresh"}:
            if candidate.get("source_state") != fixture_candidate_state(fixture, candidate.get("candidate_id")):
                quality_violations.append(str(candidate.get("candidate_id")))
    checks.update(_check("source_quality_state_is_preserved", not quality_violations,
                         details=quality_violations))

    # Source conflict/staleness/missing/duplicate scenarios must expose an
    # independent or fresh route in the evaluator expectation.  This check is
    # deliberately qualitative; it does not convert expectations into a gold
    # score or a numeric VoI value.
    if family in {"market_news_conflict", "source_staleness", "source_missing", "source_duplicate"}:
        expected = fixture.get("expected_policy", {})
        first = expected.get("expected_first_selection", []) if isinstance(expected, Mapping) else []
        checks.update(_check("degraded_source_has_conservative_alternative",
                             bool(first) and all(item in candidate_ids for item in first),
                             details={"family": family, "first": first}))

    return {"fixture_id": fixture_id, "family": family, "checks": checks}


def fixture_candidate_state(fixture: Mapping[str, Any], candidate_id: Any) -> Any:
    """Read a complete fixture candidate's quality state without actor leakage."""
    for candidate in fixture.get("candidate_menu", []) if isinstance(fixture, Mapping) else []:
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id:
            return candidate.get("source_state")
    return None


def _context_routing_audit() -> dict[str, Any]:
    replay = replay_cases()
    checks = replay.get("checks", {}) if isinstance(replay, Mapping) else {}
    expected = {
        "qualified_case_queries_reserve_context",
        "qualified_case_lineup_waits_for_unknowns",
        "draw_case_queries_tactical_context",
        "unknown_case_queries_qualification",
        "qualification_does_not_infer_rotation",
        "qualification_does_not_infer_tactics",
        "all_contexts_are_pre_match",
        "hidden_result_flag_false",
    }
    missing = sorted(expected - set(checks))
    failed = sorted(name for name in expected if checks.get(name) is not True)
    return {"checks": {
        "context_routing_cases_pass": _check("context_routing_cases_pass",
                                              not missing and not failed,
                                              details={"missing": missing, "failed": failed}),
        "context_routing_has_no_hidden_result": _check(
            "context_routing_has_no_hidden_result",
            checks.get("hidden_result_flag_false") is True and checks.get("all_contexts_are_pre_match") is True,
            details={"hidden_result_flag_false": checks.get("hidden_result_flag_false"),
                     "all_contexts_are_pre_match": checks.get("all_contexts_are_pre_match")}),
    }, "source": replay.get("status") if isinstance(replay, Mapping) else None}


def _match_contract_audit() -> dict[str, Any]:
    now = datetime(2026, 10, 8, 0, 0, tzinfo=timezone.utc)
    bundle = build_match_bundle({
        "match": "Alpha FC vs Beta FC", "competition": "synthetic-league",
        "kickoff": "2026-10-09T12:00:00Z",
    }, now=now)
    state = bundle["state"]
    catalog = state.get("candidate_catalog", [])
    value_fields = [item.get("action_id") for item in catalog
                    if item.get("conditional_value") is not None or item.get("declared_cost") is not None
                    or any(key in item for key in ("voi", "priority", "value"))]
    unknown_ok = all(item.get("status") == "unknown" for item in state.get("unknown_information", []))
    checks = {
        "match_input_has_no_user_scope_question": _check(
            "match_input_has_no_user_scope_question",
            state.get("contract", {}).get("no_user_scope_question") is True and
            state.get("next_step", {}).get("user_question") is None),
        "match_input_has_no_result_leakage": _check(
            "match_input_has_no_result_leakage",
            not any(path.rsplit("/", 1)[-1].casefold() in _RESULT_KEYS
                    for path in _walk_keys(bundle))),
        "candidate_catalog_starts_unscored": _check(
            "candidate_catalog_starts_unscored",
            bool(catalog) and all(item.get("status") == "unscored" for item in catalog) and not value_fields,
            details=value_fields),
        "baseline_is_not_fabricated": _check(
            "baseline_is_not_fabricated",
            state.get("baseline", {}).get("probabilities") is None and
            state.get("baseline", {}).get("status") == "not_computed"),
        "unknowns_are_explicit": _check("unknowns_are_explicit", unknown_ok),
    }
    return {"checks": checks, "target_key": state.get("target_key"), "state_schema": state.get("schema")}


def _capability_binding_audit() -> dict[str, Any]:
    bundle = build_match_bundle({
        "home": "Alpha FC", "away": "Beta FC", "kickoff": "2026-10-09T12:00:00Z",
    }, now=datetime(2026, 10, 8, tzinfo=timezone.utc))
    bound = bind_source_capabilities(bundle["state"], {
        "competition_rules": "available", "standings": "available",
        "official_lineup": "stale", "team_news": "conflicting",
        "injury_report": "unavailable", "weather": "unknown",
    })
    catalog = {item["information_need"]: item for item in bound["candidate_catalog"]}
    checks = {
        "binding_preserves_unscored_values": _check(
            "binding_preserves_unscored_values",
            bound.get("capability_binding", {}).get("value_scored") is False and
            bound.get("capability_binding", {}).get("priority_assigned") is False and
            all(item.get("conditional_value") is None and item.get("eligibility") == "unscored"
                for item in bound.get("candidate_catalog", []))),
        "binding_maps_available_capabilities": _check(
            "binding_maps_available_capabilities",
            catalog.get("qualification_constraint", {}).get("availability") == "available"),
        "binding_maps_degraded_capabilities_conservatively": _check(
            "binding_maps_degraded_capabilities_conservatively",
            catalog.get("rotation_lineup", {}).get("availability") == "stale_or_conflicting" and
            catalog.get("injury_availability", {}).get("availability") in
            {"stale_or_conflicting", "unknown", "unavailable"} and
            catalog.get("injury_availability", {}).get("availability") != "available"),
        "binding_does_not_auto_resolve_prerequisites": _check(
            "binding_does_not_auto_resolve_prerequisites",
            bound.get("capability_binding", {}).get("prerequisites_auto_resolved") is False),
    }
    return {"checks": checks, "binding_version": bound.get("capability_binding", {}).get("version")}


def run_autonomous_mechanism_acceptance(
    fixtures: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all deterministic mechanism checks and return a JSON-safe report."""
    values = list(fixtures) if fixtures is not None else generate_scenario_fixtures()
    rows = [_fixture_structural_checks(item) for item in values]
    checks: dict[str, Any] = {}
    for row in rows:
        for name, result in row["checks"].items():
            checks[f"fixture:{row['fixture_id']}:{name}"] = result
    context = _context_routing_audit()
    for name, result in context["checks"].items():
        # ``_check`` returns a one-key mapping so callers can use it directly;
        # flatten it at the report boundary for a stable check schema.
        checks[f"routing:{name}"] = result.get(name, result)
    contract = _match_contract_audit()
    for name, result in contract["checks"].items():
        checks[f"contract:{name}"] = result.get(name, result)
    binding = _capability_binding_audit()
    for name, result in binding["checks"].items():
        checks[f"capability:{name}"] = result.get(name, result)
    failed = sorted(name for name, result in checks.items() if not result.get("passed"))
    passed_count = len(checks) - len(failed)
    return {
        "schema_version": VERSION,
        "status": "passed" if not failed else "failed",
        "mode": "offline_mechanism_audit",
        "policy_mode": "conditional_semantic_voi_untrained_g",
        "empirical_g": {"status": "deferred", "used_for_selection": False},
        "fixture_count": len(values),
        "checks": checks,
        "summary": {"total": len(checks), "passed": passed_count, "failed": len(failed)},
        "failed_checks": failed,
        "interpretation": "通过表示信息选择契约和保守证据处理成立，不表示比赛结果预测准确。",
    }


__all__ = ["VERSION", "run_autonomous_mechanism_acceptance"]
