"""Round133 scenario-to-policy integration for offline mechanism evaluation.

This module joins three deliberately separate artifacts:

* the actor-visible scenario fixture from :mod:`scenario_fixture_pack`;
* an independent evaluator record from :mod:`evaluator_sidecar`;
* the policy replay contract from :mod:`policy_benchmark`.

The join is synthetic and offline.  It does not call a provider, ask an LLM,
read an answer supplied by the fixture author, or write Agent runtime state.
Qualitative ``expected_policy`` fields stay evaluator-only; the actor receives
only the public snapshot and finite candidate menu.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .evaluator_sidecar import (
    DEFAULT_EVALUATOR_ID,
    DEFAULT_LOSS_UNIT,
    build_evaluator_record,
)
from .scenario_fixture_pack import (
    actor_projection,
    generate_scenario_fixtures,
    validate_scenario_fixture,
)
from .value_evaluation_contract import INDEPENDENT_GOLD


VERSION = "round133-policy-integration-v1"
SYNTHETIC_COST_UNIT = "external_calls"
DEFAULT_BUDGET_POLICY = "sum_declared_external_calls"


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("prediction_cutoff must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _external_cost(candidate: Mapping[str, Any]) -> float:
    declared = candidate.get("declared_cost")
    if not isinstance(declared, Mapping):
        raise ValueError("candidate declared_cost is required")
    value = declared.get("external_calls")
    if type(value) is not int or value < 0:
        raise ValueError("candidate external_calls cost must be a nonnegative integer")
    return float(value)


def _base_episode(fixture: Mapping[str, Any]) -> dict[str, Any]:
    report = validate_scenario_fixture(fixture)
    if not report["valid"]:
        raise ValueError("SCENARIO_FIXTURE_INVALID:" + ",".join(report["reasons"]))
    actor = actor_projection(fixture)
    cutoff = actor["prediction_cutoff"]
    _parse_iso(cutoff)
    candidate_ids = [item["candidate_id"] for item in actor["candidate_menu"]]
    # The contract requires a selection timestamp even for this offline
    # harness.  Equal-to-cutoff means no information after the as-of boundary
    # is smuggled into the pre-action snapshot.
    return {
        "schema_version": "value-evaluation-contract-v1",
        "episode_id": f"{actor['fixture_id']}:mechanism",
        "target_key": actor["target_key"],
        "prediction_cutoff": cutoff,
        "pre_action": {
            "snapshot": deepcopy(actor["snapshot"]),
            "snapshot_digest": actor["snapshot_digest"],
            "candidate_ids": candidate_ids,
            "candidate_generation_complete": True,
            "selection_operation_id": f"{actor['fixture_id']}:selection",
            "decision": "defer",
            "selected_candidate_id": None,
            "selected_at": cutoff,
        },
        "candidate_evaluations": [],
        "evaluation_outcome": None,
        "evaluator_sidecar": {
            "status": "not_attached",
            "required": True,
            "runtime_can_read": False,
        },
    }


def _record_to_sidecar(record: Mapping[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    evaluator = record.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("evaluator record is missing evaluator metadata")
    evaluator_id = evaluator.get("evaluator_id")
    if not isinstance(evaluator_id, str) or not evaluator_id.strip():
        raise ValueError("evaluator_id is required")
    rows = record.get("candidate_evaluations")
    if not isinstance(rows, list):
        raise ValueError("evaluator candidate_evaluations must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("evaluator candidate row must be an object")
        candidate_id = row.get("candidate_id")
        if candidate_id not in candidate_ids or candidate_id in by_id:
            raise ValueError("evaluator candidate is outside or duplicated in public menu")
        if row.get("label_status") != INDEPENDENT_GOLD:
            raise ValueError("evaluator candidate must be independent gold")
        by_id[candidate_id] = deepcopy(dict(row))
    if not by_id:
        raise ValueError("at least one independent candidate loss is required")
    first = next(iter(by_id.values()))
    baseline = first.get("decision_loss_before")
    loss_unit = first.get("loss_unit", DEFAULT_LOSS_UNIT)
    if not isinstance(baseline, (int, float)) or isinstance(baseline, bool) or baseline < 0:
        raise ValueError("evaluator baseline loss is required")
    if not isinstance(loss_unit, str) or not loss_unit.strip():
        raise ValueError("evaluator loss unit is required")
    sequence_losses: dict[str, float] = {"": float(baseline)}
    for candidate_id, row in by_id.items():
        after = row.get("decision_loss_after")
        if not isinstance(after, (int, float)) or isinstance(after, bool) or after < 0:
            raise ValueError("evaluator candidate loss is required")
        sequence_losses[candidate_id] = float(after)
    return {
        "evaluator_id": evaluator_id,
        "label_status": INDEPENDENT_GOLD,
        "baseline_loss": float(baseline),
        "loss_unit": loss_unit,
        "candidate_evaluations": list(by_id.values()),
        "sequence_losses": sequence_losses,
        "complete_sequence_evaluation": False,
        "mechanism_only": True,
        "world_truth_verified": False,
    }


def fixture_to_benchmark_episode(
    fixture: Mapping[str, Any], *,
    candidate_losses: Mapping[str, Any] | None = None,
    outcome_label: str | None = None,
    evaluator_record: Mapping[str, Any] | None = None,
    budget: float | None = None,
    cost_to_loss_rate: float | None = None,
) -> dict[str, Any]:
    """Build one benchmark episode without exposing fixture-only expectations."""
    episode = _base_episode(fixture)
    actor = actor_projection(fixture)
    menu: list[dict[str, Any]] = []
    for candidate in actor["candidate_menu"]:
        menu.append({
            "candidate_id": candidate["candidate_id"],
            "cost": _external_cost(candidate),
            # Qualitative fixture classes are intentionally not translated to
            # a numeric expected value.  The online gate must defer instead of
            # treating a scenario-author hypothesis as a measured VoI.
            "expected_value": None,
            "latency_ms": None,
            "reliability": None,
        })
    costs = [item["cost"] for item in menu]
    if budget is None:
        budget = sum(costs)
    if type(budget) not in (int, float) or budget < 0:
        raise ValueError("budget must be a nonnegative number")
    if cost_to_loss_rate is not None:
        if type(cost_to_loss_rate) not in (int, float) or cost_to_loss_rate < 0:
            raise ValueError("cost_to_loss_rate must be nonnegative")
        episode["cost_unit"] = SYNTHETIC_COST_UNIT
        episode["cost_to_loss_rate"] = float(cost_to_loss_rate)
    episode["public"] = {
        "candidate_menu": menu,
        "candidate_generation_complete": True,
        "budget": float(budget),
        "cost_unit": SYNTHETIC_COST_UNIT if cost_to_loss_rate is not None else None,
        "cost_to_loss_rate": float(cost_to_loss_rate) if cost_to_loss_rate is not None else None,
    }
    if evaluator_record is None:
        evaluator_record = build_evaluator_record(
            episode,
            candidate_losses=candidate_losses,
            outcome_label=outcome_label,
        )
    sidecar = _record_to_sidecar(evaluator_record, [item["candidate_id"] for item in menu])
    episode["evaluation_sidecar"] = sidecar
    episode["integration"] = {
        "version": VERSION,
        "evidence_origin": "synthetic_fixture",
        "world_truth_verified": False,
        "runtime_reads_sidecar": False,
        "public_budget_policy": DEFAULT_BUDGET_POLICY if budget == sum(costs) else "explicit",
    }
    return episode


def build_fixture_benchmark_episodes(
    fixtures: Iterable[Mapping[str, Any]] | None = None, **kwargs: Any,
) -> list[dict[str, Any]]:
    values = list(fixtures) if fixtures is not None else generate_scenario_fixtures()
    return [fixture_to_benchmark_episode(fixture, **kwargs) for fixture in values]


__all__ = [
    "VERSION", "SYNTHETIC_COST_UNIT", "fixture_to_benchmark_episode",
    "build_fixture_benchmark_episodes",
]
