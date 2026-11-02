"""A small, auditable research vertical slice for autonomous VoI selection.

The operational runtime has many boundary and persistence contracts.  This
module intentionally stays smaller: it takes one already frozen, actor-safe
historical public episode, a prior probability vector, and explicit response
models for candidate information actions.  It evaluates the candidates with
the strict one-step EVSI contract and returns an actor-safe selection trace.

This is a research harness, not a production predictor.  In particular, a
response model marked ``development_fixture`` cannot be promoted to numeric
VoI training data.  An independent result remains outside the returned actor
trace and must be attached through the evaluator sidecar.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from ..acquisition.expected_voi import (
    CandidateValue,
    ObservationBranch,
    evaluate_candidate,
    select_candidate,
    unknown_candidate,
)


VERSION = "voi-vertical-slice-v2"
_FORBIDDEN_PUBLIC_KEYS = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score", "away_score",
    "home_goals", "away_goals", "final_score", "label", "loss", "decision_loss",
    "realized_gain", "realized_loss", "evaluator", "sidecar", "sequence_losses",
    "action_returns", "post_state", "after_state", "actual_result", "target_label",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _public_key_scan(value: Any, path: str = "public") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_public_key_scan(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_public_key_scan(child, f"{path}[{index}]"))
    return found


def _candidate_ids(public: Mapping[str, Any]) -> tuple[str, ...]:
    raw = public.get("candidate_menu", public.get("candidates"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("public episode must contain a finite candidate menu")
    ids: list[str] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            candidate_id = item
        elif isinstance(item, Mapping):
            candidate_id = item.get("candidate_id")
        else:
            candidate_id = None
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"candidate_menu[{index}] has no candidate_id")
        ids.append(candidate_id.strip())
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    return tuple(ids)


def _actor_source_ids(public: Mapping[str, Any]) -> set[str]:
    """Collect source IDs already visible to the actor in a historical episode."""
    visible: set[str] = set()
    provenance = public.get("historical_provenance")
    if isinstance(provenance, Mapping):
        raw = provenance.get("actor_source_ids", [])
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            visible.update(item for item in raw if isinstance(item, str))
    snapshot = public.get("match_snapshot")
    if isinstance(snapshot, Mapping):
        observations = snapshot.get("source_observations", [])
        if isinstance(observations, Sequence) and not isinstance(observations, (str, bytes)):
            for item in observations:
                if isinstance(item, Mapping) and isinstance(item.get("source_id"), str):
                    visible.add(item["source_id"])
    return visible


def _validate_unobserved_candidates(public: Mapping[str, Any]) -> list[str]:
    """Reject a menu that silently reuses a source already in actor state."""
    raw = public.get("candidate_menu", public.get("candidates"))
    visible = _actor_source_ids(public)
    overlap: list[str] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for index, item in enumerate(raw):
            if isinstance(item, Mapping):
                declared_sources = item.get("source_ids", [])
                if (not isinstance(declared_sources, Sequence) or isinstance(declared_sources, (str, bytes))
                        or any(not isinstance(source, str) for source in declared_sources)):
                    raise ValueError(f"candidate_menu[{index}].source_ids must be a list of strings")
                sources = list(declared_sources)
                if isinstance(item.get("source_id"), str):
                    sources.append(item["source_id"])
                for source_id in sources:
                    if source_id in visible:
                        overlap.append(f"candidate_menu[{index}].source_id={source_id}")
    if overlap:
        raise ValueError(
            "candidate menu reuses actor-visible sources; prospective actions must be unobserved: "
            + ",".join(overlap)
        )
    return overlap


def _branch(raw: ObservationBranch | Mapping[str, Any], index: int) -> ObservationBranch:
    if isinstance(raw, ObservationBranch):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"branches[{index}] must be a mapping or ObservationBranch")
    posterior = raw.get("posterior")
    if not isinstance(posterior, (list, tuple)):
        raise ValueError(f"branches[{index}].posterior is required")
    return ObservationBranch(
        branch_id=str(raw.get("branch_id", "")),
        probability=raw.get("probability"),
        posterior=tuple(posterior),
    )


def _value_row(value: CandidateValue) -> dict[str, Any]:
    return value.as_dict()


def run_vertical_slice(
    public_episode: Mapping[str, Any],
    *,
    prior: Sequence[float] | Iterable[float],
    candidate_models: Mapping[str, Mapping[str, Any]],
    loss: str = "log_loss",
    decision_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a finite menu and return an actor-safe one-step trace.

    Every candidate must have either explicit branches and cost metadata or
    ``status='UNKNOWN'`` with a reason.  Missing entries are still rejected.
    Unknown response models flow to ``defer`` instead of fabricated zero EVSI.
    """
    if not isinstance(public_episode, Mapping):
        raise ValueError("public_episode must be a mapping")
    leaked = _public_key_scan(public_episode)
    if decision_context is not None and not isinstance(decision_context, Mapping):
        raise ValueError("decision_context must be a mapping")
    leaked.extend(_public_key_scan(decision_context, "decision_context"))
    if leaked:
        raise ValueError("public episode contains evaluator fields: " + ",".join(leaked))
    public = deepcopy(dict(public_episode))
    prior = tuple(prior)  # A generator must not be consumed anew per candidate.
    candidate_source_overlap = _validate_unobserved_candidates(public)
    ids = _candidate_ids(public)
    if not isinstance(candidate_models, Mapping):
        raise ValueError("candidate_models must be a mapping")
    if set(candidate_models) != set(ids):
        missing = sorted(set(ids) - set(candidate_models))
        extra = sorted(set(candidate_models) - set(ids))
        raise ValueError(f"candidate_models do not match public menu; missing={missing}, extra={extra}")

    menu = public.get("candidate_menu", public.get("candidates"))
    declared_costs = {item["candidate_id"]: item["cost"] for item in menu
                      if isinstance(item, Mapping) and "cost" in item}
    rows: list[CandidateValue] = []
    for candidate_id in ids:
        model = candidate_models[candidate_id]
        if not isinstance(model, Mapping):
            raise ValueError(f"candidate_models[{candidate_id}] must be a mapping")
        if candidate_id in declared_costs:
            expected, actual = declared_costs[candidate_id], model.get("cost")
            if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) or v < 0 for v in (expected, actual))
                    or not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-12)):
                raise ValueError(f"candidate_models[{candidate_id}].cost differs from public menu")
        if model.get("status") == "UNKNOWN":
            if "branches" in model:
                raise ValueError("UNKNOWN response model must not supply branches")
            rows.append(unknown_candidate(
                candidate_id, prior, reason=model.get("reason"), cost=model.get("cost"),
                loss=loss, predictor_version=model.get("predictor_version"),
                cost_model_version=model.get("cost_model_version"),
            ))
            continue
        if model.get("status", "AVAILABLE") != "AVAILABLE":
            raise ValueError("response model status must be AVAILABLE or UNKNOWN")
        raw_branches = model.get("branches")
        if not isinstance(raw_branches, Iterable) or isinstance(raw_branches, (str, bytes, Mapping)):
            raise ValueError(f"candidate_models[{candidate_id}].branches is required")
        branches = tuple(_branch(raw, index) for index, raw in enumerate(raw_branches))
        rows.append(evaluate_candidate(
            candidate_id,
            prior,
            branches,
            cost=model.get("cost"),
            delay=model.get("delay", 0.0),
            delay_weight=model.get("delay_weight", 0.0),
            reliability=model.get("reliability", 1.0),
            reliability_penalty_weight=model.get("reliability_penalty_weight", 0.0),
            loss=loss,
            predictor_version=model.get("predictor_version", "unbound"),
            response_model_version=model.get("response_model_version", "unbound"),
            cost_model_version=model.get("cost_model_version", "unbound"),
            calibration_cutoff=model.get("calibration_cutoff"),
        ))

    decision = select_candidate(rows)
    complete = public.get("candidate_generation_complete") is True
    if decision["decision"] == "stop" and not complete:
        decision.update(decision="defer", reason="candidate_catalog_incomplete",
                        stop_certificate=None)
    decision["selection_scope"] = "declared_finite_candidate_menu"
    trace = {
        "schema_version": VERSION,
        "episode_id": public.get("historical_provenance", {}).get("episode_id"),
        "target_key": public.get("target_key"),
        "prediction_cutoff": public.get("prediction_cutoff"),
        "kickoff": public.get("kickoff"),
        "public_snapshot_digest": public.get("public_snapshot_digest"),
        "input_public_digest": _digest(public),
        "prior": list(prior),
        "loss": loss,
        "decision_context": deepcopy(dict(decision_context or {})),
        "candidate_values": [_value_row(row) for row in rows],
        "candidate_source_overlap_with_actor": candidate_source_overlap,
        "selection": decision,
        "candidate_generation_complete": complete,
        "response_model_scope": ("unavailable" if all(
            candidate_models[key].get("status") == "UNKNOWN" for key in ids)
            else "development_fixture"),
        "numeric_voi_training_eligible": False,
        "hidden_result_included": False,
        "limitations": [
            "This is a one-step model-based EVSI trace, not realised match performance.",
            "Development response models require historical calibration before training use.",
            "Independent result and realised loss belong in an evaluator sidecar.",
            "A finite menu certificate does not establish exhaustion of open-world information.",
        ],
    }
    return trace


__all__ = ["VERSION", "run_vertical_slice"]
