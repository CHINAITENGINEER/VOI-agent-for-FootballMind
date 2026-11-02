"""Safe, cutoff-bound probability snapshots for the autonomous runtime.

The acquisition policy estimates the value of obtaining information.  It does
not itself produce a match posterior.  This module defines the small contract
between a real predictor and the operational runtime so that a posterior can
be recorded without allowing evaluator data, future clocks, or cross-match
state to enter an episode journal.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping


class PredictionSnapshotError(ValueError):
    """Raised when a predictor result cannot be bound to the actor state."""


def _clock(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PredictionSnapshotError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionSnapshotError(f"{name} must be a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc)


def _iso(value: Any, name: str) -> str:
    return _clock(value, name).isoformat().replace("+00:00", "Z")


def _vector(value: Any, name: str) -> list[float]:
    if isinstance(value, Mapping):
        keys = ("home", "draw", "away")
        if not all(key in value for key in keys):
            # Accept the explicit long names used by PredictionService too.
            keys = ("home_win_probability", "draw_probability", "away_win_probability")
        if not all(key in value for key in keys):
            raise PredictionSnapshotError(
                f"{name}.probabilities must contain home/draw/away probabilities")
        value = [value[key] for key in keys]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise PredictionSnapshotError(f"{name}.probabilities must be a 3-class vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PredictionSnapshotError(f"{name}.probabilities must contain finite numbers")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise PredictionSnapshotError(f"{name}.probabilities must contain finite nonnegative numbers")
        result.append(number)
    if abs(sum(result) - 1.0) > 1e-8:
        raise PredictionSnapshotError(f"{name}.probabilities must sum to one")
    return result


def _input_ids(value: Any, name: str) -> list[str]:
    """Validate the predictor's explicit semantic-consumption ledger.

    The runtime must carry the ledger verbatim.  It must never infer consumed
    documents from a changed probability or from the existence of a receipt.
    """
    if not isinstance(value, list):
        raise PredictionSnapshotError(f"{name} must be a list when provided")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise PredictionSnapshotError(f"{name}[{index}] must be a nonempty string")
        item = item.strip()
        if item in result:
            raise PredictionSnapshotError(f"{name} contains duplicate input IDs")
        result.append(item)
    return result


def normalise_prediction_snapshot(
    snapshot: Mapping[str, Any] | list[float] | tuple[float, ...],
    *,
    match_id: str,
    prediction_cutoff: Any,
    state_digest: str | None = None,
    name: str = "prediction",
) -> dict[str, Any]:
    """Bind a predictor response to one actor-visible state.

    A mapping is required for production adapters because it must declare the
    match and cutoff.  A bare vector is accepted only for small local fixtures
    and is still bound to the caller's match and cutoff; production adapters
    should always return the mapping form.
    """
    if not isinstance(match_id, str) or not match_id.strip():
        raise PredictionSnapshotError("match_id must be a nonempty string")
    expected_cutoff = _iso(prediction_cutoff, "prediction_cutoff")
    raw: Mapping[str, Any]
    if isinstance(snapshot, Mapping):
        raw = snapshot
        probabilities = raw.get("probabilities")
        if probabilities is None:
            probabilities = raw
        declared_match = raw.get("match_id")
        if declared_match is not None and declared_match != match_id:
            raise PredictionSnapshotError(f"{name}.match_id differs from actor match")
        declared_cutoff = raw.get("prediction_cutoff")
        if declared_cutoff is None:
            raise PredictionSnapshotError(f"{name}.prediction_cutoff is required")
        if _iso(declared_cutoff, f"{name}.prediction_cutoff") != expected_cutoff:
            raise PredictionSnapshotError(f"{name}.prediction_cutoff differs from actor cutoff")
    elif isinstance(snapshot, (list, tuple)):
        raw = {}
        probabilities = snapshot
    else:
        raise PredictionSnapshotError(f"{name} must be a mapping or probability vector")
    result: dict[str, Any] = {
        "probabilities": _vector(probabilities, name),
        "match_id": match_id,
        "prediction_cutoff": expected_cutoff,
    }
    if state_digest is not None:
        declared_state = raw.get("state_digest")
        if declared_state is not None and declared_state != state_digest:
            raise PredictionSnapshotError(f"{name}.state_digest differs from actor state")
        result["state_digest"] = state_digest
    for key in ("model_version", "dataset_version", "prior_version", "artifact_sha256",
                "information_mode"):
        value = raw.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise PredictionSnapshotError(f"{name}.{key} must be a nonempty string")
            result[key] = value.strip()
    if "used_semantic_input_ids" in raw:
        result["used_semantic_input_ids"] = _input_ids(
            raw["used_semantic_input_ids"], f"{name}.used_semantic_input_ids")
    if "modeled_read_documents" in raw:
        value = raw["modeled_read_documents"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PredictionSnapshotError(
                f"{name}.modeled_read_documents must be a nonnegative integer")
        result["modeled_read_documents"] = value
    if "visible_lineage" in raw:
        result["visible_lineage"] = raw["visible_lineage"]
    return result


__all__ = ["PredictionSnapshotError", "normalise_prediction_snapshot"]
