"""Bridge strict historical as-of envelopes into Round162 replay episodes.

Round163 binds the actor-visible historical snapshot to the policy-replay
contract without manufacturing a result label.  A source archive can prove
what was observable before a prediction cutoff, while an independent result
archive is required before the episode can support a performance comparison.

The bridge is deliberately evaluator-side.  It does not fetch providers,
infer a score, estimate VoI, or turn a source report into a gold label.  The
public projection contains the pre-action snapshot and candidate menu; the
result, provenance and loss sequence remain in ``evaluation_sidecar``.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .counterfactual_policy_replay import INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED
from .historical_asof_episode import validate_historical_asof_envelope


VERSION = "round163-historical-episode-bridge-v2"
SOURCE_ONLY_NO_GOLD = "SOURCE_ONLY_NO_GOLD"
UNAVAILABLE = "UNAVAILABLE"
SUPPORTED_LABELS = frozenset({
    INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED, SOURCE_ONLY_NO_GOLD, UNAVAILABLE,
})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PUBLIC_FORBIDDEN = frozenset({
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


def _text(value: Any, name: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return result


def public_evaluator_fields(value: Any, path: str = "public") -> list[str]:
    """List structured evaluator-only fields without modifying actor inputs."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in _PUBLIC_FORBIDDEN:
                found.append(f"{path}.{key}")
            found.extend(public_evaluator_fields(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(public_evaluator_fields(child, f"{path}[{index}]"))
    return found


def _public(value: Any, path: str = "public") -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.casefold() in _PUBLIC_FORBIDDEN:
                raise ValueError(f"{path} contains evaluator-only field: {key}")
            output[key] = _public(child, f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_public(child, f"{path}[]") for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return deepcopy(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} contains an unsupported value")


def _candidate_menu(
    candidate_ids: Sequence[Any],
    candidate_menu: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = [_text(item, f"candidate_ids[{index}]") for index, item in enumerate(candidate_ids)]
    if any(item is None for item in expected):
        raise ValueError("candidate IDs must be nonempty")
    if len(set(expected)) != len(expected):
        raise ValueError("candidate IDs must be unique")
    if not isinstance(candidate_menu, Sequence) or isinstance(candidate_menu, (str, bytes)) or not candidate_menu:
        raise ValueError("candidate_menu must be a nonempty sequence")
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(candidate_menu):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate_menu[{index}] must be an object")
        candidate_id = _text(raw.get("candidate_id"), f"candidate_menu[{index}].candidate_id")
        assert candidate_id is not None
        if candidate_id in rows:
            raise ValueError("candidate_menu candidate IDs must be unique")
        cost = _number(raw.get("cost"), f"candidate_menu[{index}].cost")
        row: dict[str, Any] = {"candidate_id": candidate_id, "cost": cost}
        for key in ("source_id", "action_id", "action_kind", "description"):
            if key in raw and raw[key] is not None:
                if key in {"source_id", "action_id"}:
                    row[key] = _text(raw[key], f"candidate_menu[{index}].{key}")
                else:
                    row[key] = deepcopy(raw[key])
        rows[candidate_id] = row
    if set(rows) != set(expected):
        missing = sorted(set(expected) - set(rows))
        extra = sorted(set(rows) - set(expected))
        raise ValueError(f"candidate menu does not match historical menu; missing={missing}, extra={extra}")
    return [rows[candidate_id] for candidate_id in expected]


def _actor_source_provenance(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    observations = snapshot.get("source_observations", [])
    if not isinstance(observations, list):
        observations = []
    source_ids: list[str] = []
    digests: list[str] = []
    for item in observations:
        if not isinstance(item, Mapping):
            continue
        source_id = item.get("source_id")
        if isinstance(source_id, str) and source_id not in source_ids:
            source_ids.append(source_id)
        digest = item.get("content_digest")
        if isinstance(digest, str) and _DIGEST_RE.fullmatch(digest) and digest not in digests:
            digests.append(digest)
    return {"source_ids": source_ids, "content_digests": digests}


def _independent_sidecar(
    *,
    result: Mapping[str, Any],
    kickoff: datetime,
    cutoff: datetime,
    actor_source_ids: set[str],
    actor_digests: set[str],
    evaluator_id: str,
    base: Mapping[str, Any],
) -> dict[str, Any]:
    source_id = _text(result.get("source_id", result.get("result_source_id")), "independent_result.source_id")
    assert source_id is not None
    if source_id in actor_source_ids:
        raise ValueError("independent result source must be separate from actor sources")
    observed_at = _time(result.get("observed_at", result.get("result_observed_at")),
                        "independent_result.observed_at")
    available_at = _time(result.get("available_at", result.get("label_available_at")),
                         "independent_result.available_at")
    if observed_at < kickoff:
        raise ValueError("independent result observation must follow kickoff")
    label_status = result.get("label_status", INDEPENDENT_GOLD)
    if label_status not in {INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED}:
        raise ValueError("independent_result.label_status is unsupported")
    timing_basis = result.get("timing_basis", (
        "historical_reconstructed_result_delay" if label_status == INDEPENDENT_GOLD_RECONSTRUCTED
        else "operational_result_availability"))
    if not isinstance(timing_basis, str) or not timing_basis.strip():
        raise ValueError("independent_result.timing_basis must be a nonempty string")
    # A reconstructed archive may be ingested long after the inferred result
    # availability time.  Its historical clock is still checked against the
    # kickoff and prediction cutoff, while the operational path additionally
    # requires system availability to follow observation.
    if label_status == INDEPENDENT_GOLD and available_at < observed_at:
        raise ValueError("independent result availability must follow observation")
    if available_at <= kickoff:
        raise ValueError("independent result availability must follow kickoff")
    if available_at <= cutoff:
        raise ValueError("independent result availability must follow prediction cutoff")
    digest = _text(result.get("content_digest", result.get("raw_sha256")),
                   "independent_result.content_digest")
    assert digest is not None
    if not _DIGEST_RE.fullmatch(digest):
        raise ValueError("independent_result.content_digest must be a lowercase SHA256")
    if digest in actor_digests:
        raise ValueError("independent result digest must be separate from actor evidence")
    home_goals = result.get("home_goals", result.get("home_score"))
    away_goals = result.get("away_goals", result.get("away_score"))
    if type(home_goals) is not int or home_goals < 0 or type(away_goals) is not int or away_goals < 0:
        raise ValueError("independent result requires nonnegative integer home_goals and away_goals")
    if result.get("score_period", "regulation") != "regulation":
        raise ValueError("independent result must use regulation scores")
    if (label_status == INDEPENDENT_GOLD_RECONSTRUCTED) != (timing_basis == "historical_reconstructed_result_delay"):
        raise ValueError("independent result timing_basis conflicts with label_status")
    outcome = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    outcome_payload = {
        "match_id": base.get("match_id"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "outcome": outcome,
        "score_period": result.get("score_period", "regulation"),
        "source_id": source_id,
        "content_digest": digest,
    }
    return {
        "label_status": label_status,
        "evaluator_id": evaluator_id,
        "label_available_at": _iso(available_at),
        "timing_basis": timing_basis,
        "result_source_id": source_id,
        "result_observed_at": _iso(observed_at),
        "result_content_digest": digest,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "outcome": outcome,
        "score_period": result.get("score_period", "regulation"),
        "outcome_digest": _digest(outcome_payload),
    }


def _loss_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("baseline_loss", "cost_to_loss_rate", "sequence_losses", "action_returns"):
        if key in source:
            fields[key] = deepcopy(source[key])
    return fields


def _replay_inputs_present(sidecar: Mapping[str, Any], public: Mapping[str, Any]) -> bool:
    """Check replay prerequisites, not eligibility of every policy sequence.

    The replay engine still checks the loss for the exact path it executes.
    A label, a baseline, or a single incidental sidecar field is insufficient.
    """
    if sidecar.get("independent_result_bound") is not True:
        return False
    try:
        baseline = _number(sidecar.get("baseline_loss"), "baseline_loss")
        _number(sidecar.get("cost_to_loss_rate"), "cost_to_loss_rate")
        losses, returns = sidecar.get("sequence_losses"), sidecar.get("action_returns")
        if not isinstance(losses, Mapping) or not isinstance(returns, Mapping):
            return False
        if losses.get("") != baseline:
            return False
        cutoff = _time(public.get("prediction_cutoff"), "prediction_cutoff")
        ids = {item["candidate_id"] for item in public["candidate_menu"]}
        for key, loss in losses.items():
            _number(loss, "sequence_loss")
            if not isinstance(key, str):
                return False
            sequence = [] if key == "" else [part.strip() for part in key.replace("|", ">").split(">")]
            if len(set(sequence)) != len(sequence) or not set(sequence) <= ids:
                return False
        for candidate_id in ids:
            row = returns.get(candidate_id)
            if not isinstance(row, Mapping) or not isinstance(row.get("available"), bool):
                return False
            observed = _time(row.get("observed_at"), "action.observed_at")
            available = _time(row.get("operational_available_at", row.get("available_at")), "action.available_at")
            if not observed <= available <= cutoff:
                return False
            update = row.get("context_update", {})
            if not isinstance(update, Mapping) or public_evaluator_fields(update):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def build_round163_episode(
    envelope: Mapping[str, Any],
    *,
    candidate_menu: Sequence[Mapping[str, Any]],
    initial_context: Mapping[str, Any] | None = None,
    budget: float,
    evaluator_sidecar: Mapping[str, Any] | None = None,
    independent_result: Mapping[str, Any] | None = None,
    evaluator_id: str = "round163-independent-result-evaluator-v1",
) -> dict[str, Any]:
    """Create a Round162-compatible episode from a historical as-of envelope.

    With no ``independent_result`` the returned sidecar is explicitly
    ``SOURCE_ONLY_NO_GOLD``.  Supplying an evaluator sidecar alone never
    upgrades that status; the result source and its timing/digest must pass the
    independent binding checks.
    """
    if not isinstance(envelope, Mapping):
        raise ValueError("historical envelope must be an object")
    envelope_check = validate_historical_asof_envelope(envelope)
    if envelope_check.get("valid") is not True:
        raise ValueError("historical envelope is invalid: " + ",".join(envelope_check.get("reasons", [])))
    historical = envelope.get("episode")
    if not isinstance(historical, Mapping):
        raise ValueError("historical envelope episode is required")
    pre = historical.get("pre_action")
    if not isinstance(pre, Mapping) or not isinstance(pre.get("snapshot"), Mapping):
        raise ValueError("historical pre_action snapshot is required")
    snapshot = _public(pre["snapshot"], "public.match_snapshot")
    leaks = public_evaluator_fields(snapshot, "public.match_snapshot")
    if leaks:
        raise ValueError("historical snapshot contains evaluator fields: " + ",".join(leaks))
    historical_snapshot_digest = pre.get("snapshot_digest")
    if not isinstance(historical_snapshot_digest, str) or not _DIGEST_RE.fullmatch(historical_snapshot_digest):
        raise ValueError("historical snapshot digest is missing or malformed")
    if historical_snapshot_digest != _digest(snapshot):
        raise ValueError("historical snapshot digest mismatch")
    cutoff = _time(historical.get("prediction_cutoff"), "prediction_cutoff")
    kickoff = _time(snapshot.get("kickoff"), "match_snapshot.kickoff")
    if cutoff > kickoff:
        raise ValueError("prediction_cutoff must not follow kickoff")
    candidate_ids = pre.get("candidate_ids")
    if not isinstance(candidate_ids, Sequence) or isinstance(candidate_ids, (str, bytes)):
        raise ValueError("historical candidate_ids are required")
    menu = _candidate_menu(candidate_ids, candidate_menu)
    context = {} if initial_context is None else _public(initial_context, "public.initial_context")
    budget_value = _number(budget, "budget")
    if not isinstance(evaluator_id, str) or not evaluator_id.strip():
        raise ValueError("evaluator_id must be a nonempty string")
    evaluator_id = evaluator_id.strip()
    actor_provenance = _actor_source_provenance(snapshot)
    actor_source_ids = set(actor_provenance["source_ids"])
    actor_digests = set(actor_provenance["content_digests"])

    public = {
        "target_key": historical.get("target_key"),
        "kickoff": _iso(kickoff),
        "prediction_cutoff": _iso(cutoff),
        "match_snapshot": snapshot,
        "initial_context": context,
        "candidate_generation_complete": pre.get("candidate_generation_complete") is True,
        "budget": budget_value,
        "loss_unit": "decision_loss",
        "candidate_menu": menu,
        "historical_provenance": {
            "bridge_version": VERSION,
            "episode_id": historical.get("episode_id"),
            "historical_snapshot_digest": historical_snapshot_digest,
            "actor_source_ids": actor_provenance["source_ids"],
            "actor_content_digests": actor_provenance["content_digests"],
            "cutoff_basis": "explicit_prediction_cutoff",
        },
    }
    public = _public(public, "public")
    public["public_snapshot_digest"] = _digest(public["match_snapshot"])
    base_sidecar = dict(evaluator_sidecar or {})
    status = SOURCE_ONLY_NO_GOLD
    sidecar: dict[str, Any] = {
        "label_status": status,
        "evaluator_id": evaluator_id,
        "source_only": True,
        "actor_snapshot_digest": public["public_snapshot_digest"],
        "actor_source_ids": actor_provenance["source_ids"],
        "independent_result_bound": False,
        "independent_result_rejection_reason": "independent_result_not_supplied",
    }
    if independent_result is not None:
        sidecar.update(_independent_sidecar(
            result=independent_result,
            kickoff=kickoff,
            cutoff=cutoff,
            actor_source_ids=actor_source_ids,
            actor_digests=actor_digests,
            evaluator_id=evaluator_id,
            base=snapshot,
        ))
        status = sidecar["label_status"]
        sidecar["label_status"] = status
        sidecar["source_only"] = False
        sidecar["independent_result_bound"] = True
        sidecar.pop("independent_result_rejection_reason", None)
        sidecar.update(_loss_fields(base_sidecar))
    else:
        # Loss declarations without an independently bound result are kept out
        # of the sidecar rather than allowing a caller to smuggle synthetic
        # metrics into a source-only episode.
        sidecar["loss_fields_ignored"] = sorted(_loss_fields(base_sidecar))

    episode_id = f"round163:{historical.get('episode_id')}"
    return {
        "schema_version": VERSION,
        "episode_id": episode_id,
        "public": public,
        "evaluation_sidecar": sidecar,
        "audit": {
            "bridge_version": VERSION,
            "historical_envelope_valid": True,
            "public_snapshot_digest": public["public_snapshot_digest"],
            "candidate_menu_digest": _digest(menu),
            "actor_source_count": len(actor_provenance["source_ids"]),
            "independent_result_bound": status in {INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED},
            "label_status": status,
            "counterfactual_replay_ready": _replay_inputs_present(sidecar, public),
            "policy_sequence_eligibility_requires_replay": True,
            "numeric_voi_calibrated": False,
            "real_source_requests": 0,
            "real_model_calls": 0,
        },
    }


def validate_round163_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bridge boundary without replaying a policy."""
    reasons: list[str] = []
    if not isinstance(episode, Mapping):
        return {"valid": False, "label_status": None, "reasons": ["episode_not_mapping"]}
    public = episode.get("public")
    sidecar = episode.get("evaluation_sidecar")
    if not isinstance(public, Mapping):
        reasons.append("public_missing")
    else:
        reasons.extend(public_evaluator_fields(public))
        try:
            cutoff = _time(public.get("prediction_cutoff"), "public.prediction_cutoff")
            kickoff = _time(public.get("kickoff"), "public.kickoff")
            if cutoff > kickoff:
                reasons.append("cutoff_after_kickoff")
        except ValueError as exc:
            reasons.append(str(exc))
        snapshot = public.get("match_snapshot")
        if isinstance(snapshot, Mapping):
            digest = public.get("public_snapshot_digest")
            if digest != _digest(snapshot):
                reasons.append("public_snapshot_digest_mismatch")
        else:
            reasons.append("match_snapshot_missing")
    label_status = sidecar.get("label_status") if isinstance(sidecar, Mapping) else None
    if label_status not in SUPPORTED_LABELS:
        reasons.append("unsupported_label_status")
    independent = label_status in {INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED}
    if independent and (not isinstance(sidecar, Mapping) or sidecar.get("independent_result_bound") is not True):
        reasons.append("independent_result_not_bound")
    if isinstance(sidecar, Mapping) and label_status == SOURCE_ONLY_NO_GOLD:
        if sidecar.get("independent_result_bound") is True:
            reasons.append("source_only_marked_bound")
        if any(key in sidecar for key in ("home_goals", "away_goals", "outcome")):
            reasons.append("source_only_contains_result")
    return {
        "valid": not reasons,
        "episode_id": episode.get("episode_id"),
        "label_status": label_status,
        "independent_gold": label_status == INDEPENDENT_GOLD and not reasons,
        "reconstructed_result": label_status == INDEPENDENT_GOLD_RECONSTRUCTED and not reasons,
        "counterfactual_replay_eligible": independent and not reasons and _replay_inputs_present(sidecar, public),
        "policy_sequence_eligibility_requires_replay": True,
        "reasons": reasons,
    }


__all__ = [
    "VERSION", "SOURCE_ONLY_NO_GOLD", "UNAVAILABLE", "INDEPENDENT_GOLD",
    "INDEPENDENT_GOLD_RECONSTRUCTED",
    "build_round163_episode", "validate_round163_episode", "public_evaluator_fields",
]
