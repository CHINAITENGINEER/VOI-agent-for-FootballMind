"""Actor-facing context adapter for PlatformReader exports (Round136).

The adapter is deliberately local and deterministic.  It prepares the bounded
information that an acquisition policy may inspect; it does not select a
candidate, call a provider, or infer a numeric value of information.  Current
operational clocks and historical reconstruction clocks are separate views.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

VERSION = "round136-platform-acquisition-context-v1"
OPERATIONAL = "operational"
HISTORICAL = "historical_reconstruction"
SYNTHETIC = "synthetic"
MODES = frozenset({OPERATIONAL, HISTORICAL, SYNTHETIC})

_RESULT_KEYS = frozenset({
    "result", "outcome", "score", "scores", "goals", "home_score",
    "away_score", "home_goals", "away_goals", "final_score", "label",
    "labels", "target", "target_label", "status", "score_period",
    "label_status", "result_available_at", "label_available_at",
    "settlement_status", "post_state", "after_state", "realized_loss",
    "decision_loss", "gold", "evaluator", "sidecar",
    "future_result", "hidden_return", "hidden_returns", "hidden_features",
    "full_snapshot", "response_body", "raw_data", "actual_result",
    "actual_outcome", "decision_loss_after", "realized_gain",
    "label_sidecar_digest", "outcome_digest", "evaluator_sidecar",
})
_TIME_KEYS = ("observed_at", "ingested_at", "available_at", "operational_available_at")


class PlatformAcquisitionContextError(ValueError):
    """Raised when a platform export cannot form an actor-visible context."""


def _row(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return deepcopy(dict(dumped))
    raise PlatformAcquisitionContextError(f"{name}_MUST_BE_MAPPING")


def _time(value: Any, name: str, *, required: bool = False) -> datetime | None:
    if value is None:
        if required:
            raise PlatformAcquisitionContextError(f"{name}_REQUIRED")
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise PlatformAcquisitionContextError(f"{name}_INVALID_TIME") from exc
    else:
        raise PlatformAcquisitionContextError(f"{name}_INVALID_TIME")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PlatformAcquisitionContextError(f"{name}_TIMEZONE_REQUIRED")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _safe(value: Any) -> Any:
    """Recursively remove labels/results and non-finite values from actor data."""
    if isinstance(value, Mapping):
        return {
            str(key): _safe(child)
            for key, child in value.items()
            if str(key).casefold() not in _RESULT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe(child) for child in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise PlatformAcquisitionContextError("NONFINITE_ACTOR_VALUE")
    return deepcopy(value)


def _has_private(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).casefold() in _RESULT_KEYS or _has_private(child)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_private(child) for child in value)
    return False


def _values(item: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("values", "payload", "record"):
        if isinstance(item.get(key), Mapping):
            return item[key]
    return {}


def _identity(item: Mapping[str, Any]) -> dict[str, Any]:
    values = _values(item)
    keys = ("match_id", "competition_id", "competition_name", "home_team_id",
            "away_team_id", "home_team_name", "away_team_name", "kickoff")
    identity = {key: values.get(key, item.get(key)) for key in keys}
    identity["match_id"] = identity.get("match_id") or item.get("canonical_match_id") or item.get("id")
    identity["kickoff"] = identity.get("kickoff") or values.get("event_time", item.get("event_time"))
    return {key: value for key, value in identity.items() if value is not None}


def _source_id(value: Mapping[str, Any], fallback: str) -> str:
    result = value.get("source_id", value.get("evidence_id", value.get("observation_id", fallback)))
    if not isinstance(result, str) or not result.strip():
        raise PlatformAcquisitionContextError("SOURCE_ID_REQUIRED")
    return result.strip()


def _clocks(item: Mapping[str, Any]) -> tuple[dict[str, datetime | None], datetime | None]:
    parsed = {key: _time(item.get(key), key) for key in _TIME_KEYS}
    actual = max((value for value in parsed.values() if value is not None), default=None)
    return parsed, actual


def _fixture_matches(item: Mapping[str, Any], target: Mapping[str, Any], match_id: str) -> bool:
    identity = _identity(item)
    if identity.get("match_id") != match_id:
        return False
    try:
        if _time(identity.get("kickoff"), "kickoff", required=True) != _time(target.get("kickoff", target.get("event_time")), "kickoff", required=True):
            return False
    except PlatformAcquisitionContextError:
        return False
    for key in ("home_team_id", "away_team_id", "home_team_name", "away_team_name"):
        if target.get(key) is not None:
            if identity.get(key) != target[key]:
                return False
    return True


def _valid_clock(item: Mapping[str, Any], *, mode: str, cutoff: datetime,
                 reconstructed: datetime | None, is_fixture: bool) -> tuple[bool, str, dict[str, datetime | None], datetime | None]:
    clocks, actual = _clocks(item)
    if mode == HISTORICAL:
        effective = reconstructed if reconstructed is not None else None
        if effective is None:
            return False, "missing_reconstructed_availability", clocks, actual
        if effective > cutoff:
            return False, "reconstructed_after_cutoff", clocks, actual
        return True, "explicit_reconstructed" if not is_fixture else "fixture_reconstruction_assumption", clocks, actual
    # Operational rows need both clocks.  A stale warehouse row with only an
    # ingestion timestamp must not become historical-visible by accident.
    if clocks["observed_at"] is None or clocks["ingested_at"] is None:
        return False, "missing_required_operational_clock", clocks, actual
    if actual is None or actual > cutoff:
        return False, "after_cutoff" if actual is not None else "missing_availability", clocks, actual
    return True, "max_platform_clocks", clocks, actual


def _sidecar(match: Mapping[str, Any], match_id: str, kickoff: datetime) -> dict[str, Any]:
    home = match.get("home_goals", match.get("home_score"))
    away = match.get("away_goals", match.get("away_score"))
    status = match.get("label_status", "UNAVAILABLE")
    result: dict[str, Any] = {
        "match_id": match_id, "kickoff": _iso(kickoff),
        "label_status": status, "result_available_at": match.get("result_available_at", match.get("label_available_at")),
    }
    if type(home) is int and home >= 0 and type(away) is int and away >= 0:
        result.update(home_goals=home, away_goals=away,
                      outcome=0 if home > away else 1 if home == away else 2)
    return result


def build_platform_acquisition_context(
    match: Mapping[str, Any] | Any,
    evidence_rows: Iterable[Mapping[str, Any] | Any] | None = None,
    *, prediction_cutoff: datetime | str | None = None,
    information_mode: str | None = None,
    reconstruction_assumptions: Mapping[str, Any] | None = None,
    candidate_catalog: Any = None,
    candidate_catalog_complete: bool | None = None,
) -> dict[str, Any]:
    """Build an actor-safe context plus audit and evaluator-only sidecar.

    Candidate declarations are intentionally not embedded.  The acquisition
    policy receives them from its separate catalog/whitelist interface.
    ``candidate_catalog`` is accepted for migration compatibility and ignored.
    """
    target = _row(match, "match")
    match_id = target.get("match_id", target.get("canonical_match_id", target.get("id")))
    if not isinstance(match_id, str) or not match_id.strip():
        raise PlatformAcquisitionContextError("MATCH_ID_REQUIRED")
    match_id = match_id.strip()
    kickoff = _time(target.get("kickoff", target.get("event_time")), "kickoff", required=True)
    cutoff = _time(prediction_cutoff if prediction_cutoff is not None else target.get("prediction_cutoff"), "prediction_cutoff", required=True)
    assert kickoff is not None and cutoff is not None
    if cutoff > kickoff:
        raise PlatformAcquisitionContextError("CUTOFF_AFTER_KICKOFF")
    mode = information_mode or target.get("information_mode", OPERATIONAL)
    if mode not in MODES:
        raise PlatformAcquisitionContextError("INFORMATION_MODE_INVALID")
    assumptions = dict(reconstruction_assumptions or {})
    fixture_reconstructed: datetime | None = None
    if mode == HISTORICAL:
        explicit = assumptions.get("fixture_reconstructed_available_at")
        lead = assumptions.get("fixture_lead_hours")
        if explicit is not None:
            fixture_reconstructed = _time(explicit, "fixture_reconstructed_available_at")
        elif type(lead) in (int, float) and math.isfinite(lead) and lead >= 0:
            try:
                fixture_reconstructed = kickoff - timedelta(hours=float(lead))
            except OverflowError as exc:
                raise PlatformAcquisitionContextError("FIXTURE_RECONSTRUCTION_INVALID") from exc
        if fixture_reconstructed is None or fixture_reconstructed > cutoff:
            raise PlatformAcquisitionContextError("FIXTURE_RECONSTRUCTION_REQUIRED")

    raw = evidence_rows
    if raw is None:
        raw = target.get("source_observations", target.get("observations", []))
    rows = [_row(item, f"evidence_rows[{index}]") for index, item in enumerate(raw or [])]
    # The match record itself is the target fixture identity.  It can authorize
    # identity only when its clocks meet the mode contract.
    fixture_source = dict(target)
    fixture_source["kind"] = "fixture_identity"
    fixture_source.setdefault("source_id", f"fixture:{match_id}")
    # The synthetic observation is a projection of the platform match record,
    # not a claim that every other field in that record was pre-match known.
    fixture_source["values"] = _identity(target)
    candidates = [fixture_source] + rows
    observations: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    actual_clock_audit: list[dict[str, Any]] = []
    visible_fixtures: list[tuple[str, str, dict[str, Any]]] = []
    for index, item in enumerate(candidates):
        try:
            sid = _source_id(item, "")
        except PlatformAcquisitionContextError:
            excluded.append({"index": index, "reason": "missing_source_id"})
            continue
        is_fixture = str(item.get("kind", "")).casefold() in {"fixture", "fixture_identity"}
        if is_fixture and not _fixture_matches(item, target, match_id):
            excluded.append({"source_id": sid, "reason": "foreign_fixture_identity"})
            continue
        try:
            reconstructed = None
            if mode == HISTORICAL:
                reconstructed = _time(item.get("reconstructed_available_at"), "reconstructed_available_at")
                if reconstructed is None and is_fixture:
                    reconstructed = fixture_reconstructed
            valid, basis, clocks, actual = _valid_clock(item, mode=mode, cutoff=cutoff, reconstructed=reconstructed, is_fixture=is_fixture)
        except PlatformAcquisitionContextError as exc:
            excluded.append({"source_id": sid, "reason": str(exc)})
            continue
        actual_clock_audit.append({"source_id": sid, **{key: _iso(value) for key, value in clocks.items()}, "actual_available_at": _iso(actual),
                                   "original_content_digest": item.get("digest", item.get("content_digest", item.get("content_hash")))})
        if not valid:
            excluded.append({"source_id": sid, "reason": basis})
            continue
        payload = _values(item)
        safe_payload = _safe(payload) if isinstance(payload, Mapping) else {}
        obs: dict[str, Any] = {"source_id": sid, "kind": item.get("kind", item.get("source", "platform")), "available_at": _iso(reconstructed if mode == HISTORICAL else actual), "availability_basis": basis}
        if mode in {OPERATIONAL, SYNTHETIC}:
            obs["observed_at"] = _iso(clocks["observed_at"])
            obs["ingested_at"] = _iso(clocks["ingested_at"])
        else:
            obs["observed_clock_status"] = "unknown_in_reconstructed_view"
            obs["reconstruction_clock_is_assumed"] = True
        for key in ("quality", "freshness", "availability", "valid_until", "expires_at", "observation_id",
                    "claim_conflict_group", "claim_key", "claim_id", "conflict_group"):
            if key in item and item[key] is not None:
                obs[key] = _safe(item[key])
        # Digests of mixed outcome/evidence records encode private labels even
        # after their fields are removed.  Preserve them for audit only.
        if index != 0 and not _has_private(payload) and not _has_private(item):
            for key in ("digest", "content_digest", "content_hash"):
                if key in item and item[key] is not None:
                    obs[key] = deepcopy(item[key])
        if safe_payload:
            obs["values"] = safe_payload
        observations.append(obs)
        if is_fixture:
            visible_fixtures.append((obs["available_at"], sid, _identity(item)))
    if not visible_fixtures:
        raise PlatformAcquisitionContextError("TARGET_FIXTURE_NOT_CUTOFF_VISIBLE")
    # Sort and deduplicate visible rows only, so later or missing-clock records
    # never alter selection of an already visible version with the same id.
    observations = list({_digest(item): item for item in observations}.values())
    observations.sort(key=lambda item: (item["available_at"] or "", item["source_id"], _digest(item)))
    visible_fixtures.sort(key=lambda item: (item[0], item[1], _digest(item[2])))
    identity = visible_fixtures[-1][2]
    identity.update({"kickoff": _iso(kickoff), "prediction_cutoff": _iso(cutoff), "information_mode": mode})
    public_snapshot = _safe({**identity, "source_observations": observations})
    actor = {"schema_version": VERSION, "match_id": match_id, "prediction_cutoff": _iso(cutoff), "information_mode": mode, "snapshot": public_snapshot, "source_observations": observations, "context_digest": _digest(public_snapshot)}
    audit = {"schema_version": VERSION, "match_id": match_id, "prediction_cutoff": _iso(cutoff), "information_mode": mode, "excluded_source_observations": excluded, "actual_clocks": actual_clock_audit, "fixture_visible": True, "candidate_catalog_embedded": False, "candidate_catalog_complete_ignored": candidate_catalog_complete, "historical_assumptions": assumptions if mode == HISTORICAL else None, "result_fields_sidecar_only": True, "unclocked_known_ignored": bool(target.get("known"))}
    return {"actor_context": actor, "audit": audit, "evaluator_sidecar": _sidecar(target, match_id, kickoff)}


__all__ = ["VERSION", "OPERATIONAL", "HISTORICAL", "SYNTHETIC", "PlatformAcquisitionContextError", "build_platform_acquisition_context"]
