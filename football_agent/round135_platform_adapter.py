"""Offline adapter for PlatformReader match/evidence rows.

The platform reader returns operational rows whose clocks describe when a
record was observed and committed.  This module turns those rows into the
plain JSON-like contract consumed by :mod:`historical_asof_episode`; it never
opens the platform database and never calls a provider.

Two clocks are kept distinct throughout the adapter:

* operational mode uses ``max(observed_at, ingested_at, available_at)``;
* historical reconstruction uses an explicit, caller-supplied assumption for
  the reconstructed availability clock and preserves the actual clocks beside
  it for audit.

The match result is passed through only so the frozen as-of builder can place
it in its evaluator sidecar.  It is never copied into the public episode by
this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Iterable, Mapping

from .historical_asof_episode import build_historical_asof_episode


VERSION = "platform-history-adapter-v1"
OPERATIONAL = "operational"
HISTORICAL = "historical_reconstruction"
MODES = frozenset({OPERATIONAL, HISTORICAL})


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return deepcopy(dict(dumped))
    if is_dataclass(value) and not isinstance(value, type):
        return deepcopy(asdict(value))
    raise ValueError(f"{name} must be mapping-like")


def _text(value: Any, name: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _time(value: Any, name: str, *, required: bool = False) -> datetime | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be ISO-8601") from exc
    else:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _record_values(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return provider values without inventing a schema.

    ``record`` is the shape emitted by ``PlatformReader``.  Other collectors
    may already expose ``values`` or ``payload``; all are accepted and copied
    for the frozen builder to redact result-like keys at the actor boundary.
    """
    for key in ("values", "payload", "record"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return deepcopy(dict(value))
    return {}


def _clock_bundle(row: Mapping[str, Any], *, cutoff: datetime,
                  mode: str, reconstruction_available_at: datetime | None,
                  path: str) -> tuple[dict[str, Any], list[str]]:
    """Build a source timing bundle and explicit timing audit entries."""
    observed = _time(row.get("observed_at"), f"{path}.observed_at")
    ingested = _time(row.get("ingested_at"), f"{path}.ingested_at")
    supplied = _time(row.get("available_at"), f"{path}.available_at")
    operational_supplied = _time(row.get("operational_available_at"),
                                 f"{path}.operational_available_at")
    issues: list[str] = []
    if observed is None:
        issues.append(f"{path}.observed_at")
    if ingested is None:
        issues.append(f"{path}.ingested_at")

    # A missing operational clock remains unknown.  In particular, do not use
    # today's ingest time as evidence that an old fixture was visible earlier.
    actual_values = [value for value in (observed, ingested, supplied,
                                         operational_supplied) if value is not None]
    actual_available = max(actual_values) if actual_values else None
    if mode == OPERATIONAL:
        effective = actual_available if not issues else None
        basis = "max(observed_at,ingested_at,available_at)" if effective else "unknown_missing_clock"
    else:
        effective = reconstruction_available_at
        basis = "explicit_reconstruction_assumption" if effective else "unknown_missing_assumption"

    if effective is not None and effective > cutoff:
        issues.append(f"{path}.available_at_after_cutoff")
    if observed is not None and observed > cutoff:
        issues.append(f"{path}.observed_at_after_cutoff")

    result: dict[str, Any] = {
        "observed_at": _iso(observed),
        "ingested_at": _iso(ingested),
        "available_at": _iso(effective),
        "availability_basis": basis,
        # Actual clocks remain visible to the audit and downstream provenance.
        "actual_available_at": _iso(actual_available),
        "operational_available_at": _iso(actual_available),
    }
    if mode == HISTORICAL:
        result["reconstructed_available_at"] = _iso(reconstruction_available_at)
        result["reconstruction_clock_is_assumed"] = True
    else:
        result["reconstruction_clock_is_assumed"] = False
    return result, issues


def _candidate_menu(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("candidate_menu must be a list")
    return deepcopy(value)


def platform_match_to_episode_row(
    match: Mapping[str, Any] | Any,
    evidence_rows: Iterable[Mapping[str, Any] | Any] | None = None,
    *,
    prediction_cutoff: datetime | str | None = None,
    information_mode: str | None = None,
    reconstruction_assumptions: Mapping[str, Any] | None = None,
    candidate_menu: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert a PlatformReader match and evidence export into a pure row.

    No database or network object is accepted or touched.  ``historical`` mode
    requires ``fixture_lead_hours`` in ``reconstruction_assumptions`` and uses
    it only for the reconstructed fixture clock; actual observation and ingest
    clocks are retained separately.
    """
    source_row = _mapping(match, "match")
    mode = information_mode or source_row.get("information_mode", OPERATIONAL)
    if mode not in MODES:
        raise ValueError(f"information_mode must be one of {sorted(MODES)}")

    match_id = _text(source_row.get("match_id", source_row.get("canonical_match_id", source_row.get("id"))), "match_id")
    assert match_id is not None
    kickoff = _time(source_row.get("kickoff", source_row.get("event_time")), "kickoff", required=True)
    cutoff = _time(prediction_cutoff if prediction_cutoff is not None else source_row.get("prediction_cutoff"),
                   "prediction_cutoff", required=True)
    assert kickoff is not None and cutoff is not None
    if cutoff > kickoff:
        raise ValueError("prediction_cutoff must not follow kickoff")

    assumptions = deepcopy(dict(reconstruction_assumptions or {}))
    if mode == HISTORICAL:
        lead = assumptions.get("fixture_lead_hours")
        if isinstance(lead, bool) or not isinstance(lead, (int, float)) or lead < 0:
            raise ValueError("historical_reconstruction requires finite fixture_lead_hours")
        reconstruction_fixture_at = kickoff - timedelta(hours=float(lead))
        if reconstruction_fixture_at > cutoff:
            raise ValueError("reconstructed fixture availability follows prediction_cutoff")
    else:
        reconstruction_fixture_at = None

    actual_match_clock, match_clock_issues = _clock_bundle(
        source_row, cutoff=cutoff, mode=mode,
        reconstruction_available_at=reconstruction_fixture_at, path="match")

    raw_evidence = evidence_rows
    if raw_evidence is None:
        raw_evidence = source_row.get("source_observations", source_row.get("observations", []))
    if raw_evidence is None:
        raw_evidence = []
    evidence = [_mapping(item, f"evidence_rows[{index}]") for index, item in enumerate(raw_evidence)]

    # Target fixture identity must have a source provenance row of its own.
    # Add one when an export contains no explicit fixture evidence; a missing
    # clock remains unknown and is reported rather than backdated.
    fixture_source_id = str(source_row.get("observation_id") or source_row.get("source_id") or f"fixture:{match_id}")
    has_fixture = any(str(item.get("source_id") or item.get("evidence_id") or "") == fixture_source_id
                      or str(item.get("kind", "")).casefold() in {"fixture", "fixture_identity"}
                      for item in evidence)
    if not has_fixture:
        evidence.insert(0, {
            "source_id": fixture_source_id,
            "kind": "fixture_identity",
            "observed_at": source_row.get("observed_at"),
            "ingested_at": source_row.get("ingested_at"),
            "available_at": source_row.get("available_at", source_row.get("operational_available_at")),
            "values": {
                "match_id": match_id,
                "kickoff": _iso(kickoff),
                "competition_id": source_row.get("competition_id"),
                "home_team_id": source_row.get("home_team_id"),
                "away_team_id": source_row.get("away_team_id"),
            },
            "content_hash": source_row.get("content_hash", source_row.get("data_version")),
        })

    observations: list[dict[str, Any]] = []
    unknown_clock_rows: list[str] = list(match_clock_issues)
    future_rows: list[str] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(evidence):
        source_id = _text(item.get("source_id", item.get("evidence_id", item.get("version_id"))),
                          f"evidence_rows[{index}].source_id")
        assert source_id is not None
        if source_id in seen_ids:
            raise ValueError(f"duplicate evidence source_id: {source_id}")
        seen_ids.add(source_id)
        if mode == HISTORICAL:
            reconstructed = _time(item.get("reconstructed_available_at"),
                                  f"evidence_rows[{index}].reconstructed_available_at")
            # The fixture identity is the sole record that may inherit the
            # explicit fixture-lead assumption. Injury, odds, xG and news
            # observations require their own reconstructed publication clock;
            # a final warehouse ingest timestamp is never a substitute.
            is_fixture = (str(item.get("kind", "")).casefold() in
                          {"fixture", "fixture_identity"} or source_id == fixture_source_id)
            if reconstructed is None and is_fixture:
                reconstructed = reconstruction_fixture_at
        else:
            reconstructed = None
        clocks, issues = _clock_bundle(item, cutoff=cutoff, mode=mode,
                                       reconstruction_available_at=reconstructed,
                                       path=f"evidence_rows[{index}]")
        if ("unknown_missing" in clocks["availability_basis"] or
                (clocks["observed_at"] is None or clocks["ingested_at"] is None)):
            unknown_clock_rows.append(source_id)
        if any("after_cutoff" in issue for issue in issues):
            future_rows.append(source_id)
        obs: dict[str, Any] = {
            "source_id": source_id,
            "kind": item.get("kind", item.get("source", "platform")),
            "observed_at": clocks["observed_at"],
            "ingested_at": clocks["ingested_at"],
            "available_at": clocks["available_at"],
            "actual_available_at": clocks["actual_available_at"],
            "operational_available_at": clocks["operational_available_at"],
            "availability_basis": clocks["availability_basis"],
            "information_mode": mode,
            "quality": item.get("quality", item.get("status", "observed")),
            "values": _record_values(item),
        }
        if mode == HISTORICAL:
            obs["reconstructed_available_at"] = clocks["reconstructed_available_at"]
            obs["reconstruction_clock_is_assumed"] = True
        digest = item.get("content_hash", item.get("content_digest", item.get("digest")))
        if digest is not None:
            obs["content_digest"] = str(digest)
        if item.get("observation_id") is not None:
            obs["observation_id"] = str(item["observation_id"])
        observations.append(obs)

    row: dict[str, Any] = {
        "match_id": match_id,
        "kickoff": _iso(kickoff),
        "prediction_cutoff": _iso(cutoff),
        "information_mode": mode,
        "source": source_row.get("source", "platform"),
        "competition_id": source_row.get("competition_id"),
        "competition_name": source_row.get("competition_name"),
        "home_team_id": source_row.get("home_team_id"),
        "away_team_id": source_row.get("away_team_id"),
        "home_team_name": source_row.get("home_team_name"),
        "away_team_name": source_row.get("away_team_name"),
        # Preserve actual match clocks as first-class fields.
        "observed_at": _iso(_time(source_row.get("observed_at"), "match.observed_at")),
        "ingested_at": _iso(_time(source_row.get("ingested_at"), "match.ingested_at")),
        "available_at": actual_match_clock["available_at"],
        "actual_available_at": actual_match_clock["actual_available_at"],
        "operational_available_at": actual_match_clock["actual_available_at"],
        "source_observations": observations,
        "candidate_menu": _candidate_menu(candidate_menu if candidate_menu is not None else source_row.get("candidate_menu")),
        "candidate_generation_complete": source_row.get("candidate_generation_complete"),
        "known": deepcopy(source_row.get("known", {})) if isinstance(source_row.get("known", {}), Mapping) else {},
        "historical_assumptions": assumptions if mode == HISTORICAL else None,
        # Result fields are retained for the frozen builder's sidecar only.
        "home_goals": source_row.get("home_goals", source_row.get("home_score")),
        "away_goals": source_row.get("away_goals", source_row.get("away_score")),
        "status": source_row.get("status"),
        "score_period": source_row.get("score_period"),
        "label_status": source_row.get("label_status"),
        "result_available_at": source_row.get("result_available_at", source_row.get("label_available_at")),
        "adapter_version": VERSION,
    }
    if mode == HISTORICAL:
        row["fixture_available_at"] = _iso(reconstruction_fixture_at)
        row["reconstructed_fixture_available_at"] = _iso(reconstruction_fixture_at)
    required_time_keys = {"kickoff", "prediction_cutoff", "observed_at",
                          "ingested_at", "available_at", "operational_available_at"}
    row = {key: value for key, value in row.items()
           if value is not None or key in required_time_keys}
    fixture_obs = next((item for item in observations
                        if item["source_id"] == fixture_source_id or
                        str(item.get("kind", "")).casefold() in {"fixture", "fixture_identity"}), None)
    fixture_cutoff_visible = bool(fixture_obs and fixture_obs.get("available_at") is not None and
                                  _time(fixture_obs["available_at"], "fixture.available_at") <= cutoff)
    audit = {
        "adapter_version": VERSION,
        "information_mode": mode,
        "prediction_cutoff": _iso(cutoff),
        "kickoff": _iso(kickoff),
        "operational_availability_basis": "max(observed_at,ingested_at,available_at)",
        "actual_clocks_preserved": True,
        "today_ingest_clock_used_as_historical_visibility": False,
        "unknown_clock_rows": sorted(set(unknown_clock_rows)),
        "future_evidence_source_ids": sorted(set(future_rows)),
        "target_fixture_identity_source_id": fixture_source_id,
        "target_fixture_identity_evidence_present": bool(fixture_obs),
        "target_fixture_identity_cutoff_visible": fixture_cutoff_visible,
        "historical_assumptions": assumptions if mode == HISTORICAL else None,
        "result_fields_passed_for_sidecar_only": True,
        "source_observation_count": len(observations),
        "source_row_digest": _digest(row),
    }
    return {"match": row, "audit": audit}


def build_platform_asof_episode(
    match: Mapping[str, Any] | Any,
    evidence_rows: Iterable[Mapping[str, Any] | Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Adapt platform rows, then invoke the frozen actor-safe builder."""
    adapted = platform_match_to_episode_row(match, evidence_rows, **kwargs)
    row = adapted["match"]
    audit = adapted["audit"]
    if not audit["target_fixture_identity_cutoff_visible"]:
        raise ValueError("target fixture identity has no cutoff-visible evidence")
    replay_row = deepcopy(row)
    # Historical reconstruction is a separate view.  Actual PlatformReader
    # clocks stay in ``adapted['match']`` and ``adapted['audit']``; removing
    # them from this replay projection prevents a current ingest timestamp (or
    # a late warehouse observation) from making a reconstruction claim.
    if row.get("information_mode") == HISTORICAL:
        for observation in replay_row.get("source_observations", []):
            observation.pop("observed_at", None)
            observation.pop("ingested_at", None)
            observation.pop("operational_available_at", None)
            observation.pop("actual_available_at", None)
            observation["reconstruction_clock_is_assumed"] = True
    result = build_historical_asof_episode(
        replay_row,
        prediction_cutoff=replay_row["prediction_cutoff"],
        candidate_catalog_complete=replay_row.get("candidate_generation_complete"),
    )
    result["platform_adapter"] = audit
    return result


__all__ = [
    "VERSION", "platform_match_to_episode_row", "build_platform_asof_episode",
]
