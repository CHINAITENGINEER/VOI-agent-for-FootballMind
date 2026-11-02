"""Strict registry and join boundary for independent historical results.

Round164 does not collect a new provider.  It turns an already frozen result
archive into evaluator-only records and requires an explicit cross-source
identity link before a result can be attached to an actor-visible episode.

The registry keeps two timing modes separate:

* ``INDEPENDENT_GOLD`` is reserved for an operationally observed result
  source whose publication/availability clock is part of the evidence chain.
* ``INDEPENDENT_GOLD_RECONSTRUCTED`` is a post-match label reconstructed from
  an archive.  It is useful for retrospective utility replay, but does not
  support a claim that the source was available online at the historical
  cutoff.

This module never projects a score into the public episode.  It returns an
evaluator result payload and an audit report that callers may pass to the
Round163 bridge explicitly.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .counterfactual_policy_replay import INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED
from .round163_historical_episode_bridge import public_evaluator_fields


VERSION = "round164-independent-result-registry-v3"
RESULT_SOURCE_ONLY = "RESULT_SOURCE_ONLY"
UNMATCHED = "UNMATCHED"
REJECTED = "REJECTED"

_DIGEST_LENGTH = 64
_SUPPORTED_MODES = {"operational", "historical_reconstruction"}


def reconstruct_spftll_fixture_public(
    payload: bytes,
    observation: Mapping[str, Any],
    *,
    match_id: str,
    prediction_lead_hours: int = 1,
) -> dict[str, Any]:
    """Project fixture identity from its own verified archive, preserving clocks.

    This is a retrospective identity projection, not a historical as-of
    snapshot. The scenario cutoff is computed from kickoff, but the observed
    and available clocks always retain the actual collection time. Scores,
    match outcomes and post-match status are never projected. Result records
    and cross-source review decisions are not inputs to this function.
    """
    if not isinstance(payload, bytes) or not isinstance(observation, Mapping):
        raise ValueError("payload bytes and observation metadata are required")
    if observation.get("source") != "spftll" or observation.get("http_status") != 200:
        raise ValueError("a successful spftll observation is required")
    if observation.get("error_class") is not None:
        raise ValueError("failed observations are not fixture evidence")
    observation_id = _text(observation.get("id"), "observation.id")
    observed = _time(observation.get("observed_at"), "observation.observed_at")
    payload_digest = _digest_or_none(observation.get("content_hash"), "observation.content_hash")
    if payload_digest is None or hashlib.sha256(payload).hexdigest() != payload_digest:
        raise ValueError("fixture payload digest differs from observation metadata")
    if type(prediction_lead_hours) is not int or prediction_lead_hours < 0:
        raise ValueError("prediction_lead_hours must be a nonnegative integer")
    match_id = _text(match_id, "match_id")
    body = json.loads(payload)
    if not isinstance(body, Mapping) or body.get("code") != 0 or not isinstance(body.get("data"), Mapping):
        raise ValueError("a successful spftll detail object is required")
    data = body["data"]
    rows = [data]
    for key in ("battleList", "homeBattleList", "awayBattleList"):
        history = data.get(key) or []
        if not isinstance(history, list) or not all(isinstance(item, Mapping) for item in history):
            raise ValueError(f"{key} must contain records")
        rows.extend(history)
    identities = []
    for row in rows:
        if str(row.get("matchId")) != match_id:
            continue
        raw_time = _text(row.get("matchTime"), "fixture.matchTime")
        kickoff = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        home_id, away_id, competition_id = (row.get(key) for key in ("homeId", "awayId", "leagueId"))
        if any(value is None or isinstance(value, bool) for value in (home_id, away_id, competition_id)):
            raise ValueError("fixture requires source team and competition IDs")
        if str(home_id) == str(away_id):
            raise ValueError("fixture home and away team IDs must differ")
        identities.append({
            "source": "spftll", "match_id": match_id, "kickoff": _iso(kickoff),
            "home_team_id": str(home_id), "away_team_id": str(away_id),
            "home_team_name": _text(row.get("homeChs") or row.get("home"), "fixture.home"),
            "away_team_name": _text(row.get("awayChs") or row.get("away"), "fixture.away"),
            "competition_id": str(competition_id),
            "competition_name": _text(row.get("leagueChs") or row.get("league"), "fixture.league"),
        })
    if not identities:
        raise ValueError("fixture is absent from its source observation")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("source observation has conflicting fixture identities")
    identity = identities[0]
    cutoff = _time(identity["kickoff"], "kickoff") - timedelta(hours=prediction_lead_hours)
    return {
        "schema_version": "reconstructed-fixture-public-v1",
        "target_key": f"spftll:{match_id}",
        "kickoff": identity["kickoff"],
        "prediction_cutoff": _iso(cutoff),
        "prediction_cutoff_basis": f"scenario_kickoff_minus_{prediction_lead_hours}_hours",
        "information_mode": "historical_reconstruction",
        "online_availability_claim": False,
        "match_snapshot": {
            **identity,
            "information_mode": "historical_reconstruction",
            "source_observations": [{
                "source_id": f"spftll:{match_id}:fixture",
                "observation_id": observation_id,
                "kind": "fixture_identity",
                "observed_at": _iso(observed),
                "available_at": _iso(observed),
                "availability_basis": "source_observed_at_fallback",
                "content_digest": payload_digest,
                "values": {"source_match_id": match_id},
            }],
        },
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _time(value: Any, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = _text(value, field)
    assert text is not None
    if len(text) != _DIGEST_LENGTH or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return text


def _goals(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _manifest_mode(manifest: Mapping[str, Any] | None) -> str | None:
    if not isinstance(manifest, Mapping):
        return None
    mode = manifest.get("information_mode")
    return mode if isinstance(mode, str) else None


def _manifest_payload_digest(manifest: Mapping[str, Any] | None,
                             row: Mapping[str, Any]) -> str | None:
    if not isinstance(manifest, Mapping):
        return None
    evidence = manifest.get("source_evidence")
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        return None
    records = [item for item in evidence if isinstance(item, Mapping)]
    observation_id = row.get("observation_id")
    if observation_id is not None:
        records = [item for item in records if item.get("observation_id") == observation_id]
        if not records:
            raise ValueError("result observation_id is absent from manifest source_evidence")
    if not records:
        return None
    if len(records) != 1:
        raise ValueError("manifest source_evidence is ambiguous; observation_id is required")
    record = records[0]
    if record.get("observed_at") is not None:
        if _time(record["observed_at"], "manifest.observed_at") != _time(
            row.get("observed_at", row.get("first_observed_at")), "observed_at"
        ):
            raise ValueError("manifest observation clock differs from result row")
    return _digest_or_none(record.get("content_hash"), "manifest.source_evidence.content_hash")


def registry_entry_from_row(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    mapping_evidence_digest: str | None = None,
) -> dict[str, Any]:
    """Normalize one result row into an evaluator-only registry entry."""
    if not isinstance(row, Mapping):
        raise ValueError("result row must be a mapping")
    source = _text(row.get("source"), "source")
    source_id = _text(row.get("source_id"), "source_id")
    match_id = _text(row.get("match_id"), "match_id")
    assert source is not None and source_id is not None and match_id is not None
    kickoff = _time(row.get("kickoff"), "kickoff")
    observed_at = _time(row.get("observed_at", row.get("first_observed_at")), "observed_at")
    available_at = _time(row.get("result_available_at", row.get("label_available_at")),
                         "result_available_at")
    if observed_at < kickoff:
        raise ValueError("result observed_at must follow kickoff")
    if available_at <= kickoff:
        raise ValueError("result_available_at must follow kickoff")
    home_goals = _goals(row.get("home_goals", row.get("home_score")), "home_goals")
    away_goals = _goals(row.get("away_goals", row.get("away_score")), "away_goals")
    score_period = _text(row.get("score_period"), "score_period")
    status = _text(row.get("status"), "status")
    if status not in {"finished", "completed", "final"}:
        raise ValueError("result row must be final")
    if score_period != "regulation":
        raise ValueError("only regulation results are supported")
    home_id = _text(row.get("home_team_id"), "home_team_id")
    away_id = _text(row.get("away_team_id"), "away_team_id")
    if home_id == away_id:
        raise ValueError("home and away team IDs must differ")
    home_name = _text(row.get("home_team_name"), "home_team_name")
    away_name = _text(row.get("away_team_name"), "away_team_name")
    competition_id = _text(row.get("competition_id"), "competition_id", required=False)
    competition_name = _text(row.get("competition_name"), "competition_name", required=False)
    mode = row.get("information_mode", _manifest_mode(manifest))
    if mode not in _SUPPORTED_MODES:
        raise ValueError("information_mode must be operational or historical_reconstruction")
    if isinstance(manifest, Mapping):
        if manifest.get("source") not in (None, source):
            raise ValueError("manifest source differs from result row")
        if _manifest_mode(manifest) not in (None, mode):
            raise ValueError("manifest information_mode differs from result row")
    # available_at means availability to this system, as in the Round163
    # bridge.  A provider's earlier publication time is a separate clock.
    if mode == "operational" and available_at < observed_at:
        raise ValueError("operational result availability must follow observation")
    published_at = row.get("source_published_at")
    if published_at is not None:
        published_at = _time(published_at, "source_published_at")
        if not kickoff < published_at <= observed_at:
            raise ValueError("result source publication must follow kickoff and not follow observation")
    label_status = INDEPENDENT_GOLD if mode == "operational" else INDEPENDENT_GOLD_RECONSTRUCTED
    timing_basis = (
        "operational_result_availability"
        if mode == "operational"
        else "historical_reconstructed_result_delay"
    )
    manifest_digest = _manifest_payload_digest(manifest, row)
    explicit_digest = _digest_or_none(
        row.get("source_payload_digest", row.get("content_digest")), "source_payload_digest")
    if explicit_digest and manifest_digest and explicit_digest != manifest_digest:
        raise ValueError("source payload digest differs from manifest")
    source_payload_digest = explicit_digest or manifest_digest
    mapping_digest = _digest_or_none(mapping_evidence_digest, "mapping_evidence_digest")
    outcome = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
    identity = {
        "source": source,
        "source_id": source_id,
        "match_id": match_id,
        "kickoff": _iso(kickoff),
        "home_team_id": home_id,
        "away_team_id": away_id,
        "competition_id": competition_id,
    }
    entry: dict[str, Any] = {
        "schema_version": VERSION,
        "result_source": source,
        "result_source_id": f"{source}:{source_id}",
        "source_id": source_id,
        "match_id": match_id,
        "kickoff": _iso(kickoff),
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team_name": home_name,
        "away_team_name": away_name,
        "competition_id": competition_id,
        "competition_name": competition_name,
        "observed_at": _iso(observed_at),
        "available_at": _iso(available_at),
        "source_published_at": _iso(published_at) if published_at is not None else None,
        "information_mode": mode,
        "timing_basis": timing_basis,
        "label_status": label_status,
        "source_payload_digest": source_payload_digest,
        "observation_id": row.get("observation_id"),
        "mapping_evidence_digest": mapping_digest,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "outcome": outcome,
        "score_period": score_period,
        "source_revision_bias": manifest.get("source_revision_bias") if isinstance(manifest, Mapping) else None,
    }
    entry["identity_digest"] = digest(identity)
    entry["result_digest"] = digest({
        "match_id": match_id, "home_goals": home_goals, "away_goals": away_goals,
        "score_period": score_period, "source_id": source_id,
    })
    entry["record_digest"] = digest(entry)
    return entry


def build_registry(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
    mapping_evidence_by_source_id: Mapping[str, str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build a deterministic registry and retain every rejection reason."""
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_match_ids: set[str] = set()
    evidence = mapping_evidence_by_source_id or {}
    for index, row in enumerate(rows):
        try:
            entry = registry_entry_from_row(
                row,
                manifest=manifest,
                mapping_evidence_digest=evidence.get(str(row.get("source_id"))) if isinstance(row, Mapping) else None,
            )
            if entry["result_source_id"] in seen_source_ids:
                raise ValueError("duplicate result source_id")
            if entry["match_id"] in seen_match_ids:
                raise ValueError("duplicate result match_id")
            seen_source_ids.add(entry["result_source_id"])
            seen_match_ids.add(entry["match_id"])
            entries.append(entry)
        except (TypeError, ValueError) as exc:
            rejected.append({"row_index": index, "reason": str(exc)})
    if strict and rejected:
        raise ValueError(f"result registry rejected {len(rejected)} rows: {rejected[0]}")
    entries.sort(key=lambda item: (item["kickoff"], item["result_source_id"]))
    mode_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for item in entries:
        mode_counts[item["information_mode"]] = mode_counts.get(item["information_mode"], 0) + 1
        label_counts[item["label_status"]] = label_counts.get(item["label_status"], 0) + 1
    return {
        "schema_version": VERSION,
        "registry_id": digest({"entries": [item["record_digest"] for item in entries]}),
        "source": manifest.get("source") if isinstance(manifest, Mapping) else (entries[0]["result_source"] if entries else None),
        "dataset_version": manifest.get("dataset_version") if isinstance(manifest, Mapping) else None,
        "information_mode": manifest.get("information_mode") if isinstance(manifest, Mapping) else None,
        "entries": entries,
        "rejected": rejected,
        "stats": {
            "input_rows": len(entries) + len(rejected),
            "accepted_rows": len(entries),
            "rejected_rows": len(rejected),
            "mode_counts": mode_counts,
            "label_counts": label_counts,
        },
        "claims": {
            "operational_availability_claim": any(item["label_status"] == INDEPENDENT_GOLD for item in entries),
            "operational_result_timing_present": any(item["label_status"] == INDEPENDENT_GOLD for item in entries),
            "historical_reconstruction_present": any(item["label_status"] == INDEPENDENT_GOLD_RECONSTRUCTED for item in entries),
            "scores_are_evaluator_only": True,
        },
    }


def confirmed_match_mappings(
    rows: Iterable[Mapping[str, Any]], *,
    entity_mappings: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Normalize reviewed cross-source links; fuzzy candidates are excluded."""
    mappings: list[dict[str, Any]] = []
    actor_ids: set[str] = set()
    result_ids: set[str] = set()
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    if entity_mappings is not None:
        for row in entity_mappings:
            if row.get("source") != "understat" or row.get("status") != "confirmed" or row.get("valid_to") is not None:
                continue
            key = (str(row.get("entity_type")), str(row.get("source_id")))
            if key in entities:
                raise ValueError("ambiguous confirmed entity mapping")
            _text(row.get("entity_id"), "entity_mapping.entity_id")
            _time(row.get("valid_from"), "entity_mapping.valid_from")
            entities[key] = deepcopy(dict(row))
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"mapping[{index}] must be an object")
        if row.get("basis") != "both_reviewed_team_ids_same_home_away_competition_kickoff_regulation_score":
            continue
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"mapping[{index}].evidence is required")
        actor_match_id = _text(evidence.get("spftll_id"), f"mapping[{index}].spftll_id")
        result_source_id = _text(evidence.get("understat_id"), f"mapping[{index}].understat_id")
        actor_teams, result_teams = evidence.get("spftll_teams"), evidence.get("understat_teams")
        if (not isinstance(actor_teams, (list, tuple)) or len(actor_teams) != 2
                or not isinstance(result_teams, (list, tuple)) or len(result_teams) != 2
                or not all(isinstance(team, Mapping) for team in (*actor_teams, *result_teams))):
            raise ValueError(f"mapping[{index}] must contain ordered home/away teams")
        actor_home = _text(actor_teams[0].get("source_id"), f"mapping[{index}].actor_home_team_id")
        actor_away = _text(actor_teams[1].get("source_id"), f"mapping[{index}].actor_away_team_id")
        result_home = _text(result_teams[0].get("id"), f"mapping[{index}].result_home_team_id")
        result_away = _text(result_teams[1].get("id"), f"mapping[{index}].result_away_team_id")
        assert actor_match_id and result_source_id and actor_home and actor_away and result_home and result_away
        if actor_match_id in actor_ids or result_source_id in result_ids:
            raise ValueError("mapping IDs must be unique")
        actor_ids.add(actor_match_id)
        result_ids.add(result_source_id)
        evidence_digest = digest({
            "basis": row["basis"], "evidence": evidence,
            "understat_observation_id": row.get("understat_observation_id"),
        })
        mapping = {
            "schema_version": VERSION,
            "mapping_status": "confirmed",
            "actor_source": "spftll",
            "actor_match_id": actor_match_id,
            "actor_home_team_id": actor_home,
            "actor_away_team_id": actor_away,
            "result_source": "understat",
            "result_source_id": result_source_id,
            "result_home_team_id": result_home,
            "result_away_team_id": result_away,
            "kickoff": _iso(_time(evidence.get("kickoff"), f"mapping[{index}].kickoff")),
            "basis": row["basis"],
            "actor_observation_id": _text(evidence.get("spftll_observation_id"), f"mapping[{index}].spftll_observation_id"),
            "result_observation_id": _text(row.get("understat_observation_id"), f"mapping[{index}].understat_observation_id"),
            "observation_id": _text(row.get("understat_observation_id"), f"mapping[{index}].understat_observation_id"),
            "evidence_digest": evidence_digest,
            "mapping_uses_postmatch_evidence": True,
            "mapping_available_before_cutoff": False,
        }
        if entity_mappings is not None:
            required = {
                "result_home_canonical_team_id": ("team", result_home),
                "result_away_canonical_team_id": ("team", result_away),
                "result_canonical_match_id": ("match", result_source_id),
                "result_canonical_competition_id": ("competition", "EPL"),
            }
            bindings = {}
            for field, key in required.items():
                if key not in entities:
                    raise ValueError(f"missing confirmed entity mapping: {key}")
                binding = entities[key]
                mapping[field] = binding["entity_id"]
                bindings[field] = binding
            # Resolve from the independent frozen entity table, never from the
            # result row being checked.  The full evidence stays evaluator-only.
            mapping["entity_mapping_evidence"] = bindings
            mapping["entity_mapping_digest"] = digest(bindings)
        mapping["mapping_digest"] = digest(mapping)
        mappings.append(mapping)
    return tuple(sorted(mappings, key=lambda item: item["actor_match_id"]))


def join_result_to_public(
    public: Mapping[str, Any],
    entry: Mapping[str, Any],
    mapping: Mapping[str, Any],
    *,
    actor_source_ids: Iterable[str] = (),
    actor_content_digests: Iterable[str] = (),
    kickoff_tolerance_seconds: int = 60,
) -> dict[str, Any]:
    """Attempt one explicit, auditable result-to-episode join."""
    if not isinstance(public, Mapping) or not isinstance(entry, Mapping) or not isinstance(mapping, Mapping):
        raise ValueError("public, entry and mapping must be mappings")
    if type(kickoff_tolerance_seconds) is not int or kickoff_tolerance_seconds < 0:
        raise ValueError("kickoff_tolerance_seconds must be a nonnegative integer")
    snapshot = public.get("match_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("public.match_snapshot is required")
    actor_match_id = _text(snapshot.get("match_id"), "public.match_snapshot.match_id")
    actor_kickoff = _time(public.get("kickoff", snapshot.get("kickoff")), "public.kickoff")
    cutoff = _time(public.get("prediction_cutoff"), "public.prediction_cutoff")
    if mapping.get("mapping_status") != "confirmed":
        return {"accepted": False, "status": REJECTED, "reason": "mapping_not_confirmed"}
    checks: dict[str, Any] = {}
    checks["public_excludes_evaluator_fields"] = not public_evaluator_fields(public)
    checks["entry_integrity"] = entry.get("record_digest") == digest(
        {key: value for key, value in entry.items() if key != "record_digest"})
    checks["cutoff_not_after_kickoff"] = cutoff <= actor_kickoff
    if snapshot.get("kickoff") is not None:
        checks["public_kickoff_consistent"] = actor_kickoff == _time(snapshot["kickoff"], "snapshot.kickoff")
    checks["actor_match_id"] = actor_match_id == mapping.get("actor_match_id")
    if snapshot.get("source") is not None:
        checks["actor_source"] = snapshot["source"] == mapping.get("actor_source")
    if mapping.get("mapping_digest") is not None:
        checks["mapping_integrity"] = mapping["mapping_digest"] == digest(
            {key: value for key, value in mapping.items() if key != "mapping_digest"})
    if entry.get("mapping_evidence_digest") is not None:
        checks["mapping_evidence_bound"] = entry["mapping_evidence_digest"] == mapping.get("evidence_digest")
    checks["result_source"] = entry.get("result_source") == mapping.get("result_source")
    checks["result_source_id"] = str(entry.get("source_id")) == str(mapping.get("result_source_id"))
    checks["actor_home_team_id"] = str(snapshot.get("home_team_id")) == str(mapping.get("actor_home_team_id"))
    checks["actor_away_team_id"] = str(snapshot.get("away_team_id")) == str(mapping.get("actor_away_team_id"))
    expected_result_home = mapping.get("result_home_canonical_team_id", mapping.get("result_home_team_id"))
    expected_result_away = mapping.get("result_away_canonical_team_id", mapping.get("result_away_team_id"))
    checks["result_home_team_id"] = str(entry.get("home_team_id")) == str(expected_result_home)
    checks["result_away_team_id"] = str(entry.get("away_team_id")) == str(expected_result_away)
    if "result_home_canonical_team_id" in mapping or "result_away_canonical_team_id" in mapping:
        bindings = mapping.get("entity_mapping_evidence")
        checks["mapping_integrity"] = mapping.get("mapping_digest") == digest(
            {key: value for key, value in mapping.items() if key != "mapping_digest"})
        checks["entity_mapping_integrity"] = isinstance(bindings, Mapping) and (
            mapping.get("entity_mapping_digest") == digest(bindings))
        fields = {
            "result_home_canonical_team_id": ("team", mapping.get("result_home_team_id")),
            "result_away_canonical_team_id": ("team", mapping.get("result_away_team_id")),
            "result_canonical_match_id": ("match", mapping.get("result_source_id")),
            "result_canonical_competition_id": ("competition", "EPL"),
        }
        checks["entity_mapping_bindings"] = isinstance(bindings, Mapping) and all(
            isinstance(bindings.get(field), Mapping)
            and bindings[field].get("entity_type") == kind
            and bindings[field].get("source_id") == source_id
            and bindings[field].get("source") == mapping.get("result_source")
            and bindings[field].get("status") == "confirmed"
            and bindings[field].get("valid_to") is None
            and bindings[field].get("entity_id") == mapping.get(field)
            for field, (kind, source_id) in fields.items())
        checks["canonical_match_id"] = entry.get("match_id") == mapping.get("result_canonical_match_id")
        checks["canonical_competition_id"] = entry.get("competition_id") == mapping.get("result_canonical_competition_id")
    result_kickoff = _time(entry.get("kickoff"), "entry.kickoff")
    checks["kickoff_within_tolerance"] = abs((result_kickoff - actor_kickoff).total_seconds()) <= kickoff_tolerance_seconds
    if mapping.get("kickoff") is not None:
        checks["mapping_kickoff"] = abs((actor_kickoff - _time(mapping["kickoff"], "mapping.kickoff")).total_seconds()) <= kickoff_tolerance_seconds
    result_available = _time(entry.get("available_at"), "entry.available_at")
    checks["result_after_cutoff"] = result_available > cutoff
    observations = snapshot.get("source_observations", [])
    if not isinstance(observations, list) or not all(isinstance(item, Mapping) for item in observations):
        raise ValueError("source_observations must be a list of mappings")
    if mapping.get("actor_observation_id") is not None:
        checks["actor_observation_bound"] = any(
            item.get("observation_id") == mapping["actor_observation_id"] for item in observations)
    result_observation_id = mapping.get("result_observation_id", mapping.get("observation_id"))
    if result_observation_id is not None:
        checks["result_observation_bound"] = entry.get("observation_id") == result_observation_id
    actor_ids = {str(value) for value in actor_source_ids} | {str(item.get("source_id")) for item in observations}
    actor_digests = {str(value) for value in actor_content_digests} | {str(item.get("content_digest")) for item in observations}
    result_provider = str(entry.get("result_source"))
    checks["source_separate"] = (result_provider != mapping.get("actor_source")
        and result_provider != snapshot.get("source")
        and not any(value == str(entry.get("source_id")) or value.split(":", 1)[0] == result_provider for value in actor_ids))
    source_digest = entry.get("source_payload_digest")
    checks["source_digest_present"] = _digest_or_none(source_digest, "source_payload_digest") is not None
    checks["digest_separate"] = source_digest is not None and str(source_digest) not in actor_digests
    checks["entry_result_label"] = entry.get("label_status") in {INDEPENDENT_GOLD, INDEPENDENT_GOLD_RECONSTRUCTED}
    result_observed = _time(entry.get("observed_at"), "entry.observed_at")
    checks["result_clock_semantics"] = result_observed >= result_kickoff and result_available > result_kickoff and (
        (entry.get("label_status") == INDEPENDENT_GOLD
         and entry.get("information_mode") == "operational" and result_available >= result_observed)
        or (entry.get("label_status") == INDEPENDENT_GOLD_RECONSTRUCTED
            and entry.get("information_mode") == "historical_reconstruction"))
    failures = [name for name, passed in checks.items() if passed is not True]
    if failures:
        return {
            "accepted": False,
            "status": UNMATCHED if "actor_match_id" in failures else REJECTED,
            "reason": "join_checks_failed",
            "failed_checks": failures,
            "checks": checks,
            "actor_match_id": actor_match_id,
            "result_source_id": entry.get("result_source_id"),
        }
    independent_result = {
        "source_id": entry["result_source_id"],
        "observed_at": entry["observed_at"],
        "available_at": entry["available_at"],
        "content_digest": entry["source_payload_digest"],
        "home_goals": entry["home_goals"],
        "away_goals": entry["away_goals"],
        "score_period": entry["score_period"],
        "label_status": entry["label_status"],
        "timing_basis": entry["timing_basis"],
        "mapping_evidence_digest": mapping.get("evidence_digest"),
    }
    actor_cutoff_verified = bool(observations) and all(
        _time(item.get("observed_at"), "actor.observed_at")
        <= _time(item.get("available_at"), "actor.available_at") <= cutoff
        for item in observations
    )
    actor_provenance_verified = bool(observations) and all(
        isinstance(item.get("source_id"), str) and bool(item["source_id"].strip())
        and isinstance(item.get("observation_id"), str) and bool(item["observation_id"].strip())
        and _digest_or_none(item.get("content_digest"), "actor.content_digest") is not None
        for item in observations
    )
    mapping_cutoff_verified = (
        mapping.get("mapping_available_before_cutoff") is True
        and mapping.get("mapping_uses_postmatch_evidence") is False
        and mapping.get("available_at") is not None
        and mapping.get("observed_at") is not None
        and _time(mapping["observed_at"], "mapping.observed_at")
        <= _time(mapping["available_at"], "mapping.available_at") <= cutoff
    )
    if isinstance(mapping.get("entity_mapping_evidence"), Mapping):
        mapping_cutoff_verified = mapping_cutoff_verified and all(
            binding.get("valid_from") is not None
            and _time(binding["valid_from"], "entity_mapping.valid_from") <= cutoff
            for binding in mapping["entity_mapping_evidence"].values()
        )
    mapping_provenance_verified = (
        mapping.get("mapping_digest") is not None
        and mapping.get("evidence_digest") is not None
        and entry.get("mapping_evidence_digest") == mapping.get("evidence_digest")
        and checks.get("actor_observation_bound") is True
        and checks.get("result_observation_bound") is True
    )
    online = (
        entry["label_status"] == INDEPENDENT_GOLD
        and public.get("information_mode") != "historical_reconstruction"
        and snapshot.get("information_mode") != "historical_reconstruction"
        and actor_cutoff_verified and mapping_cutoff_verified
        and actor_provenance_verified and mapping_provenance_verified
    )
    return {
        "accepted": True,
        "status": entry["label_status"],
        "checks": checks,
        "mapping": deepcopy(dict(mapping)),
        "result_entry_digest": entry.get("record_digest"),
        "independent_result": independent_result,
        # Joining a label proves neither action coverage nor online replay.
        "online_availability_claim": online,
        "actor_cutoff_verified": actor_cutoff_verified,
        "mapping_cutoff_verified": mapping_cutoff_verified,
        "actor_provenance_verified": actor_provenance_verified,
        "mapping_provenance_verified": mapping_provenance_verified,
        "retrospective_result_binding": not online,
        "retrospective_utility_claim": False,
        "evaluation_scope": "operational_result_binding" if online else "retrospective_result_binding",
        "result_label_bound": True,
        "counterfactual_replay_ready": False,
        "numeric_voi_training_eligible": False,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL archive without opening a network or database connection."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{index + 1} must contain an object")
        rows.append(value)
    return rows


__all__ = [
    "VERSION", "INDEPENDENT_GOLD", "INDEPENDENT_GOLD_RECONSTRUCTED",
    "RESULT_SOURCE_ONLY", "UNMATCHED", "REJECTED", "digest",
    "registry_entry_from_row", "build_registry", "confirmed_match_mappings",
    "join_result_to_public", "load_jsonl", "reconstruct_spftll_fixture_public",
]
