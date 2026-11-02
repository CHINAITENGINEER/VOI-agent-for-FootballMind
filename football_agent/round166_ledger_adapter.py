"""Round166 adapter from the verified action ledger to the Round165 contract.

The action ledger already records the actor's selection boundary, declared
value and resource reservations.  It does not, by itself, prove that a
posterior prediction changed, that a returned observation was available before
the cutoff, or that a terminal stop was justified.  This adapter joins only
fields that are explicitly present in the ledger (or in the optional
``episode_evidence`` extension) and delegates completeness checks to the
Round165 contract.

It is intentionally a projection layer.  It does not call providers, infer
missing timestamps, manufacture predictions, or convert resource reservations
into decision-loss units.  A normal pre-Round166 ledger therefore remains a
useful but partial episode until the runtime records the additional evidence.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from .round165_episode_evidence_contract import validate_round165_episode_evidence


VERSION = "round166-action-ledger-adapter-v1"
_QUERY_ACTIONS = frozenset({"search_web", "read_url", "query_acquired"})
_TERMINAL_ACTIONS = frozenset({"stop", "defer", "ask_user"})
_RESULT_KEYS = frozenset({"result", "outcome", "score", "goals", "label", "loss",
                          "home_score", "away_score", "actual_result", "sequence_losses",
                          "action_returns"})


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _evidence(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return only the explicit per-action evidence extension."""
    raw = entry.get("episode_evidence")
    return raw if isinstance(raw, Mapping) else {}


def _enriched_entry(entry: Mapping[str, Any],
                    trace_enrichment: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    """Overlay explicitly supplied evidence without changing the ledger.

    Enrichment is keyed by the stable selection operation ID (with entry ID as
    a fallback).  Only the ``episode_evidence`` namespace is accepted; this
    prevents an external evaluator from silently rewriting the actor ledger.
    """
    if not isinstance(trace_enrichment, Mapping):
        return dict(entry)
    key = entry.get("selection_operation_id") or entry.get("entry_id")
    extra = trace_enrichment.get(key) if key is not None else None
    if not isinstance(extra, Mapping):
        return dict(entry)
    current = dict(entry)
    evidence = dict(_evidence(entry))
    evidence.update(dict(extra))
    current["episode_evidence"] = evidence
    return current


def _field(entry: Mapping[str, Any], key: str, default: Any = None) -> Any:
    evidence = _evidence(entry)
    if key in evidence:
        return evidence[key]
    return entry.get(key, default)


def _request_id(entry: Mapping[str, Any]) -> str | None:
    explicit = _field(entry, "request_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    acquisition_id = entry.get("acquisition_id")
    if isinstance(acquisition_id, str) and acquisition_id.strip():
        return acquisition_id
    request_ids = entry.get("request_ids")
    if isinstance(request_ids, Sequence) and not isinstance(request_ids, (str, bytes)):
        if len(request_ids) > 1:
            # Multiple transport requests form one acquisition.  Binding the
            # first HTTP ID would lose the action-level identity.
            return None
        for value in request_ids:
            if isinstance(value, str) and value.strip():
                return value
    local = entry.get("local_read_operation_id")
    return local if isinstance(local, str) and local.strip() else None


def _candidate_rows(entries: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for entry in entries:
        if entry.get("action") not in _QUERY_ACTIONS:
            # Terminal rows may carry an opaque historical candidate ID.  It
            # is not part of the information-action menu.
            continue
        assessments = entry.get("candidate_value_assessments")
        if not isinstance(assessments, list):
            assessments = []
        for assessment in assessments:
            if not isinstance(assessment, Mapping):
                reasons.append("candidate_assessment_not_mapping")
                continue
            candidate_id = assessment.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                reasons.append("candidate_assessment_id_missing")
                continue
            value = assessment.get("value") if isinstance(assessment.get("value"), Mapping) else assessment
            row = rows.setdefault(candidate_id, {"candidate_id": candidate_id})
            if "cost" in value:
                if "cost" in row and row["cost"] != value.get("cost"):
                    reasons.append(f"candidate_cost_conflict:{candidate_id}")
                else:
                    row["cost"] = value.get("cost")
            if "source_id" in assessment:
                if "source_id" in row and row["source_id"] != assessment.get("source_id"):
                    reasons.append(f"candidate_source_conflict:{candidate_id}")
                else:
                    row["source_id"] = assessment.get("source_id")
        candidate_id = entry.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id.strip() and candidate_id not in rows:
            # The selected candidate is retained, but its missing menu cost is
            # left explicit so the Round165 validator can mark the episode
            # incomplete instead of silently assigning zero.
            rows[candidate_id] = {"candidate_id": candidate_id}
    return list(rows.values()), sorted(set(reasons))


def _public_snapshot(entries: Sequence[Mapping[str, Any]], target_key: Any) -> dict[str, Any]:
    """Build a result-free public snapshot bound to the ledger projections."""
    digests = [entry.get("pre_action_snapshot_digest") for entry in entries
               if isinstance(entry.get("pre_action_snapshot_digest"), str)]
    return {
        "target_key": target_key,
        "pre_action_snapshot_digests": list(dict.fromkeys(digests)),
        "projection": "round166_action_value_ledger",
    }


def _forbidden_result_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).casefold() in _RESULT_KEYS or _forbidden_result_key(child)
                   for key, child in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_forbidden_result_key(child) for child in value)
    return False


def _stop_reason(entry: Mapping[str, Any]) -> str | None:
    explicit = _field(entry, "stop_reason")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    formal = entry.get("formal_query_value_decision")
    if isinstance(formal, Mapping):
        certificate = formal.get("stop_certificate")
        if isinstance(certificate, Mapping):
            for key in ("reason", "stop_reason", "basis"):
                value = certificate.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return None


def _trace_for_entry(entry: Mapping[str, Any], position: int, *, episode_id: str,
                     target_key: str) -> dict[str, Any]:
    action = entry.get("action")
    evidence = _evidence(entry)
    if action in _TERMINAL_ACTIONS:
        terminal_id = "STOP" if action == "stop" else (
            "DEFER" if action == "defer" else "USER_INPUT_REQUIRED")
        trace: dict[str, Any] = {
            "episode_id": episode_id,
            "target_key": target_key,
            "step": position,
            "action_id": terminal_id,
            "candidate_id": "STOP",
            "request_id": _request_id(entry),
            # Selection time is not a source-observation time.  Leave this
            # absent unless the runtime explicitly records a terminal capture
            # clock under episode_evidence.
            "captured_at": _field(entry, "captured_at"),
            "expected_cost": 0,
            "state_before_digest": _field(entry, "state_before_digest",
                                           entry.get("pre_action_snapshot_digest")),
            "state_after_digest": _field(entry, "state_after_digest"),
            "stop_reason": _stop_reason(entry),
        }
        prediction_before = _field(entry, "prediction_before")
        if prediction_before is not None:
            trace["prediction_before"] = deepcopy(prediction_before)
        return trace

    return {
        "episode_id": episode_id,
        "target_key": target_key,
        "step": position,
        "action_id": _field(entry, "action_id", entry.get("candidate_id")),
        "candidate_id": entry.get("candidate_id"),
        "request_id": _request_id(entry),
        "captured_at": _field(entry, "captured_at"),
        "expected_cost": entry.get("expected_cost"),
        "state_before_digest": _field(entry, "state_before_digest",
                                       entry.get("pre_action_snapshot_digest")),
        "state_after_digest": _field(entry, "state_after_digest"),
        "prediction_before": deepcopy(_field(entry, "prediction_before")),
        "prediction_after": deepcopy(_field(entry, "prediction_after")),
        "observed_outcome": deepcopy(_field(entry, "observed_outcome")),
    }


def _aggregate_for_entry(entry: Mapping[str, Any], trace: Mapping[str, Any], *,
                         episode_id: str, target_key: str, cutoff: Any) -> dict[str, Any] | None:
    evidence = _evidence(entry)
    explicit = evidence.get("aggregate")
    if isinstance(explicit, Mapping):
        aggregate = deepcopy(dict(explicit))
        aggregate.setdefault("episode_id", episode_id)
        aggregate.setdefault("target_key", target_key)
        aggregate.setdefault("candidate_id", entry.get("candidate_id"))
        aggregate.setdefault("request_id", trace.get("request_id"))
        aggregate.setdefault("prediction_cutoff", cutoff)
        return aggregate
    # A result digest or a technical ``return_observation`` is insufficient to
    # establish source lineage and cutoff timing.  The runtime must explicitly
    # supply an aggregate record in ``episode_evidence`` (or via enrichment).
    # Returning ``None`` keeps the Round165 report partial instead of turning a
    # successful transport into usable evidence.
    return None


def adapt_action_value_ledger(
    ledger: Mapping[str, Any], *,
    candidate_menu: Sequence[Mapping[str, Any]] | None = None,
    trace_enrichment: Mapping[str, Mapping[str, Any]] | None = None,
    evaluator_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map one action-value ledger into a Round165 episode and validate it.

    The returned ``episode`` is safe to expose to the actor-side evaluator;
    the optional sidecar is passed only to the isolated validator and is never
    copied into the public episode.  Missing fields stay missing and therefore
    produce a partial report.
    """
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("entries"), list):
        return {"schema_version": VERSION, "status": "INVALID_LEDGER",
                "numeric_voi_training_eligible": False,
                "reasons": ["ledger.entries must be a list"]}
    original_entries = [row for row in ledger["entries"] if isinstance(row, Mapping)]
    raw_entries = [_enriched_entry(row, trace_enrichment) for row in original_entries]
    reasons: list[str] = []
    if len(raw_entries) != len(ledger["entries"]):
        reasons.append("ledger_entry_not_mapping")
    if not raw_entries:
        return {"schema_version": VERSION, "status": "INVALID_LEDGER",
                "numeric_voi_training_eligible": False,
                "reasons": ["ledger.entries is empty"]}
    episode_id = ledger.get("episode_id") or ledger.get("run_id")
    target_values = {entry.get("target_key") for entry in raw_entries if entry.get("target_key")}
    cutoff_values = {entry.get("prediction_cutoff") for entry in raw_entries if entry.get("prediction_cutoff")}
    for index, (entry, original_entry) in enumerate(zip(raw_entries, original_entries)):
        if not isinstance(entry.get("target_key"), str) or not entry.get("target_key", "").strip():
            reasons.append(f"entry_{index}_target_key_missing")
        if not isinstance(entry.get("prediction_cutoff"), str) or not entry.get("prediction_cutoff", "").strip():
            reasons.append(f"entry_{index}_prediction_cutoff_missing")
        sequence = entry.get("sequence")
        if sequence is None:
            reasons.append(f"entry_{index}_sequence_missing")
        elif sequence != index + 1:
            reasons.append(f"entry_{index}_sequence_not_contiguous")
        evidence = _evidence(original_entry)
        if evidence:
            bound = entry.get("episode_evidence_digest")
            if bound != _digest(evidence):
                reasons.append(f"entry_{index}_episode_evidence_unbound")
    if trace_enrichment is not None:
        if ledger.get("episode_evidence_digest") != _digest(trace_enrichment):
            reasons.append("trace_enrichment_unbound")
    target_key = next(iter(target_values), None)
    cutoff = next(iter(cutoff_values), None)
    if len(target_values) > 1:
        reasons.append("ledger_target_key_inconsistent")
    if len(cutoff_values) > 1:
        reasons.append("ledger_prediction_cutoff_inconsistent")
    if not isinstance(episode_id, str) or not episode_id.strip():
        episode_id = None
        reasons.append("ledger_episode_id_missing")
    if not isinstance(target_key, str) or not target_key.strip():
        target_key = None
        reasons.append("ledger_target_key_missing")
    if not isinstance(cutoff, str) or not cutoff.strip():
        cutoff = None
        reasons.append("ledger_prediction_cutoff_missing")
    if candidate_menu is not None:
        if not isinstance(candidate_menu, Sequence) or isinstance(candidate_menu, (str, bytes)):
            menu, menu_reasons = [], ["candidate_menu_not_sequence"]
        else:
            menu = [deepcopy(dict(row)) for row in candidate_menu if isinstance(row, Mapping)]
            menu_reasons = [] if len(menu) == len(candidate_menu) else ["candidate_menu_row_not_mapping"]
    else:
        menu, menu_reasons = _candidate_rows(raw_entries)
    reasons.extend(menu_reasons)
    complete_menu = all(entry.get("candidate_generation_complete") is True for entry in raw_entries)
    if not complete_menu:
        reasons.append("candidate_generation_incomplete")
    episode = {
        "episode_id": episode_id,
        "target_key": target_key,
        "prediction_cutoff": cutoff,
        "match_snapshot": _public_snapshot(raw_entries, target_key),
        "candidate_menu": menu,
        "candidate_generation_complete": complete_menu,
        "public_snapshot_digest": _digest(_public_snapshot(raw_entries, target_key)),
        "ledger_binding": {
            "run_id": ledger.get("run_id"),
            "ledger_digest": ledger.get("ledger_digest"),
            "state_digest": ledger.get("state_digest"),
            "entry_ids": [entry.get("entry_id") for entry in raw_entries],
            "entry_digests": [entry.get("entry_digest") for entry in raw_entries],
        },
    }
    traces: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    if episode_id and target_key:
        for position, entry in enumerate(raw_entries):
            trace = _trace_for_entry(entry, position, episode_id=episode_id, target_key=target_key)
            traces.append(trace)
            aggregate = _aggregate_for_entry(entry, trace, episode_id=episode_id,
                                              target_key=target_key, cutoff=cutoff)
            if aggregate is not None:
                aggregates.append(aggregate)
            if entry.get("action") in _QUERY_ACTIONS:
                if _field(entry, "captured_at") is None:
                    reasons.append(f"entry_{position}_capture_clock_missing")
                if _field(entry, "prediction_before") is None or _field(entry, "prediction_after") is None:
                    reasons.append(f"entry_{position}_prediction_trace_missing")
                if not isinstance(_evidence(entry).get("aggregate"), Mapping):
                    reasons.append(f"entry_{position}_aggregate_evidence_missing")
                if _forbidden_result_key(_field(entry, "observed_outcome")):
                    reasons.append(f"entry_{position}_observed_outcome_result_leak")
    report = validate_round165_episode_evidence(
        episode, traces, aggregates, evaluator_sidecar=evaluator_sidecar)
    terminal_actions = [entry.get("action") for entry in raw_entries
                        if entry.get("action") in _TERMINAL_ACTIONS]
    if any(action == "ask_user" for action in terminal_actions):
        reasons.append("ask_user_terminal_is_not_autonomous_stop")
    if any(action == "defer" for action in terminal_actions):
        reasons.append("defer_terminal_is_not_autonomous_stop")
    if not complete_menu:
        reasons.append("open_world_candidate_menu_not_complete")
    if not any(entry.get("action") in _QUERY_ACTIONS for entry in raw_entries):
        reasons.append("no_query_action_in_episode")
    if terminal_actions and raw_entries[-1].get("action") not in _TERMINAL_ACTIONS:
        reasons.append("terminal_action_not_last")
    for index, entry in enumerate(raw_entries):
        requests = entry.get("request_ids")
        if isinstance(requests, Sequence) and not isinstance(requests, (str, bytes)) and len(requests) > 1:
            if not isinstance(entry.get("acquisition_id"), str) or not entry.get("acquisition_id", "").strip():
                reasons.append(f"entry_{index}_multi_request_action_id_missing")
    if terminal_actions:
        terminal = raw_entries[-1]
        if terminal.get("action") in _TERMINAL_ACTIONS:
            terminal_evidence = _evidence(terminal)
            if "remaining_candidate_ids" not in terminal_evidence:
                reasons.append("stop_remaining_candidates_missing")
    if reasons:
        report["status"] = "PARTIAL_ACTION_EVIDENCE"
        report["eligible_for_replay"] = False
        report["reasons"] = sorted(set(list(report.get("reasons", [])) + reasons))
    report["numeric_voi_training_eligible"] = False
    status = "MAPPED" if report.get("status") != "INVALID" else "INVALID_EPISODE"
    cost_projection = [{
        "selection_operation_id": entry.get("selection_operation_id"),
        "declared_cost": entry.get("expected_cost"),
        "declared_cost_unit": _field(entry, "declared_cost_unit"),
        "selection_resource_cost": deepcopy(entry.get("selection_resource_cost", {})),
        "acquisition_resource_cost": deepcopy(entry.get("resource_cost", {})),
        "cost_basis": "preserved_separately; no conversion to decision loss",
    } for entry in raw_entries]
    return {
        "schema_version": VERSION,
        "status": status,
        "mapping_reasons": sorted(set(reasons)),
        "episode": episode,
        "traces": traces,
        "aggregates": aggregates,
        "report": report,
        "cost_projection": cost_projection,
        "ledger_cost_basis": _mapping(ledger.get("summary")).get("cost_basis"),
        "numeric_voi_training_eligible": False,
    }


__all__ = ["VERSION", "adapt_action_value_ledger"]
