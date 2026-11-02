"""Offline contract for evaluating autonomous information selection.

The runtime owns the public pre-action snapshot and the actor's provisional
assessment.  This module is deliberately a separate evaluator boundary: it
checks that an external evaluator can compare a selected action with a frozen
finite menu without leaking a later result into the selection state.

It does not call a provider, calculate match truth, or turn an actor judgement
into a training label.  A complete finite menu can be marked eligible for a
counterfactual audit; numeric VoI calibration remains a separate, explicit
research step.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re
from typing import Any, Iterable, Mapping


VERSION = "value-evaluation-contract-v1"
ADAPTER_VERSION = "value-evaluation-adapter-v1"
COUNTERFACTUAL_ELIGIBLE = "counterfactual_eligible"
PARTIAL_OBSERVATION = "partial_observation"
BLOCKED = "blocked"
INDEPENDENT_GOLD = "INDEPENDENT_EVALUATOR_GOLD"
ACTOR_PROVISIONAL = "ACTOR_PROVISIONAL_NOT_GOLD"
ALLOWED_LABEL_STATUS = {INDEPENDENT_GOLD, ACTOR_PROVISIONAL, "UNAVAILABLE"}
ALLOWED_DECISIONS = {"query", "stop", "defer"}
ALLOWED_LOSS_UNITS = {"log_loss", "brier", "decision_loss"}

# A public snapshot may describe what is known and which actions are available,
# but it must not carry a result that only becomes available after selection.
FORBIDDEN_PRE_ACTION_KEYS = {
    "outcome", "outcomes", "label", "labels", "target_label", "result",
    "score", "scores", "goals", "home_goals", "away_goals", "future_result",
    "hidden_return", "hidden_returns", "hidden_features", "full_snapshot",
    "post_state", "after_state", "response_body", "raw_data", "actual_result",
    "counterfactual_return", "decision_loss_after", "realized_gain",
}


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(body).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _finite_nonnegative(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _walk_forbidden(value: Any, path: str = "snapshot") -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_PRE_ACTION_KEYS:
                reasons.append({
                    "code": "PRE_ACTION_RESULT_LEAK",
                    "path": f"{path}/{key_text}",
                    "message": "A post-selection result field is present in the public pre-action snapshot",
                    "hard": True,
                })
            reasons.extend(_walk_forbidden(child, f"{path}/{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reasons.extend(_walk_forbidden(child, f"{path}/{index}"))
    return reasons


def _reason(code: str, path: str, message: str, *, hard: bool = True) -> dict[str, Any]:
    return {"code": code, "path": path, "message": message, "hard": hard}


def _validate_cost(value: Any, path: str, reasons: list[dict[str, Any]]) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or not value:
        reasons.append(_reason("INVALID_DECLARED_COST", path,
                               "Declared cost must be a nonempty named-resource mapping"))
        return
    for key, amount in value.items():
        if not _nonempty(key) or type(amount) is not int or amount < 0:
            reasons.append(_reason("INVALID_DECLARED_COST", f"{path}/{key}",
                                   "Each cost slot must have a nonnegative integer reservation"))


def _validate_pre_action(pre: Any, cutoff: datetime | None,
                         reasons: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(pre, Mapping):
        reasons.append(_reason("MISSING_PRE_ACTION", "pre_action",
                               "A frozen public pre-action snapshot is required"))
        return None
    required = {"snapshot", "snapshot_digest", "candidate_ids", "candidate_generation_complete",
                "selection_operation_id", "decision", "selected_candidate_id", "selected_at"}
    missing = sorted(required - set(pre))
    for key in missing:
        reasons.append(_reason("MISSING_PRE_ACTION_FIELD", f"pre_action/{key}",
                               "The pre-action binding is incomplete"))
    snapshot = pre.get("snapshot")
    if not isinstance(snapshot, Mapping):
        reasons.append(_reason("INVALID_PRE_ACTION_SNAPSHOT", "pre_action/snapshot",
                               "The public snapshot must be a JSON object"))
    else:
        reasons.extend(_walk_forbidden(snapshot))
    if not _is_digest(pre.get("snapshot_digest")):
        reasons.append(_reason("INVALID_PRE_ACTION_DIGEST", "pre_action/snapshot_digest",
                               "A lowercase SHA256 digest is required"))
    elif isinstance(snapshot, Mapping) and pre["snapshot_digest"] != _digest(snapshot):
        reasons.append(_reason("PRE_ACTION_DIGEST_MISMATCH", "pre_action/snapshot_digest",
                               "The digest does not bind the supplied public snapshot"))
    ids = pre.get("candidate_ids")
    if not isinstance(ids, list) or not ids or any(not _nonempty(item) for item in ids):
        reasons.append(_reason("INVALID_CANDIDATE_MENU", "pre_action/candidate_ids",
                               "A nonempty list of candidate IDs is required"))
        ids = []
    elif len(ids) != len(set(ids)):
        reasons.append(_reason("DUPLICATE_CANDIDATE_ID", "pre_action/candidate_ids",
                               "Candidate IDs must be unique"))
    if type(pre.get("candidate_generation_complete")) is not bool:
        reasons.append(_reason("INVALID_CANDIDATE_COMPLETENESS",
                               "pre_action/candidate_generation_complete",
                               "Candidate completeness must be explicit"))
    if not _nonempty(pre.get("selection_operation_id")):
        reasons.append(_reason("INVALID_SELECTION_OPERATION_ID",
                               "pre_action/selection_operation_id",
                               "A stable selection operation ID is required"))
    decision = pre.get("decision")
    if decision not in ALLOWED_DECISIONS:
        reasons.append(_reason("INVALID_SELECTION_DECISION", "pre_action/decision",
                               "Decision must be query, stop, or defer"))
    selected = pre.get("selected_candidate_id")
    if decision == "query" and selected not in ids:
        reasons.append(_reason("SELECTED_CANDIDATE_OUTSIDE_MENU",
                               "pre_action/selected_candidate_id",
                               "A query must select an ID from the frozen menu"))
    if decision in {"stop", "defer"} and selected is not None:
        reasons.append(_reason("TERMINAL_SELECTION_HAS_CANDIDATE",
                               "pre_action/selected_candidate_id",
                               "A terminal decision cannot name a candidate"))
    selected_at = _time(pre.get("selected_at"))
    if selected_at is None:
        reasons.append(_reason("INVALID_SELECTION_TIME", "pre_action/selected_at",
                               "Selection time must be timezone-aware"))
    elif cutoff is not None and selected_at < cutoff:
        reasons.append(_reason("SELECTION_BEFORE_CUTOFF", "pre_action/selected_at",
                               "The selection must occur at or after the prediction cutoff"))
    return {"ids": ids, "decision": decision, "selected": selected,
            "selected_at": selected_at,
            "complete": pre.get("candidate_generation_complete") is True}


def _validate_candidate_evaluations(rows: Any, pre: Mapping[str, Any],
                                    reasons: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(rows, list):
        reasons.append(_reason("INVALID_CANDIDATE_EVALUATIONS", "candidate_evaluations",
                               "Candidate evaluator rows must be a list"))
        rows = []
    ids = set(pre.get("ids", []))
    seen: set[str] = set()
    valid_rows = []
    for index, row in enumerate(rows):
        path = f"candidate_evaluations/{index}"
        if not isinstance(row, Mapping):
            reasons.append(_reason("INVALID_CANDIDATE_EVALUATION", path,
                                   "Each evaluator row must be an object"))
            continue
        candidate_id = row.get("candidate_id")
        if not _nonempty(candidate_id) or candidate_id in seen:
            reasons.append(_reason("DUPLICATE_OR_INVALID_EVALUATION_ID", f"{path}/candidate_id",
                                   "Each evaluated candidate must be named once"))
        elif candidate_id not in ids:
            reasons.append(_reason("EVALUATION_CANDIDATE_OUTSIDE_MENU", f"{path}/candidate_id",
                                   "Evaluator rows must belong to the frozen candidate menu"))
        seen.add(candidate_id)
        status = row.get("label_status")
        if status not in ALLOWED_LABEL_STATUS:
            reasons.append(_reason("INVALID_LABEL_STATUS", f"{path}/label_status",
                                   "Labels must distinguish independent gold from actor or unavailable evidence"))
        if not _nonempty(row.get("evaluator_id")):
            reasons.append(_reason("MISSING_INDEPENDENT_EVALUATOR", f"{path}/evaluator_id",
                                   "An evaluator identity is required"))
        if not _is_digest(row.get("return_digest")):
            reasons.append(_reason("INVALID_RETURN_DIGEST", f"{path}/return_digest",
                                   "The evaluator must bind the returned evidence by digest"))
        observed = _time(row.get("return_observed_at"))
        if observed is None:
            reasons.append(_reason("INVALID_RETURN_TIME", f"{path}/return_observed_at",
                                   "Return observation time must be timezone-aware"))
        elif pre.get("selected_at") is not None and observed < pre["selected_at"]:
            reasons.append(_reason("RETURN_BEFORE_SELECTION", f"{path}/return_observed_at",
                                   "A returned observation cannot precede the action selection"))
        unit = row.get("loss_unit")
        if status == INDEPENDENT_GOLD and unit not in ALLOWED_LOSS_UNITS:
            reasons.append(_reason("INVALID_LOSS_UNIT", f"{path}/loss_unit",
                                   "Independent evaluator loss must use an explicitly supported unit"))
        elif unit is not None and unit not in ALLOWED_LOSS_UNITS:
            reasons.append(_reason("INVALID_ACTOR_LOSS_UNIT", f"{path}/loss_unit",
                                   "A supplied non-gold loss unit must use an explicitly supported unit"))
        before = row.get("decision_loss_before")
        after = row.get("decision_loss_after")
        if status == INDEPENDENT_GOLD:
            if not _finite_nonnegative(before) or not _finite_nonnegative(after):
                reasons.append(_reason("INVALID_DECISION_LOSS", path,
                                       "Independent evaluator losses must be finite and nonnegative"))
        elif before is not None or after is not None:
            # Actor or unavailable rows may carry an explanatory loss-like
            # field, but it can never be treated as a gold label.  Keep the
            # episode observable when those fields are absent; reject malformed
            # numeric claims only when they are supplied.
            if ((before is not None and not _finite_nonnegative(before)) or
                    (after is not None and not _finite_nonnegative(after))):
                reasons.append(_reason("INVALID_ACTOR_DECISION_LOSS", path,
                                       "Non-gold loss claims must be finite and nonnegative"))
        _validate_cost(row.get("declared_cost"), f"{path}/declared_cost", reasons)
        if status == ACTOR_PROVISIONAL and (before is not None or after is not None):
            reasons.append(_reason("ACTOR_LOSS_NOT_GOLD", path,
                                   "Actor-provided losses cannot be treated as evaluator labels", hard=False))
        valid_rows.append({"candidate_id": candidate_id, "status": status,
                           "evaluator_id": row.get("evaluator_id"),
                           "observed": observed, "unit": unit,
                           "before": before, "after": after})
    return {"rows": valid_rows, "ids": seen}


def validate_evaluation_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an evaluator-only episode without reading files or the network.

    `counterfactual_eligible` means that the finite menu, independent labels,
    and time ordering are sufficient for a later evaluator to compare choices.
    It does not claim that the evaluator's labels are world truth or that a
    numeric VoI model has been calibrated.
    """
    reasons: list[dict[str, Any]] = []
    if not isinstance(episode, Mapping):
        return {"schema_version": VERSION, "status": BLOCKED,
                "reasons": [_reason("INVALID_EPISODE", "episode", "Episode must be an object")],
                "eligibility": _empty_eligibility()}
    if episode.get("schema_version") != VERSION:
        reasons.append(_reason("SCHEMA_VERSION_MISMATCH", "schema_version",
                               "The evaluator schema version is not recognized"))
    for key in ("episode_id", "target_key"):
        if not _nonempty(episode.get(key)):
            reasons.append(_reason("MISSING_EPISODE_IDENTITY", key,
                                   "Episode and target identity are required"))
    cutoff = _time(episode.get("prediction_cutoff"))
    if cutoff is None:
        reasons.append(_reason("INVALID_PREDICTION_CUTOFF", "prediction_cutoff",
                               "Prediction cutoff must be timezone-aware"))
    pre = _validate_pre_action(episode.get("pre_action"), cutoff, reasons)
    if pre is None:
        pre = {"ids": [], "decision": None, "selected": None,
               "selected_at": None, "complete": False}
    binding = episode.get("ledger_binding")
    if binding is not None:
        if not isinstance(binding, Mapping):
            reasons.append(_reason("INVALID_LEDGER_BINDING", "ledger_binding",
                                   "Ledger provenance binding must be an object"))
        else:
            menu_digest = binding.get("candidate_menu_digest")
            if not _is_digest(menu_digest):
                reasons.append(_reason("INVALID_CANDIDATE_MENU_BINDING",
                                       "ledger_binding/candidate_menu_digest",
                                       "Candidate menu provenance must use a SHA256 digest"))
            elif menu_digest != _digest(pre.get("ids", [])):
                reasons.append(_reason("CANDIDATE_MENU_BINDING_MISMATCH",
                                       "ledger_binding/candidate_menu_digest",
                                       "Ledger provenance does not bind the supplied candidate menu"))
            for name in ("entry_digest", "ledger_digest", "state_digest", "prospective_digest",
                         "decision_digest"):
                value = binding.get(name)
                if value is not None and not _is_digest(value):
                    reasons.append(_reason("INVALID_LEDGER_BINDING_DIGEST",
                                           f"ledger_binding/{name}",
                                           "Ledger provenance digest must be lowercase SHA256"))
    row_data = _validate_candidate_evaluations(episode.get("candidate_evaluations"), pre, reasons)
    outcome = episode.get("evaluation_outcome")
    outcome_time = None
    outcome_evaluator = None
    outcome_independent = False
    if not isinstance(outcome, Mapping):
        reasons.append(_reason("MISSING_EVALUATION_OUTCOME", "evaluation_outcome",
                               "An evaluator outcome envelope is required", hard=False))
    else:
        outcome_time = _time(outcome.get("available_at"))
        outcome_evaluator = outcome.get("evaluator_id")
        if outcome_time is None:
            reasons.append(_reason("INVALID_OUTCOME_TIME", "evaluation_outcome/available_at",
                                   "Outcome availability must be timezone-aware"))
        if not _nonempty(outcome_evaluator):
            reasons.append(_reason("MISSING_OUTCOME_EVALUATOR", "evaluation_outcome/evaluator_id",
                                   "The outcome envelope needs an evaluator identity"))
        if outcome.get("label_status") not in ALLOWED_LABEL_STATUS:
            reasons.append(_reason("INVALID_OUTCOME_LABEL_STATUS", "evaluation_outcome/label_status",
                                   "Outcome status must be explicit"))
        outcome_independent = outcome.get("label_status") == INDEPENDENT_GOLD
        if outcome.get("label_status") not in {INDEPENDENT_GOLD, None}:
            reasons.append(_reason("OUTCOME_NOT_INDEPENDENT_GOLD", "evaluation_outcome/label_status",
                                   "A counterfactual comparison requires an independent outcome label",
                                   hard=False))
        if not _is_digest(outcome.get("outcome_digest")):
            reasons.append(_reason("INVALID_OUTCOME_DIGEST", "evaluation_outcome/outcome_digest",
                                   "The evaluator must bind the target outcome by digest"))
    if outcome_time is not None:
        for row in row_data["rows"]:
            if row["observed"] is not None and outcome_time < row["observed"]:
                reasons.append(_reason("OUTCOME_BEFORE_RETURN", "evaluation_outcome/available_at",
                                       "The outcome envelope cannot precede an evaluated return"))
    independent = bool(row_data["rows"]) and all(row["status"] == INDEPENDENT_GOLD for row in row_data["rows"])
    evaluator_bound = bool(outcome_evaluator) and all(row["evaluator_id"] == outcome_evaluator
                                                      for row in row_data["rows"])
    complete_coverage = (pre["complete"] and row_data["ids"] == set(pre["ids"])
                         and len(row_data["ids"]) == len(pre["ids"]))
    selected_covered = pre["selected"] is not None and pre["selected"] in row_data["ids"]
    time_valid = not any(item["code"] in {"RETURN_BEFORE_SELECTION", "OUTCOME_BEFORE_RETURN",
                                          "SELECTION_BEFORE_CUTOFF", "INVALID_RETURN_TIME",
                                          "INVALID_OUTCOME_TIME"} for item in reasons)
    pre_valid = not any(item["hard"] and (item["path"].startswith("pre_action") or
                                          item["code"] == "PRE_ACTION_RESULT_LEAK") for item in reasons)
    hard_invalid = any(item["hard"] for item in reasons)
    counterfactual = (pre_valid and time_valid and complete_coverage and independent
                      and evaluator_bound and outcome_independent)
    eligibility = {
        "pre_action_snapshot_valid": pre_valid,
        "candidate_coverage": "complete" if complete_coverage else ("selected_only" if selected_covered else "none"),
        "independent_gold_labels": independent,
        "independent_outcome_label": outcome_independent,
        "evaluator_identity_bound": evaluator_bound,
        "time_order_valid": time_valid,
        "counterfactual_selection_eligible": counterfactual,
        "numeric_voi_training_eligible": False,
        "actor_assessment_is_gold": False,
    }
    if hard_invalid:
        status = BLOCKED
    elif counterfactual:
        status = COUNTERFACTUAL_ELIGIBLE
    else:
        status = PARTIAL_OBSERVATION
    return {
        "schema_version": VERSION,
        "episode_id": episode.get("episode_id"),
        "target_key": episode.get("target_key"),
        "status": status,
        "reasons": reasons,
        "eligibility": eligibility,
        "candidate_count": len(pre["ids"]),
        "evaluated_candidate_count": len(row_data["ids"]),
        "selected_candidate_id": pre["selected"],
        "scope": "offline evaluator metadata; no provider/network calls; no numeric VoI calibration",
    }


def _empty_eligibility() -> dict[str, Any]:
    return {
        "pre_action_snapshot_valid": False,
        "candidate_coverage": "none",
        "independent_gold_labels": False,
        "independent_outcome_label": False,
        "evaluator_identity_bound": False,
        "time_order_valid": False,
        "counterfactual_selection_eligible": False,
        "numeric_voi_training_eligible": False,
        "actor_assessment_is_gold": False,
    }


def summarize_evaluations(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate eligibility only; no loss or gain is inferred here."""
    reports = [validate_evaluation_episode(item) for item in episodes]
    counts = {BLOCKED: 0, PARTIAL_OBSERVATION: 0, COUNTERFACTUAL_ELIGIBLE: 0}
    eligible_ids = []
    for report in reports:
        counts[report["status"]] = counts.get(report["status"], 0) + 1
        if report["status"] == COUNTERFACTUAL_ELIGIBLE:
            eligible_ids.append(report.get("episode_id"))
    return {
        "schema_version": VERSION,
        "episode_count": len(reports),
        "status_counts": counts,
        "counterfactual_eligible_episode_ids": eligible_ids,
        "numeric_voi_training_eligible": False,
        "numeric_gain": None,
        "scope": "Eligibility inventory only; independent loss labels require a separate analysis protocol",
    }


def _adapter_snapshot(entry: Mapping[str, Any], public_snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only the actor-visible, before-action snapshot for one ledger row.

    The ledger is a projection of a runtime archive.  A caller may provide the
    exact public snapshot captured before selection; when it is absent we use a
    small, ledger-contained projection.  In both cases this function never
    copies technical returns, result labels, evaluator metadata, or post-action
    state into the episode.
    """
    if public_snapshot is not None:
        if not isinstance(public_snapshot, Mapping):
            raise ValueError("VALUE_ADAPTER_PUBLIC_SNAPSHOT_OBJECT_REQUIRED")
        return json.loads(json.dumps(public_snapshot, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False))
    projection = {
        "selection_operation_id": entry.get("selection_operation_id"),
        "candidate_id": entry.get("candidate_id"),
        "question": entry.get("question"),
        "unknown_to_resolve": entry.get("unknown_to_resolve"),
        "useful_if": entry.get("useful_if"),
        "selection_basis": entry.get("selection_basis"),
        "selection_scope": entry.get("selection_scope"),
        "value_status": entry.get("value_status"),
    }
    return {key: value for key, value in projection.items() if value is not None}


def _adapter_candidate_ids(entry: Mapping[str, Any]) -> list[str]:
    rows = entry.get("candidate_value_assessments")
    ids: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and _nonempty(row.get("candidate_id")):
                candidate_id = row["candidate_id"]
                if candidate_id not in ids:
                    ids.append(candidate_id)
    selected = entry.get("candidate_id")
    if _nonempty(selected) and selected not in ids:
        ids.append(selected)
    return ids


def _adapter_binding(entry: Mapping[str, Any], candidate_ids: list[str], *,
                     run_id: str | None = None, ledger_digest: str | None = None,
                     state_digest: str | None = None) -> dict[str, Any]:
    """Bind an episode to the exact ledger projection that produced it.

    These identifiers are provenance metadata only.  They do not expose a
    return body or outcome and are never interpreted as an evaluation label.
    """
    return {
        "run_id": run_id,
        "entry_id": entry.get("entry_id"),
        "entry_digest": entry.get("entry_digest"),
        "ledger_digest": ledger_digest,
        "state_digest": state_digest,
        "prospective_digest": entry.get("prospective_digest"),
        "decision_digest": entry.get("decision_digest"),
        "policy_fingerprint": entry.get("policy_fingerprint"),
        "candidate_menu_digest": _digest(candidate_ids),
    }


def ledger_entry_to_evaluation_episode(
    entry: Mapping[str, Any], *,
    run_id: str | None = None,
    target_key: str | None = None,
    prediction_cutoff: str | None = None,
    public_snapshot: Mapping[str, Any] | None = None,
    ledger_digest: str | None = None,
    state_digest: str | None = None,
) -> dict[str, Any]:
    """Adapt one verified action-ledger row into a public evaluation episode.

    This is a pure adapter.  It emits an episode with no evaluator labels and
    no technical return body, so the resulting episode is intentionally
    ``partial_observation`` until a separately managed evaluator sidecar is
    joined by an offline evaluator.  The runtime never calls a provider or
    reads a result through this function.
    """
    if not isinstance(entry, Mapping):
        raise ValueError("VALUE_ADAPTER_LEDGER_ENTRY_OBJECT_REQUIRED")
    operation_id = entry.get("selection_operation_id")
    if not _nonempty(operation_id):
        raise ValueError("VALUE_ADAPTER_SELECTION_OPERATION_REQUIRED")
    episode_id = entry.get("entry_id") or operation_id
    if not _nonempty(episode_id):
        raise ValueError("VALUE_ADAPTER_EPISODE_ID_REQUIRED")
    action = entry.get("action")
    if action in {"search_web", "read_url", "query_acquired"}:
        decision = "query"
        selected = entry.get("candidate_id")
    elif action in {"stop", "defer"}:
        decision = action
        selected = None
    else:
        # Other terminal actions (for example legacy ask_user) are kept as a
        # deferred evaluation boundary rather than treated as a query.
        decision = "defer"
        selected = None
    cutoff = prediction_cutoff or entry.get("prediction_cutoff")
    selected_at = entry.get("selection_at") or entry.get("selected_at")
    snapshot = _adapter_snapshot(entry, public_snapshot)
    candidate_ids = _adapter_candidate_ids(entry)
    complete = entry.get("candidate_generation_complete") is True
    # The contract requires an explicit candidate menu.  A terminal row with
    # no menu is represented by a stable sentinel and remains ineligible until
    # an evaluator supplies a real finite menu.
    if not candidate_ids:
        candidate_ids = ["__terminal__"]
        complete = False
    if not isinstance(cutoff, str):
        cutoff = ""
    if not isinstance(selected_at, str):
        selected_at = ""
    return {
        "schema_version": VERSION,
        "adapter_version": ADAPTER_VERSION,
        "episode_id": str(episode_id),
        "target_key": target_key or entry.get("target_key") or run_id or "",
        "ledger_binding": _adapter_binding(
            entry, candidate_ids, run_id=run_id, ledger_digest=ledger_digest,
            state_digest=state_digest),
        "prediction_cutoff": cutoff,
        "pre_action": {
            "snapshot": snapshot,
            "snapshot_digest": _digest(snapshot),
            "candidate_ids": candidate_ids,
            "candidate_generation_complete": complete,
            "selection_operation_id": operation_id,
            "decision": decision,
            "selected_candidate_id": selected,
            "selected_at": selected_at,
        },
        # Evaluator rows and outcome envelopes belong to a private sidecar and
        # are deliberately absent from the runtime/public adaptation.
        "candidate_evaluations": [],
        "evaluation_outcome": None,
        "evaluator_sidecar": {
            "status": "not_attached",
            "required": True,
            "runtime_can_read": False,
        },
    }


def ledger_to_evaluation_episodes(
    ledger: Mapping[str, Any], *,
    run_id: str | None = None,
    target_key: str | None = None,
    prediction_cutoff: str | None = None,
) -> list[dict[str, Any]]:
    """Adapt all ledger entries without reading files, tools, or evaluator data."""
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("entries"), list):
        raise ValueError("VALUE_ADAPTER_LEDGER_ENTRIES_REQUIRED")
    effective_run_id = run_id or ledger.get("run_id")
    return [ledger_entry_to_evaluation_episode(
        entry, run_id=effective_run_id, target_key=target_key,
        prediction_cutoff=prediction_cutoff,
        ledger_digest=ledger.get("ledger_digest"),
        state_digest=ledger.get("state_digest"),
        public_snapshot=entry.get("pre_action_snapshot")
        if isinstance(entry, Mapping) and isinstance(entry.get("pre_action_snapshot"), Mapping)
        else None,
    ) for entry in ledger["entries"]]


__all__ = [
    "VERSION", "ADAPTER_VERSION", "COUNTERFACTUAL_ELIGIBLE", "PARTIAL_OBSERVATION", "BLOCKED",
    "INDEPENDENT_GOLD", "ACTOR_PROVISIONAL", "validate_evaluation_episode",
    "summarize_evaluations", "ledger_entry_to_evaluation_episode",
    "ledger_to_evaluation_episodes",
]
