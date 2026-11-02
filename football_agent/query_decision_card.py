"""Auditable, non-generative cards from completed acquisition event snapshots."""
from __future__ import annotations

from copy import deepcopy
import math

from .audit import stored_digest
from .research_bridge import public_state


VERSION = "auditable-query-decision-card-v1"
FAILURES = {"FAILED", "TIMEOUT", "PROVIDER_ERROR", "BAD_DATA", "STALE", "UNAVAILABLE"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _no_tokens(value):
    if isinstance(value, dict):
        _require("joint_return_hypothesis" not in value, "Hypothetical tokens entered a real decision snapshot")
        for item in value.values():
            _no_tokens(item)
    elif isinstance(value, list):
        for item in value:
            _no_tokens(item)


def _probabilities(prediction):
    if prediction is None:
        return None
    probabilities = prediction.get("probabilities")
    _require(isinstance(probabilities, dict) and set(probabilities) == {"home", "draw", "away"}, "Three named prediction probabilities required")
    _require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
             and abs(sum(probabilities.values()) - 1.) <= 1e-10, "Invalid prediction probabilities")
    return deepcopy(probabilities)


def _context(snapshot):
    state = public_state(snapshot["visible_state"])
    _no_tokens(state)
    prediction = snapshot.get("prediction")
    if prediction:
        _no_tokens(prediction)
        _require(prediction.get("match_id") == state["match_id"], "Prediction and visible fixture differ")
    return {"probabilities": _probabilities(prediction), "public_mask": deepcopy(state["mask"]),
            "group_statuses": deepcopy(state.get("quality", {}).get("group_statuses", {})),
            "observed_return_categories": deepcopy((prediction or {}).get("observed_return_categories", {})),
            "evidence_versions": deepcopy(state.get("evidence_versions", {})),
            "predictor_version": (prediction or {}).get("model_version"),
            "information_mode": state["information_mode"], "prediction_cutoff": state["prediction_cutoff"]}


def _reference(event):
    return {"seq": event["seq"], "event_hash": event["event_hash"], "type": event["type"]}


def _candidate(candidate, risk):
    value = candidate.get("value", {})
    native = value.get("research_estimate", {})
    metric = "conservative_net_gain" if risk == "conservative" else "net_gain"
    return {"action_id": candidate["action_id"], "group_ids": deepcopy(candidate.get("group_ids", [])),
            "catalog_status": candidate.get("status"), "unavailable_reason": candidate.get("unavailable_reason"),
            "source": candidate.get("source"), "dependency_key": candidate.get("dependency_key"),
            "cost_scope": candidate.get("cost_scope"), "cost_vector": deepcopy(candidate.get("cost_vector")),
            "cost_components": deepcopy(candidate.get("cost_components")),
            "estimate_status": value.get("status", "NOT_EVALUATED"),
            "model_evsi": native.get("model_evsi"), "expected_gain": value.get("expected_loss_improvement"),
            "lower_gain": value.get("lower_gain"), "cost_in_loss_units": value.get("cost_in_loss_units"),
            "mean_net_gain": native.get("mean_net_gain"), "lower_net_gain": native.get("lower_net_gain"),
            "selection_score": value.get(metric), "decision_metric": metric,
            "support": value.get("support"), "parent_support": value.get("parent_support"),
            "effective_support": value.get("effective_support"), "support_status": value.get("support_status"),
            "fallback_reason": value.get("fallback_reason"), "gate_reason": value.get("gate_reason"),
            "standard_error": value.get("standard_error"), "uncertainty_kind": value.get("uncertainty_kind"),
            "estimator_version": value.get("estimator_version"), "acquired_action_ids": deepcopy(native.get("acquired_action_ids")),
            "physical_budget": deepcopy(native.get("physical_budget")), "lambda_cost": deepcopy(native.get("lambda_cost"))}


def _receipt(event, after):
    result = event["data"]["result"]
    state = public_state(result["visible_state"])
    _no_tokens(state)
    status = result.get("status")
    statuses = state.get("quality", {}).get("group_statuses", {})
    kind = ("provider_failure" if status in FAILURES else "summary_missing" if status == "NO_DATA" else
            "summary_observed" if status in {"OK", "ACQUIRED", "SUCCESS"} else "unchanged_or_other")
    return {"audit": _reference(event), "action_id": event["data"]["action_id"],
            "request_id": result.get("request_id"), "status": status, "return_kind": kind,
            "reason": result.get("reason"), "charged_cost": deepcopy(result.get("cost")),
            "charged_cost_components": deepcopy(result.get("cost_components")),
            "evidence_refs": deepcopy(result.get("evidence_refs", [])), "public_mask_after_return": deepcopy(state["mask"]),
            "group_statuses_after_return": deepcopy(statuses),
            "prediction_after_return": _context(after["snapshot"]) if after else None,
            "prediction_update_audit": _reference(after) if after else None}


def build_query_decision_cards(run):
    """Build cards after execution; never recompute a gate or infer missing data.

    Verify the inline event chain here. Caller RunStore.verify additionally
    validates external tool input/output blobs before publishing the cards.
    """
    events = run.get("events", [])
    _require(bool(events) and run.get("state") in {"STOPPED", "DEFERRED"}, "A completed audited run is required")
    previous = None
    for seq, event in enumerate(events, 1):
        body = {k: v for k, v in event.items() if k != "event_hash"}
        _require(event["seq"] == seq and event.get("previous_hash") == previous and stored_digest(body) == event["event_hash"], "Audit event chain mismatch")
        previous = event["event_hash"]
    _require(all(run.get(k) == v for k, v in events[-1]["snapshot"].items()), "Final run differs from its audited snapshot")
    configuration = run["configuration"]["backend"]
    cards = []
    decision_indices = [i for i, e in enumerate(events) if e["type"] == "value_estimated"]
    for pos, index in enumerate(decision_indices):
        event = events[index]
        snapshot = event["snapshot"]
        context = _context(snapshot)
        _require(context["probabilities"] is not None, "A query decision requires its pre-action prediction")
        following = events[index+1:decision_indices[pos+1] if pos+1 < len(decision_indices) else len(events)]
        selections = [e for e in following if e["type"] == "action_selected"]
        returns = [e for e in following if e["type"] == "acquisition_result"]
        _require(len(selections) <= 1 and len(returns) <= 1, "Ambiguous action-to-decision audit linkage")
        selected = selections[0]["data"]["action_id"] if selections else None
        candidates = [_candidate(c, run["risk_mode"]) for c in event["data"]["candidates"]]
        _require(selected is None or selected in {c["action_id"] for c in candidates}, "Chosen action lacks a candidate record")
        receipt = None
        if returns:
            returned = returns[0]
            _require(selected == returned["data"]["action_id"], "Actual receipt belongs to a different action")
            after = next((e for e in following if e["type"] == "prediction_updated" and e["seq"] > returned["seq"]), None)
            receipt = _receipt(returned, after)
        terminal = next((e for e in following if e["snapshot"]["state"] in {"STOPPED", "DEFERRED"}), None)
        _require(selected is not None or terminal is not None, "Missing STOP or selected action after estimation")
        cards.append({"step": len(snapshot["history"]), "kind": "query_decision", "audit": _reference(event),
                      "current": context, "candidates": candidates,
                      "decision": {"kind": "acquire" if selected is not None else "stop" if terminal["snapshot"]["state"] == "STOPPED" else "defer", "selected_action": selected,
                                   "reason": "numeric_candidate_gate" if selected is not None else terminal["snapshot"]["reason"],
                                   "audit": _reference(selections[0]) if selected is not None else _reference(terminal)},
                      "receipt": receipt})
    terminal = events[-1]
    cards.append({"step": len(run["history"]), "kind": "terminal", "audit": _reference(terminal),
                  "current": _context(terminal["snapshot"]), "candidates": [],
                  "decision": {"kind": "stop" if run["state"] == "STOPPED" else "defer", "selected_action": None, "reason": run["reason"]},
                  "receipt": None})
    budget = run["budget"]
    return {"schema": VERSION, "run_id": run["run_id"], "match_id": run["match_id"],
            "prediction_cutoff": run["prediction_cutoff"], "risk_mode": run["risk_mode"], "cards": cards,
            "artifact_hashes": deepcopy(configuration.get("artifact_hashes", {})),
            "reference_joint": deepcopy(configuration.get("reference_joint", {})),
            "final_probabilities": _probabilities(run.get("prediction")),
            "actual_actions": [h["action_id"] for h in run["history"]],
            "physical_budget": {k: budget[k] for k in ("max_simulated_cost", "simulated_cost", "max_actions", "actions")},
            "audit_head_hash": previous, "source_run_sha256": stored_digest(run),
            "scope": "Observed historical execution trace, not a generated explanation or effectiveness proof. Inline chain checked; external blobs require RunStore.verify.",
            "tool_scope": "Two controlled views of one historical response; open tool discovery and independent provider acquisition are not implemented.",
            "llm_generated_explanation": False}
