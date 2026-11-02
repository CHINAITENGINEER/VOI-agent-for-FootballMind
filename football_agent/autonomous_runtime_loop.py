"""Run the autonomous state-bound information acquisition loop."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import math
from typing import Any, Callable, Iterable, Mapping

from .candidate_action_registry import build_execution_request, build_state_from_menu, replan_after_execution_receipt
from .execution_action_adapter import complete_execution_request
from .execution_receipt_bridge import LATE, accept_action
from .state_bound_context_value import StateBoundContextValueEstimator
from .audit import digest
from .episode_persistence import EpisodePersistence
from .prediction_snapshot import normalise_prediction_snapshot

VERSION = "autonomous-runtime-loop-v2"


PredictionAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any] | list[float] | tuple[float, ...]]
PredictionStateProjector = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _provider_result(value: Any) -> dict[str, Any]:
    raw = _mapping(value, "provider_result")
    required = ("result_payload", "evidence_rows", "actual_resource_cost", "observed_at", "operational_available_at")
    for field in required:
        if field not in raw:
            raise ValueError(f"provider_result.{field} is required")
    return {
        "result_payload": raw["result_payload"], "evidence_rows": raw["evidence_rows"],
        "actual_resource_cost": raw["actual_resource_cost"], "observed_at": raw["observed_at"],
        "operational_available_at": raw["operational_available_at"],
        **{key: raw[key] for key in ("captured_at", "status", "error_code") if key in raw},
    }


def _resources(value: Any, name: str) -> dict[str, float] | None:
    if value is None:
        return None
    result = {}
    for key, amount in _mapping(value, name).items():
        key = _text(key, f"{name}.key")
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise ValueError(f"{name}.{key} must be finite and nonnegative")
        if key in result:
            raise ValueError(f"{name} contains duplicate normalized resource names")
        result[key] = float(amount)
    return result


def _clock(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("runtime clock must include timezone")
    return parsed.astimezone(timezone.utc)


def _prediction_for_state(
    prediction_adapter: PredictionAdapter | None,
    prediction_state_projector: PredictionStateProjector | None,
    *,
    actor_context: Mapping[str, Any],
    state_bundle: Mapping[str, Any],
    name: str,
) -> dict[str, Any] | None:
    """Evaluate one actor-visible state through an explicitly supplied adapter.

    The runtime deliberately does not guess how its state-bound menu maps to a
    research predictor's public state.  Callers must provide that projection
    when enabling a predictor, which prevents an adapter from silently reading
    evaluator fields or an unbounded raw receipt.
    """
    if prediction_adapter is None:
        return None
    if prediction_state_projector is None:
        raise ValueError("prediction_state_projector is required with prediction_adapter")
    projected = prediction_state_projector(deepcopy(dict(actor_context)), deepcopy(dict(state_bundle)))
    if not isinstance(projected, Mapping):
        raise ValueError("prediction_state_projector must return a mapping")
    raw = prediction_adapter(deepcopy(dict(projected)))
    state = state_bundle.get("state") if isinstance(state_bundle.get("state"), Mapping) else {}
    return normalise_prediction_snapshot(
        raw,
        match_id=actor_context.get("match_id"),
        prediction_cutoff=actor_context.get("prediction_cutoff"),
        state_digest=state.get("state_digest"),
        name=name,
    )


def _prediction_metadata(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {key: deepcopy(value) for key, value in snapshot.items() if key != "probabilities"}


def _consumption_ledger(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project only explicit f consumption fields onto an action handoff.

    An empty list is an explicit proof that the predictor read no semantic
    inputs; absence of the field remains ``unknown``.  No inference from
    probabilities, receipts, or document existence is allowed here.
    """
    if not isinstance(snapshot, Mapping) or "used_semantic_input_ids" not in snapshot:
        return {
            "f_consumed_input_ids": None,
            "f_modeled_read_documents": None,
            "f_consumption_status": "unknown",
            "f_consumption_basis": None,
        }
    ids = snapshot.get("used_semantic_input_ids")
    if not isinstance(ids, list):
        return {
            "f_consumed_input_ids": None,
            "f_modeled_read_documents": None,
            "f_consumption_status": "unknown",
            "f_consumption_basis": "invalid_predictor_ledger",
        }
    return {
        "f_consumed_input_ids": deepcopy(ids),
        "f_modeled_read_documents": snapshot.get("modeled_read_documents"),
        "f_consumption_status": "consumed" if ids else "not_consumed",
        "f_consumption_basis": "prediction.used_semantic_input_ids",
    }


def _settle_budget(before: Mapping[str, float] | None, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Settle observed spend; hold reservations where actual costs are unknown."""
    actual = _resources(receipt.get("actual_resource_cost"), "actual_resource_cost")
    reserved = _resources(receipt.get("declared_resource_cost"), "declared_resource_cost") or {}
    missing = sorted(key for key, value in reserved.items() if value > 0 and key not in (actual or {}))
    held = {key: reserved[key] for key in missing}
    charged = {**(actual or {}), **held}
    after = None if before is None else dict(before)
    overrun = {}
    unbudgeted = []
    for key, amount in charged.items():
        if before is None or key not in before:
            if amount > 0:
                unbudgeted.append(key)
            continue
        # Decimal arithmetic avoids an exact .3 - .1 budget rejecting a .2 action.
        difference = Decimal(str(before[key])) - Decimal(str(amount))
        after[key] = float(max(Decimal(0), difference))
        if difference < 0:
            overrun[key] = float(-difference)
    status = ("budget_overrun" if overrun else "cost_reconciliation_required"
              if actual is None or missing or unbudgeted else "settled")
    return {"status": status, "before": deepcopy(before), "after": after,
            "actual_resource_cost": actual, "held_resource_cost": held,
            "overrun": overrun, "unbudgeted_resources": sorted(unbudgeted),
            "missing_actual_resources": missing}


def _existing_rows(actor_context: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # The actor's prior visible evidence is authoritative even when callers omit
    # the optional raw rows. Rebuilding from just the newest receipt loses memory.
    result = deepcopy(list(actor_context["snapshot"].get("source_observations", [])))
    for row in rows:
        copied = deepcopy(dict(row))
        if copied not in result:
            result.append(copied)
    return result


def _persist_cycle(store: EpisodePersistence | None, *, episode_id: str | None,
                   actor_context: Mapping[str, Any], cycle: Mapping[str, Any],
                   step: int, target_key: str | None,
                   selected_at: str | datetime | None = None) -> None:
    """Persist explicit runtime facts and optional bound posterior snapshots.

    Posterior fields live at cycle level so adding them cannot invalidate the
    selection trace digest.  Callers without a predictor continue to persist
    explicit ``None`` values and remain ``PARTIAL``.
    """
    if store is None:
        return
    episode = episode_id or actor_context.get("episode_id")
    if not isinstance(episode, str) or not episode.strip():
        raise ValueError("episode_persistence requires an explicit episode_id")
    trace = cycle.get("selection") if isinstance(cycle.get("selection"), Mapping) else {}
    receipt = cycle.get("receipt") if isinstance(cycle.get("receipt"), Mapping) else {}
    request = cycle.get("request") if isinstance(cycle.get("request"), Mapping) else {}
    decision = trace.get("decision")
    terminal = decision != "acquire"
    action_id = "STOP" if decision == "stop" else "DEFER" if decision == "defer" else (
        "USER_INPUT_REQUIRED" if decision == "ask_user" else request.get("action_id"))
    candidate_id = "STOP" if terminal else request.get("candidate_id", trace.get("selected_candidate_id"))
    selection_id = trace.get("trace_digest")
    if not isinstance(selection_id, str) or not selection_id.strip():
        raise ValueError("runtime selection trace_digest is required for episode persistence")
    replan = cycle.get("replan") if isinstance(cycle.get("replan"), Mapping) else {}
    after_state = replan.get("state") if isinstance(replan.get("state"), Mapping) else {}
    prediction_before = (cycle["prediction_before"] if "prediction_before" in cycle
                         else trace.get("prediction_before"))
    prediction_after = (cycle["prediction_after"] if "prediction_after" in cycle
                        else trace.get("prediction_after"))
    row = {
        "episode_id": episode.strip(), "target_key": target_key or actor_context.get("match_id"),
        "sequence": step, "action_id": action_id, "candidate_id": candidate_id,
        "selection_operation_id": selection_id, "selection_digest": digest(trace),
        "state_before_digest": cycle.get("state_digest"),
        "state_after_digest": after_state.get("state_digest"),
        "prediction_before": prediction_before,
        "prediction_after": prediction_after,
        "prediction_cutoff": cycle.get("prediction_cutoff", actor_context.get("prediction_cutoff")),
        "selected_at": selected_at,
        "captured_at": receipt.get("captured_at"),
        "acquisition_id": receipt.get("acquisition_id") if receipt else None,
        "acquisition_digest": receipt.get("receipt_digest") if receipt else None,
        # Preserve the predictor's explicit semantic-consumption ledger at the
        # same action boundary as the posterior.  These fields are copied
        # verbatim; persistence must never infer consumption from a changed
        # probability or from the receipt alone.
        "f_consumed_input_ids": deepcopy(cycle.get("f_consumed_input_ids")),
        "f_consumption_status": cycle.get("f_consumption_status", "unknown"),
        "f_consumption_basis": cycle.get("f_consumption_basis"),
        "f_modeled_read_documents": cycle.get("f_modeled_read_documents"),
        "f_consumption_required": cycle.get("f_consumption_required") is True,
        # This loop has no separate ExperienceService admission boundary.  Keep
        # that distinction explicit instead of claiming a receipt was admitted.
        "admission_status": "unknown" if receipt else None,
    }
    store.append(row)


def run_runtime_cycle(
    *,
    actor_context: Mapping[str, Any],
    state_bundle: Mapping[str, Any],
    menu_report: Mapping[str, Any],
    catalog: Iterable[Mapping[str, Any]],
    match: Mapping[str, Any],
    estimator: StateBoundContextValueEstimator,
    context: Mapping[str, Any],
    loss_unit: str,
    remaining_budget: Mapping[str, float] | None,
    provider_id: str,
    accepted_at: str,
    execute_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    existing_evidence_rows: Iterable[Mapping[str, Any]] = (),
    target_key: str | None = None,
    episode_store: EpisodePersistence | None = None,
    episode_id: str | None = None,
    episode_step: int = 0,
    prediction_adapter: PredictionAdapter | None = None,
    prediction_state_projector: PredictionStateProjector | None = None,
) -> dict[str, Any]:
    """Execute one autonomous evaluate-select-acquire-replan cycle."""
    bundle = _mapping(state_bundle, "state_bundle")
    state = _mapping(bundle.get("state"), "state_bundle.state")
    registry = _mapping(bundle.get("registry"), "state_bundle.registry")
    if actor_context.get("information_mode") == "historical_reconstruction":
        raise ValueError("historical reconstruction requires the replay adapter, not the operational loop")
    if state.get("context_digest") != actor_context.get("context_digest"):
        raise ValueError("state context differs from actor context")
    match_id = match.get("match_id", match.get("canonical_match_id", match.get("id")))
    if state.get("match_id") != actor_context.get("match_id") or match_id != actor_context.get("match_id"):
        raise ValueError("runtime match identity mismatch")
    snapshot = _mapping(actor_context.get("snapshot"), "actor_context.snapshot")
    if _clock(snapshot["kickoff"]) != _clock(match.get("kickoff", match.get("event_time"))):
        raise ValueError("runtime match kickoff mismatch")
    for key in ("home_team_id", "away_team_id", "home_team_name", "away_team_name"):
        if snapshot.get(key) is not None and snapshot[key] != match.get(key):
            raise ValueError(f"runtime match {key} mismatch")
    if target_key is not None and target_key != actor_context.get("match_id"):
        raise ValueError("target_key differs from actor match")
    if episode_store is not None and not (episode_id or actor_context.get("episode_id")):
        raise ValueError("episode_persistence requires an explicit episode_id")
    expected = build_state_from_menu(actor_context, menu_report,
                                    candidate_generation_complete=state.get("candidate_generation_complete", False))
    if expected["state"]["state_digest"] != state.get("state_digest") or expected["registry"]["registry_digest"] != registry.get("registry_digest"):
        raise ValueError("runtime state or registry differs from current menu")
    budget = _resources(remaining_budget, "remaining_budget")
    # Materialize iterable inputs before dispatch; providers must not consume or
    # mutate the catalog used to rebuild the next action state.
    catalog_rows = deepcopy(list(catalog))
    prior_rows = _existing_rows(actor_context, existing_evidence_rows)
    target = target_key or actor_context.get("match_id")
    prediction_before_snapshot = _prediction_for_state(
        prediction_adapter, prediction_state_projector,
        actor_context=actor_context, state_bundle=bundle, name="prediction_before")
    evaluation = estimator.select(state, context, loss_unit=loss_unit,
                                  remaining_budget=deepcopy(budget), target_key=target)
    trace = _mapping(evaluation.get("selection"), "evaluation.selection")
    base = {
        "schema_version": VERSION, "state_digest": state.get("state_digest"),
        "evaluation": evaluation, "selection": trace,
        "prediction_before": (None if prediction_before_snapshot is None
                               else deepcopy(prediction_before_snapshot["probabilities"])),
        "prediction_after": None,
        "prediction_cutoff": actor_context.get("prediction_cutoff"),
        "prediction_meta_before": _prediction_metadata(prediction_before_snapshot),
        "prediction_meta_after": None,
        **_consumption_ledger(prediction_before_snapshot),
        "f_consumption_required": prediction_before_snapshot is not None,
        "contract": {
            "agent_selected_without_user_question": True,
            "autonomous_completion": False,
            "replay_eligible": False,
            "human_input_required": False,
            "selection_delegated_to_state_bound_voi": True,
            "real_source_requests": 0, "real_model_calls": 0, "provider_invocations": 0,
            "evaluator_fields_used": False,
        },
        "budget": {"status": "unchanged", "before": deepcopy(budget), "after": deepcopy(budget),
                   "actual_resource_cost": {}, "held_resource_cost": {}, "overrun": {},
                   "unbudgeted_resources": [], "missing_actual_resources": []},
    }
    decision = trace.get("decision")
    if decision not in {"acquire", "stop", "defer", "ask_user"}:
        # An unrecognized planner output must never be represented as a
        # successful terminal decision. Fail closed before any provider call.
        raise ValueError("unsupported selection decision")
    if decision != "acquire":
        is_user_input = decision == "ask_user"
        is_autonomous_stop = decision == "stop"
        base["contract"].update({
            "agent_selected_without_user_question": not is_user_input,
            "autonomous_completion": is_autonomous_stop,
            "replay_eligible": is_autonomous_stop,
            "human_input_required": is_user_input,
        })
        base.update({
            "status": "human_input_required" if is_user_input else "terminal",
            "terminal_reason": decision,
            "request": None, "receipt": None, "replan": None,
        })
        _persist_cycle(episode_store, episode_id=episode_id, actor_context=actor_context,
                       cycle=base, step=episode_step, target_key=target, selected_at=accepted_at)
        return base
    request = build_execution_request(trace, registry, parameters={"match_id": actor_context["match_id"]})
    accepted = accept_action(trace, provider_id=provider_id, accepted_at=accepted_at,
                             execution_request=request)
    base.update({"request": request, "receipt": accepted, "replan": None})
    # An injected callback can perform network/model calls; the runtime does not
    # measure them and must not report hard-coded zeros after invoking it.
    base["contract"].update({"provider_invocations": 1, "real_source_requests": None,
                             "real_model_calls": None, "external_call_counts_measured": False})
    try:
        raw_provider = execute_fn(deepcopy(request))
    except Exception:
        # The attempt may have consumed resources. Keep the accepted receipt and
        # hold its reservation; no implicit retry and no invented failure cost.
        base.update({"status": "execution_unresolved", "terminal_reason": "provider_exception",
                     "budget": _settle_budget(budget, accepted)})
        _persist_cycle(episode_store, episode_id=episode_id, actor_context=actor_context,
                       cycle=base, step=episode_step, target_key=target, selected_at=accepted_at)
        return base
    try:
        provider = _provider_result(raw_provider)
        if provider["actual_resource_cost"] is None:
            raise ValueError("provider actual_resource_cost must be a mapping")
        provider["actual_resource_cost"] = _resources(provider["actual_resource_cost"], "actual_resource_cost")
        completion = complete_execution_request(accepted, request, **provider)
    except Exception:
        # Deferred provider iterables can fail during normalization as well as
        # during the callback itself. An attempt still requires an audit trail.
        base.update({"status": "execution_unresolved", "terminal_reason": "provider_response_rejected",
                     "budget": _settle_budget(budget, accepted)})
        _persist_cycle(episode_store, episode_id=episode_id, actor_context=actor_context,
                       cycle=base, step=episode_step, target_key=target, selected_at=accepted_at)
        return base
    completed_receipt = completion["receipt"]
    settlement = _settle_budget(budget, completed_receipt)
    base.update({"receipt": completed_receipt, "budget": settlement,
                 "provider_contract": {**completion["contract"], "measurement_scope": "local_receipt_adapter_only"}})
    try:
        replanned = replan_after_execution_receipt(
            actor_context, completed_receipt, match, menu_report=menu_report, catalog=catalog_rows,
            existing_evidence_rows=prior_rows,
            candidate_generation_complete=state.get("candidate_generation_complete", False),
        )
    except (ValueError, TypeError, KeyError):
        # The execution already happened. Retain its settled spend and receipt
        # even when the returned evidence fails to form a new public state.
        final_status = settlement["status"] if settlement["status"] != "settled" else "replan_required"
        base.update({"status": final_status,
                     "terminal_reason": final_status if final_status != "replan_required" else "public_state_rebuild_rejected",
                     "replan_error": "public_state_rebuild_rejected"})
        _persist_cycle(episode_store, episode_id=episode_id, actor_context=actor_context,
                       cycle=base, step=episode_step, target_key=target, selected_at=accepted_at)
        return base
    late_cutoff = completed_receipt.get("status") == LATE
    base.update({
        "status": "execution_deadline_reached" if late_cutoff else ("replanned" if replanned.get("state") is not None else "replan_required"),
        "terminal_reason": (completed_receipt.get("late_reason") or "execution_deadline_reached") if late_cutoff else (None if replanned.get("state") is not None else "receipt_requires_replan"),
        "request": request, "receipt": completed_receipt,
        "replan": replanned,
    })
    if settlement["status"] != "settled":
        base.update({"status": settlement["status"], "terminal_reason": settlement["status"]})
    if (prediction_adapter is not None and not late_cutoff
            and settlement["status"] == "settled"
            and replanned.get("state") is not None):
        after_bundle = {"state": replanned["state"], "registry": replanned["registry"]}
        try:
            prediction_after_snapshot = _prediction_for_state(
                prediction_adapter, prediction_state_projector,
                actor_context=replanned["actor_context"], state_bundle=after_bundle,
                name="prediction_after")
        except Exception:
            # The acquisition is already durable, but an invalid or unavailable
            # posterior must not be fabricated from the receipt.  Keep the
            # episode auditable and let its status remain PARTIAL.
            prediction_after_snapshot = None
        if prediction_after_snapshot is not None:
            if prediction_after_snapshot["prediction_cutoff"] != base["prediction_cutoff"]:
                prediction_after_snapshot = None
            else:
                base["prediction_after"] = deepcopy(prediction_after_snapshot["probabilities"])
                base["prediction_meta_after"] = _prediction_metadata(prediction_after_snapshot)
                base.update(_consumption_ledger(prediction_after_snapshot))
                base["f_consumption_required"] = True
    _persist_cycle(episode_store, episode_id=episode_id, actor_context=actor_context,
                   cycle=base, step=episode_step, target_key=target, selected_at=accepted_at)
    return base


def run_information_loop(
    *,
    actor_context: Mapping[str, Any],
    state_bundle: Mapping[str, Any],
    menu_report: Mapping[str, Any],
    catalog: Iterable[Mapping[str, Any]],
    match: Mapping[str, Any],
    estimator: StateBoundContextValueEstimator,
    context: Mapping[str, Any],
    loss_unit: str,
    remaining_budget: Mapping[str, float] | None,
    provider_id: str,
    accepted_at: str,
    execute_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    existing_evidence_rows: Iterable[Mapping[str, Any]] = (),
    target_key: str | None = None,
    max_steps: int = 8,
    episode_store: EpisodePersistence | None = None,
    episode_id: str | None = None,
    prediction_adapter: PredictionAdapter | None = None,
    prediction_state_projector: PredictionStateProjector | None = None,
) -> dict[str, Any]:
    """Repeat autonomous cycles until stop/defer, failed replan, or a step cap."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    current_actor = deepcopy(dict(actor_context))
    current_bundle = deepcopy(dict(state_bundle))
    current_menu = deepcopy(dict(menu_report))
    current_context = deepcopy(dict(context))
    current_rows = _existing_rows(current_actor, existing_evidence_rows or [])
    catalog_rows = deepcopy(list(catalog))
    current_budget = _resources(remaining_budget, "remaining_budget")
    initial_budget = deepcopy(current_budget)
    current_time = _clock(accepted_at)
    spent: dict[str, float] = {}
    held: dict[str, float] = {}
    history = []
    terminal_status = "max_steps"
    for step in range(max_steps):
        cycle = run_runtime_cycle(
            actor_context=current_actor, state_bundle=current_bundle, menu_report=current_menu,
            catalog=catalog_rows, match=match, estimator=estimator, context=current_context,
            loss_unit=loss_unit, remaining_budget=current_budget, provider_id=provider_id,
            accepted_at=current_time.isoformat(), execute_fn=execute_fn, existing_evidence_rows=current_rows,
            target_key=target_key, episode_store=episode_store, episode_id=episode_id,
            episode_step=step,
            prediction_adapter=prediction_adapter,
            prediction_state_projector=prediction_state_projector,
        )
        cycle["step"] = step
        history.append(cycle)
        current_budget = deepcopy(cycle["budget"]["after"])
        for total, amounts in ((spent, cycle["budget"]["actual_resource_cost"]),
                               (held, cycle["budget"]["held_resource_cost"])):
            for key, amount in (amounts or {}).items():
                total[key] = float(Decimal(str(total.get(key, 0))) + Decimal(str(amount)))
        replanned = cycle.get("replan")
        if replanned and replanned.get("state") is not None:
            current_actor = deepcopy(replanned["actor_context"])
            current_menu = deepcopy(replanned["menu_report"])
            current_bundle = {"registry": deepcopy(replanned["registry"]), "state": deepcopy(replanned["state"])}
            current_context = deepcopy(replanned["context_report"]["binding"]["context"])
            # Only the actor projection is carried to subsequent steps. Excluded
            # late/private rows cannot enter the contextual estimator indirectly.
            current_rows = deepcopy(current_actor["snapshot"].get("source_observations", []))
        receipt = cycle.get("receipt")
        if receipt and receipt.get("captured_at"):
            current_time = max(current_time, _clock(receipt["captured_at"]))
        if cycle["status"] != "replanned":
            terminal_status = cycle["status"]
            break
        if current_time > _clock(current_actor["prediction_cutoff"]):
            terminal_status = "execution_deadline_reached"
            break
    invocations = sum(item["contract"]["provider_invocations"] for item in history)
    return {
        "schema_version": VERSION, "terminal_status": terminal_status,
        "steps": len(history), "history": history,
        "final_actor_context": current_actor, "final_state": current_bundle.get("state"),
        "final_context": current_context, "initial_budget": initial_budget,
        "remaining_budget": current_budget, "actual_resource_cost": spent,
        "held_resource_cost": held,
        "terminal_reason": history[-1].get("terminal_reason") or terminal_status,
        "contract": {
            "agent_selected_without_user_question": all(item["contract"].get("agent_selected_without_user_question", True) for item in history),
            "selection_delegated_to_state_bound_voi": True,
            "stop_or_defer_is_explicit": True, "real_source_requests": 0 if not invocations else None,
            "real_model_calls": 0 if not invocations else None, "evaluator_fields_used": False,
            "provider_invocations": invocations, "external_call_counts_measured": not bool(invocations),
            "resource_cost_settled_per_step": True,
            "autonomous_completion": bool(history and history[-1]["contract"].get("autonomous_completion", False)),
            "replay_eligible": bool(history and history[-1]["contract"].get("replay_eligible", False)),
            "human_input_required": bool(history and history[-1]["contract"].get("human_input_required", False)),
        },
    }


__all__ = ["VERSION", "PredictionAdapter", "PredictionStateProjector",
           "run_runtime_cycle", "run_information_loop"]
