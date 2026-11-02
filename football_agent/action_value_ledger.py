"""Deterministic action-level value/evidence ledger for one inquiry run.

The runtime already persists the selection boundary, acquisition receipt and
provisional return assessment as separate records.  This module joins those
records into one replayable row per accepted selection.  It is deliberately a
projection: it records declared value assumptions and observed costs, while
leaving realized information gain and calibrated VoI unset.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from .audit import atomic_write, stored_digest as digest
from .acquisition_boundary import VERSION as BOUNDARY_VERSION
from .locks import run_lock
from .query_value_contract import VALUE_UNMEASURED
from . import acquisition_admission as admission
from .episode_persistence import EpisodePersistence


VERSION = "action-value-ledger-v1"
ACQUISITION_ACTIONS = frozenset({"search_web", "read_url", "query_acquired"})
TERMINAL_ACTIONS = frozenset({"stop", "defer", "ask_user"})
RESOURCE_SLOTS = {"http": "http_calls", "model": "llm_calls", "local_read": "local_reads"}


def _read(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"ACTION_LEDGER_MISSING_FILE:{path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"ACTION_LEDGER_INVALID_JSON:{path.name}") from exc


def _copy(value: Any) -> Any:
    return deepcopy(value)


def _resource_limits(state: Mapping[str, Any]) -> dict[str, int]:
    limits = state.get("config", {}).get("limits", {})
    result = {key: limits[key] for key in ("http_calls", "llm_calls")}
    capability = state.get("config", {}).get("local_query_capability")
    if isinstance(capability, Mapping):
        result["local_reads"] = capability.get("max_reads")
    if any(type(amount) is not int or amount < 0 for amount in result.values()):
        raise ValueError("ACTION_LEDGER_INVALID_RESOURCE_LIMIT")
    return result


def _resource_cost(operations, limits):
    result = {key: 0 for key in limits}
    for operation in operations:
        slot = RESOURCE_SLOTS.get(operation.get("kind"))
        if slot not in result:
            raise ValueError("ACTION_LEDGER_UNKNOWN_RESOURCE_SLOT")
        result[slot] += 1
    return result


def _remaining_before(operations: list[Mapping[str, Any]], limits: Mapping[str, int], index: int) -> dict[str, int]:
    """Compute independent remaining slots before an operation index.

    Every reservation consumes its own named slot, including failed or pending
    reservations.  HTTP calls and local reads are intentionally never added
    together.
    """
    return _subtract(limits, _resource_cost(operations[:index], limits))


def _subtract(before: Mapping[str, int], cost: Mapping[str, int]) -> dict[str, int]:
    result = dict(before)
    for key, amount in cost.items():
        if amount < 0:
            raise ValueError("ACTION_LEDGER_NEGATIVE_RESOURCE_COST")
        result[key] = result.get(key, 0) - amount
        if result[key] < 0:
            raise ValueError("ACTION_LEDGER_BUDGET_UNDERFLOW")
    return result


def _question(response: Mapping[str, Any]) -> dict[str, Any] | None:
    question_id = response.get("active_question_id")
    if not question_id:
        return None
    for item in response.get("analysis_plan", []) or []:
        if isinstance(item, Mapping) and item.get("id") == question_id:
            return {
                "id": question_id,
                "question": item.get("question"),
                "decision_relevance": item.get("decision_relevance"),
                "status": item.get("status"),
            }
    return {"id": question_id}


def _value_projection(choice: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = choice.get("diagnostics") if isinstance(choice.get("diagnostics"), Mapping) else {}
    candidate_id = choice.get("candidate_id")
    rows = diagnostics.get("candidates") if isinstance(diagnostics.get("candidates"), list) else []
    row = next((item for item in rows if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id), None)
    value = row.get("value") if isinstance(row, Mapping) and isinstance(row.get("value"), Mapping) else None
    selected = diagnostics.get("selected_value") if isinstance(diagnostics.get("selected_value"), Mapping) else None
    value = value or selected
    formal = diagnostics.get("query_value_decision")
    formal_projection = None
    if isinstance(formal, Mapping):
        formal_projection = {
            "schema_version": formal.get("schema_version"),
            "action": formal.get("action"),
            "candidate_id": formal.get("candidate_id"),
            "selection_basis": formal.get("selection_basis"),
            "stop_certificate": _copy(formal.get("stop_certificate")),
        }
    action = response.get("action")
    if action not in ACQUISITION_ACTIONS:
        value = None
    return {
        "selection_basis": diagnostics.get("selection_basis") or choice.get("reason"),
        "selection_scope": diagnostics.get("selection_scope") or diagnostics.get("stop_scope"),
        "value_status": value.get("value_status") if value else (
            VALUE_UNMEASURED if action in ACQUISITION_ACTIONS else "NOT_APPLICABLE"),
        "expected_information_value": value.get("expected_value") if value else None,
        "expected_cost": value.get("cost") if value else None,
        "expected_net_value": value.get("net_value") if value else None,
        "formal_query_value_decision": formal_projection,
        "candidate_value_assessments": _copy(rows),
        "value_support_scope": value.get("support_scope") if value else None,
        "calibration_status": value.get("calibration_status") if value else "NOT_CALIBRATED",
    }


def _compact_return_context(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    keys = ("version", "acquisition_id", "selection_operation_id", "context_digest", "question",
            "unknown_to_resolve", "useful_if", "action", "document_id")
    return {key: _copy(value[key]) for key in keys if key in value}


def _compact_assessment(record: Mapping[str, Any] | None) -> Any:
    if not isinstance(record, Mapping):
        return None
    assessment = record.get("assessment") if isinstance(record.get("assessment"), Mapping) else {}
    return {
        "assessment_operation_id": record.get("assessment_operation_id"),
        "outcome": assessment.get("outcome"),
        "reason": assessment.get("reason"),
        "citations": _copy(assessment.get("citations", [])),
        "remaining_unknowns": _copy(assessment.get("remaining_unknowns", [])),
        "label_status": record.get("label_status"),
        "world_truth_verified": record.get("world_truth_verified"),
        "independent_semantic_validation": record.get("independent_semantic_validation"),
        "realized_decision_gain": None,
        "numeric_voi_training_eligible": False,
    }


def _acquisition_index(folder: Path, state: Mapping[str, Any], operations: list[Mapping[str, Any]],
                      accepted: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read and validate acquisition links, keyed by selection operation."""
    by_selection: dict[str, dict[str, Any]] = {}
    links_dir = folder / "acquisitions"
    if not links_dir.exists():
        return by_selection
    limits = _resource_limits(state)
    positions = {operation.get("operation_id"): index for index, operation in enumerate(operations)}
    for link_path in sorted(links_dir.glob("*/selection-link.json")):
        link = _read(link_path)
        selection_id = link.get("operation_id")
        if selection_id not in accepted:
            raise ValueError("ACTION_LEDGER_ACQUISITION_WITHOUT_ACCEPTED_SELECTION")
        if selection_id in by_selection:
            raise ValueError("ACTION_LEDGER_MULTIPLE_ACQUISITIONS")
        group_id = link_path.parent.name
        action_path = link_path.parent / "input.json"
        action = _read(action_path)
        if link.get("action_digest") != digest(action):
            raise ValueError("ACTION_LEDGER_ACTION_LINK_MISMATCH")
        if link.get("decision_digest") != accepted[selection_id]["decision_digest"]:
            raise ValueError("ACTION_LEDGER_DECISION_LINK_MISMATCH")
        group_ops = [(index, operation) for index, operation in enumerate(operations)
                     if operation.get("group_id") == group_id]
        # A source tool can return without reserving a request (e.g. cancelled
        # before transport). The persisted zero-request return is still a real
        # acquisition attempt; it must not invent a paid HTTP reservation.
        selection_position = positions[selection_id]
        first_index = min((index for index, _ in group_ops), default=selection_position + 1)
        last_index = max((index for index, _ in group_ops), default=selection_position)
        if group_ops and (first_index <= selection_position or
                [index for index, _ in group_ops] != list(range(first_index, last_index + 1))):
            raise ValueError("ACTION_LEDGER_ACQUISITION_ORDER_MISMATCH")
        before = _remaining_before(operations, limits, first_index)
        cost = {key: 0 for key in before}
        request_ids = []
        local_operation_id = None
        for _, operation in group_ops:
            kind = operation.get("kind")
            if operation.get("selection_operation_id") != selection_id or kind not in {"http", "local_read"}:
                raise ValueError("ACTION_LEDGER_GROUP_LINK_MISMATCH")
            key = {"http": "http_calls", "local_read": "local_reads"}.get(kind)
            if key is not None:
                cost[key] = cost.get(key, 0) + 1
            if kind == "http":
                request_ids.append(operation.get("operation_id"))
            if kind == "local_read":
                if local_operation_id is not None:
                    raise ValueError("ACTION_LEDGER_MULTIPLE_LOCAL_READS")
                local_operation_id = operation.get("operation_id")
        after = _subtract(before, cost)
        result_path = link_path.parent / "result.json"
        result = _read(result_path) if result_path.exists() else None
        admission_record = (_read(link_path.parent / "admission.json")
                            if (link_path.parent / "admission.json").exists() else None)
        document = result.get("document") if isinstance(result, Mapping) and isinstance(result.get("document"), Mapping) else None
        return_context = result.get("return_context") if isinstance(result, Mapping) else None
        receipt = result.get("local_receipt") if isinstance(result, Mapping) and isinstance(result.get("local_receipt"), Mapping) else None
        statuses = {str(operation.get("status")) for _, operation in group_ops}
        if admission_record is not None and admission_record.get('status') == 'rejected':
            acquisition_status = "rejected_late_return"
        elif result is None:
            acquisition_status = "failed" if "failed" in statuses else "pending"
        elif receipt and receipt.get("status") == "failed":
            acquisition_status = "failed"
        elif "failed" in statuses and not document and not result.get("requests"):
            acquisition_status = "failed"
        else:
            acquisition_status = "returned"
        if admission_record is not None and admission_record.get('status') == 'rejected':
            document = None
            return_context = None
        observation = document.get("return_observation") if document else None
        transport_status = (receipt.get('status') if receipt else
                            'failed' if 'failed' in statuses else
                            'returned' if result is not None else 'pending')
        by_selection[selection_id] = {
            "acquisition_id": group_id,
            "action_digest": link.get("action_digest"),
            "decision_digest": link.get("decision_digest"),
            "request_ids": request_ids,
            "operation_ids": [operation['operation_id'] for _, operation in group_ops],
            "local_read_operation_id": local_operation_id,
            "resource_cost": cost,
            "budget_before": before,
            "budget_after": after,
            "acquisition_status": acquisition_status,
            "admission_status": admission_record.get('status') if admission_record else (
                'legacy_unspecified' if result is not None else None),
            "admission_reason": admission_record.get('reason') if admission_record else None,
            "admission_digest": admission_record.get('admission_digest') if admission_record else None,
            "transport_status": transport_status,
            "document_ids": [document.get("document_id")] if document and document.get("document_id") else [],
            "return_observation": _copy(observation),
            "return_context": _compact_return_context(return_context),
            "result_digest": digest(result) if result is not None else None,
            "last_operation_index": last_index,
        }
    return by_selection


def _project_verified_ledger(folder: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Project the snapshot already validated by acquisition_replay.

    This internal entry is called only while replay owns execution/state locks.
    Public callers must use build_action_value_ledger for archive verification.
    """
    folder = Path(folder)
    operations = list(state.get("operations", []))
    positions = {operation.get("operation_id"): index for index, operation in enumerate(operations)}
    if len(positions) != len(operations):
        raise ValueError("ACTION_LEDGER_DUPLICATE_OPERATION")
    accepted_operations = [operation for operation in operations
                           if operation.get("kind") == "model" and operation.get("decision_digest")
                           and operation.get('selection_boundary_version') == BOUNDARY_VERSION]
    accepted = {operation["operation_id"]: operation for operation in accepted_operations}
    acquisitions = _acquisition_index(folder, state, operations, accepted)
    history = state.get("question_return_assessments", [])
    assessments: dict[str, Mapping[str, Any]] = {}
    for record in history:
        context = record.get("context") if isinstance(record, Mapping) else None
        selection_id = context.get("selection_operation_id") if isinstance(context, Mapping) else None
        if selection_id:
            if selection_id in assessments:
                raise ValueError("ACTION_LEDGER_MULTIPLE_RETURN_ASSESSMENTS")
            assessments[selection_id] = record
    limits = _resource_limits(state)
    total_reserved = _resource_cost(operations, limits)
    if any(total_reserved[key] != state['spent'][key] for key in ('http_calls', 'llm_calls')):
        raise ValueError("ACTION_LEDGER_RESOURCE_ACCOUNTING_MISMATCH")
    _subtract(limits, total_reserved)
    entries = []
    for sequence, operation in enumerate(accepted_operations, 1):
        operation_id = operation["operation_id"]
        directory = folder / "operations" / operation_id
        selection = _read(directory / "selection.json")
        decision = _read(directory / "decision.json")
        if selection.get("decision_digest") != operation.get("decision_digest") or digest(decision) != operation.get("decision_digest"):
            raise ValueError("ACTION_LEDGER_DECISION_MISMATCH")
        choice = selection.get("choice") if isinstance(selection.get("choice"), Mapping) else {}
        selection_input = _read(directory / "input.json")
        selection_before = _copy(selection_input.get("remaining", {}))
        if selection_before != _remaining_before(operations, limits, positions[operation_id]):
            raise ValueError("ACTION_LEDGER_SELECTION_BUDGET_MISMATCH")
        selection_after = _subtract(selection_before, {'llm_calls': 1})
        value = _value_projection(choice, decision)
        acquisition = acquisitions.get(operation_id)
        response_action = decision.get("action")
        assessment = assessments.get(operation_id)
        if acquisition:
            next_position = acquisition["last_operation_index"]
        else:
            next_position = positions.get(operation_id, -1)
        next_operation = next((candidate for candidate in accepted_operations
                               if positions.get(candidate.get("operation_id"), -1) > next_position), None)
        observation = acquisition.get("return_observation") if acquisition else None
        semantic_status = "provisional:" + str((assessment.get("assessment") or {}).get("outcome")) if assessment else (
            "unassessed" if observation else "not_applicable")
        evidence_scope = "technical_return_only" if observation else (
            "acquisition_without_document" if acquisition else "no_acquisition")
        resource_cost = acquisition["resource_cost"] if acquisition else {key: 0 for key in limits}
        body = {
            "sequence": sequence,
            "selection_operation_id": operation_id,
            "candidate_id": choice.get("candidate_id"),
            "action": response_action,
            "selection_basis": value["selection_basis"],
            "selection_scope": value["selection_scope"],
            "value_status": value["value_status"],
            "expected_information_value": value["expected_information_value"],
            "expected_cost": value["expected_cost"],
            "expected_net_value": value["expected_net_value"],
            "value_support_scope": value['value_support_scope'],
            "calibration_status": value['calibration_status'],
            "candidate_value_assessments": value['candidate_value_assessments'],
            "diagnostics_status": selection.get('diagnostics_status'),
            "numeric_voi_calibrated": False,
            "selection_budget_before": selection_before,
            "selection_budget_after": selection_after,
            "selection_resource_cost": {"llm_calls": 1},
            "acquisition_id": acquisition.get("acquisition_id") if acquisition else None,
            "request_ids": acquisition.get("request_ids", []) if acquisition else [],
            "local_read_operation_id": acquisition.get("local_read_operation_id") if acquisition else None,
            "resource_cost": resource_cost,
            "acquisition_budget_before": acquisition.get("budget_before") if acquisition else None,
            "acquisition_budget_after": acquisition.get("budget_after") if acquisition else None,
            "acquisition_status": acquisition.get("acquisition_status", "not_requested") if acquisition else "not_requested",
            "document_ids": acquisition.get("document_ids", []) if acquisition else [],
            "return_observation": _copy(observation),
            "result_digest": acquisition.get("result_digest") if acquisition else None,
            "action_digest": acquisition.get("action_digest") if acquisition else None,
            "decision_digest": operation.get("decision_digest"),
            "policy_fingerprint": selection.get("policy", {}).get("fingerprint"),
            "question": _question(decision),
            "unknown_to_resolve": decision.get("unknown_to_resolve"),
            "useful_if": decision.get("useful_if"),
            "return_context": acquisition.get("return_context") if acquisition else None,
            "return_assessment": _compact_assessment(assessment),
            "return_assessment_operation_id": assessment.get("assessment_operation_id") if assessment else None,
            "next_selection_operation_id": next_operation.get("operation_id") if next_operation else None,
            "intervening_unaccepted_model_operation_ids": [candidate['operation_id'] for candidate in operations
                if candidate.get('kind') == 'model' and candidate['operation_id'] not in accepted
                and next_position < positions[candidate['operation_id']]
                and (next_operation is None or positions[candidate['operation_id']] < positions[next_operation['operation_id']])],
            "observed_information_gain": None,
            "realized_decision_gain": None,
            "numeric_voi_training_eligible": False,
            "evidence_scope": evidence_scope,
            "semantic_status": semantic_status,
            "formal_query_value_decision": value["formal_query_value_decision"],
            # Round132 public evaluation bridge.  This is the exact actor-visible
            # input captured before this selection; no return/result/evaluator
            # sidecar is copied here.
            "pre_action_snapshot": _copy(selection_input),
            "pre_action_snapshot_digest": digest(selection_input),
            "selection_at": operation.get("started_at"),
            "prediction_cutoff": selection_input.get("reference_utc"),
            "target_key": selection_input.get("match_input") or state.get("run_id"),
            "candidate_generation_complete": bool(
                isinstance(choice.get("diagnostics"), Mapping)
                and choice["diagnostics"].get("candidate_generation_complete") is True
            ),
        }
        # Stable identity and pre-action digest survive later return/assessment
        # updates. The entry digest identifies this particular ledger snapshot.
        prospective = {key: body[key] for key in (
            'selection_operation_id', 'candidate_id', 'action', 'decision_digest',
            'policy_fingerprint', 'question', 'unknown_to_resolve', 'useful_if',
            'selection_basis', 'selection_scope', 'candidate_value_assessments',
            'value_status', 'expected_information_value', 'expected_cost', 'expected_net_value',
            'selection_budget_before', 'formal_query_value_decision')}
        body['prospective_digest'] = digest(prospective)
        entry = {"entry_id": digest({'run_id': state['run_id'], 'operation_id': operation_id,
                                    'decision_digest': body['decision_digest']}),
                 "entry_digest": digest(body), **body}
        entries.append(entry)
    total_cost = {key: 0 for key in limits}
    for entry in entries:
        for key, amount in entry["resource_cost"].items():
            total_cost[key] = total_cost.get(key, 0) + amount
    accounted_ids = set(accepted)
    for acquisition in acquisitions.values():
        accounted_ids.update(acquisition['operation_ids'])
    unassigned = [{key: operation.get(key) for key in ('operation_id', 'kind', 'status', 'group_id',
                   'selection_operation_id')} for operation in operations if operation['operation_id'] not in accounted_ids]
    # A current runtime's source reservation always belongs to an accepted
    # selection. Historical unbound operations remain explicitly unassigned.
    if state['config'].get('selection_boundary_version') == BOUNDARY_VERSION and any(
            operation['kind'] != 'model' for operation in unassigned):
        raise ValueError('ACTION_LEDGER_ORPHAN_ACQUISITION_OPERATION')
    summary = {
        "entry_count": len(entries),
        "query_count": sum(entry["action"] in ACQUISITION_ACTIONS for entry in entries),
        "terminal_count": sum(entry["action"] in TERMINAL_ACTIONS for entry in entries),
        "returned_count": sum(entry["acquisition_status"] == "returned" for entry in entries),
        "failed_count": sum(entry["acquisition_status"] == "failed" or
                             entry.get("transport_status") == "failed" for entry in entries),
        "pending_count": sum(entry["acquisition_status"] == "pending" for entry in entries),
        "assessment_count": sum(entry["return_assessment"] is not None for entry in entries),
        "resource_cost": total_cost,
        "selection_llm_calls": len(entries),
        "unassigned_resource_cost": _resource_cost(unassigned, limits),
        "total_reserved_resource_cost": total_reserved,
        "remaining_budget": _subtract(limits, total_reserved),
        "cost_basis": "runtime_reservations_including_failed_and_pending; not money_or_loss_units",
        "numeric_voi_calibrated": False,
        "realized_decision_gain": None,
    }
    body = {
        "schema_version": VERSION,
        "run_id": state.get("run_id"),
        "state_digest": digest(state),
        "entries": entries,
        "unassigned_operations": unassigned,
        "summary": summary,
        "numeric_voi_calibrated": False,
        "realized_decision_gain": None,
    }
    return {**body, "ledger_digest": digest(body)}


def build_action_value_ledger(folder: str | Path) -> dict[str, Any]:
    """Verify then project a stable snapshot, without provider/network calls."""
    from .acquisition_replay import verify_run
    ledger = verify_run(folder, include_action_value_ledger=True)['action_value_ledger']
    # Episode persistence is an optional extension.  Older runs do not have a
    # journal and therefore retain the exact legacy projection.  If present,
    # carry the immutable snapshot through the same ledger/export path; a
    # PARTIAL snapshot remains PARTIAL and never upgrades the ledger's VoI
    # eligibility or invents missing posterior/loss fields.
    episode_path = Path(folder) / "episode-actions.jsonl"
    if episode_path.exists():
        snapshot = EpisodePersistence(folder).snapshot()
        ledger = {**ledger,
                  "episode_persistence": snapshot,
                  "episode_persistence_status": snapshot["report"]["status"],
                  "episode_persistence_eligible_for_replay": bool(
                      snapshot["report"].get("eligible_for_replay", False))}
        unsigned = {key: value for key, value in ledger.items() if key != "ledger_digest"}
        ledger["ledger_digest"] = digest(unsigned)
    return ledger


def write_action_value_ledger(folder: str | Path) -> dict[str, Any]:
    """Persist the current deterministic projection beside the run archive."""
    folder = Path(folder)
    with run_lock(folder/'execution.lock', folder.name, timeout=0):
        ledger = build_action_value_ledger(folder)
        atomic_write(folder / "action-value-ledger.json", ledger)
        return ledger


__all__ = ["VERSION", "build_action_value_ledger", "write_action_value_ledger"]
