"""Read-only projections used by the FootballMind demo UI.

The workflow journal remains the source of truth.  This module deliberately
does not run tools, calculate a new forecast, or fill missing values with
presentation defaults.  It turns the existing journal into a stable, audited
read model for the browser.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import math
from typing import Any, Mapping


RUN_VIEW_SCHEMA = "footballmind-ui-run-view-v1"
MATCHES_SCHEMA = "footballmind-ui-matches-v1"
PROBABILITY_FIELDS = ("home_win_probability", "draw_probability", "away_win_probability")
CHANGE_TOLERANCE = 1e-9


def _plain(value: Any) -> Any:
    """Convert existing dataclasses/envelopes without changing their meaning."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _events(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(event) for event in (run.get("events") or []) if isinstance(event, Mapping)]


def _event_data(event: Mapping[str, Any]) -> dict[str, Any]:
    return _dict(event.get("data"))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_prediction(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(_finite(value.get(field)) for field in PROBABILITY_FIELDS)


def _prediction_compatible(before: Mapping[str, Any], after: Mapping[str, Any], run: Mapping[str, Any]) -> bool:
    if not (_valid_prediction(before) and _valid_prediction(after)):
        return False
    # These are the identity fields that make a probability delta meaningful.
    for key in ("model_version", "prediction_cutoff", "information_mode"):
        left = before.get(key, run.get(key) if key != "model_version" else None)
        right = after.get(key, run.get(key) if key != "model_version" else None)
        if key == "model_version" and (left is None or right is None):
            return False
        if left != right:
            return False
    # A prediction schema marker is optional in historical journals.  When it
    # exists on either side it must agree.
    for key in ("schema_version", "prediction_schema", "schema"):
        if before.get(key) is not None or after.get(key) is not None:
            if before.get(key) != after.get(key):
                return False
    return True


def _delta(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None,
           run: Mapping[str, Any]) -> dict[str, float | None] | None:
    if not before or not after or not _prediction_compatible(before, after, run):
        return None
    result: dict[str, float | None] = {
        "home": float(after[PROBABILITY_FIELDS[0]]) - float(before[PROBABILITY_FIELDS[0]]),
        "draw": float(after[PROBABILITY_FIELDS[1]]) - float(before[PROBABILITY_FIELDS[1]]),
        "away": float(after[PROBABILITY_FIELDS[2]]) - float(before[PROBABILITY_FIELDS[2]]),
        "l1": None,
    }
    if _finite(before.get("l1")) and _finite(after.get("l1")):
        result["l1"] = float(after["l1"]) - float(before["l1"])
    return result


def _manifest_ids(value: Any) -> set[str]:
    """Extract explicit evidence IDs from a feature manifest.

    Raw refs, URLs and hashes are intentionally ignored.  They identify an
    input artifact, but do not prove that f consumed an evidence record.
    """
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key in ("evidence_id", "evidence_ids", "consumed_evidence_ids", "f_consumed_evidence_ids"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                found.add(item)
            elif isinstance(item, (list, tuple, set)):
                found.update(str(x) for x in item if isinstance(x, str) and x.strip())
        for key, item in value.items():
            if key not in {"evidence_id", "evidence_ids", "consumed_evidence_ids", "f_consumed_evidence_ids"}:
                found.update(_manifest_ids(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found.update(_manifest_ids(item))
    return found


def _run_evidence_ids(run: Mapping[str, Any]) -> set[str]:
    return {str(item.get("evidence_id")) for item in (run.get("evidence") or [])
            if isinstance(item, Mapping) and item.get("evidence_id")}


def _feature_payloads(run: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if event:
        data = _event_data(event)
        for key in ("features", "feature_snapshot", "feature_manifest"):
            item = data.get(key)
            if isinstance(item, Mapping):
                payloads.append(dict(item))
        snapshot = event.get("snapshot")
        if isinstance(snapshot, Mapping) and isinstance(snapshot.get("features"), Mapping):
            payloads.append(dict(snapshot["features"]))
    # For an event-scoped decision, only the event snapshot or explicit event
    # data can prove which feature input was consumed.  Falling back to the
    # current run-level feature object would make an old/new event look as if
    # it consumed whatever happens to be present at the end of the run.
    if event is None and isinstance(run.get("features"), Mapping):
        payloads.append(dict(run["features"]))
    return payloads


def _consumed_for_event(run: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[set[str], bool]:
    data = _event_data(event)
    ids: set[str] = set()
    explicit = False
    for key in ("consumed_evidence_ids", "f_consumed_evidence_ids", "evidence_ids_consumed"):
        if key in data:
            explicit = True
            ids.update(_manifest_ids({key: data.get(key)}))
    for payload in _feature_payloads(run, event):
        for key in ("consumed_evidence_ids", "f_consumed_evidence_ids", "input_manifest", "feature_manifest"):
            if key in payload:
                explicit = True
                ids.update(_manifest_ids(payload.get(key)))
    # Some older journals did not copy the feature snapshot into every event,
    # but did persist a run-level manifest.  Accept that manifest only when it
    # contains explicit evidence IDs; raw URLs/hashes still cannot establish
    # f consumption.
    if not ids and not explicit and event.get("type") == "prediction_updated":
        run_features = run.get("features")
        if isinstance(run_features, Mapping):
            legacy_ids = _manifest_ids(run_features.get("input_manifest"))
            if legacy_ids:
                ids.update(legacy_ids)
                explicit = True
    # Restrict the public list to evidence records actually present in this run
    # when possible.  A valid audit may refer to a compact external evidence
    # ledger, so preserve an explicit ID if the run itself does not have rows.
    available = _run_evidence_ids(run)
    if available:
        ids = {item for item in ids if item in available}
    return ids, explicit


def _consumed_evidence(run: Mapping[str, Any]) -> tuple[set[str], bool, dict[str, Any] | None]:
    ids: set[str] = set()
    explicit = False
    latest: dict[str, Any] | None = None
    for event in _events(run):
        if event.get("type") not in {"features_ready", "prediction_updated"}:
            continue
        event_ids, event_explicit = _consumed_for_event(run, event)
        if event_ids or event_explicit:
            ids.update(event_ids)
            latest = event
            explicit = True
    return ids, explicit, latest


def _prediction_projection(run: Mapping[str, Any]) -> tuple[dict[str, Any], set[str], bool, list[dict[str, Any]]]:
    events = _events(run)
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    after_events: list[dict[str, Any]] = []
    consumed_ids, consumed_explicit, _ = _consumed_evidence(run)

    # The first valid prediction_updated.after is the initial prediction.  A
    # later update is an after only when its own audited feature input exists.
    for event in events:
        if event.get("type") != "prediction_updated":
            continue
        data = _event_data(event)
        candidate = data.get("after")
        if not _valid_prediction(candidate):
            continue
        candidate = deepcopy(dict(candidate))
        reason = data.get("update_reason")
        if before is None:
            initial = data.get("before") if _valid_prediction(data.get("before")) else candidate
            before = deepcopy(dict(initial))
            if reason not in {None, "INITIAL_PREDICTION"} and _consumed_for_event(run, event)[0]:
                after = candidate
                after_events.append(event)
            continue
        if reason == "INITIAL_PREDICTION":
            continue
        event_ids, event_explicit = _consumed_for_event(run, event)
        if event_ids or event_explicit:
            after = candidate
            after_events.append(event)

    current = None
    for candidate in (run.get("prediction"), run.get("last_prediction"), after, before):
        if _valid_prediction(candidate):
            current = deepcopy(dict(candidate))
            break

    change_status = "unknown"
    delta = _delta(before, after, run)
    if after is None:
        change_status = "not_applicable" if before is not None else "unknown"
    elif delta is None:
        change_status = "unknown"
    elif any(abs(float(delta[key])) > CHANGE_TOLERANCE for key in ("home", "draw", "away")):
        change_status = "changed"
    else:
        change_status = "unchanged_after_f"

    return ({"before": before, "after": after, "current": current, "delta": delta,
             "change_status": change_status}, consumed_ids, consumed_explicit, after_events)


def _f_projection(run: Mapping[str, Any], prediction: Mapping[str, Any], consumed_ids: set[str],
                  consumed_explicit: bool, after_events: list[dict[str, Any]]) -> dict[str, Any]:
    events = _events(run)
    f_events = [e for e in events if e.get("type") == "features_ready"]
    configured = _dict(run.get("tool_configuration"))
    tools = _dict(configured.get("tools"))
    if tools.get("get_features") in {"NEEDS_CONFIGURATION", "NOT_CONFIGURED"}:
        status = "not_configured"
    elif any(_event_data(e).get("status") in {"FAILED", "MODEL_UNAVAILABLE"} for e in events
             if e.get("type") in {"features_ready", "prediction_updated", "tool_returned"}):
        status = "failed"
    elif not f_events and not after_events and not isinstance(run.get("features"), Mapping):
        status = "not_run"
    elif after_events and prediction.get("after") is not None:
        status = {"changed": "executed_changed", "unchanged_after_f": "executed_no_change"}.get(
            prediction.get("change_status"), "unknown")
    elif f_events or isinstance(run.get("features"), Mapping):
        # A first prediction may have used f, but without an audited before /
        # after pair it cannot support a change claim.
        status = "unknown" if consumed_explicit or consumed_ids else "not_run"
    else:
        status = "unknown"

    feature = None
    for event in reversed(events):
        if event.get("type") == "features_ready":
            feature = _event_data(event)
            snapshot = event.get("snapshot")
            if isinstance(snapshot, Mapping) and isinstance(snapshot.get("features"), Mapping):
                feature = dict(snapshot["features"])
            break
    if feature is None and isinstance(run.get("features"), Mapping):
        feature = dict(run["features"])
    model = prediction.get("after") or prediction.get("current") or {}
    missing_reason = None
    if status == "not_run":
        missing_reason = "没有 f 消费证据的审计记录"
    elif status == "unknown":
        missing_reason = "已有特征或预测，但缺少可确认的 before/after 消费链"
    return {"status": status,
            "model_version": model.get("model_version") if isinstance(model, Mapping) else None,
            "feature_version": feature.get("feature_version") if isinstance(feature, Mapping) else None,
            "consumed_evidence_ids": sorted(consumed_ids),
            "feature_snapshot_hash": (feature or {}).get("feature_snapshot_hash") if isinstance(feature, Mapping) else None,
            "modeled_claim_count": (feature or {}).get("modeled_claim_count") if isinstance(feature, Mapping) else None,
            "missing_reason": missing_reason}


def _candidate_sources(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for key in ("candidate_actions", "candidates", "candidate_menu", "action_candidates"):
        value = run.get(key)
        if isinstance(value, list):
            sources.extend(item for item in value if isinstance(item, Mapping))
    for event in _events(run):
        data = _event_data(event)
        for key in ("candidate_actions", "candidates", "candidate_menu", "action_candidates"):
            value = data.get(key)
            if isinstance(value, list):
                sources.extend(item for item in value if isinstance(item, Mapping))
    return [dict(item) for item in sources]


def _g_projection(run: Mapping[str, Any], consumed_ids: set[str]) -> dict[str, Any]:
    candidates = _candidate_sources(run)
    decisions = [_event_data(e) for e in _events(run) if e.get("type") == "policy_decision"]
    explicit = _dict(run.get("g"))
    authority = (explicit.get("selection_authority") or run.get("selection_authority") or
                 (decisions[-1].get("selection_authority") if decisions else None))
    policy_version = str(run.get("policy_version") or "")
    synthetic = bool(run.get("test_mode")) or run.get("mode") in {"test", "synthetic", "demo"}
    has_empirical = False
    for item in candidates:
        has_empirical |= _finite(item.get("expected_signed_gain")) or _finite(item.get("expected_net_voi"))
    has_empirical |= _finite(explicit.get("expected_signed_gain")) or _finite(explicit.get("expected_net_voi"))
    if authority not in {"empirical_g_only", "fixed_order", "none", "unknown"}:
        authority = "unknown"
    if synthetic:
        status = "synthetic_diagnostic_only"
        authority = "none" if authority in {None, "unknown", "empirical_g_only"} else authority
    elif has_empirical and authority == "empirical_g_only":
        status = "empirical_supported"
    elif policy_version.startswith("fixed-order") or any(d.get("reason") == "FIXED_QUERY_ORDER" for d in decisions):
        status = "unknown"
        authority = "fixed_order"
    else:
        status = explicit.get("status") if explicit.get("status") in {"empirical_supported", "unknown", "synthetic_diagnostic_only", "not_configured"} else "unknown"
    selected_ids = {str(d.get("action_id")) for d in decisions if d.get("action_id")}
    queried = {str(topic) for topic in (run.get("queried_topics") or [])}
    normalized: list[dict[str, Any]] = []
    for item in candidates:
        action_id = item.get("action_id")
        topic = item.get("information_key") or item.get("topic")
        item_ids = [str(x) for x in item.get("evidence_ids", []) if isinstance(x, str)]
        selected = item.get("status") == "selected" or (action_id and str(action_id) in selected_ids)
        executed = item.get("request_executed")
        if executed is None:
            executed = bool(topic and str(topic) in queried)
        f_status = item.get("f_status")
        if f_status not in {"not_configured", "not_run", "executed_changed", "executed_no_change", "failed", "unknown"}:
            f_status = "unknown"
        if item_ids and consumed_ids.intersection(item_ids):
            f_status = "executed_changed" if item.get("prediction_changed") is True else "executed_no_change"
        gain = item.get("expected_signed_gain") if status == "empirical_supported" and _finite(item.get("expected_signed_gain")) else None
        net_voi = item.get("expected_net_voi") if status == "empirical_supported" and _finite(item.get("expected_net_voi")) else None
        normalized.append({"action_id": action_id, "information_key": topic, "topic": topic,
                           "status": "selected" if selected else (item.get("status") if item.get("status") in {"unselected", "unavailable", "not_executed"} else "not_executed"),
                           "expected_signed_gain": gain, "expected_net_voi": net_voi,
                           "gain_status": "empirical" if gain is not None or net_voi is not None else "unknown",
                           "propensity": item.get("propensity") if _finite(item.get("propensity")) else None,
                           "cost": item.get("cost") if _finite(item.get("cost")) else None,
                           "cost_status": "known" if _finite(item.get("cost")) else "unknown",
                           "evidence_ids": item_ids, "f_status": f_status,
                           "request_executed": bool(executed) if isinstance(executed, bool) else None,
                           "selection_reason": item.get("selection_reason") or item.get("reason")})
    stop_reason = run.get("reason")
    return {"status": status, "selection_authority": authority or "unknown",
            "stop_reason": stop_reason, "candidate_availability": "returned" if normalized else "not_returned",
            "candidate_actions": normalized}


def _evidence_projection(run: Mapping[str, Any], consumed_ids: set[str], consumed_explicit: bool) -> list[dict[str, Any]]:
    result = []
    for raw in (run.get("evidence") or []):
        if not isinstance(raw, Mapping):
            continue
        item = deepcopy(dict(raw))
        evidence_id = item.get("evidence_id")
        claims = item.get("claims")
        if not isinstance(claims, list):
            claims = []
        item["claims"] = claims
        item["claim_count"] = item.get("claim_count") if item.get("claim_count") is not None else (len(claims) if claims else None)
        item["raw_ref"] = item.get("raw_ref") or item.get("raw_payload_ref")
        item["f_consumed"] = True if evidence_id in consumed_ids else (False if consumed_explicit else None)
        supplied = item.get("admission")
        item["admission"] = supplied if supplied in {"verified", "unmodeled", "rejected", "unknown"} else ("verified" if evidence_id in consumed_ids else "unknown")
        item["availability_status"] = "known" if item.get("available_at") else "unknown"
        result.append(item)
    return result


def _source_label(run: Mapping[str, Any]) -> str:
    mode = run.get("mode")
    if mode in {"test", "synthetic"} or run.get("test_mode"):
        return "测试模式 · 合成压力测试"
    if mode == "demo":
        return "演示数据 · 接口无可用比赛"
    if run.get("time_mode", "archive_replay") == "live":
        return "真实在线决策 · 关联时点运行"
    if run.get("information_mode") == "historical_reconstruction":
        return "真实历史重建 · 明示可用时间假设"
    return "真实档案回放 / 本地逐批披露"


def build_run_view(run: Mapping[str, Any]) -> dict[str, Any]:
    """Build ``footballmind-ui-run-view-v1`` from one existing run record."""
    source = _plain(run)
    if not isinstance(source, Mapping):
        raise TypeError("run must be a mapping")
    source = dict(source)
    prediction, consumed_ids, consumed_explicit, after_events = _prediction_projection(source)
    f_view = _f_projection(source, prediction, consumed_ids, consumed_explicit, after_events)
    g_view = _g_projection(source, consumed_ids)
    time_mode = source.get("time_mode") or _dict(source.get("tool_configuration")).get("time_mode") or "archive_replay"
    information_mode = source.get("information_mode") or _dict(source.get("tool_configuration")).get("information_mode") or "operational"
    mode = source.get("mode") or "unknown"
    is_demo = bool(source.get("is_demo") or source.get("test_mode") or mode in {"test", "synthetic", "demo"})
    configured_external = _dict(source.get("tool_configuration")).get("external_requests_enabled")
    if configured_external is None and time_mode == "archive_replay":
        configured_external = False
    state = source.get("state") or "unknown"
    lifecycle = {"state": state, "reason": source.get("reason"), "created_at": source.get("created_at"),
                 "updated_at": source.get("updated_at"), "is_terminal": str(state).upper() in {"STOPPED", "DEFERRED", "COMPLETED", "FAILED", "BUDGET_EXHAUSTED", "DEADLINE_EXPIRED"},
                 "is_demo": is_demo, "is_live": time_mode == "live"}
    uncertainty: list[dict[str, Any]] = []
    if prediction["before"] is None:
        uncertainty.append({"code": "prediction_before_missing", "message": "没有可核验的初始预测"})
    if prediction["after"] is None and prediction["before"] is not None:
        uncertainty.append({"code": "prediction_after_not_applicable", "message": "本次运行没有经过已确认的额外信息与 f 更新"})
    if f_view["status"] in {"not_run", "unknown"}:
        uncertainty.append({"code": "f_consumption_unknown", "message": f_view["missing_reason"] or "无法确认 f 是否消费了证据"})
    if g_view["status"] in {"unknown", "synthetic_diagnostic_only"}:
        uncertainty.append({"code": "g_unavailable", "message": "empirical g 数值或选择权限不可用"})
    if any(item.get("available_at") is None for item in source.get("evidence", []) if isinstance(item, Mapping)):
        uncertainty.append({"code": "evidence_availability_unknown", "message": "至少一条证据缺少可用时间"})
    view = {
        "schema_version": RUN_VIEW_SCHEMA,
        "run_id": source.get("run_id"), "match_id": source.get("match_id"), "mode": mode,
        "information_mode": information_mode, "time_mode": time_mode,
        "source_label": _source_label(source), "prediction_cutoff": source.get("prediction_cutoff"),
        "is_live": time_mode == "live", "is_demo": is_demo,
        "external_requests_enabled": configured_external,
        "match": deepcopy(source.get("match")), "lifecycle": lifecycle,
        "prediction": prediction, "f": f_view, "g": g_view,
        "evidence": _evidence_projection(source, consumed_ids, consumed_explicit),
        "uncertainty": uncertainty, "budget": deepcopy(source.get("budget")),
        "quality": deepcopy(source.get("quality") or []),
        # Audit fields remain available to the UI.  They are copies of the
        # existing journal, not a second mutable state store.
        "events": deepcopy(source.get("events") or []), "state": state, "reason": source.get("reason"),
        "queried_topics": deepcopy(source.get("queried_topics") or []),
        "policy_version": source.get("policy_version"), "policy_configuration": deepcopy(source.get("policy_configuration") or {}),
        "tool_configuration": deepcopy(source.get("tool_configuration") or {}),
        "features": deepcopy(source.get("features")), "prediction_current": prediction["current"],
        "last_prediction": prediction["after"] or prediction["before"],
        "report": deepcopy(source.get("report")),
    }
    return view


def build_matches_view(payload: Any, *, mode: str = "real", information_mode: str = "operational",
                       time_mode: str = "archive_replay", cutoff: str | None = None) -> dict[str, Any]:
    """Add the versioned metadata and nullable eligibility fields to matches."""
    plain = _plain(payload)
    if isinstance(plain, list):
        base, status, rows, error = {}, "OK", plain, None
    elif isinstance(plain, Mapping):
        base, status, rows, error = dict(plain), plain.get("status"), plain.get("data") or [], plain.get("error")
    else:
        base, status, rows, error = {}, "FAILED", [], "invalid match result"
    rows = [deepcopy(dict(row)) for row in rows if isinstance(row, Mapping)]
    for row in rows:
        existing = row.get("eligibility") if isinstance(row.get("eligibility"), Mapping) else {}
        row["eligibility"] = {"prematch_at_cutoff": existing.get("prematch_at_cutoff"),
                               "fixture_available_at_cutoff": existing.get("fixture_available_at_cutoff"),
                               "identity_verified": existing.get("identity_verified"),
                               "odds_available_at_cutoff": existing.get("odds_available_at_cutoff"),
                               "run_allowed": existing.get("run_allowed"), "reason": existing.get("reason")}
    is_demo = mode in {"test", "synthetic", "demo"}
    source_label = ("测试模式 · 合成压力测试" if is_demo else
                    "真实在线决策 · 关联时点运行" if time_mode == "live" else
                    "真实历史重建 · 明示可用时间假设" if information_mode == "historical_reconstruction" else
                    "真实档案回放 / 本地逐批披露")
    result: dict[str, Any] = {**base, "schema_version": MATCHES_SCHEMA, "status": status, "data": rows,
                              "meta": {"mode": mode, "time_mode": time_mode, "information_mode": information_mode,
                                       "cutoff": cutoff, "source_label": source_label, "is_demo": is_demo,
                                       "is_live": time_mode == "live", "total": len(rows)}}
    if error is not None:
        result["error"] = error
    return result


# Descriptive aliases make the projection convenient for embedders and tests.
project_run = build_run_view
project_matches = build_matches_view
