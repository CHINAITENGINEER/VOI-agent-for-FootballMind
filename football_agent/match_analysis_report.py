"""User-facing match analysis from the acquired run, never from outcome labels.

The projection is deliberately a whitelist: it neither reads hidden outcomes
nor recomputes probabilities. Missing measurements remain unknown. A completed
defer is a useful analysis with gaps, whereas unresolved execution is incomplete.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlsplit

from football_spider.storage.raw_archive import atomic_write

VERSION = "footballmind-match-analysis-report-v1"
_EVIDENCE_ONLY_SCHEMA = "learned-voi-open-execution-v5"
_EVIDENCE_ONLY_NOTE = "缺少可用市场先验，仅整理已获取证据，不提供胜平负概率或数值信息增益"
_EVIDENCE_ONLY_REASONS = {
    "defer_prior_unavailable": "缺少可用市场先验，本轮仅整理已获取证据",
    "prior_unavailable": "缺少可用市场先验",
    "physical_budget_exhausted": "本轮可用资源预算已耗尽",
}
_UNITS = ("money", "requests", "latency_ms", "tokens")
_TOPICS = {"fixture_confirmation": "比赛及日期", "availability": "伤情与身体状态",
           "lineup": "阵容", "rotation": "轮换", "qualification": "晋级条件",
           "tactics": "战术", "other_match_context": "其他比赛背景"}
_REASONS = {
    "candidate_generation_incomplete": "候选信息发现尚不完整",
    "candidate_value_incomplete": "部分候选尚无可用的信息价值估计",
    "forecast_cutoff_elapsed": "本场预测的信息截止时间已到",
    "exploration_action_allowance_exhausted": "探索次数额度已用完",
    "exploration_resource_allowance_exhausted": "探索资源额度已用完",
    "computed_nonpositive_value": "已计算的净 VoI 不为正",
    "modeled_response_support_unavailable": "数值返回模型缺少适用支持",
    "response_model_unavailable": "尚无对应的信息返回模型",
    "execution_reconciliation_required": "外部执行结果尚需核对",
    "receipt_reconciliation_required": "返回记录未通过核验",
    "actual_resource_budget_overrun": "实际资源消耗超出额度",
    "step_limit": "已达到本轮步骤上限",
    "already_attempted": "同一信息已经尝试获取",
    "not_sampled_this_cycle": "本轮有限探索未抽中该候选",
    "not_selected_this_cycle": "本轮未选择该候选",
    "not_assessed_this_cycle": "本轮没有可用的数值评估",
    "source_explicitly_leaves_status_unresolved": "来源明确表示该状态尚未确定",
    "applicability_to_this_fixture_not_established": "尚未确认信息是否适用于本场比赛",
    "opposed_source_statements_need_scope_time_or_source_review": "来源陈述相反，需核对时间、对象与来源",
    "no_substantive_match_evidence_observed": "尚未取得实质性的比赛证据",
    "provider_no_data": "返回没有可用内容", "provider_failed": "获取失败",
    "after_information_cutoff": "返回晚于信息截止时间",
    "document_not_usable": "返回内容不可用于分析",
    "unmodeled_document_for_replanning_only": "已取得正文，供后续信息选择使用",
    "modeled_actual_return_verified": "数值模型返回已通过核验",
    "learned_text_return_verified": "正文返回与查询前冻结的响应分支一致",
    "text_return_support_unknown": "正文已保存，条件响应模型尚不支持该返回",
    "after_actual_return_clock": "正文可用时间晚于本次实际返回时钟",
    "empirical_current_menu_nonpositive": "当前候选的经验净改善估计均不为正；这不是全局风险保证",
    "context_gain_support_unknown": "剩余候选尚缺可用的查询收益支持，本轮有限探索已结束",
    "insufficient_independent_local_support": "相近状态与候选的独立历史比赛支持不足",
    "candidate_title_not_yet_observed": "尚未从实际读取的页面观察到候选标题",
    "predictor_text_response_not_supported": "当前预测器未学出正文响应，不能据此判定外部信息无价值",
    "small_sample_empirical_context": "依据少量独立历史比赛估计的条件查询收益",
    "positive_empirical_context_gain": "预计查询收益高于预先设定的获取成本",
    "unmodeled_tool": "当前收益模型尚未覆盖这一工具类型",
    "declared_menu_nonpositive_empirical_mean": "当前已声明候选的经验净收益均不为正",
    "uncalibrated_state_pattern_generalization": "历史相近候选上的状态泛化估计，当前状态组合未经局部验证",
    "frozen_context_predictor_observed_return": "已将返回后的可见状态交给冻结预测器重新评估；正文是否入模需查看消费记录",
    "no_admitted_body_segments": "没有通过正文规则的段落",
    "unknown_publication_time": "来源发布时间未知，正文未进入预测器",
    "invalid_or_ambiguous_contentful_publication": "来源发布时间无效或无法唯一绑定，正文未通过抽取检查",
}


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _probabilities(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("A probability vector must contain home/draw/away in that order")
    if any(_number(v) is None or not 0 <= v <= 1 for v in value) or abs(sum(value) - 1) > 1e-7:
        raise ValueError("Probabilities must be finite, nonnegative and sum to one")
    # Keep the runtime's original numbers; display rounding is not a new model.
    return deepcopy(list(value))


def _name(value):
    if isinstance(value, Mapping):
        value = value.get("name", value.get("canonical_name"))
    return value.strip() if isinstance(value, str) and value.strip() else None


def _url(value):
    if not isinstance(value, str) or any(ord(c) < 32 for c in value):
        return None
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return None
    if any(key.lower() in {"api_key", "key", "token", "password", "authorization"} for key, _ in parse_qsl(parts.query)):
        return None
    return value


def _action(action):
    value = _mapping(action)
    return {"action_id": value.get("action_id"), "tool": value.get("tool"),
            "source": value.get("source"), "url": _url(value.get("url")),
            "query": value.get("query") if isinstance(value.get("query"), str) else None,
            "cost_status": value.get("cost_status"),
            "declared_cost": {unit: _number(_mapping(value.get("cost")).get(unit)) for unit in _UNITS}}


def _value(candidate, *, evidence_only=False):
    value = _mapping(_mapping(candidate).get("value"))
    computed = value.get("status") == "COMPUTED" and _number(value.get("net_voi")) is not None
    result = {"status": "computed" if computed else "unknown", "model_status": value.get("status"),
            "net_voi": value.get("net_voi") if computed else None,
            "gross_evsi": _number(value.get("gross_evsi")) if computed else None,
            "converted_cost": _number(value.get("cost")),
            "loss": value.get("loss"), "reason": value.get("reason"),
            "meaning": "model_expected_risk_reduction_minus_declared_cost" if computed else "no_usable_numeric_value_estimate"}
    if _mapping(candidate).get("continuation_enumeration") is not None:
        immediate = _mapping(_mapping(candidate.get("response_enumeration")).get("value"))
        result.update(planning_horizon=2, immediate_gross_evsi=_number(immediate.get("gross_evsi")),
                      immediate_net_voi=_number(immediate.get("net_voi")),
                      meaning="bounded_two_step_expected_risk_reduction_minus_both_query_costs" if computed
                              else "no_usable_two_step_value_estimate")
    if _mapping(candidate).get("empirical_risk_estimate") is not None:
        empirical = _mapping(candidate["empirical_risk_estimate"])
        result.update(expected_predictor_gain=_number(empirical.get("expected_gain")),
                      meaning="empirical_fixed_predictor_risk_reduction_minus_declared_cost" if computed
                              else "no_supported_empirical_predictor_gain",
                      conditional_coverage_guaranteed=False)
    if value.get("schema_version") in {"context-empirical-query-value-v1", "official-optional-context-query-value-v1"}:
        result.update(expected_predictor_gain=_number(value.get("expected_signed_gain")),
            meaning="conditional_empirical_signed_gain_minus_declared_cost" if computed else "context_gain_support_unknown",
            conditional_coverage_guaranteed=False)
        estimate = _mapping(value.get("estimate"))
        if estimate.get("support_status") == "prototype_state_generalization":
            result.update(state_pattern_extrapolation=True, calibration_status="not_established",
                          support=deepcopy(estimate.get("support")),
                          meaning="uncalibrated_state_pattern_gain_estimate_minus_declared_cost")
    if evidence_only and any(result.get(key) is not None for key in (
            "net_voi", "gross_evsi", "expected_predictor_gain", "immediate_gross_evsi", "immediate_net_voi")):
        raise ValueError("Evidence-only reports cannot include numeric information gain")
    return result


def _reason(value, *, evidence_only=False):
    if evidence_only and value in _EVIDENCE_ONLY_REASONS:
        return _EVIDENCE_ONLY_REASONS[value]
    return _REASONS.get(value, value or "未记录原因")


def _resources(run, cycles):
    first = _mapping(cycles[0].get("public_before")) if cycles else {}
    budget = _mapping(first.get("budget"))
    limits = _mapping(budget.get("limits"))
    reserved = _mapping(run.get("reserved_cost"))
    actual = _mapping(run.get("measured_resources"))
    unknown = _mapping(run.get("unmeasured_resource_receipts"))
    measured = {}
    for unit in _UNITS:
        amount = _number(actual.get(unit))
        missing = unknown.get(unit)
        missing = missing if type(missing) is int and missing >= 0 else None
        measured[unit] = {"measured_part": amount, "unmeasured_receipts": missing,
                          "total": amount if missing == 0 else None,
                          "status": "complete" if amount is not None and missing == 0 else "partial" if amount is not None and missing else "unknown"}
    return {"declared_limits": {u: _number(limits.get(u)) for u in _UNITS},
            "reserved_cost": {u: _number(reserved.get(u)) for u in _UNITS},
            "exploration_reserved_cost": {u: _number(_mapping(run.get("exploration_reserved_cost")).get(u)) for u in _UNITS},
            "measured_resources": measured,
            "money_semantics": "declared_policy_credits_are_not_an_actual_currency_bill"}


def _evidence_summary(record):
    topics = record.get("topics", [])
    kinds = "、".join(_TOPICS.get(topic, str(topic)) for topic in topics) or "来源背景"
    if record.get("applicability") == "match_scoped_source_assertion":
        summary = f"来源包含与本场明确绑定的{kinds}陈述；仍属于来源陈述。"
    else:
        summary = f"来源包含{kinds}背景，尚未确认其对本场的适用性。"
    if record.get("modality") == "uncertain_source_statement":
        summary += "原文明确保留了不确定性。"
    return summary


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _gain_model_summary(prediction):
    value = _mapping(_mapping(prediction).get("gain_model_summary"))
    constant = value.get("constant_baseline_selected")
    benefit = value.get("empirical_decision_benefit_established")
    response = value.get("predictor_text_response_supported")
    return {"telemetry_status": "recorded" if value else "unknown",
            "fit_scheme": value.get("fit_scheme") if isinstance(value.get("fit_scheme"), str) else None,
            "constant_baseline_selected": constant if type(constant) is bool else None,
            "predictor_text_response_supported": response if type(response) is bool else None,
            "selected_regularization": _number(value.get("selected_regularization")),
            "selection_independent_matches": _count(value.get("selection_independent_matches")),
            "conditioning_claim": value.get("conditioning_claim") if isinstance(value.get("conditioning_claim"), str) else None,
            "empirical_decision_benefit_established": benefit if type(benefit) is bool else None}


def _text_input(prediction):
    """Project consumption telemetry, not a claim that a returned page was used."""
    prediction = _mapping(prediction)
    telemetry = _mapping(prediction.get("text_model_input"))
    current = telemetry.get("schema") == "historical-text-input-v1" or telemetry.get("schema_version") == "historical-text-input-v1"
    result = {"telemetry_status": "recorded" if current else "legacy_count_only" if
              _count(prediction.get("modeled_read_documents")) is not None else "unknown",
              "modeled_body_documents": _count(telemetry.get("modeled_body_documents")) if current
              else _count(prediction.get("modeled_read_documents")),
              "modeled_body_paragraphs": _count(telemetry.get("modeled_body_paragraphs")) if current else None,
              "modeled_link_labels": _count(telemetry.get("modeled_link_labels")) if current else None,
              "excluded_link_labels": _count(telemetry.get("excluded_link_labels")) if current else None,
              "modeled_body_document_ids": None, "modeled_publication_times": None,
              "unmodeled_documents": None}
    ids = telemetry.get("modeled_body_document_ids") if current else None
    if isinstance(ids, list) and all(isinstance(item, str) for item in ids):
        result["modeled_body_document_ids"] = list(ids)
    times = telemetry.get("modeled_publication_times") if current else None
    if isinstance(times, list):
        result["modeled_publication_times"] = []
        for item in times:
            if isinstance(item, str):
                result["modeled_publication_times"].append(item)
            elif isinstance(item, Mapping):
                result["modeled_publication_times"].append({key: deepcopy(item[key]) for key in (
                    "document_id", "url", "source_published_at", "source_modified_at", "available_at",
                    "publication_upper_bound", "publication_basis") if key in item})
    omitted = telemetry.get("unmodeled_documents") if current else None
    if isinstance(omitted, list):
        result["unmodeled_documents"] = [{key: deepcopy(item[key]) for key in (
            "document_id", "url", "reason", "reasons", "publication_status", "source_published_at")
            if key in item} for item in omitted if isinstance(item, Mapping)]
    return result


def _text_consumption(cycle):
    before_public, after_public = (_mapping(cycle.get(key)) for key in ("public_before", "public_after"))
    before, after = (_text_input(public.get("prediction")) for public in (before_public, after_public))
    before_count, after_count = before["modeled_body_documents"], after["modeled_body_documents"]
    delta = after_count - before_count if before_count is not None and after_count is not None else None
    before_ids, after_ids = before["modeled_body_document_ids"], after["modeled_body_document_ids"]
    added = len(set(after_ids) - set(before_ids)) if before_ids is not None and after_ids is not None else None
    prior_documents = before_public.get("documents")
    new_documents = after_public.get("documents")
    receipts = None
    if isinstance(prior_documents, list) and isinstance(new_documents, list):
        prior_ids = {item.get("document_id") for item in prior_documents if isinstance(item, Mapping)}
        receipts = []
        for document in new_documents:
            if not isinstance(document, Mapping) or not document.get("document_id") or document["document_id"] in prior_ids:
                continue
            structured = _mapping(document.get("structured_news"))
            segments = structured.get("segments")
            paragraph_count = len(segments) if isinstance(segments, list) else None
            http = document.get("http_status")
            receipts.append({"document_id": document["document_id"], "url": _url(document.get("url")),
                "return_status": document.get("status"), "http_status": http if type(http) is int else None,
                "http_success": 200 <= http < 300 if type(http) is int else None,
                "body_extraction_status": "extracted" if paragraph_count else "no_admitted_body_segments" if paragraph_count == 0 else "unknown",
                "extracted_body_segments": paragraph_count,
                "publication_status": document.get("publication_status"),
                "source_published_at": document.get("source_published_at"),
                "body_exclusion_reasons": [item.get("reason") for item in structured.get("excluded", [])
                                           if isinstance(item, Mapping) and item.get("reason")]})
    return {"before": before, "after": after, "modeled_body_document_delta": delta,
            "newly_modeled_body_documents": added, "new_document_returns": receipts,
            "probability_change_attribution": "not_identified_by_input_counts",
            "source_text_implies_match_fact": False, "prediction_improvement_established": False}


def build_analysis_report(run, *, match):
    """Build a label-free user result from a reconciled or partial runtime run.

    Older runs may lack ``evidence`` and cost completeness fields. They receive
    explicit gaps/unknowns; absence is never displayed as zero cost or zero risk.
    A missing forecast or terminal decision makes the analysis incomplete,
    except for the explicit v5 evidence-only contract with no available prior.
    """
    if not isinstance(run, Mapping) or not isinstance(match, Mapping):
        raise ValueError("run and match must be mappings")
    evidence_only = run.get("schema") == _EVIDENCE_ONLY_SCHEMA
    if evidence_only and (run.get("prior_status") != "prior_unavailable"
                          or run.get("analysis_mode") != "evidence_only"):
        raise ValueError("Evidence-only v5 requires explicit prior_unavailable and evidence_only markers")
    state = _mapping(run.get("final_state"))
    base = _mapping(state.get("base"))
    supplied_id = match.get("match_id", match.get("id"))
    if supplied_id and state.get("match_id") and supplied_id != state["match_id"]:
        raise ValueError("Report match differs from the runtime match")
    identity = {"match_id": supplied_id or state.get("match_id"),
                "home_team": _name(match.get("home_team", match.get("home_name", match.get("home")))),
                "away_team": _name(match.get("away_team", match.get("away_name", match.get("away")))),
                "kickoff_at": match.get("kickoff_at", match.get("kickoff", base.get("kickoff"))),
                "prediction_cutoff": match.get("prediction_cutoff", state.get("prediction_cutoff"))}
    initial, final = (_probabilities(run.get(key)) for key in ("initial_probabilities", "final_probabilities"))
    cycles = run.get("cycles", [])
    if not isinstance(cycles, list) or any(not isinstance(c, Mapping) for c in cycles):
        raise ValueError("Runtime cycles must be a list of mappings")
    if evidence_only and (initial is not None or final is not None or any(
            cycle.get(key) is not None for cycle in cycles for key in ("prediction_before", "prediction_after"))):
        raise ValueError("Evidence-only reports require all predictions to be None")
    attempts = {a.get("action_id"): a for a in run.get("attempts", []) if isinstance(a, Mapping)}
    decisions, incomplete_reasons = [], []
    for index, cycle in enumerate(cycles):
        strict = _mapping(cycle.get("strict_decision"))
        evaluated = {c.get("action_id"): c for c in strict.get("candidates", []) if isinstance(c, Mapping)}
        selected = _mapping(cycle.get("selected_action"))
        chosen = selected.get("action_id")
        basis = cycle.get("selection_basis")
        picked_value = _value(evaluated.get(chosen), evidence_only=evidence_only)
        if chosen and basis in {"computed_positive_voi", "computed_positive_empirical_gain"} and not (picked_value["status"] == "computed" and picked_value["net_voi"] > 0):
            incomplete_reasons.append("selected_numeric_value_not_verified_in_cycle")
        previous = {a.get("action_id") for a in _mapping(cycle.get("public_before")).get("attempts", []) if isinstance(a, Mapping)}
        exclusions = _mapping(cycle.get("exploration_exclusions"))
        eligible = _mapping(cycle.get("exploration_record")).get("eligible_ids", [])
        unselected = []
        for proposed in cycle.get("generated_actions", []):
            if not isinstance(proposed, Mapping) or proposed.get("action_id") == chosen:
                continue
            aid = proposed.get("action_id")
            value = _value(evaluated.get(aid), evidence_only=evidence_only)
            reason = exclusions.get(aid)
            if reason is None:
                reason = ("already_attempted" if aid in previous else
                          "computed_nonpositive_value" if value["status"] == "computed" and value["net_voi"] <= 0 else
                          "not_sampled_this_cycle" if aid in eligible else
                          "not_assessed_this_cycle" if aid not in evaluated else "not_selected_this_cycle")
            unselected.append({"action": _action(proposed), "value": value, "reason": reason,
                               "reason_zh": _reason(reason, evidence_only=evidence_only)})
        execution = cycle.get("execution_status", "not_recorded")
        if (chosen and execution != "returned") or (evidence_only and execution in {
                "reserved_before_io", "execution_outcome_unknown", "receipt_rejected"}):
            incomplete_reasons.append("selected_action_execution_not_reconciled")
        admission = _mapping(cycle.get("admission"))
        decisions.append({"step": cycle.get("step", index), "selected_action": _action(selected) if chosen else None,
                          "selection_basis": basis, "value": picked_value if chosen else None,
                          "exploration_probability": _number(_mapping(cycle.get("exploration_record")).get("selected_p")),
                          "exploration_scope": "conditional_on_generated_eligible_candidates_only" if basis == "bounded_unknown_information_exploration" else None,
                          "execution_status": execution, "return_status": _mapping(attempts.get(chosen)).get("status"),
                          "admission": {"status": admission.get("status"), "reason": admission.get("reason"),
                                        "reason_zh": _reason(admission.get("reason"), evidence_only=evidence_only)},
                          "prediction_before": _probabilities(cycle.get("prediction_before")),
                          "prediction_after": _probabilities(cycle.get("prediction_after")),
                          "unselected_actions": unselected})
    source = _mapping(run.get("evidence"))
    if source.get("match_id") and source["match_id"] != identity["match_id"]:
        raise ValueError("Evidence belongs to a different match")
    evidence, sources = [], {}
    for row in source.get("evidence", []):
        if not isinstance(row, Mapping):
            continue
        anchor = _mapping(row.get("anchor"))
        url = _url(anchor.get("url"))
        if url:
            sources.setdefault(url, []).append(row.get("evidence_id"))
        evidence.append({"evidence_id": row.get("evidence_id"), "topics": deepcopy(row.get("topics", [])),
                         "applicability": row.get("applicability"), "modality": row.get("modality"),
                         "summary_zh": _evidence_summary(row), "source_url": url,
                         "source_document_id": anchor.get("document_id"), "raw_sha256": anchor.get("raw_sha256"),
                         "source_quote": anchor.get("quote"), "quote_start": anchor.get("start"), "quote_end": anchor.get("end"),
                         "observed_at": row.get("observed_at"), "available_at": row.get("available_at"),
                         "source_published_at": row.get("source_published_at"),
                         "calibrated_confidence": None, "independently_verified": False})
    evidence_state = _mapping(source.get("state"))
    unresolved = []
    for item in evidence_state.get("unresolved", []):
        if not isinstance(item, Mapping):
            continue
        reasons = item.get("reasons", [])
        unresolved.append({"evidence_id": item.get("evidence_id"), "conflict_id": item.get("conflict_id"),
                           "reasons": list(reasons), "reasons_zh": [_reason(r) for r in reasons]})
    followups = [{"candidate_id": item.get("candidate_id"), "query": item.get("query"),
                  "based_on_evidence_ids": deepcopy(item.get("based_on_evidence_ids", [])),
                  "reason": item.get("reason"), "expected_gain": None, "mandatory": False}
                 for item in evidence_state.get("followup_candidates", []) if isinstance(item, Mapping)]
    terminal = _mapping(run.get("terminal_selection", run.get("terminal")))
    if evidence_only and terminal.get("stop_certificate") is not None:
        raise ValueError("Evidence-only reports cannot include a numeric stop certificate")
    terminal_kind = terminal.get("decision")
    terminal_valid = terminal_kind in {"defer", "stop", "wait"}
    if evidence_only:
        terminal_valid = ((terminal_kind == "defer" and terminal.get("reason") == "defer_prior_unavailable")
                          or (terminal_kind == "halt" and terminal.get("reason") in {
                              "physical_budget_exhausted", "forecast_cutoff_elapsed", "step_limit"}))
    if not terminal_valid:
        incomplete_reasons.append(terminal.get("reason") or "terminal_decision_missing")
    if not evidence_only and (initial is None or final is None):
        incomplete_reasons.append("forecast_missing")
    gaps = ["prior_unavailable"] if evidence_only else []
    if not source:
        gaps.append("evidence_projection_not_available")
    if not evidence:
        gaps.append("no_substantive_match_evidence_observed")
    if unresolved:
        gaps.append("source_or_applicability_questions_remain")
    if evidence_state.get("coverage_complete") is not True:
        gaps.append("information_coverage_not_established")
    status = "incomplete" if incomplete_reasons else "completed_with_gaps" if gaps or terminal_kind in {"defer", "wait"} else "completed"
    report = {"schema_version": VERSION, "run_id": run.get("run_id"), "status": status, "match": identity,
            "probabilities": {"order": ["home", "draw", "away"], "initial": initial, "final": final,
                              "changed": initial != final if initial is not None and final is not None else None,
                              "information_mode": state.get("information_mode"), "calibration_established": False},
            "decisions": decisions, "budget": _resources(run, cycles), "evidence": evidence,
            "sources": [{"url": url, "evidence_ids": ids} for url, ids in sources.items()],
            "source_count_semantics": "distinct_urls_not_independently_confirmed_sources",
            "unresolved": unresolved, "followup_candidates": followups,
            "terminal": {"decision": terminal_kind, "reason": terminal.get("reason"),
                         "reason_zh": _reason(terminal.get("reason"), evidence_only=evidence_only),
                         "stop_certificate_present": terminal.get("stop_certificate") is not None,
                         "revisit_at": terminal.get("revisit_at")},
            "gaps": gaps, "incomplete_reasons": sorted(set(incomplete_reasons)),
            "semantics": {"outcome_labels_read": False, "probabilities_recomputed": False,
                          "open_text_probability_updates_claimed": False, "policy_benefit_proven": False,
                          "unknown_value_is_zero": False, "source_claim_is_verified_fact": False}}
    first_prediction = _mapping(_mapping(cycles[0].get("public_before")).get("prediction")) if cycles else {}
    if not evidence_only and first_prediction.get("probability_basis") == "observed_market_snapshot":
        report["probabilities"]["initial_probability_basis"] = "observed_market_snapshot"
    if evidence_only:
        report.update(analysis_mode="evidence_only", prior_status="prior_unavailable",
                      analysis_note_zh=_EVIDENCE_ONLY_NOTE)
        report["probabilities"]["initial_probability_basis"] = "unavailable"
        report["semantics"]["numeric_information_gain_available"] = False
    learning = _mapping(run.get("text_response_learning"))
    if run.get("schema") == "learned-voi-open-execution-v3" and learning:
        report["text_response_learning"] = {key: deepcopy(learning.get(key)) for key in (
            "model_version", "evidence_origin", "modeled_returns", "unknown_returns",
            "probability_updates", "query_and_update_share_response_model",
            "calibration_status", "production_ready")}
        count = learning.get("probability_updates")
        report["semantics"]["open_text_probability_updates_claimed"] = type(count) is int and count > 0
        empirical = _mapping(run.get("text_risk_learning"))
        if empirical:
            report["text_risk_learning"] = {key: deepcopy(empirical.get(key)) for key in (
                "model_version", "evidence_origin", "prediction_model_version", "value_scope",
                "calibration_status", "production_ready", "real_world_benefit_proven")}
        discovery = _mapping(run.get("discovery_response_learning"))
        if discovery:
            report["discovery_response_learning"] = {key: deepcopy(discovery.get(key)) for key in (
                "model_version", "evidence_origin", "planning_horizon", "continuation_enabled_selections",
                "continuation_enabled_steps", "execution_scope", "value_scope", "open_world_optimality_proven",
                "calibration_status", "production_ready")}
    context = _mapping(run.get("context_response_learning"))
    if run.get("schema") == "learned-voi-open-execution-v4" and context:
        report["context_response_learning"] = deepcopy(context)
        report["semantics"]["open_text_probability_updates_claimed"] = context.get("probability_updates", 0) > 0
        report["probabilities"]["initial_probability_basis"] = first_prediction.get("probability_basis", "observed_market_snapshot")
        if first_prediction.get("probability_basis") == "historical_text_f_with_A_learned_base":
            report["analysis_mode"] = "historical_empirical_context"
            report["gain_model_summary"] = _gain_model_summary(first_prediction)
            report["probabilities"].update(original_prior=None,
                model_base=deepcopy(first_prediction.get("base_probabilities")))
            for decision, cycle in zip(report["decisions"], cycles):
                decision["text_model_consumption"] = _text_consumption(cycle)
                if report["gain_model_summary"]["constant_baseline_selected"] is True:
                    for item in [decision, *decision["unselected_actions"]]:
                        value = _mapping(item.get("value"))
                        if value.get("status") == "computed":
                            value.update(meaning="constant_empirical_gain_minus_declared_cost",
                                         gain_model_conditioning="constant_baseline",
                                         support_gate_is_learned_conditional_gain=False)
            # Legacy probability_updates counts any state update, including link labels.
            # It cannot establish that a returned body entered f, let alone caused a gain.
            report["semantics"].update(open_text_probability_updates_claimed=False,
                body_probability_change_attribution_established=False,
                text_state_probability_updates_recorded=context.get("probability_updates", 0) > 0,
                conditional_gain_learning_established=False if report["gain_model_summary"]["constant_baseline_selected"] is True else None,
                support_gate_action_change_proves_learned_conditional_gain=False)
    if run.get("schema") == "learned-voi-open-execution-v6" and context:
        if first_prediction.get("prior_probabilities") is not None or not first_prediction.get("base_probabilities"):
            raise ValueError("Official context reports require the explicit learned base and missing original prior")
        report["context_response_learning"] = deepcopy(context)
        report["semantics"]["open_text_probability_updates_claimed"] = context.get("probability_updates", 0) > 0
        report["analysis_mode"] = "official_optional_context"
        report["probabilities"].update(initial_probability_basis=first_prediction.get("forecast_basis"),
            original_prior=None, model_base=deepcopy(first_prediction["base_probabilities"]),
            base_source=deepcopy(first_prediction.get("base_source")))
    return report


def _md(value):
    value = str(value if value is not None else "未记录").replace("\r", " ").replace("\n", " ")
    value = value.replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]()|])", r"\\\1", value)


def _display_action(action):
    if action.get("tool") == "search_web":
        return "搜索：" + str(action.get("query") or "未记录查询词")
    if action.get("tool") == "read_url":
        return "读取：" + str(action.get("url") or "未记录来源")
    return {"results_view": "获取历史战绩", "xg_view": "获取 xG 数据"}.get(action.get("action_id"), str(action.get("action_id")))


def _distribution(value):
    return "未取得概率" if value is None else f"主胜 {value[0]:.2%} · 平局 {value[1]:.2%} · 客胜 {value[2]:.2%}"


def _consumption_lines(decisions):
    lines = ["", "## 实际进入预测器的信息", "",
             "HTTP 成功、正文抽取成功、正文实际入模分别记录。缺少统计时显示未知；入模次数也不证明该信息适用于本场。",
             "标题/链接状态变化不证明比赛事实或预测改善；这里不把概率变化归因于返回正文。", "",
             "| 评估 | 入模正文篇数（前→后） | 新增正文篇数 | 正文篇数净变化 | 入模段落（后） | 入模标题/链接（后） |",
             "| --- | --- | --- | --- | --- | --- |"]
    display = lambda value: "未知" if value is None else str(value)
    for step in decisions:
        consumption = step["text_model_consumption"]
        before, after = consumption["before"], consumption["after"]
        lines.append(f'| {step["step"] + 1} | {display(before["modeled_body_documents"])}→{display(after["modeled_body_documents"])} | '
                     f'{display(consumption["newly_modeled_body_documents"])} | {display(consumption["modeled_body_document_delta"])} | '
                     f'{display(after["modeled_body_paragraphs"])} | {display(after["modeled_link_labels"])} |')
    lines.append("")
    for step in decisions:
        consumption = step["text_model_consumption"]
        after = consumption["after"]
        prefix = f'第 {step["step"] + 1} 次评估'
        receipts = consumption["new_document_returns"]
        if receipts is None:
            lines.append(f"- {prefix}：新增页面返回记录未知。")
        for receipt in receipts or []:
            extraction = {"extracted": "已抽取正文段落", "no_admitted_body_segments": "未抽取到通过正文规则的段落",
                          "unknown": "正文抽取状态未知"}[receipt["body_extraction_status"]]
            lines.append(f'- {prefix}：HTTP {display(receipt["http_status"])}；{extraction}（{display(receipt["extracted_body_segments"])} 段）；'
                         f'来源声称发布时间：{_md(receipt["source_published_at"])}。来源：{_md(receipt["url"])}。')
            if receipt["body_exclusion_reasons"]:
                lines.append("  正文未通过的原因：" + "；".join(_md(_reason(reason)) for reason in receipt["body_exclusion_reasons"]) + "。")
        times = after["modeled_publication_times"]
        if times is None:
            lines.append(f"- {prefix}：入模正文的来源时间统计未知。")
        elif not times:
            lines.append(f"- {prefix}：没有记录入模正文的来源时间。")
        else:
            for item in times:
                detail = item if isinstance(item, str) else f'{item.get("url", "来源未记录")}，发布 {item.get("source_published_at", "未知")}，修改 {item.get("source_modified_at", "未知")}'
                lines.append(f"- {prefix}：通过时间检查的来源声称时间：{_md(detail)}；不是独立核验的发布时间。")
        omitted = after["unmodeled_documents"]
        if omitted is None:
            lines.append(f"- {prefix}：未入模文档原因统计未知。")
        for item in omitted or []:
            reasons = item.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = [reasons]
            reasons = list(dict.fromkeys([item.get("reason"), *reasons]))
            reasons = [reason for reason in reasons if reason] or [None]
            lines.append(f'- {prefix}：{_md(item.get("url", item.get("document_id")))} 未入模；'
                         + "；".join(_md(_reason(reason)) for reason in reasons) + "。")
    return lines


def _render(report):
    match = report["match"]
    title = f'{match["home_team"] or "主队"} vs {match["away_team"] or "客队"}'
    status = {"completed": "本轮分析已完成", "completed_with_gaps": "本轮分析已完成，仍有未决信息", "incomplete": "本轮分析尚未完整完成"}[report["status"]]
    learning = report.get("text_response_learning")
    gain_summary = _mapping(report.get("gain_model_summary"))
    constant_gain = gain_summary.get("constant_baseline_selected") is True
    if report.get("analysis_mode") == "evidence_only":
        explanation = _EVIDENCE_ONLY_NOTE + "。"
    elif report.get("context_response_learning"):
        explanation = ("本轮使用冻结的可见信息预测器，按实际返回时刻的正文、来源和查询历史更新概率。"
                       "信息选择依据是条件查询收益减去预先登记的成本；收益估计尚未支持的候选只可进入有限探索。"
                       "概率发生变化不等于预测改善，真实比赛收益仍需独立验证。")
        if report.get("analysis_mode") == "official_optional_context":
            explanation = "本次没有市场先验；初始基准来自训练阶段的独立比赛结果分布。" + explanation
        elif report.get("analysis_mode") == "historical_empirical_context":
            explanation = ("初始分布来自训练集比赛结果频率；文本预测与查询收益均为小样本研究模型。"
                           "冻结的预测器根据实际可见且通过入模检查的输入更新；页面返回并不意味着正文已进入预测器。"
                           "信息选择使用收益估计减去预先登记的成本；收益估计尚未支持的候选只可进入有限探索。"
                           "标题/链接状态变化不证明比赛事实或预测改善，概率变化的来源需查看逐周期消费记录。")
            if constant_gain:
                count = gain_summary.get("selection_independent_matches")
                sample = f"（{count} 场独立比赛）" if count is not None else "（独立比赛数未记录）"
                explanation = (f"本轮收益模型在 B 分区内按整场验证选择了常数均值基线{sample}。"
                               "数值收益估计不区分当前状态和候选，自主条件信息价值尚未学出，也未验证。"
                               "支持门和有限探索可以改变动作选择，但不能把这种变化当作条件收益模型已学会信息选择的证据。"
                               + explanation)
            elif gain_summary.get("constant_baseline_selected") is None:
                explanation += "未记录收益模型是否选择常数基线，不能据此推断已经学到条件信息价值。"
            if gain_summary.get("predictor_text_response_supported") is False:
                explanation = ("A 分区内选模得到不使用正文的基率预测器。它产生的零增益只表示模型没有内容响应，"
                               "不能说明外部信息没有价值；相应候选保留 UNKNOWN，不能据此签发信息充分性停止结论。" + explanation)
        if any(_mapping(item.get('value')).get('state_pattern_extrapolation')
               for decision in report['decisions']
               for item in [decision, *decision.get('unselected_actions', [])]):
            explanation += ("本轮部分候选通过相近历史候选的支持规则取得数值估计资格，当前状态组合尚未局部验证。" if constant_gain else
                            "本轮部分估值是在相近历史候选上对未见状态组合进行的模型外推，尚未校准。")
            explanation += "据其均值跳过查询不代表这些信息已被证明没有价值，也不构成信息充分性证书。"
    elif learning:
        origin = "合成机制实验" if learning.get("evidence_origin") == "constructed_mechanism" else "尚待独立验证的正文响应模型"
        explanation = (f"本轮使用{origin}。查询前的信息价值与取得正文后的概率更新使用同一条件响应模型；"
                       "数值一致性不等于真实比赛预测收益已经成立。")
        if report.get("text_risk_learning"):
            explanation = (f"本轮使用{origin}。查询选择依据是固定正文预测器在样本外实际更新后的经验风险改善，"
                           "改善可以为负；实际正文仍由绑定的预测器更新概率。"
                           "经验估计和停止规则不构成条件风险保证，真实比赛收益尚未验证。")
        if report.get("discovery_response_learning"):
            explanation += ("信息选择还考虑了训练轨迹支持的两步后续价值，并计入两次查询成本；"
                            "搜索页可以因其发现正文的价值而被选择。实际读取仍只使用已经发现的链接，"
                            "此估值不代表已经穷尽开放环境中的信息。")
    elif report["probabilities"].get("initial_probability_basis") == "observed_market_snapshot":
        explanation = ("初始概率来自实际取得的赛前胜平负报价，按赔率倒数归一化。"
                       "本轮正文信息价值仍未知，系统使用有限探索采集轨迹；"
                       "未取得可用的概率更新模型，不能将网页获取描述为预测收益。")
    else:
        explanation = "概率沿用本轮结构化模型输出。未建模网页内容用于提出后续查询，不能据此宣称预测准确率已经提高。"
    lines = [f"# {_md(title)}", "", status + "。", "",
             f'开球时间：{_md(match["kickoff_at"])}；信息截止：{_md(match["prediction_cutoff"])}。', "",
             f'初始判断：{_distribution(report["probabilities"]["initial"])}。',
             f'当前判断：{_distribution(report["probabilities"]["final"])}。', "",
             explanation, "",
             "## 系统实际选择了什么", ""]
    selected_count = 0
    for step in report["decisions"]:
        selected = step["selected_action"]
        if selected:
            selected_count += 1
            value = step["value"]
            if step["selection_basis"] == "computed_positive_voi" and value["status"] == "computed":
                rationale = f'按数值模型选择，净 VoI 为 {value["net_voi"]:.6f}'
                if value.get("planning_horizon") == 2:
                    rationale = f'按两步信息价值模型选择，净 VoI 为 {value["net_voi"]:.6f}'
                    if value.get("immediate_net_voi") is not None and value["immediate_net_voi"] <= 1e-12:
                        rationale += f'；当前一步净值为 {value["immediate_net_voi"]:.6f}，选择依据包含后续查询价值'
            elif step["selection_basis"] == "bounded_unknown_information_exploration":
                rationale = "使用有限探索额度；该动作的信息价值尚无可用数值估计"
            elif step["selection_basis"] == "computed_positive_empirical_gain" and value["status"] == "computed":
                rationale = f'按{"常数收益基线" if constant_gain else "经验风险改善"}选择，扣除声明成本后的净改善估计为 {value["net_voi"]:.6f}'
            else:
                rationale = "选择依据尚需核对"
            lines.append(f'- 第 {step["step"] + 1} 次评估，{_md(_display_action(selected))}。{rationale}；{_md(step["admission"]["reason_zh"])}。')
    if not selected_count:
        lines.append("本轮没有记录已选取的外部信息动作。")
    if report.get("analysis_mode") == "historical_empirical_context":
        lines.extend(_consumption_lines(report["decisions"]))
    skipped = {}
    selected_ids = {d["selected_action"]["action_id"] for d in report["decisions"] if d["selected_action"]}
    for step in report["decisions"]:
        for item in step["unselected_actions"]:
            if item["action"]["action_id"] not in selected_ids:
                skipped[item["action"]["action_id"]] = item
    if skipped:
        lines.extend(["", "本轮始终未选的候选：", ""])
        for item in skipped.values():
            value = item["value"]
            amount = f'，净 VoI {value["net_voi"]:.6f}' if value["status"] == "computed" else "，数值价值未知"
            if value.get('state_pattern_extrapolation'):
                amount = f'，净收益估计 {value["net_voi"]:.6f}（状态外推，未校准）'
            if constant_gain and value["status"] == "computed":
                amount = f'，常数基线净收益估计 {value["net_voi"]:.6f}（未校准，未学到状态与候选差异）'
            lines.append(f'- {_md(_display_action(item["action"]))}：{_md(item["reason_zh"])}{amount}。')
    lines.extend(["", "## 已取得的信息", ""])
    if not report["evidence"]:
        lines.append("尚未提取到可用于本场分析的实质性来源证据；这不代表现实中没有相关信息。")
    for row in report["evidence"]:
        url = row["source_url"]
        source = f'[来源](<{url.replace("<", "%3C").replace(">", "%3E")}>)' if url else "来源链接未记录"
        lines.append(f'- {_md(row["summary_zh"])} {source} 获取于 {_md(row["available_at"])}。')
    if report["sources"]:
        lines.append(f'\n以上 {len(report["evidence"])} 条引文来自 {len(report["sources"])} 个不同页面地址；页面或引文数量不等于独立证实次数。')
    lines.extend(["", "## 还存在哪些具体未知", ""])
    if report["followup_candidates"]:
        lines.append("系统从已获取内容提出以下后续查询，尚未证明每项都值得执行：")
        lines.append("")
        for item in report["followup_candidates"]:
            lines.append("- " + _md(item["query"]) + "。")
    else:
        reasons = sorted({reason for item in report["unresolved"] for reason in item["reasons_zh"]})
        lines.extend(["- " + _md(reason) + "。" for reason in reasons] or ["尚无有据可查的具体后续问题；候选覆盖也未因此得到证明。"])
    lines.extend(["", "## 预算和终态", ""])
    budget = report["budget"]
    spent, limit = budget["reserved_cost"]["money"], budget["declared_limits"]["money"]
    lines.append(f'策略声明成本额度已占用 {_md(spent)}，上限 {_md(limit)}；该额度不是实际货币账单。')
    labels = {"requests": "HTTP 请求数", "latency_ms": "累计请求耗时（毫秒）", "tokens": "模型 token", "money": "实际货币费用"}
    for unit, label in labels.items():
        measured = budget["measured_resources"][unit]
        if measured["total"] is not None:
            detail = str(measured["total"])
        elif measured["measured_part"] is not None:
            detail = f'已测部分 {measured["measured_part"]}，总量未知'
        else:
            detail = "未测量"
        lines.append(f'- {label}：{detail}。')
    terminal = report["terminal"]
    lines.extend(["", f'终态：{_md(terminal["decision"])}；{_md(terminal["reason_zh"])}。'])
    if not terminal["stop_certificate_present"]:
        lines.append("本轮没有给出“所有剩余信息都不值得查询”的数值停止证明。")
    if report["status"] == "incomplete":
        lines.append("未完成原因：" + "；".join(_md(_reason(r)) for r in report["incomplete_reasons"]) + "。")
    return "\n".join(lines) + "\n"


def write_analysis_report(run, *, match, output):
    """Write the structured result and a Chinese reader report atomically each."""
    report = build_analysis_report(run, match=match)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    analysis_path, report_path = root / "analysis.json", root / "report.md"
    atomic_write(analysis_path, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    atomic_write(report_path, _render(report).encode("utf-8"))
    return {"analysis_path": str(analysis_path.resolve()), "report_path": str(report_path.resolve()),
            "status": report["status"], "run_id": report["run_id"]}


__all__ = ["VERSION", "build_analysis_report", "write_analysis_report"]
