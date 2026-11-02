"""Readable audit of saved semantic assumptions, source returns and finite VoI.

This module renders saved values; it does not query sources, classify evidence,
invent missing probabilities, or certify forecasting performance.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

VERSION = "semantic-scenario-report-v1"
_REASONS = {
    "positive_declared_lower_net_value": "预声明模型集合的净价值下界为正",
    "nonpositive_declared_lower_net_value": "预声明模型集合的净价值下界不为正",
    "no_positive_declared_lower_net_value": "当前可行候选均没有正的净价值下界",
    "request_budget_exhausted": "HTTP 请求预算耗尽",
    "insufficient_request_budget": "剩余请求不足以执行候选工具合同",
    "no_unattempted_observation_groups": "当前菜单没有尚未尝试的观测组",
    "already_attempted": "该动作已经尝试",
    "observation_group_already_attempted_or_blocked": "同一事实观测组已经尝试或被阻止",
    "semantic_call_budget_exhausted": "语义模型调用预算耗尽",
    "prediction_cutoff_reached": "到达预测截点",
    "runtime_incomplete": "运行未完整结束，不能作为正常停止证明",
}


def _pick(value, keys):
    return {key: deepcopy(value[key]) for key in keys if key in value}


def _cell(value):
    if value is None:
        return "未记录"
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;").replace("|", "&#124;").replace("\r\n", "<br>").replace("\n", "<br>")


def _number(value):
    return "未记录" if value is None else f"{value:.9g}"


def _probabilities(values):
    if values is None:
        return "未记录"
    return " / ".join(f"{value:.4%}" for value in values)


def _reason(value):
    return _REASONS.get(value, value or "未记录")


def _numeric_delta(before, after):
    """Compare saved model predictions/posteriors without refitting assumptions."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if not before or set(before) != set(after):
            return None
        differences = [_numeric_delta(before[key], after[key]) for key in before]
    elif isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        if not before or len(before) != len(after):
            return None
        differences = [_numeric_delta(a, b) for a, b in zip(before, after)]
    elif (isinstance(before, (int, float)) and not isinstance(before, bool)
          and isinstance(after, (int, float)) and not isinstance(after, bool)
          and math.isfinite(before) and math.isfinite(after)):
        return abs(before - after)
    else:
        return None
    return None if any(value is None for value in differences) else max(differences)


def _history(observations):
    return [(item.get("action_id"), item.get("outcome")) for item in observations]


def _directional_summary(observations, *, required=False):
    """Expose opt-in directional annotations without feeding the VoI solver.

    Legacy ``supports/refutes/no_answer`` observations are intentionally not
    mapped to a team direction. The independent directional extension owns
    validation; this report layer only preserves saved auditable fields.
    """
    records = []
    for item in observations:
        if "direction" not in item and item.get("schema") != "semantic-directional-observation-v1":
            continue
        records.append(_pick(item, (
            "action_id", "outcome", "direction", "subject", "direction_basis",
            "directional_claim", "applicability", "evidence", "reason", "directional_gate")))
    return {
        "version": "semantic-directional-observation-v1",
        "available": bool(records),
        "records": records,
        "required_for_live_update": bool(required),
        "used_for_update_gate": bool(required),
        "used_for_voi": False,
        "interpretation": (
            ("新运行只有显式本场方向性正文才允许保留 supports/refutes；"
             "方向性门控参与实际观察更新资格，但不改变查询前的 VoI 或提供校准效应大小。"
             if required else
             "方向性记录是来源语义注释；旧 supports/refutes 不自动映射为主客队利空或利好。"
             "旧协议运行不启用方向性门控。")
        ),
    }


def _no_answer_checks(run):
    checks = []
    observations = run.get("observations", [])
    for index, observation in enumerate(observations):
        if observation.get("outcome") not in ("no_answer", "unknown"):
            continue
        before_history, after_history = _history(observations[:index]), _history(observations[:index + 1])
        before = next((cycle for cycle in run.get("cycles", [])
                       if _history(cycle.get("observations", [])) == before_history), None)
        after = next((cycle for cycle in run.get("cycles", [])
                      if _history(cycle.get("observations", [])) == after_history), None)
        prediction_delta = posterior_delta = None
        if before is not None and after is not None:
            prediction_delta = _numeric_delta(before["decision"].get("model_predictions"),
                                              after["decision"].get("model_predictions"))
            posterior_delta = _numeric_delta(before["decision"].get("model_posteriors"),
                                             after["decision"].get("model_posteriors"))
        verified = None if prediction_delta is None or posterior_delta is None else (
            prediction_delta <= 1e-12 and posterior_delta <= 1e-12)
        checks.append({"action_id": observation.get("action_id"), "outcome": observation.get("outcome"),
                       "scope": "saved_adjacent_cycles_all_declared_models_not_semantic_truth_verification",
                       "before_cycle": None if before is None else before.get("index"),
                       "after_cycle": None if after is None else after.get("index"),
                       "max_prediction_delta": prediction_delta, "max_posterior_delta": posterior_delta,
                       "unchanged_within_1e_12": verified})
    return checks


def _token_usage(calls):
    totals, reported_calls = {}, 0
    for call in calls:
        usage = call.get("receipt", {}).get("usage")
        items = [usage] if isinstance(usage, Mapping) else usage or []
        reported = False
        for item in items:
            if not isinstance(item, Mapping):
                continue
            for key, value in item.items():
                if (key.endswith("tokens") and isinstance(value, (int, float))
                        and not isinstance(value, bool) and math.isfinite(value) and value >= 0):
                    totals[key] = totals.get(key, 0) + value
                    reported = True
        reported_calls += reported
    return {"reported_token_totals": totals, "calls_with_token_usage": reported_calls,
            "calls_without_token_usage": len(calls) - reported_calls,
            "scope": "sum_of_reported_receipt_fields_cached_tokens_may_overlap_input_tokens"}


def _source(receipt):
    returned = receipt.get("returned", {})
    document = receipt.get("admitted", returned.get("document", {}))
    observation = returned.get("observation", {})
    return {"receipt_index": receipt.get("index"), "purpose": receipt.get("purpose"),
            "receipt_status": receipt.get("status"), "receipt_error": receipt.get("error"),
            "reserved_requests": receipt.get("reserved_requests"),
            "tool": receipt.get("candidate", {}).get("tool"),
            "query": receipt.get("candidate", {}).get("query"),
            "request_target": receipt.get("candidate", {}).get("url") or receipt.get("candidate", {}).get("query"),
            **_pick(document, ("document_id", "url", "title", "status", "kind", "observed_at",
                               "available_at", "source_published_at", "source_modified_at", "admission")),
            "admitted_body_characters": len(document.get("text") or "") if "text" in document else None,
            "observed_link_count": len(document.get("links") or []) if "links" in document else None,
            "http_status": observation.get("http_status"),
            "http_attempts": returned.get("http_attempts"),
            "latency_seconds": returned.get("latency_seconds"),
            "payload_path": observation.get("payload_path"),
            "content_hash": observation.get("content_hash")}


def _analysis(run):
    configuration = run.get("configuration", {})
    calls = run.get("semantic_calls", [])
    proposal = run.get("proposal")
    if not proposal:
        proposal = next((call["validated"] for call in calls
                         if call.get("validated", {}).get("schema") == "semantic-scenario-proposal-v1"), {})
    discovery = [{"call_index": call.get("index"), **deepcopy(query)} for call in calls
                 for query in call.get("validated", {}).get("queries", [])]
    sources = [_source(receipt) for receipt in run.get("http_receipts", [])]
    observations = deepcopy(run.get("observations", []))
    cycles = []
    for cycle in run.get("cycles", []):
        decision = cycle.get("decision", {})
        selected = decision.get("selected_action_id")
        candidates = []
        for item in decision.get("candidates", []):
            binding = cycle.get("candidate_bindings", {}).get(item["action_id"], {})
            candidates.append({**deepcopy(item), "binding": _pick(binding, (
                "tool", "url", "title", "query", "information_need", "request_cost", "return_contract")),
                "selected_by_solver": item["action_id"] == selected})
        actual = next((item for item in observations if item.get("action_id") == selected), None)
        returned_ids = set((actual or {}).get("returned_document_ids", []))
        cycles.append({"index": cycle.get("index"), "requests_spent": cycle.get("requests_spent"),
                       "remaining_requests": cycle.get("remaining_requests"),
                       **_pick(decision, ("prediction_nominal", "posterior_nominal", "selected_action_id",
                                          "stop_reason", "model_predictions", "model_posteriors")),
                       "candidates": candidates, "actual_observation": deepcopy(actual),
                       "actual_returns": [source for source in sources if source.get("document_id") in returned_ids]})
    first_decision = run.get("cycles", [{}])[0].get("decision", {}) if run.get("cycles") else {}
    resources = deepcopy(run.get("resources", {}))
    resources.update(_token_usage(calls))
    resources["http_requests_confirmed"] = resources.get("http_requests")
    if resources.get("http_accounting_complete") is False or any(source["http_attempts"] is None for source in sources):
        # The runtime's older http_requests field is a sum of known receipts,
        # not a complete count when a projection failed after network access.
        resources["http_accounting_complete"] = False
        resources["http_requests"] = None
    return {"version": VERSION, "run_id": run.get("run_id"), "status": run.get("status"),
            "scope": configuration.get("scope"), "study_role": configuration.get("study_role"),
            "policy_mode": configuration.get("policy_mode", "conditional_semantic_voi_untrained_g"),
            "empirical_g": deepcopy(configuration.get("empirical_g", {
                "status": "unspecified", "used_for_selection": False})),
            "match": deepcopy(configuration.get("match", {})),
            "started_at": configuration.get("started_at"), "completed_at": run.get("completed_at"),
            "initial_prediction": deepcopy(run.get("initial_prediction")),
            "final_prediction": deepcopy(run.get("final_prediction")),
            "baseline_scope": configuration.get("baseline_scope"),
            "source_facts": {"meaning": "acquired_source_text_and_transport_records_not_verified_football_truth",
                             "observation_classification_scope": "LLM_semantic_judgment_with_exact_quote_checks_not_truth_verification",
                             "sources": sources, "observations": observations},
            "observation_protocol": configuration.get("observation_protocol"),
            "directional_evidence_required": bool(configuration.get("directional_evidence_required", False)),
            "directional_evidence": _directional_summary(
                observations, required=configuration.get("directional_evidence_required", False)),
            "semantic_assumptions": {"queries": discovery, "hypotheses": deepcopy(proposal.get("hypotheses", [])),
                                     "actions": deepcopy(proposal.get("actions", [])),
                                     "assumptions": deepcopy(proposal.get("assumptions", [])),
                                     "aligned_outcome_probabilities": first_decision.get("aligned_outcome_probabilities"),
                                     "alignment": first_decision.get("alignment"),
                                     "extensions": [_pick(extension, ("addition", "conditioned_on_observations", "history_sha256"))
                                                    for extension in run.get("extensions", [])]},
            "conditional_voi": {"meaning": ("exact_one_step_conditional_on_declared_assumptions"
                                             if configuration.get("planning_horizon", configuration.get("horizon", 1)) == 1 else
                                             "branchwise_robust_shared_two_step_policy_conditional_on_declared_assumptions"
                                             if configuration.get("planning_policy") == "shared_observation_continuation_v1" else
                                             "exact_bounded_two_step_bellman_conditional_on_declared_assumptions"),
                                "planning_horizon": configuration.get("planning_horizon", configuration.get("horizon", 1)),
                                **({"planning_policy": configuration["planning_policy"]}
                                   if configuration.get("planning_policy") else {}),
                                "contamination_levels": configuration.get("contamination_levels"),
                                "lower_interpretation": "predeclared_contamination_model_set_envelope_not_confidence_interval",
                                "request_loss_cost": configuration.get("request_loss_cost"),
                                "request_budget": configuration.get("request_budget"),
                                "semantic_call_budget": configuration.get("semantic_call_budget"),
                                "cycles": cycles},
            "no_answer_checks": _no_answer_checks(run), "terminal": deepcopy(run.get("terminal")),
            "resources": resources, "failures": deepcopy(run.get("failures", [])),
            "semantic_call_audit": [{**_pick(call, ("index", "protocol", "status", "error")),
                                      "receipt": _pick(call.get("receipt", {}), (
                                          "provider", "real_provider_call", "model_requested", "response_model",
                                          "requests", "requests_unit", "usage", "latency_seconds", "actual_money"))}
                                     for call in calls],
            "evidence_scope": deepcopy(run.get("evidence_scope", {})),
            **_pick(run, ("candidate_discovery", "resume_provenance"))}


def _catalog_lines(discovery):
    """Render opt-in initial catalog coverage without widening the stop claim."""
    if not isinstance(discovery, Mapping):
        return []
    proposal = discovery.get("proposal") or {}
    if proposal.get("schema") != "semantic-scenario-discovery-v2":
        return []
    coverage = discovery.get("catalog_coverage") or {}
    lines = ["## 初始候选目录、语义提名与覆盖范围", "",
             "本次初始候选遵循：已见目录 → 语义提名 → 数学估值。"
             "Agent 从实际观察到的文章目录中提名可读取文章，也可提出查询；"
             "进入菜单后，由数学 VoI 决定是否执行。提名本身不是数值价值评估，"
             "目录标题和链接也不构成本场正文证据。", "",
             f"目录范围：`{_cell(coverage.get('scope'))}`；"
             f"可用文章 {_cell(coverage.get('available_articles'))}，"
             f"本次向语义模型展示 {_cell(coverage.get('shown_articles'))}，"
             f"未展示 {_cell(coverage.get('omitted_articles'))}，"
             f"展示上限 {_cell(coverage.get('catalog_limit'))}。",
             f"上游来源目录是否截断：`{_cell(coverage.get('source_projection_truncated'))}`；"
             f"全局完整性声明：`{_cell(coverage.get('global_completeness'))}`。",
             f"已观察文章候选 {_cell(discovery.get('observed_article_candidates'))}；"
             f"初始菜单上限 {_cell(discovery.get('menu_cap'))}；"
             f"菜单未纳入的可见候选 {_cell(discovery.get('omitted_visible_candidates'))}；"
             f"未建模文章候选 {_cell(discovery.get('unmodeled_article_candidates'))}。", "",
             "未建模不等于低价值。未展示、未提名或未进入菜单的文章没有获得同等数学估值，"
             "不能据此判定其信息无用。上述计数描述初始发现步骤，"
             "不证明全目录覆盖，也不构成开放世界停止证明。", ""]
    nominations = proposal.get("article_candidates") or []
    if nominations:
        lines += ["语义模型保存的文章提名（入选菜单不等于已执行）：", "",
                  "| 已见文章动作 ID | 语义提名的信息需求 | 初始菜单是否纳入 |",
                  "| --- | --- | --- |"]
        menu_ids = {item.get("action_id") for item in discovery.get("menu", [])}
        for item in nominations:
            lines.append("| " + " | ".join((
                _cell(item.get("action_id")), _cell(item.get("information_need")),
                "是" if item.get("action_id") in menu_ids else "否")) + " |")
        lines += [""]
    else:
        lines += ["本次没有保存已验证的文章提名。", ""]
    return lines


def _render(analysis):
    match, resources = analysis["match"], analysis["resources"]
    conditional, semantic = analysis["conditional_voi"], analysis["semantic_assumptions"]
    directional = analysis.get("directional_evidence", {})
    terminal = analysis.get("terminal") or {}
    home, away = match.get("home_name", match.get("home_team", "主队")), match.get("away_name", match.get("away_team", "客队"))
    lines = ["# FootballMind 单场语义情景与信息选择审计", "",
             f"**{_cell(home)} vs {_cell(away)}**。运行状态：`{_cell(analysis['status'])}`；"
             f"停止原因：{_cell(_reason(terminal.get('stop_reason')))}。", "",
             f"运行标识：`{_cell(analysis['run_id'])}`；范围：`{_cell(analysis['scope'])}`。",
             f"策略模式：`{_cell(analysis.get('policy_mode'))}`；经验 g："
             f"`{_cell((analysis.get('empirical_g') or {}).get('status'))}`，"
             "本次未用于动作选择。",
             f"观察协议：`{_cell(analysis.get('observation_protocol'))}`；"
             f"方向性门控：`{_cell(analysis.get('directional_evidence_required'))}`。",
             "本报告区分实际来源返回、LLM 的概率假设和假设条件下的数学积分。"
             "没有将模型假设当成足球事实，也不以单次运行证明预测改善。", "",
             "## 场次、时间与预测", "",
             f"场次 ID：`{_cell(match.get('match_id'))}`；记录的开球时间：{_cell(match.get('kickoff'))}。",
             f"准入截点：{_cell(match.get('prediction_cutoff'))}；运行时间："
             f"{_cell(analysis['started_at'])} 至 {_cell(analysis['completed_at'])}。",
             f"场次身份来源范围：`{_cell(match.get('fixture_identity_scope'))}`。"
             "若仅来自现有单一导出，该记录不构成独立赛程核验。",
             f"初始概率来源范围：`{_cell(analysis['baseline_scope'])}`。", "",
             "| 状态 | 主胜 | 平局 | 客胜 |", "| --- | ---: | ---: | ---: |"]
    if "constructed" in str(analysis.get("scope", "")):
        lines[2:2] = ["**构造测试数据：仅验证报告与数学记录的呈现，不是足球实证。**", ""]
    observation_count = len(analysis["source_facts"]["observations"])
    answered_count = sum(item.get("outcome") in ("supports", "refutes")
                         for item in analysis["source_facts"]["observations"])
    lines[2:2] = [f"保存了 {len(conditional['cycles'])} 轮数学决策、{observation_count} 个动作观测，"
                  f"其中 {answered_count} 个被语义分类为本场支持或反驳。"
                  + ("本次没有发生基于实际正文分类的概率更新。" if answered_count == 0 else "分类本身仍需按下列出处审阅。"), ""]
    if analysis.get("study_role") == "zero_loss_cost_mechanism_exercise_not_policy_success":
        lines[2:2] = ["**零损失成本机制诊断，不能作为主策略成功证据。** "
                      "本次将请求的损失等价成本设为 0，只检查真实调用、分类与无答案循环。"
                      "真实请求和 LLM 开销仍需记录，不能将本次动作数量解释为成本敏感策略的表现。", "",
                      "运行角色：`zero_loss_cost_mechanism_exercise_not_policy_success`。", ""]
    provenance = analysis.get("resume_provenance")
    if isinstance(provenance, Mapping):
        lines[2:2] = ["**从已保存运行恢复：继续同一累计预算，不是新的独立实验。**",
                      f"原运行：`{_cell(provenance.get('source_run_path'))}`；"
                      f"原运行 SHA-256：`{_cell(provenance.get('source_run_sha256'))}`。",
                      f"保留语义调用索引：`{_cell(provenance.get('retained_semantic_call_indices'))}`；"
                      f"保留的失败语义调用数：{_cell(provenance.get('retained_failed_semantic_calls'))}。"
                      f"累计预算不退还：`{_cell(provenance.get('cumulative_budget_no_refund'))}`；"
                      f"原文件是否修改：`{_cell(provenance.get('source_files_modified'))}`。", ""]
    for label, key in (("初始", "initial_prediction"), ("终态（名义模型）", "final_prediction")):
        values = analysis[key]
        cells = ["未记录"] * 3 if values is None else [f"{value:.6%}" for value in values]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "终态表示已保存来源观测与已声明模型条件下的概率，不是未来比赛真值。"
              "基线对齐仅约束初始边际，不是统计校准。", "", "## Agent 自行提出的信息需求", ""]
    if semantic["queries"]:
        for query in semantic["queries"]:
            lines += [f"- 信息需求：{_cell(query.get('information_need'))}；查询：`{_cell(query.get('query'))}`；"
                      f"来源：语义调用 {query.get('call_index')} 的已验证结构输出。"]
    else:
        lines += ["已保存记录中没有通过验证的自主查询提案。"]
    lines += ["", "提出查询不等于执行查询。候选只在数学选择器入选后才调用真实工具。", ""]
    lines += _catalog_lines(analysis.get("candidate_discovery"))
    lines += ["## LLM 情景与概率假设", "",
              "以下先验、条件胜平负和观测似然均来自语义模型假设，未经过实证校准。"
              "表中概率顺序均为主胜 / 平局 / 客胜。", "",
              "| 情景 | 假设描述 | 先验 | 原始条件胜平负 | 对齐基线后的条件胜平负 |",
              "| --- | --- | ---: | --- | --- |"]
    aligned = semantic.get("aligned_outcome_probabilities") or {}
    for hypothesis in semantic["hypotheses"]:
        lines.append("| " + " | ".join([
            _cell(hypothesis.get("id")), _cell(hypothesis.get("description")),
            _number(hypothesis.get("prior")), _probabilities(hypothesis.get("outcome_probabilities")),
            _probabilities(aligned.get(hypothesis.get("id")))]) + " |")
    if not semantic["hypotheses"]:
        lines.append("| 未取得已验证提案 | 未记录 | 未记录 | 未记录 | 未记录 |")
    lines += [""] + [f"- {_cell(item)}" for item in semantic["assumptions"]]
    group_assumption = (
        "同一动作只尝试一次，同一观测组取得有效信息后关闭；无答案后的后备来源按冻结组策略决定；"
        if conditional.get("planning_policy") == "shared_observation_continuation_v1" else
        "同一观测组只尝试一次；")
    lines += ["", "模型还假设：给定完整情景后，目标结果与不同观测组条件独立；"
              + group_assumption + "跨组残余依赖未建模。"
              "no_answer 的似然在情景间相同，因此不能作为支持或反驳证据。", "",
              "## 条件数学 VoI 与逐轮选择", "",
              ("每个候选对 supports / refutes / no_answer 的尚未见返回作有限求和，计算一步期望对数损失下降；"
               "名义净价值 = 名义期望损失下降 − 声明请求成本。"
               if conditional.get("planning_horizon", 1) == 1 else
               "每个候选按可能返回分支选择一项所有模型共享的稳健续动作，再逐模型计算同一策略的净价值；"
               "已尝试动作、同事实组约束及剩余请求预算在第二步继续生效。"
               "这是一套可执行的逐分支稳健策略，不声称是全局最优策略树。"
               if conditional["meaning"].startswith("branchwise_robust_shared") else
               "每个候选先积分 supports / refutes / no_answer 分支，再在每个分支内计算一个最优后续动作；"
               "报告同时保留即时 net 与前视 planning_net，选择依据是声明模型集合下的前视净价值下界。")
              +
              "这不是 LLM 自报的信息价值分数。", "",
              f"每个 HTTP 请求的声明损失等价成本为 {_number(conditional['request_loss_cost'])} nats；"
              f"预声明污染强度为 {_cell(conditional['contamination_levels'])}。"
              "污染模型将观测似然与不区分情景的边际分布混合，逐模型更新并积分。"
              "净价值下界是这些预声明模型上的最小值，是集合包络，不是置信区间。"
              "其保守范围不覆盖未声明的情景、全部模型误差或开放世界。", ""]
    diagnostics_present = any(
        "branch_risk_increase_probability" in candidate
        for cycle in conditional["cycles"] for candidate in cycle.get("candidates", [])
    )
    if diagnostics_present:
        lines += [
            "本报告额外列出风险分支诊断：`branch_risk_increase_probability` 是在当前声明模型和观测律下，"
            "终端风险高于查询前风险的返回分支概率；`expected_risk_change` 是所有返回的加权风险变化。"
            "它们不是历史频率、预测错误率或置信区间，也不替代鲁棒净 VoI 的动作选择。", ""
        ]
    action_assumptions = {item["action_id"]: item for item in semantic["actions"]}
    contracts = [item for item in semantic["actions"] if item.get("direction_contract")]
    if contracts:
        lines += [
            "本运行还冻结了查询命题的方向合同：每个动作分别声明哪些来源方向算 `supports` 或 `refutes`。"
            "实际正文方向若未命中合同，会被记录为 `direction_contract_mismatch` 并降为 `no_answer`，"
            "不会被重新解释为另一个命题的证据。", ""
        ]
    for cycle in conditional["cycles"]:
        selected = cycle.get("selected_action_id")
        lines += [f"### 第 {cycle['index'] + 1} 轮", "",
                  f"已占用 {_cell(cycle['requests_spent'])} 个 HTTP 请求预算，剩余 {_cell(cycle['remaining_requests'])} 个；"
                  f"当前名义三概率：{_probabilities(cycle.get('prediction_nominal'))}。", ""]
        if diagnostics_present:
            lines += [
                "| 候选 / 问题 | 工具及请求数 | 名义净价值（nats） | 扰动下界（nats） | 名义风险增大分支概率 | 名义期望风险变化 | 稳健风险增大概率范围 | 可行性 / 入选情况 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
            ]
        else:
            lines += [
                "| 候选 / 问题 | 工具及请求数 | 名义净价值（nats） | 扰动下界（nats） | 可行性 / 入选情况 |",
                "| --- | --- | ---: | ---: | --- |"
            ]
        for candidate in cycle["candidates"]:
            binding = candidate["binding"]
            disposition = "入选；" if candidate["selected_by_solver"] else "未入选；"
            target = binding.get("query") or binding.get("url")
            tool_description = f"{_cell(binding.get('tool'))} / {_cell(candidate.get('request_cost'))}"
            if binding.get("title"):
                tool_description += "<br>已见标题：" + _cell(binding["title"])
            if target:
                tool_description += "<br>" + _cell(target)
            cells = [
                f"{_cell(candidate['action_id'])}<br>{_cell(candidate.get('question'))}",
                tool_description,
                _number(candidate.get("net")), _number(candidate.get("lower")),
            ]
            if diagnostics_present:
                robust = candidate.get("robust_branch_risk_diagnostics") or {}
                lower = robust.get("branch_risk_increase_probability_lower")
                upper = robust.get("branch_risk_increase_probability_upper")
                probability = candidate.get("branch_risk_increase_probability")
                cells.extend([
                    "未记录" if probability is None else f"{probability:.4%}",
                    _number(candidate.get("expected_risk_change")),
                    ("未记录" if lower is None or upper is None else
                     f"{lower:.4%} – {upper:.4%}"),
                ])
            cells.append(_cell(disposition + _reason(candidate.get("reason"))))
            lines.append("| " + " | ".join(cells) + " |")
        lines += [""]
        if conditional.get("planning_horizon", 1) == 2:
            lines += ["一步与两步净价值分开列示（以下续动作是查询前的条件计划，不是已执行记录）：", "",
                      "| 候选 | 一步名义净值 | 一步下包络 | 两步名义净值 | 两步下包络 | 条件续动作 |",
                      "| --- | ---: | ---: | ---: | ---: | --- |"]
            for item in cycle["candidates"]:
                policy = item.get("continuation_policy", [])
                continuation = "; ".join(f"{p['observation']} → {p['action_id']}" for p in policy)
                lines.append("| " + " | ".join((_cell(item["action_id"]), _number(item.get("net")),
                    _number(item.get("immediate_lower")), _number(item.get("planning_net")),
                    _number(item.get("planning_lower")), _cell(continuation or "未记录"))) + " |")
            lines += [""]
        if selected:
            lines += [f"求解器选择 `{_cell(selected)}`：在可行候选中净价值下界最高且为正；"
                      "价值相同按保存的动作 ID 排序规则决定。"]
            chosen = next((item for item in cycle["candidates"] if item["action_id"] == selected), {})
            binding = chosen.get("binding", {})
            lines += [f"工具目标：{_cell(binding.get('query') or binding.get('url'))}。"
                      f"返回合同：`{_cell(binding.get('return_contract'))}`。", ""]
            assumption = action_assumptions.get(selected, {})
            if assumption.get("direction_contract"):
                contract = assumption["direction_contract"]
                lines += [f"方向合同：supports=`{_cell(contract.get('supports'))}`；"
                          f"refutes=`{_cell(contract.get('refutes'))}`。", ""]
            if assumption.get("likelihoods"):
                lines += ["该动作的 LLM 观测似然（supports / refutes / no_answer）：", ""]
                lines += [f"- `{_cell(row['hypothesis_id'])}`：{_probabilities(row['probabilities'])}。"
                          for row in assumption["likelihoods"]]
                lines += [""]
            observation = cycle.get("actual_observation")
            if observation:
                for source in cycle["actual_returns"]:
                    lines += [f"实际返回 HTTP {source['receipt_index']}：`{_cell(source.get('status'))}` / "
                              f"`{_cell(source.get('kind'))}`；准入正文字数 {_cell(source['admitted_body_characters'])}；"
                              f"已观察链接数 {_cell(source['observed_link_count'])}；{_cell(source.get('url'))}。"]
                lines += [f"语义分类：`{_cell(observation.get('outcome'))}`；适用性："
                          f"`{_cell(observation.get('applicability'))}`。理由：{_cell(observation.get('reason'))}。"]
                if observation.get("evidence"):
                    lines += ["", "正文引文（逐字匹配只核对出处；本场适用性仍是语义判断）：", ""]
                    for evidence in observation["evidence"]:
                        lines += [f"来源文档 `{_cell(evidence.get('document_id'))}`：", "",
                                  *["> " + _cell(line) for line in evidence.get("quote", "").splitlines()], ""]
                else:
                    lines += ["没有可供本次更新引用的本场正文证据。"]
            else:
                lines += ["没有保存该动作的完成观测；求解器入选不能据此写成已执行或已获得证据。"]
        else:
            lines += [f"本轮停止：{_cell(_reason(cycle.get('stop_reason')))}。"]
        lines += [""]
    if not conditional["cycles"]:
        lines += ["没有保存完整数学决策轮次。", ""]
    lines += ["## 方向性证据", ""]
    if not directional.get("available"):
            lines += [("本运行没有保存方向性记录；当前使用旧观察协议，`supports/refutes` 不自动解释为主队或客队利空/利好。"
                   if not directional.get("required_for_live_update") else
                   "本运行要求方向性观察，但没有获得可用的方向性记录。")]
    else:
        lines += [("方向性门控已参与本次实时更新资格判断；它只判断正文是否明确给出本场影响方向，"
                   "不提供校准的影响大小。" if directional.get("required_for_live_update") else
                   "方向性记录只作为来源语义注释保存，当前未进入后验或 VoI。"), "",
                  "| 动作 | 结果 | 方向 | 主体 | 依据 | 门控 |", "| --- | --- | --- | --- | --- | --- |"]
        for item in directional["records"]:
            lines.append("| " + " | ".join(_cell(item.get(key)) for key in (
                "action_id", "outcome", "direction", "subject", "direction_basis", "directional_gate")) + " |")
            if item.get("directional_claim"):
                lines.append(f"方向性说明：{_cell(item['directional_claim'])}。")
    lines += ["", "## no_answer 不更新核对", ""]
    if analysis["no_answer_checks"]:
        for check in analysis["no_answer_checks"]:
            verified = check["unchanged_within_1e_12"]
            label = "通过" if verified is True else "不一致，需检查" if verified is False else "记录不足，未核对"
            lines += [f"- `{_cell(check['action_id'])}`：{label}。所有已保存污染模型的预测最大变化 "
                      f"{_number(check['max_prediction_delta'])}，情景后验最大变化 {_number(check['max_posterior_delta'])}；"
                      "数值容差 1e-12。"]
    else:
        lines += ["本次没有需要核对的 no_answer / unknown 观测。"]
    lines += ["", "no_answer 可以消耗请求并关闭已尝试观测组，但不能凭空改变情景权重。"
              "上述检查只比较保存的相邻轮次，不证明分类内容真实。", "", "## 停止与边界", "",
              f"最终停止码：`{_cell(terminal.get('stop_reason'))}`；{_cell(_reason(terminal.get('stop_reason')))}。",
              ("停止结论仅限当前已声明模型集合、可见候选菜单、一步规划和剩余预算。"
               if conditional.get("planning_horizon", 1) == 1 else
               "停止结论仅限当前已声明模型集合、可见候选菜单、两步有限前视和剩余预算；"
               "两步模式只覆盖一个已建模后续动作，不能评价更长链路。")
              +
              "它不是全开放世界停止证书，未排除尚未生成的问题或未发现来源。"
              "本运行不声称完成实证校准、预测效果验证或未来赛果核验。"]
    if analysis["failures"]:
        lines += [""] + [f"- 运行问题（{_cell(item.get('stage'))}）：{_cell(item.get('error'))}" for item in analysis["failures"]]
    http_label = (f"实际 HTTP 请求总量未知；已确认回执请求数 {_cell(resources.get('http_requests_confirmed'))}，"
                  f"已预留请求预算 {_cell(resources.get('request_budget_reserved'))} / {_cell(conditional['request_budget'])}；"
                  "HTTP 核账尚未完成，缺失回执不能视为零次请求。"
                  if resources.get("http_accounting_complete") is False else
                  f"实际 HTTP 请求：{_cell(resources.get('http_requests'))} / {_cell(conditional['request_budget'])}；")
    lines += ["", "## 请求与 LLM 成本", "",
              http_label,
              f"语义调用：{_cell(resources.get('semantic_calls'))} / {_cell(conditional['semantic_call_budget'])}。",
              f"HTTP 已记录延迟合计 {_number(resources.get('http_latency_seconds'))} 秒；"
              f"LLM 延迟合计 {_number(resources.get('semantic_latency_seconds'))} 秒。",
              "入口来源发现请求属于预声明启动开销，并非数学 VoI 入选；它计入 HTTP 总请求数。"
              "search_then_read 按至多两个请求预留成本，实际搜索无可用链接时可能只花费一个请求。",
              "LLM 开销单独记录，未纳入当前 VoI 目标。HTTP 的 nats 成本是声明的决策权衡参数，不是价格。",
              "实际货币费用：" + ("未报告，不能视为零。" if resources.get("actual_money") is None
                                  else f"{_cell(resources['actual_money'])}（按原始回执记录，币种以回执为准）。"), ""]
    if resources["reported_token_totals"]:
        lines += ["| 回执 token 字段 | 已报告合计 |", "| --- | ---: |"]
        lines += [f"| {_cell(key)} | {_cell(value)} |" for key, value in sorted(resources["reported_token_totals"].items())]
    lines += ["", f"有 token 回执的调用 {resources['calls_with_token_usage']} 个，缺少 token 回执的调用 "
              f"{resources['calls_without_token_usage']} 个。缓存 token 可能包含于输入 token，不将各字段相加当作总量。", "",
              "## 来源与审计文件", "",
              "下列列出请求回执及已取得的来源返回。核账缺口不等于零次请求；"
              "HTTP 成功、标题、搜索片段或链接数量均不直接构成本场事实。", ""]
    for source in analysis["source_facts"]["sources"]:
        index = source["receipt_index"]
        label = source.get("title") or source.get("url") or "来源未记录"
        url = source.get("url")
        link = f"[{_cell(label)}](<{url}>)" if isinstance(url, str) and url.startswith(("https://", "http://")) else _cell(label)
        lines += [f"- HTTP {index}：{link}；用途 `{_cell(source.get('purpose'))}`；"
                  f"状态 `{_cell(source.get('status'))}` / `{_cell(source.get('kind'))}`；"
                  f"准入正文 {_cell(source['admitted_body_characters'])} 字符。"
                  f"观察时间 {_cell(source.get('observed_at'))}，可用时间 {_cell(source.get('available_at'))}，"
                  f"来源发布时间 {_cell(source.get('source_published_at'))}。"
                  f"回执：[request-{index:03d}.json](http/request-{index:03d}.json)。"]
        if source.get("receipt_error"):
            lines += [f"  请求目标：{_cell(source.get('request_target'))}；回执状态 "
                      f"`{_cell(source.get('receipt_status'))}`；已预留 {_cell(source.get('reserved_requests'))} 个请求；"
                      f"实际请求次数 {_cell(source.get('http_attempts'))}。错误：{_cell(source['receipt_error'])}。"]
        admission = source.get("admission", {})
        if admission.get("reasons") or admission.get("excluded_segments"):
            lines += [f"  准入限制：{_cell(admission.get('reasons'))}；排除段落原因：{_cell(admission.get('excluded_segments'))}。"]
    lines += ["", "完整精度数值、全部候选似然和引用见 [analysis.json](analysis.json)；"
              "原始运行与回执见 [run.json](run.json)。报告只摘录审计所需字段，保留原始记录。", ""]
    return "\n".join(lines)


def write_semantic_report(run, output):
    """Write report.md and compact analysis.json; return a JSON-safe summary."""
    if not isinstance(run, Mapping):
        raise ValueError("run must be a saved semantic runtime mapping")
    analysis = _analysis(run)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    report_path, analysis_path = destination / "report.md", destination / "analysis.json"
    # Serialize first: invalid numbers must not leave a misleading partial report.
    payload = json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    markdown = _render(analysis)
    analysis_path.write_text(payload + "\n", encoding="utf-8")
    report_path.write_text(markdown, encoding="utf-8")
    return {"version": VERSION, "run_id": analysis["run_id"], "status": analysis["status"],
            "study_role": analysis["study_role"],
            "report_path": str(report_path.resolve()), "analysis_path": str(analysis_path.resolve()),
            "initial_prediction": analysis["initial_prediction"], "final_prediction": analysis["final_prediction"],
            "decision_cycles": len(analysis["conditional_voi"]["cycles"]),
            "observations": len(analysis["source_facts"]["observations"]),
            "stop_reason": (analysis.get("terminal") or {}).get("stop_reason"),
            "http_requests": analysis["resources"].get("http_requests"),
            "http_requests_confirmed": analysis["resources"].get("http_requests_confirmed"),
            "http_accounting_complete": analysis["resources"].get("http_accounting_complete"),
            "semantic_calls": analysis["resources"].get("semantic_calls"),
            "global_stop_certificate": False, "probability_improvement_proven": False}
