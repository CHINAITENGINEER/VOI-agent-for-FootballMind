from copy import deepcopy
from hashlib import sha256
import json

import pytest

from football_agent.match_analysis_report import build_analysis_report, write_analysis_report
from football_agent.match_evidence import build_match_evidence


MATCH = {"match_id": "match-1", "home_team": "Arsenal", "away_team": "Leeds",
         "kickoff_at": "2026-10-10T11:30:00Z", "prediction_cutoff": "2026-10-09T11:30:00Z"}


def action(aid, *, query=None):
    return {"action_id": aid, "tool": "search_web" if query else "frozen_evidence_projection",
            "query": query, "source": "public_source", "cost_status": "simulated",
            "cost": {"requests": 1, "money": .01}}


def candidate(aid, net):
    return {"action_id": aid, "value": {"status": "COMPUTED" if net is not None else "INELIGIBLE",
             "net_voi": net, "gross_evsi": .05 if net is not None else None,
             "cost": .01, "loss": "log_loss", "reason": "response_model_unavailable" if net is None else "computed_coherent_evsi"}}


def run():
    chosen = action("open-1", query="Arsenal Leeds 2026-10-10 update")
    return {"run_id": "test-run", "final_state": {"match_id": "match-1", "information_mode": "historical_reconstruction"},
            "initial_probabilities": [.5, .3, .2], "final_probabilities": [.5, .3, .2],
            "terminal_selection": {"decision": "defer", "reason": "candidate_generation_incomplete", "stop_certificate": None},
            "reserved_cost": {"money": .01, "requests": 1}, "measured_resources": {"money": 0., "requests": 1., "latency_ms": 24., "tokens": 0.},
            "unmeasured_resource_receipts": {"money": 1, "requests": 0, "latency_ms": 0, "tokens": 0},
            "attempts": [{"action_id": "open-1", "status": "NO_DATA"}],
            "cycles": [{"step": 0, "selected_action": chosen, "selection_basis": "bounded_unknown_information_exploration",
                        "generated_actions": [chosen, action("results_view")],
                        "strict_decision": {"candidates": [candidate("open-1", None), candidate("results_view", -.03)]},
                        "exploration_record": {"selected_p": 1., "eligible_ids": ["open-1"]},
                        "execution_status": "returned", "admission": {"status": "not_admitted", "reason": "provider_no_data"},
                        "prediction_before": [.5, .3, .2], "prediction_after": [.5, .3, .2],
                        "public_before": {"budget": {"limits": {"money": .03, "requests": 3}}, "attempts": []}}]}


def test_defer_is_complete_with_gaps_and_exploration_never_becomes_numeric_voi():
    original = run()
    before = deepcopy(original)
    report = build_analysis_report(original, match=MATCH)
    assert original == before
    assert report["status"] == "completed_with_gaps"
    assert report["probabilities"]["initial"] == report["probabilities"]["final"] == [.5, .3, .2]
    choice = report["decisions"][0]
    assert choice["value"]["net_voi"] is None
    assert choice["selection_basis"] == "bounded_unknown_information_exploration"
    assert choice["unselected_actions"][0]["value"]["net_voi"] == -.03
    assert choice["unselected_actions"][0]["reason"] == "computed_nonpositive_value"
    assert report["budget"]["measured_resources"]["money"]["total"] is None
    assert report["budget"]["measured_resources"]["requests"]["total"] == 1.


def test_historical_text_base_is_not_reported_as_an_observed_market_quote():
    value = run()
    value['schema'] = 'learned-voi-open-execution-v4'
    value['context_response_learning'] = {'probability_updates': 0, 'evidence_origin': 'historical_reconstruction'}
    value['cycles'][0]['public_before']['prediction'] = {
        'probability_basis': 'historical_text_f_with_A_learned_base',
        'base_probabilities': [.5, .3, .2]}
    report = build_analysis_report(value, match=MATCH)
    assert report['probabilities']['initial_probability_basis'] == 'historical_text_f_with_A_learned_base'
    assert report['probabilities']['original_prior'] is None
    assert report['probabilities']['model_base'] == [.5, .3, .2]


def historical_run():
    value = run()
    value["schema"] = "learned-voi-open-execution-v4"
    value["context_response_learning"] = {"probability_updates": 1, "evidence_origin": "historical_reconstruction"}
    prediction = {"probability_basis": "historical_text_f_with_A_learned_base", "base_probabilities": [.5, .3, .2]}
    value["cycles"][0]["public_before"].update(prediction=deepcopy(prediction), documents=[])
    value["cycles"][0]["public_after"] = {"prediction": deepcopy(prediction), "documents": []}
    return value


def test_http_success_and_changed_prediction_do_not_claim_body_consumption(tmp_path):
    value = historical_run()
    cycle = value["cycles"][0]
    for public in (cycle["public_before"], cycle["public_after"]):
        public["prediction"]["modeled_read_documents"] = 0
    cycle["prediction_after"] = value["final_probabilities"] = [.51, .29, .2]
    cycle["public_after"]["documents"] = [{"document_id": "returned-page", "url": "https://club.example/news/international-duty",
        "http_status": 200, "status": "OK", "text": "Raw text can be nonempty without being admitted as a body.",
        "publication_status": "unknown", "source_published_at": None,
        "structured_news": {"role": "unknown", "segments": [],
                            "excluded": [{"reason": "invalid_or_ambiguous_contentful_publication"}]}}]
    report = build_analysis_report(value, match=MATCH)
    consumption = report["decisions"][0]["text_model_consumption"]
    assert consumption["before"]["modeled_body_documents"] == consumption["after"]["modeled_body_documents"] == 0
    assert consumption["newly_modeled_body_documents"] is None  # Counts alone do not identify additions.
    assert consumption["modeled_body_document_delta"] == 0
    assert consumption["after"]["modeled_link_labels"] is None
    receipt = consumption["new_document_returns"][0]
    assert receipt["http_success"] is True and receipt["body_extraction_status"] == "no_admitted_body_segments"
    assert receipt["source_published_at"] is None
    assert report["semantics"]["open_text_probability_updates_claimed"] is False
    assert report["semantics"]["text_state_probability_updates_recorded"] is True
    write_analysis_report(value, match=MATCH, output=tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "不把概率变化归因于返回正文" in markdown
    assert "标题/链接状态变化不证明比赛事实或预测改善" in markdown
    assert "来源发布时间无效或无法唯一绑定" in markdown
    assert "按实际返回时刻的正文、来源和查询历史更新概率" not in markdown


def test_consumption_telemetry_tracks_actual_new_ids_and_retains_source_times():
    value = historical_run()
    cycle = value["cycles"][0]
    times = [{"url": "https://club.example/news/update", "source_published_at": "2026-10-06T09:00:00Z",
              "source_modified_at": None, "outcome": "DO_NOT_PROJECT"}]
    telemetry = {"schema": "historical-text-input-v1", "modeled_body_documents": 1,
                 "modeled_body_paragraphs": 3, "modeled_link_labels": 5, "excluded_link_labels": 2,
                 "modeled_body_document_ids": ["body-old"], "modeled_publication_times": [], "unmodeled_documents": []}
    cycle["public_before"]["prediction"].update(modeled_read_documents=99, text_model_input=deepcopy(telemetry))
    cycle["public_after"]["prediction"].update(modeled_read_documents=99, text_model_input={
        **telemetry, "modeled_body_document_ids": ["body-new"], "modeled_publication_times": times,
        "unmodeled_documents": [{"document_id": "omitted-page", "reason": "missing_source_publication",
                                  "outcome": "DO_NOT_PROJECT"}]})
    consumption = build_analysis_report(value, match=MATCH)["decisions"][0]["text_model_consumption"]
    assert consumption["after"]["modeled_body_documents"] == 1  # New telemetry takes precedence.
    assert consumption["modeled_body_document_delta"] == 0
    assert consumption["newly_modeled_body_documents"] == 1  # A replacement is not zero additions.
    assert consumption["after"]["modeled_link_labels"] == 5
    assert consumption["after"]["excluded_link_labels"] == 2
    assert consumption["after"]["modeled_publication_times"][0]["source_published_at"] == "2026-10-06T09:00:00Z"
    assert consumption["after"]["unmodeled_documents"][0]["reason"] == "missing_source_publication"
    assert "DO_NOT_PROJECT" not in json.dumps(consumption)
    assert consumption["probability_change_attribution"] == "not_identified_by_input_counts"


def test_missing_consumption_telemetry_stays_unknown_and_other_modes_are_unchanged():
    report = build_analysis_report(historical_run(), match=MATCH)
    consumption = report["decisions"][0]["text_model_consumption"]
    assert consumption["before"]["telemetry_status"] == "unknown"
    assert consumption["after"]["modeled_body_documents"] is None
    assert consumption["after"]["modeled_publication_times"] is None
    assert consumption["after"]["unmodeled_documents"] is None
    assert consumption["modeled_body_document_delta"] is None
    assert "text_model_consumption" not in build_analysis_report(run(), match=MATCH)["decisions"][0]


def test_base_only_forecast_does_not_explain_zero_gain_as_useless_external_information(tmp_path):
    value = historical_run()
    value['cycles'][0]['public_before']['prediction']['gain_model_summary'] = {
        'constant_baseline_selected': True, 'predictor_text_response_supported': False,
        'selection_independent_matches': 5}
    report = build_analysis_report(value, match=MATCH)
    assert report['gain_model_summary']['predictor_text_response_supported'] is False
    write_analysis_report(value, match=MATCH, output=tmp_path)
    text = (tmp_path/'report.md').read_text(encoding='utf-8')
    assert '零增益只表示模型没有内容响应' in text
    assert '相应候选保留 UNKNOWN' in text


def test_constant_gain_baseline_is_not_reported_as_learned_state_conditional_value(tmp_path):
    value = historical_run()
    summary = {"fit_scheme": "independent_match_cross_validation", "constant_baseline_selected": True,
               "selected_regularization": None, "selection_independent_matches": 5,
               "conditioning_claim": "not_learned_constant_gain_baseline",
               "empirical_decision_benefit_established": False, "outcome": "DO_NOT_PROJECT"}
    value["cycles"][0]["public_before"]["prediction"]["gain_model_summary"] = summary
    report = build_analysis_report(value, match=MATCH)
    assert report["gain_model_summary"]["constant_baseline_selected"] is True
    assert report["gain_model_summary"]["selection_independent_matches"] == 5
    assert report["gain_model_summary"]["conditioning_claim"] == "not_learned_constant_gain_baseline"
    assert report["semantics"]["conditional_gain_learning_established"] is False
    assert report["semantics"]["support_gate_action_change_proves_learned_conditional_gain"] is False
    computed = report["decisions"][0]["unselected_actions"][0]["value"]
    assert computed["meaning"] == "constant_empirical_gain_minus_declared_cost"
    assert computed["gain_model_conditioning"] == "constant_baseline"
    assert "DO_NOT_PROJECT" not in json.dumps(report)
    write_analysis_report(value, match=MATCH, output=tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "B 分区内按整场验证选择了常数均值基线（5 场独立比赛）" in markdown
    assert "数值收益估计不区分当前状态和候选" in markdown
    assert "自主条件信息价值尚未学出，也未验证" in markdown
    assert "不能把这种变化当作条件收益模型已学会信息选择的证据" in markdown
    assert "常数基线净收益估计" in markdown
    assert "信息选择依据是条件查询收益" not in markdown


def test_missing_gain_summary_does_not_infer_constant_selection_or_conditioning_success():
    report = build_analysis_report(historical_run(), match=MATCH)
    assert report["gain_model_summary"]["telemetry_status"] == "unknown"
    assert report["gain_model_summary"]["constant_baseline_selected"] is None
    assert report["gain_model_summary"]["conditioning_claim"] is None
    assert report["gain_model_summary"]["empirical_decision_benefit_established"] is None
    assert report["semantics"]["conditional_gain_learning_established"] is None
    assert "gain_model_summary" not in build_analysis_report(run(), match=MATCH)


def test_numeric_selection_is_displayed_only_with_a_computed_positive_value():
    value = run()
    cycle = value["cycles"][0]
    cycle["selected_action"] = action("results_view")
    cycle["selection_basis"] = "computed_positive_voi"
    cycle["strict_decision"]["candidates"] = [candidate("results_view", .025)]
    report = build_analysis_report(value, match=MATCH)
    assert report["decisions"][0]["value"]["net_voi"] == .025
    assert report["status"] == "completed_with_gaps"
    cycle["strict_decision"]["candidates"] = [candidate("results_view", None)]
    assert build_analysis_report(value, match=MATCH)["status"] == "incomplete"


def test_halted_or_unreconciled_run_cannot_be_reported_as_completed():
    value = run()
    value["terminal_selection"] = {"decision": "halt", "reason": "execution_reconciliation_required"}
    value["cycles"][0]["execution_status"] = "execution_outcome_unknown"
    report = build_analysis_report(value, match=MATCH)
    assert report["status"] == "incomplete"
    assert "selected_action_execution_not_reconciled" in report["incomplete_reasons"]
    minimal = {"run_id": "partial", "terminal_selection": {"decision": "defer"}}
    report = build_analysis_report(minimal, match=MATCH)
    assert report["status"] == "incomplete"
    assert report["probabilities"]["final"] is None
    assert all(v["total"] is None for v in report["budget"]["measured_resources"].values())


def test_hidden_result_fields_are_never_read_or_projected():
    class GuardedRun(dict):
        def get(self, key, default=None):
            assert key not in {"outcome", "labels", "result", "realized_gain", "hidden_returns"}
            return super().get(key, default)

    base = run()
    guarded = GuardedRun({**base, "outcome": "LEAK", "realized_gain": "LEAK", "hidden_returns": {"secret": "LEAK"}})
    report = build_analysis_report(guarded, match=MATCH)
    assert report == build_analysis_report(base, match=MATCH)
    assert "LEAK" not in json.dumps(report)


def test_source_context_remains_unconfirmed_and_multiple_quotes_are_not_independent_sources(tmp_path):
    value = run()
    text = ("Arsenal vs Leeds on 2026-10-10. "
            "Arsenal have given no official update on Kai Havertz's fitness ahead of the game.")
    doc = {"document_id": "page-1", "kind": "page", "status": "OK", "url": "https://club.example/news/team-update",
           "raw_sha256": sha256(text.encode()).hexdigest(), "text": text,
           "observed_at": "2026-10-06T09:00:00Z", "available_at": "2026-10-06T09:00:01Z"}
    value["evidence"] = build_match_evidence(MATCH, [doc], as_of="2026-10-06T10:00:00Z")
    report = build_analysis_report(value, match=MATCH)
    assert len(report["evidence"]) == 2 and len(report["sources"]) == 1
    assert all(not item["independently_verified"] for item in report["evidence"])
    assert report["followup_candidates"][0]["expected_gain"] is None
    assert report["semantics"]["policy_benefit_proven"] is False
    paths = write_analysis_report(value, match=MATCH, output=tmp_path)
    assert paths["status"] == "completed_with_gaps"
    assert json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8")) == report
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "https://club.example/news" in markdown
    assert "Havertz" in markdown and "伤情" in markdown
    assert "2 条引文来自 1 个不同页面地址" in markdown
    assert "Arsenal have given no official update" not in markdown
    assert "本轮分析已完成，仍有未决信息" in markdown


def test_mismatched_identity_and_invalid_probability_cannot_be_silently_reported():
    with pytest.raises(ValueError, match="differs"):
        build_analysis_report(run(), match={**MATCH, "match_id": "another-match"})
    with pytest.raises(ValueError, match="sum to one"):
        build_analysis_report({**run(), "final_probabilities": [.8, .4, .1]}, match=MATCH)
