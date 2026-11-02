from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace

import pytest

from football_agent.audit import digest
from football_agent.decision_risk_experiment import proposal
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_choice import InformationChoiceProtocol
from football_agent.inquiry_continuation import ContinuationPolicy
from football_agent.inquiry_hypotheses import get_protocol
from football_agent.inquiry_probability import (OUTCOMES, ReturnProbabilityProtocol,
    evaluate_forecast, score_distribution, summarize_reviews, validate_distribution)
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)
URLS = ["https://example.com/report_a", "https://example.com/report_b"]
TASK = {"task_id": "constructed-probability", "objective": "Check information before a personnel decision.",
        "deadline": (NOW+timedelta(hours=1)).isoformat(), "initial_facts": [], "seed_urls": URLS}


def distribution(outcome=None):
    return {k: float(k == outcome) if outcome else 1/len(OUTCOMES) for k in OUTCOMES}


def forecast(outcome=None, *, abstain=False):
    return {"status": "abstain" if abstain else "elicited",
            "probabilities": None if abstain else distribution(outcome),
            "basis_document_ids": [], "reason": "Declared test forecast, not fitted or calibrated."}


def propose(actions=("report_a", "report_b"), *, selected="report_b", kind="query", predicted=None, abstain=False):
    response = proposal(list(actions), chosen=selected, kind=kind)
    response["choice_review"] = {"alternatives": [{"candidate_id": c["candidate_id"],
        "information_advantage": "Could address the personnel question.",
        "source_limitations": "Answer availability unknown.", "evidence_document_ids": [],
        "not_selected_reason": None if c["candidate_id"] == response["decision"]["candidate_id"] else "Defer this route."}
        for c in response["candidates"]], "tradeoff": "Compare possible information roles.",
        "missing_routes": "The action set is not exhaustive."}
    for candidate in response["candidates"]:
        candidate["outcome_forecast"] = forecast(predicted, abstain=abstain)
    return response


def extracted_answer(public):
    response = propose((), kind="stop")
    document = public["documents"][-1]
    response["claim_updates"] = [{"claim_id": "c1", "question_id": "q1", "status": "source_claim",
        "text": "The source reports Player K starting.",
        "citations": [{"document_id": document["document_id"], "quote": document["text"]}],
        "evidence": {"kind": "official_lineup", "subject": "Team-A", "scope": "match", "scope_key": "fixture-M"},
        "hypothesis_links": [{"hypothesis_id": "h1", "relation": "supports"}]}]
    return response


def setup(tmp_path, responses, *, status=200):
    calls, inputs = [], []
    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return SimpleNamespace(status_code=status, content=b"For fixture M, Team A reports that Player K will start.",
                                   headers={}, close=lambda: None)
        def close(self): pass
    class Tools(OpenWebTools):
        def configuration(self):
            return {**super().configuration(), "execution_mode": "constructed_transport_no_network"}
    class Provider:
        def configuration(self): return {"provider": "constructed-probability-test"}
        def decide(self, public):
            inputs.append(deepcopy(public))
            response = responses[len(inputs)-1]
            return {"response": response(public) if callable(response) else deepcopy(response),
                    "receipt": {"real_provider_call": False}}
    tools = Tools(RawArchive(tmp_path/"evidence"), ["example.com"], session_factory=Session, clock=lambda: NOW)
    workflow = InquiryWorkflow(tmp_path/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=ReturnProbabilityProtocol(), continuation=ContinuationPolicy(), source_catalog=[])
    return workflow, calls, inputs


def test_scoring_convention_and_zero_probability_infinite_loss_are_explicit():
    uniform = score_distribution(distribution(), "failure")
    assert uniform["raw_multiclass_brier"] == pytest.approx(6/7)
    assert uniform["log_loss_nats"] == pytest.approx(math.log(7))
    correct = score_distribution(distribution("failure"), "failure")
    assert correct["raw_multiclass_brier"] == correct["log_loss_nats"] == 0
    wrong = score_distribution(distribution("source_support"), "failure")
    assert wrong["raw_multiclass_brier"] == 2 and wrong["log_loss_nats"] is None
    assert wrong["log_loss_status"] == "positive_infinity"
    json.dumps(wrong, allow_nan=False)


def test_abstention_and_infinite_loss_have_different_denominators():
    reviews = [{"evaluation": evaluate_forecast(f, "failure")} for f in
               [forecast("source_support"), forecast("failure"), forecast(abstain=True)]]
    summary = summarize_reviews(reviews)
    assert summary["observed_reviews"] == 3 and summary["scored_forecasts"] == 2
    assert summary["abstentions"] == summary["zero_probability_observations"] == 1
    assert summary["mean_raw_brier"] == 1 and summary["mean_log_loss_nats"] is None
    assert summary["mean_log_loss_status"] == "positive_infinity"
    assert reviews[-1]["evaluation"]["forecast_scores"] is None
    assert reviews[-1]["evaluation"]["brier_difference_vs_uniform"] is None
    empty = summarize_reviews([])
    assert empty["mean_log_loss_status"] == "not_scored" and empty["mean_raw_brier"] is None


@pytest.mark.parametrize("change", ["missing", "extra", "sum", "bool", "nan", "infinity", "negative", "above_one"])
def test_invalid_distributions_are_rejected_without_normalizing(change):
    p = distribution()
    if change == "missing": p.pop("failure")
    elif change == "extra": p["not_a_label"] = 0
    else: p["failure"] = {"sum": .8, "bool": True, "nan": float("nan"), "infinity": float("inf"),
                          "negative": -.1, "above_one": 1.1}[change]
    with pytest.raises(ValueError): validate_distribution(p)


def test_selected_forecast_is_frozen_before_query_and_only_scored_after_extraction(tmp_path):
    first = propose(predicted="source_support")
    first["candidates"][0]["outcome_forecast"] = forecast("failure")
    workflow, calls, inputs = setup(tmp_path, [first, extracted_answer])
    run = workflow.start(TASK, max_requests=1, max_llm_calls=2)
    run = workflow.step(run["run_id"])
    assert calls == [URLS[1]] and run["return_probability_reviews"] == []
    intent = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_intent")
    assert intent["data"]["inquiry_contract"]["outcome_forecast"] == first["candidates"][1]["outcome_forecast"]
    assert intent["seq"] < next(e["seq"] for e in run["events"] if e["type"] == "inquiry_acquisition_returned")
    run = workflow.step(run["run_id"])
    assert len(run["return_probability_reviews"]) == 1
    review = run["return_probability_reviews"][0]
    assert review["observed_outcome"] == "source_support"
    assert review["evaluation"]["forecast_scores"]["raw_multiclass_brier"] == 0
    assert review["forecast"] == first["candidates"][1]["outcome_forecast"]
    assert review["realized_decision_utility"] is None and run["decisions"][0]["expected_gain"] is None
    assert inputs[0]["return_probability_reviews"] == inputs[1]["return_probability_reviews"] == []
    assert workflow.public(run)["return_probability_reviews"] == [review]
    assert verify(tmp_path, run["run_id"])["decisions_verified"] == 2
    report = tmp_path/"report.md"
    workflow.export(run["run_id"], report)
    assert "查询返回概率的前瞻核对" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("abstain", [False, True])
def test_failed_transport_is_not_negative_football_evidence_and_abstention_is_preserved(tmp_path, abstain):
    workflow, calls, _ = setup(tmp_path, [propose(predicted="source_support", abstain=abstain), propose((), kind="defer")], status=503)
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert run["claims"] == [] and calls == [URLS[1]]
    review = run["return_probability_reviews"][0]
    assert review["observed_outcome"] == "failure"
    score = review["evaluation"]["forecast_scores"]
    if abstain:
        assert score is None and review["forecast"]["probabilities"] is None
    else:
        assert score["log_loss_status"] == "positive_infinity" and score["log_loss_nats"] is None
    json.dumps(run, allow_nan=False)
    verify(tmp_path, run["run_id"])


@pytest.mark.parametrize("mutation", ["missing", "sum", "unknown_document", "abstain_with_values", "elicited_null"])
def test_bad_forecast_blocks_acquisition_before_http(tmp_path, mutation):
    response = propose()
    f = response["candidates"][0]["outcome_forecast"]
    if mutation == "missing": response["candidates"][0].pop("outcome_forecast")
    elif mutation == "sum": f["probabilities"]["failure"] = .9
    elif mutation == "unknown_document": f["basis_document_ids"] = ["unknown"]
    elif mutation == "abstain_with_values": f["status"] = "abstain"
    else: f["probabilities"] = None
    workflow, calls, _ = setup(tmp_path, [response])
    run = workflow.run(workflow.start(TASK)["run_id"])
    assert run["reason"] == "INVALID_INQUIRY_RESPONSE" and not calls
    proof = verify(tmp_path, run["run_id"])
    assert proof["status"] == "VERIFIED_REJECTED_RESPONSE_TRACE_NOT_COMPLETED_INQUIRY"
    assert proof["rejected_model_responses_verified"] == 1 and proof["decisions_verified"] == 0


def test_later_acquisition_does_not_label_an_earlier_unselected_forecast(tmp_path):
    first = propose(predicted="failure")
    later = propose(("report_a",), selected="report_a", predicted="no_answer")
    workflow, calls, inputs = setup(tmp_path, [first, later, propose((), kind="defer")], status=503)
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    assert calls == [URLS[1], URLS[0]]
    assert len(run["return_probability_reviews"]) == 2
    assert [r["forecast"] for r in run["return_probability_reviews"]] == [
        first["candidates"][1]["outcome_forecast"], later["candidates"][0]["outcome_forecast"]]
    assert len(inputs[2]["return_probability_reviews"]) == 1
    assert len({r["request_id"] for r in run["return_probability_reviews"]}) == 2
    verify(tmp_path, run["run_id"])


@pytest.mark.parametrize("mutation", ["score", "contract"])
def test_rehashing_does_not_hide_modified_frozen_probabilities_or_scores(tmp_path, mutation):
    workflow, _, _ = setup(tmp_path, [propose(), extracted_answer])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    for event in run["events"]:
        if mutation == "score":
            for review in event["snapshot"].get("return_probability_reviews", []):
                review["evaluation"]["forecast_scores"]["raw_multiclass_brier"] = .123
            for decision in event["snapshot"]["decisions"]:
                for review in decision["hypothesis_assessment"].get("return_probability_reviews", []):
                    review["evaluation"]["forecast_scores"]["raw_multiclass_brier"] = .123
            if event["type"] == "inquiry_decision":
                for review in event["data"]["hypothesis_assessment"].get("return_probability_reviews", []):
                    review["evaluation"]["forecast_scores"]["raw_multiclass_brier"] = .123
        else:
            for receipt in event["snapshot"]["receipts"]:
                receipt["inquiry_contract"]["outcome_forecast"]["probabilities"] = distribution("failure")
    previous = None
    for event in run["events"]:
        event["previous_hash"] = previous
        event["event_hash"] = digest({k: v for k, v in event.items() if k != "event_hash"})
        previous = event["event_hash"]
    run.update(deepcopy(run["events"][-1]["snapshot"]))
    workflow.store.save(run)
    with pytest.raises(ValueError): verify(tmp_path, run["run_id"])


def test_protocol_extension_is_opt_in_and_preserves_the_old_schema():
    assert isinstance(get_protocol("return-probability-inquiry-v1"), ReturnProbabilityProtocol)
    assert "outcome_forecast" not in InformationChoiceProtocol.schema["properties"]["candidates"]["items"]["properties"]


def rejected_extraction(public):
    response = extracted_answer(public)
    response["claim_updates"][0]["citations"][0]["quote"] = "This quote was never acquired."
    return response


def test_invalid_extraction_exports_acquired_return_without_a_label_or_score(tmp_path):
    from football_agent.return_dataset import build_dataset
    workflow, calls, _ = setup(tmp_path, [propose(), rejected_extraction])
    run = workflow.run(workflow.start(TASK, max_requests=2, max_llm_calls=3)["run_id"])
    proof = verify(tmp_path, run["run_id"])
    assert proof["rejected_model_responses_verified"] == proof["acquired_returns_without_valid_review"] == 1
    assert run["claims"] == run["return_reviews"] == run["return_probability_reviews"] == []
    assert len(calls) == len(run["decisions"]) == 1
    reference = {"root": str(tmp_path), "run_id": run["run_id"], "head_hash": run["events"][-1]["event_hash"],
        "cluster_id": "constructed-one-investigation", "origin": "constructed",
        "cluster_basis": "Constructed offline validation failure."}
    data = build_dataset([reference], as_of=NOW+timedelta(hours=2))
    assert len(data["rows"]) == 1
    ret = data["rows"][0]["observed_return"]
    assert ret["document_status"] == "OK" and ret["query_review"] is None and ret["return_review"] is None
    assert data["rows"][0]["provenance"]["verification_status"] == proof["status"]


@pytest.mark.parametrize("mutation", ["wrong_reason", "actually_valid_reply"])
def test_a_rejection_marker_cannot_hide_an_unreproduced_or_valid_reply(tmp_path, mutation):
    workflow, _, _ = setup(tmp_path, [propose(), rejected_extraction])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    rejection = next(e for e in run["events"] if e["type"] == "inquiry_response_rejected")
    if mutation == "wrong_reason":
        rejection["data"]["reason"] = "A different fabricated rejection."
    else:
        returned = [e for e in run["events"] if e["type"] == "inquiry_reasoning_returned"][-1]
        public = workflow.store.get_blob(run["run_id"], returned["data"]["input_ref"])
        reply = workflow.store.get_blob(run["run_id"], returned["data"]["output_ref"])
        reply["response"] = extracted_answer(public)
        returned["data"]["output_ref"] = workflow.store.put_blob(run["run_id"], reply)
    previous = None
    for event in run["events"]:
        event["previous_hash"] = previous
        event["event_hash"] = digest({k: v for k, v in event.items() if k != "event_hash"})
        previous = event["event_hash"]
    run.update(deepcopy(run["events"][-1]["snapshot"]))
    workflow.store.save(run)
    workflow.store.verify(run["run_id"])
    with pytest.raises(ValueError): verify(tmp_path, run["run_id"])
