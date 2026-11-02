"""Regression for hash-valid, preinserted extraction and forecast histories.

All requests use the probability suite's constructed transport. The forged
history changes the published input and recomputes later projections, so these
checks exercise provenance rather than a stale hash or inconsistent summary.
"""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from football_agent.audit import digest
from football_agent.inquiry_verify import verify


# Load the sibling helpers by path so isolated importlib collection does not
# rely on pytest putting this directory on sys.path or collecting it first.
_spec = spec_from_file_location("_probability_prefix_fixtures", Path(__file__).with_name("test_inquiry_probability.py"))
_fixtures = module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
TASK, extracted_answer, propose, probability_setup = (
    _fixtures.TASK, _fixtures.extracted_answer, _fixtures.propose, _fixtures.setup)


def _inject_before_second_extraction(workflow, run, field, forged):
    events = run["events"]
    reserved = [i for i, event in enumerate(events) if event["type"] == "inquiry_reasoning_reserved"][1]
    returned = next(i for i in range(reserved + 1, len(events))
                    if events[i]["type"] == "inquiry_reasoning_returned")
    decided = next(i for i in range(returned + 1, len(events)) if events[i]["type"] == "inquiry_decision")
    old_ref = events[reserved]["data"]["input_ref"]
    public = workflow.store.get_blob(run["run_id"], old_ref)
    assert public[field] == []
    public[field] = [deepcopy(forged)]
    new_ref = workflow.store.put_blob(run["run_id"], public)
    assert new_ref != old_ref

    # Recompute downstream state from the poisoned input. This is precisely
    # why replay must bind public histories to its previously verified prefix.
    response = workflow.store.get_blob(run["run_id"], events[returned]["data"]["output_ref"])["response"]
    claims = {claim["claim_id"]: claim for claim in public["claims"]}
    claims.update({claim["claim_id"]: claim for claim in response["claim_updates"]})
    replay = {**deepcopy(public), "claims": list(claims.values()),
              "receipts": deepcopy(events[decided]["snapshot"]["receipts"])}
    assessment = workflow.protocol.advance(replay, response)
    decision = deepcopy(events[decided]["data"])
    decision["hypothesis_assessment"] = assessment
    stored_claims = {claim["claim_id"]: claim for claim in public["claims"]}
    stored_claims.update({claim["claim_id"]: claim for claim in events[decided]["snapshot"]["claims"]})

    for i in range(reserved, len(events)):
        event = events[i]
        event["snapshot"][field] = [deepcopy(forged)]
        if event["data"].get("input_ref") == old_ref:
            event["data"]["input_ref"] = new_ref
        pending = event["snapshot"].get("pending")
        if pending and pending.get("input_ref") == old_ref:
            pending["input_ref"] = new_ref
        if i >= decided:
            for key in workflow.protocol.projection_fields:
                event["snapshot"][key] = deepcopy(replay[key])
            event["snapshot"]["claims"] = [deepcopy(stored_claims[key]) for key in claims]
            event["snapshot"]["decisions"][-1] = deepcopy(decision)
        if i == decided:
            event["data"] = deepcopy(decision)

    previous = None
    for event in events:
        event["previous_hash"] = previous
        event["event_hash"] = digest({key: value for key, value in event.items() if key != "event_hash"})
        previous = event["event_hash"]
    run.update(deepcopy(events[-1]["snapshot"]))
    workflow.store.save(run)
    workflow.store.verify(run["run_id"])
    return assessment


@pytest.mark.parametrize("field", ["return_probability_reviews", "query_reviews", "return_reviews", "claims"])
def test_preinserted_scoring_or_label_history_is_rejected_even_after_consistent_rehash(tmp_path, field):
    workflow, calls, _ = probability_setup(tmp_path, [propose(), extracted_answer])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert verify(tmp_path, run["run_id"])["decisions_verified"] == 2
    assert len(calls) == 1 and run["configuration"]["tools"]["execution_mode"] == "constructed_transport_no_network"
    assert run["return_probability_reviews"][0]["evaluation"]["forecast_scores"]["raw_multiclass_brier"] == pytest.approx(6/7)

    forged = deepcopy(run[field][0])
    if field == "return_probability_reviews":
        forged["evaluation"]["forecast_scores"].update(
            raw_multiclass_brier=0.0, log_loss_nats=0.0, realized_outcome_probability=1.0)
        forged["evaluation"]["brier_difference_vs_uniform"] = -6/7
    elif field == "query_reviews":
        forged["outcome"] = "source_challenge"
    elif field == "return_reviews":
        forged["outcome"] = "failure"
    else:
        forged["claim_id"] = "preinserted-unreported-claim"
        forged["text"] = "This source claim was not emitted by any verified prior reply."

    assessment = _inject_before_second_extraction(workflow, run, field, forged)
    if field == "return_probability_reviews":
        # The old verifier trusted this history and skipped the real 6/7 loss.
        assert assessment["return_probability_reviews"] == []
        assert assessment["return_probability_summary"]["mean_raw_brier"] == 0
    elif field in {"query_reviews", "return_reviews"}:
        expected = "source_challenge" if field == "query_reviews" else "failure"
        assert assessment["return_probability_reviews"][0]["observed_outcome"] == expected
    else:
        assert any(claim["claim_id"] == "preinserted-unreported-claim" for claim in run["claims"])

    with pytest.raises(ValueError, match="previously verified projection prefix"):
        verify(tmp_path, run["run_id"])
