"""Independent fixed-endpoint and acquired-versus-extracted provenance checks.

All provider replies and transport returns are constructed. No external calls.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from football_agent.audit import digest
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_continuation import ContinuationPolicy
from football_agent.inquiry_impact import DecisionImpactProtocol
from football_agent.inquiry_returns import OUTCOMES
from football_agent.inquiry_rule_scope import RuleScopeProtocol
from football_agent.inquiry_tools import OpenWebTools
from football_agent.inquiry_verify import verify
from football_spider.storage.raw_archive import RawArchive


NOW = datetime(2026, 9, 27, 4, tzinfo=timezone.utc)
POSITIVE = "Coach says Team A intends to rotate the squad in fixture M."
NEGATIVE = "Coach says Team A does not intend to rotate the squad in fixture M."
TARGET = {"hypothesis_id": "h-fixed", "question_id": "q-fixed", "statement": POSITIVE,
          "hypothesis_type": "rotation_intent", "subject": "Team-A", "scope": "match",
          "scope_key": "fixture-M", "required_evidence": ["coach_statement"]}
TASK = {"task_id": "impact-adversarial", "objective": "Inspect the stated rotation position.",
        "deadline": (NOW+timedelta(hours=1)).isoformat(), "initial_facts": [],
        "seed_urls": ["https://example.com/coach", "https://example.com/failure"]}


def problem():
    return {"task_id": TASK["task_id"], "decision_id": "report-intent", "target": deepcopy(TARGET),
        "semantics": "source_position_not_world_truth",
        "state_definitions": {"positive": "The source supports the fixed statement.",
                              "negative": "The source challenges the fixed statement."},
        "loss": {"report_supported": {"positive": 0, "negative": 1},
                 "report_challenged": {"positive": 1, "negative": 0},
                 "withhold": {"positive": 0.2, "negative": 0.2}},
        "loss_basis": "Constructed declared reporting preferences, not measured utility."}


def claim(document_id, *, hypothesis_id="h-fixed", negative=False):
    return {"claim_id": "quoted-negative" if negative else "quoted-positive", "question_id": "q-fixed",
        "text": NEGATIVE if negative else POSITIVE, "status": "source_claim",
        "citations": [{"document_id": document_id, "quote": NEGATIVE if negative else POSITIVE}],
        "evidence": {"kind": "coach_statement", "subject": "Team-A", "scope": "match", "scope_key": "fixture-M"},
        "hypothesis_links": [{"hypothesis_id": hypothesis_id, "relation": "supports"}]}


def reply(kind="query", *, path="coach", updates=()):
    candidate = {"candidate_id": "read-"+path, "question_id": "q-fixed", "tool": "read_url",
        "url": "https://example.com/"+path, "query": None,
        "rationale": "Inspect a source for the fixed reporting target.",
        "answer_could_change": "A direct quoted statement could change the reported position.",
        "inquiry_intent": {"hypothesis_ids": ["h-fixed"], "purpose": "resolve",
            "expected_evidence": ["coach_statement"], "subject": "Team-A", "scope": "match", "scope_key": "fixture-M",
            "supports_change": "Retain the source position.", "challenges_change": "Preserve the counterstatement.",
            "no_answer_change": "Keep existing claims and unknowns."},
        "return_model": {"branches": [{"outcome": outcome,
            "missing_after": [{"hypothesis_id": "h-fixed", "types": ["coach_statement"]}],
            "next_action": {"kind": "reassess", "target_hypothesis_ids": [], "evidence_types": [],
                            "reason": "Inspect the actual return."}} for outcome in OUTCOMES]},
        "outcome_forecast": {"status": "abstain", "probabilities": None, "basis_document_ids": [],
                             "reason": "No fitted probability in this constructed check."},
        "terminal_effect": {"route": "direct_evidence", "could_change_to": ["report_supported", "report_challenged", "withhold"],
            "condition": "An explicit applicable statement addresses the fixed proposition.",
            "further_information_needed": "Extract the quote while preserving conflicts and source uncertainty."}}
    candidates = [candidate] if kind == "query" else []
    return {"questions": [{"question_id": "q-fixed", "question": POSITIVE,
        "decision_relevance": "The fixed reporting problem.", "answer_could_change": "The source position.",
        "based_on_documents": []}], "hypotheses": [deepcopy(TARGET)], "claim_updates": list(updates),
        "context_bindings": [], "rule_scope_bindings": [], "candidates": candidates,
        "choice_review": {"alternatives": [{"candidate_id": c["candidate_id"],
            "information_advantage": "A concrete route to source statements.", "source_limitations": "Meaning remains unverified.",
            "evidence_document_ids": [], "not_selected_reason": None} for c in candidates],
            "tradeoff": "Choose the proposed bounded route.", "missing_routes": "Other useful routes remain unknown."},
        "decision": {"kind": kind, "candidate_id": candidate["candidate_id"] if candidates else None,
                     "reason": "Constructed decision for the provenance check.", "revisit_at": None}}


def setup(root, responses, *, legacy=False):
    calls, inputs = [], []

    class Session:
        def get(self, url, **kwargs):
            calls.append(url)
            return SimpleNamespace(status_code=503 if url.endswith("failure") else 200,
                content=(POSITIVE+" "+NEGATIVE).encode(), headers={}, close=lambda: None)

        def close(self):
            pass

    class Tools(OpenWebTools):
        def configuration(self):
            return {**super().configuration(), "execution_mode": "constructed_transport_no_network"}

    class Provider:
        def configuration(self):
            return {"provider": "constructed-impact-adversarial"}

        def decide(self, public):
            inputs.append(deepcopy(public))
            value = responses[len(inputs)-1]
            response = value(public) if callable(value) else deepcopy(value)
            if legacy:
                for candidate in response["candidates"]:
                    candidate.pop("terminal_effect")
            return {"response": response, "receipt": {"real_provider_call": False}}

    tools = Tools(RawArchive(root/"evidence"), ["example.com"], session_factory=Session, clock=lambda: NOW)
    workflow = InquiryWorkflow(root/"runs", Provider(), tools, clock=lambda: NOW,
        protocol=RuleScopeProtocol() if legacy else DecisionImpactProtocol(problem()),
        continuation=ContinuationPolicy(), source_catalog=[])
    return workflow, calls, inputs


def test_reversed_statement_same_type_and_subject_cannot_replace_fixed_endpoint(tmp_path):
    workflow, _, _ = setup(tmp_path, [])
    public = workflow.public(workflow.start(TASK))
    response = reply("defer")
    response["hypotheses"][0]["statement"] = NEGATIVE
    with pytest.raises(ValueError, match="exact fixed terminal hypothesis"):
        workflow.protocol.validate(response, public)


@pytest.mark.parametrize("kind,required", [("qualification", ["competition_rules", "standings"]),
                                           ("actual_rotation", ["official_lineup", "lineup_comparison"]),
                                           ("rotation_intent", ["coach_statement", "standings"])])
def test_compound_targets_require_their_own_reporting_semantics(kind, required):
    config = problem()
    config["target"].update(hypothesis_type=kind, required_evidence=required)
    with pytest.raises(ValueError):
        DecisionImpactProtocol(config)


def test_opposite_hypothesis_support_is_not_fixed_target_support(tmp_path):
    def extract(public):
        opposite = {**deepcopy(TARGET), "hypothesis_id": "h-opposite", "statement": NEGATIVE}
        response = reply("defer", updates=[claim(public["documents"][0]["document_id"],
                                                  hypothesis_id="h-opposite", negative=True)])
        response["hypotheses"].append(opposite)
        return response

    workflow, calls, _ = setup(tmp_path, [reply(), extract])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    assert len(calls) == 1
    assert run["terminal_state"]["action"] == "withhold"
    assert run["terminal_state"]["support_claim_ids"] == []
    other = next(s for s in run["hypothesis_states"] if s["hypothesis_id"] == "h-opposite")
    assert other["status"] == "complete_source_support"
    assert run["return_reviews"][0]["outcome"] != "source_support"
    assert verify(tmp_path, run["run_id"])["decisions_verified"] == 2


@pytest.mark.parametrize("extract_before_query", [True, False])
def test_old_document_extraction_is_not_attributed_to_a_new_failed_request(tmp_path, extract_before_query):
    parent, parent_calls, _ = setup(tmp_path/"parent", [reply(), reply("defer")], legacy=True)
    prior = parent.run(parent.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    old = prior["documents"][0]

    def first(public):
        assert public["claims"] == [] and public["decision_impact_reviews"] == []
        updates = [claim(old["document_id"])] if extract_before_query else []
        return reply(path="failure", updates=updates)

    def second(public):
        updates = [] if extract_before_query else [claim(old["document_id"])]
        return reply("defer", updates=updates)

    child, calls, _ = setup(tmp_path/"child", [first, second])
    run = child.start(TASK, max_requests=1, max_llm_calls=2)
    child.inherit_evidence(run["run_id"], parent_root=tmp_path/"parent", parent_run_id=prior["run_id"],
        parent_head_hash=prior["events"][-1]["event_hash"], document_ids=[old["document_id"]])
    final = child.run(run["run_id"])
    assert len(parent_calls) == len(calls) == 1 and calls[0].endswith("/failure")
    assert final["documents"][0] == old
    assert final["receipts"][0]["status"] == "FAILED"
    review = final["decision_impact_reviews"][0]
    before = "report_supported" if extract_before_query else "withhold"
    assert review["decision_before"]["action"] == before
    assert review["decision_after"]["action"] == "report_supported"
    assert review["sole_document_target_claim_ids"] == []
    assert review["other_new_target_claim_ids"] == ([] if extract_before_query else ["quoted-positive"])
    assert review["attribution"] == ("no_action_change" if extract_before_query else "joint_or_other_evidence_change")
    assert review["actual_decision_utility"] is None and review["causal_risk_reduction"] is None
    assert final["return_reviews"][0]["outcome"] == "failure"
    assert verify(tmp_path/"child", final["run_id"])["inherited_documents_verified"] == 1


@pytest.mark.parametrize("field", ["terminal_state", "decision_ceiling"])
def test_rehashed_terminal_projection_is_rejected_outside_valid_decision(tmp_path, field):
    workflow, _, _ = setup(tmp_path, [reply(), reply("defer")])
    run = workflow.run(workflow.start(TASK, max_requests=1, max_llm_calls=2)["run_id"])
    final_event = run["events"][-1]
    if field == "terminal_state":
        final_event["snapshot"][field]["action"] = "report_supported"
    else:
        final_event["snapshot"][field]["all_action_improvement_ceiling"] = 0
    final_event["event_hash"] = digest({k: v for k, v in final_event.items() if k != "event_hash"})
    run.update(deepcopy(final_event["snapshot"]))
    workflow.store.save(run)
    workflow.store.verify(run["run_id"])
    with pytest.raises(ValueError, match="projection prefix"):
        verify(tmp_path, run["run_id"])
