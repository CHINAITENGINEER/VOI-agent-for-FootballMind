"""Isolation, missing outcomes and temporal limits of post-stop evaluation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_agent.audit import atomic_write, digest
from football_agent.inquiry_impact import terminal_state
from football_agent.inquiry_tools import OpenWebTools
from football_agent.stopping_evaluation import (
    acquire, build_plan, checked_results, read, review_input, sha, summarize, validate_review,
)


def stopped_run():
    target = {"hypothesis_id": "h-fixed", "question_id": "q-fixed", "hypothesis_type": "rotation_intent",
        "subject": "team-a", "scope": "match", "scope_key": "match-1", "statement": "Coach plans rotation due to qualification in match 1.",
        "required_evidence": ["coach_statement"]}
    problem = {"task_id": "task-1", "decision_id": "decision-1", "target": target,
        "semantics": "source_position_not_world_truth", "state_definitions": {"positive": "Source affirms.", "negative": "Source denies."},
        "loss": {"report_supported": {"positive": 0, "negative": 1}, "report_challenged": {"positive": 1, "negative": 0},
                 "withhold": {"positive": .2, "negative": .2}}, "loss_basis": "Declared reporting preference."}
    return {"run_id": "constructed-run", "state": "DEFERRED", "pending": None,
        "configuration": {"decision_problem": deepcopy(problem), "tools": {"allowed_hosts": ["club.example"]}},
        "task": {"task_id": "task-1", "seed_urls": ["https://club.example/news"]},
        "decision_problem": problem, "claims": [], "terminal_state": terminal_state(problem, []),
        "decisions": [{"kind": "defer", "at": "2026-09-27T04:00:00+00:00"}],
        "events": [{"at": "2026-09-27T04:00:01+00:00", "event_hash": "old-head"}],
        "documents": [{"document_id": "old", "url": "https://club.example/news", "kind": "page", "status": "OK",
            "observed_at": "2026-09-27T03:30:00+00:00", "available_at": "2026-09-27T03:30:01+00:00",
            "raw_sha256": "old-hash", "text": "General coach statement already visible.",
            "links": [{"url": "https://club.example/detail-1", "title": "Coach"}, {"url": "https://club.example/detail-2", "title": "More"}]}]}


def paths():
    return [{"path_id": "first", "url": "https://club.example/detail-1", "reason": "Inspect unread first detail."},
            {"path_id": "second", "url": "https://club.example/detail-2", "reason": "Inspect unread second detail."}]


def plan_for(run=None):
    return build_plan(run or stopped_run(), paths(), "2026-09-27T04:01:00+00:00")


def result(path_id, text, status="OK"):
    return {"path_id": path_id, "state": "ACQUIRED", "receipt": {"document": {
        "document_id": "new-"+path_id, "url": "https://club.example/detail-1", "kind": "page", "status": status,
        "observed_at": "2026-09-27T04:02:00+00:00", "available_at": "2026-09-27T04:02:01+00:00",
        "source_published_at": "2026-09-25T00:00:00+00:00", "publication_status": "source_reported",
        "raw_sha256": "new-hash-"+path_id, "text": text, "links": [], "text_truncated": False}}}


def annotation(plan, first_evidence=None, second_evidence=None):
    rows = []
    for identity, evidence in zip(("first", "second"), (first_evidence, second_evidence)):
        evidence = evidence or []
        relations = {e["relation"] for e in evidence}
        verdict = "conflicting" if len(relations) == 2 else "direct_support" if relations == {"supports"} else "direct_challenge" if relations else "no_direct_evidence"
        rows.append({"path_id": identity, "verdict": verdict, "rationale": "Bounded source-text annotation.", "evidence": evidence})
    return {"target_hash": digest(plan["decision_problem"]["target"]), "rows": rows}


def evidence(quote, relation="supports", context=None):
    anchors = [{"role": "statement", "quote": quote}]
    if context:
        anchors.append({"role": "match_context", "quote": context})
    return {"relation": relation, "explanation": "Source statement about the exact target.", "anchors": anchors}


def test_later_positive_signal_changes_projection_but_does_not_establish_old_miss():
    run = stopped_run(); before = deepcopy(run); plan = plan_for(run)
    quote = "Coach plans rotation due to qualification in match 1."
    records = {"first": result("first", quote), "second": result("second", "General marketing content.")}
    summary = summarize(plan, run, records, annotation(plan, [evidence(quote)]))
    row = summary["rows"][0]
    assert row["action_would_change_if_annotations_accepted"] is True
    assert row["isolated_terminal_projection"]["action"] == "report_supported"
    assert row["body_available_at_stop"] == "unknown" and row["missed_opportunity_at_stop"] is None
    assert row["evidence_anchor_overlap"][0]["novelty"] == "new_anchored_text"
    assert summary["stop_regret"] is None and not summary["optimal_stop_proven"]
    assert run == before


def test_failure_and_unattempted_remain_in_registered_denominator():
    run = stopped_run(); plan = plan_for(run)
    records = {"first": result("first", "", "FAILED")}
    summary = summarize(plan, run, records)
    assert summary["registered_paths"] == 2 and summary["attempted_paths"] == 1
    assert summary["transport_status_counts"] == {"FAILED": 1, "NO_RECEIPT": 1}
    assert summary["review_verdict_counts"] == {"NOT_REVIEWED": 2}
    assert all(r["action_would_change_if_annotations_accepted"] is None for r in summary["rows"])


def test_joint_conflict_is_separate_from_single_path_effects():
    run = stopped_run(); plan = plan_for(run)
    support, challenge = "Coach says rotation for qualification.", "Coach rules out qualification-related rotation."
    records = {"first": result("first", support), "second": result("second", challenge)}
    summary = summarize(plan, run, records, annotation(plan, [evidence(support)], [evidence(challenge, "challenges")]))
    assert [r["isolated_terminal_projection"]["action"] for r in summary["rows"]] == ["report_supported", "report_challenged"]
    assert summary["joint_terminal_projection"]["action"] == "withhold"


def test_old_quote_with_new_match_context_remains_mixed_not_zero_value():
    run = stopped_run(); plan = plan_for(run)
    old = run["documents"][0]["text"]; context = "These comments concern qualification before match 1."
    records = {"first": result("first", old+" "+context), "second": result("second", "Nothing applicable here.")}
    summary = summarize(plan, run, records, annotation(plan, [evidence(old, context=context)]))
    overlap = summary["rows"][0]["evidence_anchor_overlap"][0]
    assert overlap["anchors_previously_visible"] == [True, False]
    assert overlap["novelty"] == "mixed_old_new_anchors"


@pytest.mark.parametrize("mutation", ["target", "quote", "missing_row", "failure_negative", "context_only"])
def test_invalid_annotations_fail_closed(mutation):
    plan = plan_for(); quote = "Coach says rotation for qualification."
    records = {"first": result("first", quote), "second": result("second", "Other text.")}
    response = annotation(plan, [evidence(quote)])
    if mutation == "target": response["target_hash"] = "other-target"
    elif mutation == "quote": response["rows"][0]["evidence"][0]["anchors"][0]["quote"] = "Fabricated quote not acquired."
    elif mutation == "missing_row": response["rows"].pop()
    elif mutation == "failure_negative": records["second"]["receipt"]["document"]["status"] = "FAILED"
    else: response["rows"][0]["evidence"][0]["anchors"][0]["role"] = "match_context"
    with pytest.raises(ValueError): validate_review(plan, records, response)


def test_annotator_input_excludes_stop_reason_prior_actions_and_selection_reason():
    plan = plan_for(); public = review_input(plan, {"first": result("first", "Current text.")})
    assert set(public) == {"target", "target_hash", "paths"}
    assert all(set(p) == {"path_id", "document"} for p in public["paths"])
    assert public["paths"][1]["document"] is None


def registered_fixture(root):
    root.mkdir()
    run = stopped_run(); plan = plan_for(run)
    original = root.parent/"original-run.json"
    atomic_write(original, run); atomic_write(root/"stopped-run.json", run)
    atomic_write(root/"parent-verification.json", {"constructed": True})
    manifest = {"plan": plan, "parent_root": str(root.parent), "original_run_path": str(original),
        "parent_file_sha256": sha(original), "parent_verification_hash": digest({"constructed": True}), "sources": []}
    atomic_write(root/"manifest.json", manifest)
    return manifest


def test_real_transport_fixture_reserves_before_http_archives_once_and_preserves_parent(tmp_path):
    root = tmp_path/"evaluation"; manifest = registered_fixture(root); calls = []
    parent_hash = sha(manifest["original_run_path"])
    class Session:
        def get(self, url, **kwargs):
            intent = read(root/"first/intent.json")
            assert intent["prepared"]["endpoint"] == url
            assert read(root/"attempts.json")["attempts"][0]["state"] == "RESERVED_BEFORE_NETWORK"
            assert kwargs["allow_redirects"] is False
            calls.append(url)
            return SimpleNamespace(status_code=200, content=b'<html><article>Source text for evaluation.</article></html>', headers={}, close=lambda: None)
        def close(self): pass
    factory = lambda archive, plan: OpenWebTools(archive, plan["allowed_hosts"], session_factory=Session)
    outcome = acquire(root, "first", tools_factory=factory)
    assert outcome["state"] == "ACQUIRED" and len(calls) == 1
    assert checked_results(root, manifest)["first"] == outcome
    with pytest.raises(ValueError, match="retried"): acquire(root, "first", tools_factory=factory)
    assert sha(manifest["original_run_path"]) == parent_hash and len(calls) == 1


def test_interruption_reserves_path_without_inventing_no_http(tmp_path):
    root = tmp_path/"evaluation"; manifest = registered_fixture(root)
    class BrokenTool:
        def execute(self, *args, **kwargs): raise RuntimeError("Interrupted before receipt reconciliation")
    outcome = acquire(root, "first", tools_factory=lambda archive, plan: BrokenTool())
    assert outcome["state"] == "RECONCILIATION_REQUIRED" and outcome["receipt"] is None
    with pytest.raises(ValueError, match="retried"): acquire(root, "first", tools_factory=lambda a, p: BrokenTool())
    assert checked_results(root, manifest)["first"] == outcome
