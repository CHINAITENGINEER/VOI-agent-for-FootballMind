"""Constructed attacks on post-stop evidence, provenance and retained failures.

No real provider or network is used, including the fabricated provider archive
used to check that a valid reply cannot be hidden as a failed annotation.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_agent.audit import atomic_write, digest
from football_agent.inquiry_impact import terminal_state
from football_agent.inquiry_tools import OpenWebTools
from football_agent import stopping_evaluation as evaluation


STOP = datetime(2026, 9, 27, tzinfo=timezone.utc)
LATER = STOP+timedelta(hours=1)
STATEMENT = "The coach says Team A will rotate because its final place is fixed."
CONTEXT = "This statement concerns Team A versus Team B in fixture M."
NEW_CONTEXT = "The interview identifies fixture M and explicitly gives the fixed-place reason."
ARTICLE = "https://example.com/article?a=1&b=2"
REORDERED = "https://example.com/article?b=2&a=1"


def stopped_run():
    target = {"hypothesis_id": "h-fixed", "question_id": "q-fixed", "statement": STATEMENT,
        "hypothesis_type": "rotation_intent", "subject": "Team-A", "scope": "match",
        "scope_key": "fixture-M", "required_evidence": ["coach_statement"]}
    problem = {"task_id": "constructed-stop", "decision_id": "report-fixed", "target": target,
        "semantics": "source_position_not_world_truth", "state_definitions": {
            "positive": "The source supports the fixed statement.", "negative": "The source opposes it."},
        "loss": {"report_supported": {"positive": 0, "negative": 1},
                 "report_challenged": {"positive": 1, "negative": 0},
                 "withhold": {"positive": .2, "negative": .2}}, "loss_basis": "Constructed declared preferences."}
    doc = {"document_id": "old-list", "url": "https://example.com/list", "kind": "page", "status": "OK",
        "text": STATEMENT+" "+CONTEXT, "raw_sha256": digest("old-body"),
        "observed_at": (STOP-timedelta(minutes=1)).isoformat(),
        "available_at": (STOP-timedelta(seconds=30)).isoformat(),
        "links": [{"url": url, "title": "A visible route"} for url in
                  [ARTICLE, REORDERED, "https://example.com/failure", "https://example.com/pending"]]}
    return {"run_id": "constructed-original", "state": "DEFERRED", "pending": None,
        "decision_problem": problem, "configuration": {"decision_problem": deepcopy(problem),
            "tools": {"allowed_hosts": ["example.com"]}},
        "task": {"task_id": "constructed-stop", "seed_urls": [doc["url"]]},
        "claims": [], "terminal_state": terminal_state(problem, []), "documents": [doc],
        "decisions": [{"kind": "defer", "at": STOP.isoformat()}],
        "events": [{"at": (STOP+timedelta(seconds=1)).isoformat(), "event_hash": "f"*64}]}


def path(identity="article", url=ARTICLE):
    return {"path_id": identity, "url": url, "reason": "Finite constructed diagnostic selection."}


def register_fixture(tmp_path, monkeypatch, paths):
    run = stopped_run()
    parent = tmp_path/"original"
    original = parent/"runs"/run["run_id"]/"run.json"
    atomic_write(original, run)
    # The parent is intentionally minimal. The production registrar's existing
    # inquiry verifier is stubbed; all new evaluation checks remain real.
    monkeypatch.setattr("football_agent.inquiry_verify.verify", lambda *a, **k: {"status": "CONSTRUCTED_PARENT"})
    monkeypatch.setattr(evaluation, "now", lambda: LATER.isoformat())
    root = tmp_path/"evaluation"
    evaluation.register(root, parent, run["run_id"], paths)
    return root, run, original


def factory_with_recorded_body(root, calls, *, status=200, text=None):
    text = text or "Published 2026-09-01. "+STATEMENT+" "+CONTEXT+" "+NEW_CONTEXT

    def factory(archive, plan):
        class Session:
            def get(self, url, **kwargs):
                ledger = evaluation.read(root/"attempts.json")
                assert len(ledger["attempts"]) == len(calls)+1
                attempt = ledger["attempts"][-1]
                assert attempt["state"] == "RESERVED_BEFORE_NETWORK"
                intent = evaluation.read(root/attempt["path_id"]/"intent.json")
                assert digest(intent) == attempt["intent_hash"]
                assert ledger["manifest_hash"] == digest(evaluation.read(root/"manifest.json"))
                calls.append(url)
                return SimpleNamespace(status_code=status, content=text.encode(), headers={}, close=lambda: None)

            def close(self):
                pass

        return OpenWebTools(archive, plan["allowed_hosts"], session_factory=Session, clock=lambda: LATER)

    return factory


def annotation(plan, *, directional=True, context=CONTEXT):
    return {"target_hash": digest(plan["decision_problem"]["target"]), "rows": [{
        "path_id": p["path_id"], "verdict": "direct_support" if directional else "uncertain",
        "rationale": "Constructed annotation, not independently certified truth.",
        "evidence": [{"relation": "supports", "explanation": "Exact constructed statement and its context.",
            "anchors": [{"role": "statement", "quote": STATEMENT},
                        {"role": "match_context", "quote": context}]}] if directional else []}
        for p in plan["paths"]]}


def fabricate_annotation_archive(root, manifest, results, response, *, status):
    """Local fixture only: mimic archive shape without invoking any provider."""
    public = evaluation.review_input(manifest["plan"], results)
    atomic_write(root/"annotation-intent.json", {"manifest_hash": digest(manifest),
        "input_hash": digest(public), "started_at": LATER.isoformat()})
    folder = root/"provider"/"fabricated-provider"
    atomic_write(folder/"input.json", public)
    atomic_write(folder/"response.json", response)
    receipt = {"response_sha256": evaluation.sha(folder/"response.json"),
        "real_provider_call": True, "tool_events": []}
    atomic_write(folder/"receipt.json", receipt)
    atomic_write(folder/"process.json", {"returncode": 0, "event_types": ["turn.completed"],
        "unexpected_item_types": []})
    atomic_write(root/"annotation-reply.json", {"response": response, "receipt": receipt})
    atomic_write(root/"annotation-result.json", {"state": status, "finished_at": LATER.isoformat()})


@pytest.mark.parametrize("url", ["https://example.com/invented", "https://example.com/list"])
def test_allowed_domain_is_not_a_witness_and_an_old_page_is_not_unread(url):
    with pytest.raises(ValueError):
        evaluation.build_plan(stopped_run(), [path(url=url)], LATER.isoformat())


@pytest.mark.parametrize("already_read", [False, True])
def test_query_order_cannot_turn_the_same_physical_action_into_an_unread_path(already_read):
    run = stopped_run()
    paths = [path(), path("reordered", REORDERED)]
    if already_read:
        run["documents"].append({**deepcopy(run["documents"][0]), "document_id": "old-article", "url": ARTICLE, "links": []})
        paths = [path("reordered", REORDERED)]
    with pytest.raises(ValueError):
        evaluation.build_plan(run, paths, LATER.isoformat())


@pytest.mark.parametrize("context,novelty", [(CONTEXT, "all_anchors_previously_visible"),
                                           (NEW_CONTEXT, "mixed_old_new_anchors")])
def test_early_page_date_and_a_new_url_do_not_establish_pre_stop_acquisition_gain(context, novelty):
    run = stopped_run()
    original = deepcopy(run)
    plan = evaluation.build_plan(run, [path()], LATER.isoformat())
    doc = {"document_id": "later-article", "url": ARTICLE, "kind": "page", "status": "OK",
        "text": "Published 2026-09-01. "+STATEMENT+" "+CONTEXT+" "+NEW_CONTEXT,
        "raw_sha256": digest("later-body"), "observed_at": LATER.isoformat()}
    results = {"article": {"state": "ACQUIRED", "receipt": {"document": doc}}}
    output = evaluation.summarize(plan, run, results, annotation(plan, context=context))
    row = output["rows"][0]
    assert row["isolated_terminal_projection"]["action"] == "report_supported"
    assert row["evidence_anchor_overlap"][0]["novelty"] == novelty
    assert row["body_available_at_stop"] == "unknown" and row["missed_opportunity_at_stop"] is None
    assert row["actual_utility"] is None and output["stop_regret"] is None
    assert output["temporally_certified_paths"] == 0
    assert run == original


def test_fixed_statement_and_all_registered_rows_are_required():
    plan = evaluation.build_plan(stopped_run(), [path(), path("pending", "https://example.com/pending")], LATER.isoformat())
    response = annotation(plan, directional=False)
    opposite = {**plan["decision_problem"]["target"], "statement": "The coach rejects fixed-place rotation."}
    response["target_hash"] = digest(opposite)
    with pytest.raises(ValueError, match="fixed target"):
        evaluation.validate_review(plan, {}, response)
    response["target_hash"] = digest(plan["decision_problem"]["target"])
    response["rows"].pop()
    with pytest.raises(ValueError, match="All registered paths"):
        evaluation.validate_review(plan, {}, response)


def test_failed_interrupted_and_unattempted_paths_stay_in_denominator_without_mutating_parent(tmp_path, monkeypatch):
    paths = [path(), path("failure", "https://example.com/failure"), path("pending", "https://example.com/pending")]
    root, _, original = register_fixture(tmp_path, monkeypatch, paths)
    old_hash = evaluation.sha(original)
    calls = []
    evaluation.acquire(root, "failure", tools_factory=factory_with_recorded_body(root, calls, status=503))

    class Interrupted:
        def execute(self, *args, **kwargs):
            raise RuntimeError("Constructed interruption after reservation")

    evaluation.acquire(root, "pending", tools_factory=lambda *a: Interrupted())
    for identity in ("failure", "pending"):
        with pytest.raises(ValueError, match="must not be retried"):
            evaluation.acquire(root, identity, tools_factory=lambda *a: pytest.fail("Duplicate reached factory"))
    with pytest.raises(ValueError, match="preregistered"):
        evaluation.acquire(root, "invented", tools_factory=lambda *a: pytest.fail("Unknown path reached factory"))
    with pytest.raises(FileNotFoundError):
        evaluation.acquire(tmp_path/"unregistered", "article", tools_factory=lambda *a: pytest.fail("Unregistered factory"))
    output = evaluation.verify_evaluation(root)
    assert output["registered_paths"] == 3 and output["attempted_paths"] == 2
    assert output["transport_status_counts"] == {"NO_RECEIPT": 2, "FAILED": 1}
    assert output["acquisition_state_counts"] == {"NOT_ATTEMPTED": 1, "ACQUIRED": 1, "RECONCILIATION_REQUIRED": 1}
    assert output["review_verdict_counts"] == {"NOT_REVIEWED": 3}
    assert output["verified_physical_evaluation_http"] == 1 and len(calls) == 1
    assert evaluation.sha(original) == old_hash


@pytest.mark.parametrize("status", ["ANNOTATION_FAILED_NO_RETRY", "MADE_UP_STATUS"])
def test_a_valid_directional_reply_cannot_be_hidden_by_a_failed_or_unknown_status(tmp_path, monkeypatch, status):
    root, _, _ = register_fixture(tmp_path, monkeypatch, [path()])
    evaluation.acquire(root, "article", tools_factory=factory_with_recorded_body(root, []))
    manifest = evaluation.read(root/"manifest.json")
    results = evaluation.checked_results(root, manifest)
    fabricate_annotation_archive(root, manifest, results, annotation(manifest["plan"]), status=status)
    with pytest.raises(ValueError):
        evaluation.verify_evaluation(root)


def test_an_interrupted_reserved_path_can_be_reviewed_as_uncertain_without_losing_its_denominator(tmp_path, monkeypatch):
    root, _, _ = register_fixture(tmp_path, monkeypatch, [path()])

    def constructor_interrupted(*args):
        raise RuntimeError("Constructed tool initialization interruption")

    with pytest.raises(RuntimeError):
        evaluation.acquire(root, "article", tools_factory=constructor_interrupted)
    manifest = evaluation.read(root/"manifest.json")
    results = evaluation.checked_results(root, manifest)
    assert results["article"]["state"] == "RECONCILIATION_REQUIRED"
    fabricate_annotation_archive(root, manifest, results, annotation(manifest["plan"], directional=False),
        status="VALIDATED_SOURCE_ANNOTATION_NOT_TRUTH")
    output = evaluation.verify_evaluation(root)
    assert output["registered_paths"] == output["attempted_paths"] == 1
    assert output["review_verdict_counts"] == {"uncertain": 1}
    assert output["temporally_certified_paths"] == 0 and output["stop_regret"] is None


def test_unbound_provider_artifacts_cannot_be_reported_as_zero_annotation_attempts(tmp_path, monkeypatch):
    root, _, _ = register_fixture(tmp_path, monkeypatch, [path()])
    evaluation.acquire(root, "article", tools_factory=factory_with_recorded_body(root, []))
    manifest = evaluation.read(root/"manifest.json")
    results = evaluation.checked_results(root, manifest)
    fabricate_annotation_archive(root, manifest, results, annotation(manifest["plan"]),
        status="VALIDATED_SOURCE_ANNOTATION_NOT_TRUTH")
    (root/"annotation-intent.json").unlink()
    with pytest.raises(ValueError):
        evaluation.verify_evaluation(root)
