"""Acquisition data must preserve timing, dependence, failures and missing labels."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from football_agent.decision_risk_experiment import make_parent, ConstructedTools, proposal, task, build_selector
from football_agent.inquiry import InquiryWorkflow
from football_agent.inquiry_evidence_state import EvidenceStateProtocol
from football_agent.inquiry_exploration import UniformInquiryExploration
from football_agent.inquiry_selection import LearningExplorationSelector
from football_agent.audit import RunStore
from football_agent.return_dataset import VERSION, build_dataset, export_run


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    def __call__(self):
        self.now += timedelta(milliseconds=10)
        return self.now


def reference(root, run, **kwargs):
    return {"root": str(root), "run_id": run["run_id"], "head_hash": run["events"][-1]["event_hash"],
        "cluster_id": "one-constructed-fixture", "origin": "constructed",
        "cluster_basis": "Same constructed fixture; threads and requests are not independent tasks.", **kwargs}


@pytest.fixture
def archived(tmp_path):
    clock = Clock()
    root = tmp_path/"parent"
    run = make_parent(root, clock=clock)
    return root, run, clock


def test_one_row_per_actual_request_not_document_or_model_turn(archived):
    root, run, clock = archived
    data = build_dataset([reference(root, run)], as_of=clock())
    assert len(data["rows"]) == 2
    assert data["summary"]["constructed_rows"] == 2
    assert data["summary"]["operational_observed_requests"] == 0
    assert data["summary"]["calibrated_return_model"] is None
    assert data["schema_version"] == VERSION == "inquiry-return-dataset-v2"
    assert data["summary"]["selection_propensities_available"] is False
    for row in data["rows"]:
        pre, ret = row["pre_query"], row["observed_return"]
        assert pre["selection_probability"] is None
        assert pre["selection_probability_scope"] is None and pre["exploration_record"] is None
        assert all(c["sampling_eligible"] is None and c["conditional_selection_probability"] is None
                   for c in pre["candidate_set"])
        assert ret["document"]["document_id"] not in {d["document_id"] for d in pre["documents"]}
        assert ret["source_truth_label"] is None and ret["realized_decision_utility"] is None
        assert all(c["observed_return_label"] is None for c in pre["candidate_set"])
    assert data["rows"][1]["pre_query"]["documents"] == [data["rows"][0]["observed_return"]["document"]]


def test_cutoff_censors_return_and_later_review_without_backfilling(archived):
    root, run, clock = archived
    intent = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_intent")
    early = build_dataset([reference(root, run)], as_of=intent["at"])
    assert len(early["rows"]) == 1 and early["rows"][0]["observed_return"] is None
    assert early["rows"][0]["status"] == "CENSORED_AT_EXPORT_CUTOFF"
    received = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_returned")
    at_return = build_dataset([reference(root, run)], as_of=received["at"])
    assert at_return["rows"][0]["observed_return"]["query_review"] is None
    complete = build_dataset([reference(root, run)], as_of=clock())
    assert complete["rows"][0]["observed_return"]["query_review"] is not None
    assert complete["rows"][0]["pre_query"] == early["rows"][0]["pre_query"]


def test_repeated_input_reference_deduplicates_queries_but_rejects_conflicting_cluster(archived):
    root, run, clock = archived
    ref = reference(root, run)
    data = build_dataset([ref, ref], as_of=clock())
    assert len(data["rows"]) == 2
    with pytest.raises(ValueError, match="conflicting evidence or cluster"):
        build_dataset([ref, {**ref, "cluster_id": "pretend-independent-run"}], as_of=clock())


def test_future_completion_does_not_change_censored_status(archived):
    root, run, clock = archived
    intent = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_intent")
    full = build_dataset([reference(root, run)], as_of=intent["at"])
    prefix = deepcopy(intent["snapshot"])
    prefix["events"] = deepcopy(run["events"][:intent["seq"]])
    RunStore(root/"runs").save(prefix)
    pending = build_dataset([reference(root, prefix)], as_of=intent["at"])
    assert pending["rows"][0]["status"] == full["rows"][0]["status"] == "CENSORED_AT_EXPORT_CUTOFF"
    assert pending["rows"][0]["pre_query"] == full["rows"][0]["pre_query"]
    assert pending["rows"][0]["observed_return"] == full["rows"][0]["observed_return"] is None


def test_identical_content_in_distinct_requests_remains_distinct(archived, tmp_path):
    root, run, clock = archived
    second_root = tmp_path/"another-run"
    second = make_parent(second_root, clock=clock)
    rows = build_dataset([reference(root, run), reference(second_root, second)], as_of=clock())["rows"]
    assert len(rows) == 4 and len({r["row_id"] for r in rows}) == 4
    assert len({r["observed_return"]["raw_sha256"] for r in rows}) == 2
    assert len({r["cluster_id"] for r in rows}) == 1


def test_constructed_origin_and_pinned_heads_are_enforced(archived):
    root, run, clock = archived
    with pytest.raises(ValueError, match="Constructed transport"):
        export_run(reference(root, run, origin="operational_network"), as_of=clock())
    with pytest.raises(ValueError, match="pinned export manifest"):
        export_run(reference(root, run, head_hash="0"*64), as_of=clock())
    with pytest.raises(ValueError, match="documented dependence"):
        export_run(reference(root, run, cluster_basis=""), as_of=clock())


def test_inherited_docs_are_only_context_and_failed_call_is_retained(archived, tmp_path):
    root, parent, clock = archived
    class Provider:
        def configuration(self): return {"provider": "scripted_inheritance_export", "configured": True}
        def decide(self, public):
            answer = proposal(["verify_k", "fill_j"]) if not public["action_history"] else proposal([], kind="defer")
            return {"response": answer, "receipt": {"real_provider_call": False}}
    child_root = tmp_path/"child"
    tools = ConstructedTools(child_root, {"verify_k": "failure"}, clock=clock)
    workflow = InquiryWorkflow(child_root/"runs", Provider(), tools, clock=clock, protocol=EvidenceStateProtocol())
    run = workflow.start(task(clock(), build_selector()), max_requests=2, max_llm_calls=3)
    workflow.inherit_evidence(run["run_id"], parent_root=root, parent_run_id=parent["run_id"],
        parent_head_hash=parent["events"][-1]["event_hash"], document_ids=[d["document_id"] for d in parent["documents"]])
    run = workflow.run(run["run_id"])
    data = build_dataset([reference(child_root, run)], as_of=clock())
    before_import = build_dataset([reference(child_root, run)], as_of=run["created_at"])
    assert before_import["sources"][0]["inherited_documents_not_counted"] == 0
    assert len(data["rows"]) == 1 and data["sources"][0]["inherited_documents_not_counted"] == 2
    row = data["rows"][0]
    assert len(row["pre_query"]["documents"]) == 2
    assert row["observed_return"]["document_status"] == "FAILED"
    assert row["observed_return"]["transport_error"] == "ConnectionError"
    unchosen = next(c for c in row["pre_query"]["candidate_set"] if not c["selected"])
    assert unchosen["candidate"]["candidate_id"] == "fill_j" and unchosen["observed_return_label"] is None


def learning_run(root, *, candidates=("verify_k", "fill_j"), duplicate_candidate=False):
    clock = Clock()
    class Provider:
        def configuration(self): return {"provider": "scripted_return_sampling", "configured": True}
        def decide(self, public):
            answer = proposal(list(candidates)) if not public["action_history"] else proposal([], kind="defer")
            if duplicate_candidate and answer["candidates"]:
                duplicate = deepcopy(answer["candidates"][0])
                duplicate["candidate_id"] = "verify_copy"
                answer["candidates"].append(duplicate)
            return {"response": answer, "receipt": {"real_provider_call": False}}
    tools = ConstructedTools(root, {"verify_k": "failure", "fill_j": "failure"}, clock=clock)
    explorer = UniformInquiryExploration(135791357913579)
    selector = LearningExplorationSelector(explorer, max_exploration_requests=2)
    workflow = InquiryWorkflow(root/"runs", Provider(), tools, clock=clock,
                               protocol=EvidenceStateProtocol(), selector=selector)
    run = workflow.start(task(clock(), build_selector()), max_requests=2, max_llm_calls=3)
    return workflow.run(run["run_id"]), clock


def test_sampled_query_exports_conditional_probability_without_labeling_unchosen_actions(tmp_path):
    root = tmp_path/"learning"
    run, clock = learning_run(root)
    data = build_dataset([reference(root, run)], as_of=clock())
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    pre, ret = row["pre_query"], row["observed_return"]
    assert pre["selection_probability"] == 0.5
    assert pre["selection_probability_scope"] == "conditional_on_generated_eligible_candidates_only"
    assert pre["selection_probability_status"] == "logged_replay_verified_conditional_sampling_design"
    assert pre["exploration_record"]["chosen_id"] == pre["selected_candidate"]["candidate_id"]
    assert pre["candidate_generation_probability"] is None and pre["counterfactual_utility"] is None
    assert "seed" not in pre["exploration_record"]
    assert "135791357913579" not in str(data)
    assert all(c["sampling_eligible"] and c["conditional_selection_probability"] == 0.5
               and c["observed_return_label"] is None for c in pre["candidate_set"])
    assert sum(c["selected"] for c in pre["candidate_set"]) == 1
    assert ret["document_status"] == "FAILED"
    assert ret["realized_decision_utility"] is None
    assert data["summary"]["selection_probability_coverage"]["logged_rows"] == 1
    assert data["summary"]["global_action_propensities_available"] is False
    assert data["summary"]["candidate_generation_probabilities_available"] is False


def test_sampling_probability_is_available_at_intent_but_current_return_stays_censored(tmp_path):
    root = tmp_path/"censored-learning"
    run, clock = learning_run(root)
    intent = next(e for e in run["events"] if e["type"] == "inquiry_acquisition_intent")
    early = build_dataset([reference(root, run)], as_of=intent["at"])
    later = build_dataset([reference(root, run)], as_of=clock())
    assert early["rows"][0]["observed_return"] is None
    assert early["rows"][0]["pre_query"]["selection_probability"] == 0.5
    assert early["rows"][0]["pre_query"] == later["rows"][0]["pre_query"]
    assert early["summary"]["unobserved_rows"] == 1
    prefix = deepcopy(intent["snapshot"])
    prefix["events"] = deepcopy(run["events"][:intent["seq"]])
    RunStore(root/"runs").save(prefix)
    pending = build_dataset([reference(root, prefix)], as_of=intent["at"])
    assert pending["rows"][0]["pre_query"] == early["rows"][0]["pre_query"]
    assert pending["rows"][0]["status"] == early["rows"][0]["status"]


def test_explicitly_excluded_duplicate_has_zero_sampling_probability_and_no_return_label(tmp_path):
    root = tmp_path/"excluded-duplicate"
    run, clock = learning_run(root, duplicate_candidate=True)
    row = build_dataset([reference(root, run)], as_of=clock())["rows"][0]
    candidates = row["pre_query"]["candidate_set"]
    assert len(candidates) == 3 and sum(c["sampling_eligible"] for c in candidates) == 2
    excluded = next(c for c in candidates if not c["sampling_eligible"])
    assert excluded["conditional_selection_probability"] == 0
    assert excluded["selected"] is False and excluded["observed_return_label"] is None
    assert all(c["observed_return_label"] is None for c in candidates)
    assert row["pre_query"]["selection_probability"] == 0.5


def test_mixed_old_and_sampled_runs_preserve_missing_propensities_and_singleton_scope(archived, tmp_path):
    root, original, clock = archived
    sampled_root = tmp_path/"singleton"
    run, later_clock = learning_run(sampled_root, candidates=("verify_k",))
    refs = [reference(root, original), reference(sampled_root, run)]
    cutoff = max(clock(), later_clock()) + timedelta(seconds=1)
    data = build_dataset(refs + refs, as_of=cutoff)
    assert len(data["rows"]) == 3
    coverage = data["summary"]["selection_probability_coverage"]
    assert coverage["logged_rows"] == 1 and coverage["unlogged_rows"] == 2
    assert coverage["operational_logged_rows"] == 0
    old_rows = [r for r in data["rows"] if r["run_id"] == original["run_id"]]
    assert all(r["pre_query"]["selection_probability"] is None for r in old_rows)
    selected = next(r for r in data["rows"] if r["run_id"] == run["run_id"])
    assert selected["pre_query"]["selection_probability"] == 1
    assert len(selected["pre_query"]["exploration_record"]["eligible_ids"]) == 1
    assert selected["pre_query"]["candidate_generation_probability"] is None
    assert selected["counterfactual_response_labels_available"] is False
