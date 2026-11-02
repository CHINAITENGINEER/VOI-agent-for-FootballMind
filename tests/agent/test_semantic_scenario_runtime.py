"""Runtime integration with controlled semantics/transport and the real solver.

The probabilities below are intentionally constructed test fixtures. They are
never exported as learned models, real evidence, or competition results.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from football_agent.semantic_scenario_model import SemanticScenarioModel
from football_agent.semantic_scenario_protocol import (
    CandidateDiscoveryProtocol, ProposalProtocol, ObservationProtocol,
    ActionExtensionProtocol,
)
from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime, replay_math


NOW = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)
WHEN = NOW.isoformat()
SEED = "https://example.org/en"
WEAK = "https://example.org/en/news/general-update"
STRONG = "https://example.org/en/news/match-specific-update"
NEW = "https://example.org/en/news/newly-discovered-update"
FOUND = "https://example.org/en/news/search-found-article"
BASELINE = [.475, .275, .25]
BODY = "The manager confirms the striker is available for the specified fixture."
TITLE_ONLY = "TITLE CLAIM: everyone will start in this fixture."
CASE = {
    "match": {"match_id": "case:test", "home_team_id": "home", "away_team_id": "away",
              "kickoff": "2026-10-10T12:00:00Z", "prediction_cutoff": "2026-10-09T12:00:00Z"},
    "source_profile": {"teams": [
        {"side": "home", "canonical_name": "Home FC"},
        {"side": "away", "canonical_name": "Away FC"}]},
    "structured_baseline": {"probabilities": BASELINE},
    "baseline_scope": "constructed_test_only",
}


def link(url):
    return {"url": url, "title": "Observed article " + url.rsplit("/", 1)[-1]}


def document(identity, url, *, kind="article", text=BODY, links=(), status="OK", role="paragraph"):
    return {"document_id": identity, "url": url, "title": TITLE_ONLY,
            "kind": kind, "status": status, "text": text,
            "observed_at": WHEN, "available_at": WHEN,
            "source_published_at": "2026-10-07T12:00:00Z", "links": list(links),
            "structured_news": {"segments": [
                {"content_role": role, "text": text, "source_published_at": "2026-10-07T12:00:00Z"},
                {"content_role": "title", "text": TITLE_ONLY}]}}


class FakeTransport:
    def __init__(self, pages=None, *, source_links=(WEAK, STRONG), search_links=(FOUND,)):
        self.pages = {SEED: document("index", SEED, kind="index", links=[link(url) for url in source_links]),
                      WEAK: document("weak", WEAK), STRONG: document("strong", STRONG),
                      FOUND: document("found", FOUND), NEW: document("new", NEW)}
        self.pages.update(pages or {})
        self.search_links = search_links
        self.prepared, self.executions = [], []

    def configuration(self):
        return {"provider": "controlled_test_transport", "actual_network": False}

    def prepare(self, candidate, state):
        if candidate["tool"] == "read_url":
            observed = {url for url in state["task"]["seed_urls"]}
            observed.update(item["url"] for doc in state["documents"] for item in doc.get("links", []))
            assert candidate["url"] in observed, "Unobserved URL bypassed execution capability"
        else:
            assert candidate["tool"] == "search_web"
        self.prepared.append(deepcopy(candidate))
        return deepcopy(candidate)

    def execute(self, prepared, *, run_id, request_id):
        self.executions.append(deepcopy(prepared))
        if prepared["tool"] == "search_web":
            doc = document("search", "https://example.org/search", kind="search_results",
                text="Search snippet claims the striker starts, but this is not evidence.",
                links=[link(url) for url in self.search_links])
        else:
            doc = deepcopy(self.pages[prepared["url"]])
        return {"http_attempts": 1, "document": doc, "latency_seconds": .01,
                "run_id": run_id, "request_id": request_id, "test_only": True}


class FakeProviderFactory:
    def __init__(self, *, queries=(), answer="supports", applicability="match_specific",
                 rediscovery_queries=(), invalid_quote=False, fail_protocol=None,
                 invalid_numeric=False):
        self.queries, self.answer, self.applicability = list(queries), answer, applicability
        self.rediscovery_queries = list(rediscovery_queries)
        self.invalid_quote, self.fail_protocol = invalid_quote, fail_protocol
        self.invalid_numeric = invalid_numeric
        self.inputs = []

    @staticmethod
    def action(candidate):
        strong = candidate["tool"] == "search_then_read" or candidate.get("url") == STRONG
        distributions = ([.8, .1, .1], [.1, .8, .1]) if strong else ([.45, .45, .1], [.45, .45, .1])
        return {"action_id": candidate["action_id"], "question": "本场主力可用吗？",
                "observation_group": "group:" + candidate["action_id"],
                "likelihoods": [
                    {"hypothesis_id": "ready", "probabilities": distributions[0]},
                    {"hypothesis_id": "limited", "probabilities": distributions[1]}]}

    def __call__(self, workspace, protocol):
        factory = self
        class Provider:
            def configuration(self):
                return {"provider": "controlled_test_semantics", "protocol": protocol.version}

            def decide(self, public):
                factory.inputs.append({"protocol": protocol.version, "public": deepcopy(public)})
                if factory.fail_protocol == protocol.version:
                    raise RuntimeError("controlled provider unavailable")
                if protocol.version == CandidateDiscoveryProtocol.version:
                    discovery_count = sum(item["protocol"] == CandidateDiscoveryProtocol.version
                                          for item in factory.inputs)
                    selected_queries = (factory.queries if discovery_count == 1
                                        else factory.rediscovery_queries)
                    response = {"schema": protocol.version, "queries": [
                        {"query": query, "information_need": "本场尚未确认的信息。"} for query in selected_queries]}
                elif protocol.version == ProposalProtocol.version:
                    response = {"schema": protocol.version,
                        "hypotheses": [
                            {"id": "ready", "description": "主力可用", "prior": .5,
                             "outcome_probabilities": [.8, .15, .05]},
                            {"id": "limited", "description": "主力不可用", "prior": .5,
                             "outcome_probabilities": [.15, .4, .45]}],
                        "actions": [factory.action(candidate) for candidate in public["candidates"]],
                        "assumptions": ["仅用于集成测试的构造情景。"]}
                    if factory.invalid_numeric:
                        response["hypotheses"][0]["outcome_probabilities"] = [.8, .15, .06]
                elif protocol.version == ActionExtensionProtocol.version:
                    response = {"schema": protocol.version,
                        "actions": [factory.action(candidate) for candidate in public["candidates"]],
                        "assumptions": ["扩展保持原情景参数。"]}
                elif protocol.version == ObservationProtocol.version:
                    doc = public["documents"][0]
                    response = {"schema": protocol.version, "action_id": public["action"]["action_id"],
                        "outcome": factory.answer, "applicability": factory.applicability,
                        "evidence": [] if factory.answer == "no_answer" else [{
                            "document_id": doc["document_id"],
                            "quote": "No such quote exists in this document." if factory.invalid_quote else doc["text"]}],
                        "reason": "仅按实际返回正文分类的测试输出。"}
                else:
                    raise AssertionError("Unexpected semantic protocol")
                return {"response": response, "receipt": {"real_provider_call": False,
                         "usage": {}, "latency_seconds": .01}}
        return Provider()


def run_case(tmp_path, *, factory=None, transport=None, request_budget=4, max_extensions=0,
             max_rediscoveries=0, semantic_call_budget=12, planning_horizon=1):
    factory = factory or FakeProviderFactory()
    transport = transport or FakeTransport()
    runtime = SemanticScenarioRuntime(provider_factory=factory, transport=transport,
        output=tmp_path / "run", case=CASE, seed_urls=[SEED], bootstrap_url=SEED,
        request_budget=request_budget, max_extensions=max_extensions,
        max_rediscoveries=max_rediscoveries, semantic_call_budget=semantic_call_budget,
        planning_horizon=planning_horizon,
        clock=lambda: NOW)
    return runtime.run(), factory, transport


def test_real_solver_selects_more_valuable_later_candidate_and_replays_without_new_calls(tmp_path):
    run, factory, transport = run_case(tmp_path)
    assert run["status"] == "completed"
    assert [item["url"] for item in transport.executions] == [SEED, STRONG]
    first = run["cycles"][0]
    selected = first["decision"]["selected_action_id"]
    assert first["candidate_bindings"][selected]["url"] == STRONG
    values = {first["candidate_bindings"][item["action_id"]]["url"]: item for item in first["decision"]["candidates"]}
    assert values[STRONG]["lower"] > 0
    assert values[WEAK]["lower"] < 0
    assert run["final_prediction"][0] > BASELINE[0]
    before = (len(factory.inputs), len(transport.executions))
    replay = replay_math(run)
    assert replay["status"] == "passed" and replay["http_requests"] == replay["semantic_calls"] == 0
    assert before == (len(factory.inputs), len(transport.executions))
    assert not run["terminal"]["global_stop_certificate"]


def test_runtime_persists_bounded_planning_horizon_and_replays_it(tmp_path):
    run, _, transport = run_case(tmp_path, request_budget=3, planning_horizon=2)
    assert run["configuration"]["planning_horizon"] == 2
    assert run["configuration"]["planning_policy"] == "shared_observation_continuation_v1"
    assert run["cycles"][0]["model"]["planning_policy"] == run["configuration"]["planning_policy"]
    assert run["cycles"][0]["decision"]["global_policy_optimality_claim"] is False
    assert run["cycles"][0]["decision"]["horizon"] == 2
    assert run["evidence_scope"]["planning_horizon"] == 2
    before = list(transport.executions)
    assert replay_math(run)["status"] == "passed"
    assert transport.executions == before


def test_math_replay_rejects_tampered_planning_horizon(tmp_path):
    run, _, _ = run_case(tmp_path, request_budget=3, planning_horizon=2)
    altered = deepcopy(run)
    altered["configuration"]["planning_horizon"] = 1
    with pytest.raises(ValueError, match="Planning horizon accounting"):
        replay_math(altered)


def test_classifier_only_sees_typed_body_of_selected_return(tmp_path):
    run, factory, _ = run_case(tmp_path)
    calls = [item["public"] for item in factory.inputs if item["protocol"] == ObservationProtocol.version]
    assert len(calls) == 1
    assert [doc["document_id"] for doc in calls[0]["documents"]] == ["strong"]
    assert calls[0]["documents"][0]["text"] == BODY
    assert TITLE_ONLY not in calls[0]["documents"][0]["text"]
    assert run["observations"][0]["returned_document_ids"] == ["strong"]
    assert run["observations"][0]["evidence"][0]["quote"] == BODY


@pytest.mark.parametrize("answer,applicability", [("no_answer", "unresolved"), ("supports", "team_context")])
def test_no_answer_and_team_context_keep_beliefs_but_consume_the_attempt(tmp_path, answer, applicability):
    run, _, transport = run_case(tmp_path, factory=FakeProviderFactory(answer=answer, applicability=applicability))
    assert run["observations"][0]["outcome"] == "no_answer"
    assert run["final_prediction"] == pytest.approx(BASELINE, abs=1e-12)
    for posterior in run["cycles"][-1]["decision"]["model_posteriors"].values():
        assert posterior == pytest.approx({"ready": .5, "limited": .5}, abs=1e-12)
    assert len(transport.executions) == 2
    assert len(run["observations"]) == 1


def test_title_only_success_cannot_trigger_classifier_or_probability_update(tmp_path):
    transport = FakeTransport(pages={STRONG: document("strong", STRONG, text=TITLE_ONLY, role="title")})
    run, factory, _ = run_case(tmp_path, transport=transport)
    assert all(item["protocol"] != ObservationProtocol.version for item in factory.inputs)
    assert run["observations"][0]["outcome"] == "no_answer"
    assert run["final_prediction"] == pytest.approx(BASELINE, abs=1e-12)
    assert run["documents"][-1]["text"] == ""


def test_future_published_body_is_not_classified_or_used_as_evidence(tmp_path):
    page = document("strong", STRONG)
    page["structured_news"]["segments"][0]["source_published_at"] = "2026-10-11T12:00:00Z"
    run, factory, _ = run_case(tmp_path, transport=FakeTransport(pages={STRONG: page}))
    assert all(item["protocol"] != ObservationProtocol.version for item in factory.inputs)
    assert run["documents"][-1]["admission"]["excluded_segments"] == ["published_after_decision_boundary"]
    assert run["observations"][0]["outcome"] == "no_answer"
    assert run["final_prediction"] == pytest.approx(BASELINE, abs=1e-12)


def test_composite_search_reserves_two_requests_and_only_body_is_classified(tmp_path):
    run, factory, transport = run_case(tmp_path, request_budget=3,
        factory=FakeProviderFactory(queries=["Home FC Away FC fixture confirmed lineup"]),
        transport=FakeTransport(source_links=(WEAK,)))
    assert run["status"] == "completed"
    assert [item["tool"] for item in transport.executions] == ["read_url", "search_web", "read_url"]
    assert transport.executions[-1]["url"] == FOUND
    assert run["resources"]["http_requests"] == 3
    selected = run["cycles"][0]["decision"]["selected_action_id"]
    assert run["cycles"][0]["candidate_bindings"][selected]["request_cost"] == 2
    calls = [item["public"] for item in factory.inputs if item["protocol"] == ObservationProtocol.version]
    assert [doc["document_id"] for doc in calls[0]["documents"]] == ["found"]
    assert run["observations"][0]["returned_document_ids"] == ["search", "found"]
    assert run["terminal"]["stop_reason"] == "request_budget_exhausted"


def test_composite_search_cannot_start_when_only_one_request_remains(tmp_path):
    run, factory, transport = run_case(tmp_path, request_budget=2,
        factory=FakeProviderFactory(queries=["Home FC Away FC match availability"]),
        transport=FakeTransport(source_links=(WEAK,)))
    assert len(transport.executions) == 1
    search = next(item for item in run["cycles"][0]["decision"]["candidates"] if item["action_id"].startswith("search-"))
    assert not search["feasible"] and search["reason"] == "insufficient_request_budget"
    assert not run["observations"]
    assert all(item["protocol"] != ObservationProtocol.version for item in factory.inputs)


def test_empty_search_spends_only_attempted_http_and_never_uses_snippet(tmp_path):
    run, factory, transport = run_case(tmp_path, request_budget=3,
        factory=FakeProviderFactory(queries=["Home FC Away FC match availability"]),
        transport=FakeTransport(source_links=(WEAK,), search_links=()))
    assert [item["tool"] for item in transport.executions] == ["read_url", "search_web"]
    assert run["resources"]["http_requests"] == 2
    assert run["observations"][0]["outcome"] == "no_answer"
    assert run["final_prediction"] == pytest.approx(BASELINE, abs=1e-12)
    assert all(item["protocol"] != ObservationProtocol.version for item in factory.inputs)


def test_new_candidate_extension_preserves_already_conditioned_posterior(tmp_path):
    transport = FakeTransport(pages={STRONG: document("strong", STRONG, links=[link(NEW)])})
    run, factory, _ = run_case(tmp_path, transport=transport, max_extensions=1)
    assert run["status"] == "completed" and len(run["extensions"]) == 1
    initial_model = SemanticScenarioModel.from_manifest(run["cycles"][0]["model"])
    expected = initial_model.evaluate(run["observations"], 2)
    final = run["cycles"][-1]["decision"]
    assert final["model_id"] != expected["model_id"]
    assert final["posterior_nominal"] == pytest.approx(expected["posterior_nominal"], abs=1e-12)
    assert final["prediction_nominal"] == pytest.approx(expected["prediction_nominal"], abs=1e-12)
    assert any(candidate["url"] == NEW for candidate in run["extensions"][0]["new_candidates"])
    extension_input = next(item["public"] for item in factory.inputs if item["protocol"] == ActionExtensionProtocol.version)
    assert extension_input["actual_observations"][0]["outcome"] == "supports"
    assert extension_input["frozen_proposal"]["hypotheses"] == initial_model.proposal["hypotheses"]


def test_bounded_rediscovery_after_first_no_answer_models_query_without_prefetch(tmp_path):
    factory = FakeProviderFactory(answer="no_answer",
        rediscovery_queries=["Home FC Away FC confirmed lineup after first report"])
    run, factory, transport = run_case(tmp_path, factory=factory, max_rediscoveries=1)
    discovery_calls = [item for item in factory.inputs
                       if item["protocol"] == CandidateDiscoveryProtocol.version]
    assert len(discovery_calls) == 2
    assert len(run["rediscoveries"]) == 1
    rediscovery = run["rediscoveries"][0]
    assert rediscovery["conditioned_on_observations"][0]["outcome"] == "no_answer"
    query = rediscovery["new_candidates"][0]
    assert query["tool"] == "search_then_read" and query["request_cost"] == 2
    # Candidate discovery and action extension do not read the new target. The
    # search/read pair appears only after the solver selects the modeled query.
    assert [item["tool"] for item in transport.executions] == [
        "read_url", "read_url", "search_web", "read_url"]
    extension = rediscovery["action_extension"]
    assert extension["source"] == "rediscovery"
    extension_input = next(item["public"] for item in factory.inputs
                           if item["protocol"] == ActionExtensionProtocol.version)
    assert extension_input["actual_observations"][0]["outcome"] == "no_answer"
    assert query["action_id"] in [item["action_id"] for item in extension["new_candidates"]]


def test_rediscovery_keeps_frozen_state_and_default_is_compatible(tmp_path):
    factory = FakeProviderFactory(answer="supports",
        rediscovery_queries=["Home FC Away FC tactical plan after first report"])
    run, factory, _ = run_case(tmp_path, factory=factory, max_rediscoveries=1)
    initial = run["cycles"][0]["model"]["proposal"]
    rediscovery = run["rediscoveries"][0]
    assert rediscovery["action_extension"]["addition"]["actions"]
    assert rediscovery["conditioned_on_observations"][0]["outcome"] == "supports"
    assert rediscovery["action_extension"]["addition"]["actions"][0]["action_id"] not in {
        action["action_id"] for action in initial["actions"]}
    frozen = rediscovery["action_extension"]["addition"]
    assert run["proposal"]["hypotheses"] == initial["hypotheses"]
    assert run["proposal"]["actions"][:len(initial["actions"])] == initial["actions"]
    assert frozen["assumptions"]
    assert run["configuration"]["rediscovery"]["max_rounds"] == 1

    default_run, default_factory, _ = run_case(tmp_path / "default",
        factory=FakeProviderFactory(answer="no_answer",
            rediscovery_queries=["should never be called"]), max_rediscoveries=0)
    assert default_run["rediscoveries"] == []
    assert sum(item["protocol"] == CandidateDiscoveryProtocol.version
               for item in default_factory.inputs) == 1


def test_rediscovery_requires_two_remaining_http_requests(tmp_path):
    factory = FakeProviderFactory(answer="no_answer",
        rediscovery_queries=["Home FC Away FC late availability"])
    run, factory, transport = run_case(tmp_path, factory=factory,
        request_budget=2, max_rediscoveries=1)
    assert run["rediscoveries"] == []
    assert sum(item["protocol"] == CandidateDiscoveryProtocol.version
               for item in factory.inputs) == 1
    assert [item["tool"] for item in transport.executions] == ["read_url", "read_url"]


def test_bad_classification_gets_bounded_repair_then_no_answer_with_reported_failure(tmp_path):
    run, factory, _ = run_case(tmp_path, factory=FakeProviderFactory(invalid_quote=True))
    classifiers = [item for item in factory.inputs if item["protocol"] == ObservationProtocol.version]
    assert len(classifiers) == 2
    assert classifiers[1]["public"]["validation_errors"]
    assert run["observations"][0]["outcome"] == "no_answer"
    assert run["final_prediction"] == pytest.approx(BASELINE, abs=1e-12)
    assert run["failures"][0]["stage"] == "classification"
    assert "exact body substring" in run["failures"][0]["error"]


def test_semantic_repair_policy_closes_numeric_contract_without_retry():
    assert SemanticScenarioRuntime._semantic_repair_policy(
        "Scenario outcome must sum to one; no silent normalization"
    ) == (False, "numeric_contract")
    assert SemanticScenarioRuntime._semantic_repair_policy(
        "Observation quote is not an exact body substring"
    ) == (True, "schema_or_evidence")


def test_invalid_numeric_proposal_closes_without_expensive_repair(tmp_path):
    factory = FakeProviderFactory(invalid_numeric=True)
    run, factory, _ = run_case(tmp_path, factory=factory)
    proposal_calls = [item for item in factory.inputs
                      if item["protocol"] == ProposalProtocol.version]
    assert len(proposal_calls) == 1
    assert run["status"] == "incomplete"
    assert run["proposal"] is None
    assert run["failures"][0]["stage"] == "runtime"
    saved = json.loads((tmp_path / "run" / "semantic-calls" / "record-001.json").read_text(encoding="utf-8"))
    assert saved["repair_eligible"] is False
    assert saved["failure_code"] == "numeric_contract"


def test_provider_failure_is_saved_as_incomplete_without_fabricated_proposal(tmp_path):
    run, _, transport = run_case(tmp_path, factory=FakeProviderFactory(fail_protocol=ProposalProtocol.version))
    assert run["status"] == "incomplete"
    assert run["proposal"] is None and run["observations"] == []
    assert run["final_prediction"] == BASELINE
    assert run["terminal"]["stop_reason"] == "runtime_incomplete"
    assert run["failures"][0]["stage"] == "runtime"
    assert len(transport.executions) == 1
    persisted = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert persisted["failures"] == run["failures"]
    assert persisted["semantic_calls"][-1]["status"] == "failed"


def test_extension_failure_preserves_successful_observation_in_final_report(tmp_path):
    transport = FakeTransport(pages={STRONG: document("strong", STRONG, links=[link(NEW)])})
    run, _, _ = run_case(tmp_path, transport=transport, max_extensions=1,
        factory=FakeProviderFactory(fail_protocol=ActionExtensionProtocol.version))
    assert run["status"] == "incomplete"
    assert run["observations"][0]["outcome"] == "supports"
    current = SemanticScenarioModel.from_manifest(run["cycles"][0]["model"]).evaluate(
        run["observations"], run["configuration"]["request_budget"] - run["resources"]["http_requests"])
    assert run["final_prediction"] == pytest.approx(current["prediction_nominal"], abs=1e-12)
    assert run["final_prediction"][0] > BASELINE[0]
    assert "controlled provider unavailable" in run["failures"][-1]["error"]


@pytest.mark.parametrize("mutation", ["decision", "observation", "budget"])
def test_math_replay_rejects_tampered_decision_observation_or_accounting(tmp_path, mutation):
    run, _, _ = run_case(tmp_path)
    altered = deepcopy(run)
    if mutation == "decision":
        altered["cycles"][0]["decision"]["prediction_nominal"][0] += .01
    elif mutation == "observation":
        altered["cycles"][-1]["observations"][0]["outcome"] = "refutes"
    else:
        altered["cycles"][0]["requests_spent"] += 1
    with pytest.raises(ValueError, match="replay mismatch|accounting"):
        replay_math(altered)


def test_existing_run_directory_is_never_overwritten(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "previous.json"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        SemanticScenarioRuntime(provider_factory=FakeProviderFactory(), transport=FakeTransport(),
            output=output, case=CASE, seed_urls=[SEED], bootstrap_url=SEED, clock=lambda: NOW)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_projection_failure_keeps_budget_reserved_and_request_total_uncertain(tmp_path):
    class ProjectionFailure(FakeTransport):
        def execute(self, prepared, *, run_id, request_id):
            raise RuntimeError("projection failed after possible HTTP attempt")

    run, _, _ = run_case(tmp_path, transport=ProjectionFailure())
    assert run["status"] == "incomplete"
    assert run["resources"]["request_budget_reserved"] == 1
    assert run["resources"]["http_accounting_complete"] is False
    assert run["http_receipts"][0]["returned"]["http_attempts"] is None
    assert run["http_receipts"][0]["status"] == "reconciliation_required"
    assert not run["semantic_calls"]


def test_live_run_after_prediction_cutoff_cannot_acquire_data(tmp_path):
    transport = FakeTransport()
    case = deepcopy(CASE)
    case["match"]["prediction_cutoff"] = "2026-10-07T12:00:00Z"
    with pytest.raises(ValueError, match="now <= prediction_cutoff < kickoff"):
        SemanticScenarioRuntime(provider_factory=FakeProviderFactory(), transport=transport,
            output=tmp_path/"late", case=case, seed_urls=[SEED], bootstrap_url=SEED,
            clock=lambda: NOW)
    assert transport.executions == []
