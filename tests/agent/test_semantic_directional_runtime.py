"""New-run directional gate: answered status is not automatically an effect."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from football_agent.semantic_scenario_protocol import (
    CandidateDiscoveryProtocol, ProposalProtocol, DirectionalProposalProtocol,
    DirectionalObservationProtocol,
)
from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime


NOW = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)
WHEN = NOW.isoformat()
SEED = "https://example.org/en"
ARTICLE = "https://example.org/en/news/fixture"
BODY = "The manager confirmed the striker is available for this fixture."
BASELINE = [.475, .275, .25]
CASE = {
    "match": {"match_id": "directional:test", "home_team_id": "home", "away_team_id": "away",
              "kickoff": "2026-10-10T12:00:00Z", "prediction_cutoff": "2026-10-09T12:00:00Z"},
    "source_profile": {"teams": [{"side": "home", "canonical_name": "Home FC"},
                                    {"side": "away", "canonical_name": "Away FC"}]},
    "structured_baseline": {"probabilities": BASELINE},
    "baseline_scope": "constructed_test_only",
}


def _doc(identity, url, *, kind="article", links=()):
    text = BODY if kind == "article" else "index"
    return {"document_id": identity, "url": url, "title": "fixture", "kind": kind,
            "status": "OK", "text": text, "observed_at": WHEN, "available_at": WHEN,
            "source_published_at": "2026-10-07T12:00:00Z", "links": list(links),
            "structured_news": {"segments": [{"content_role": "paragraph", "text": text,
                                                "source_published_at": "2026-10-07T12:00:00Z"}]}}


class Transport:
    def configuration(self):
        return {"provider": "directional_test", "actual_network": False}

    def prepare(self, candidate, state):
        return candidate

    def execute(self, prepared, *, run_id, request_id):
        if prepared["tool"] == "read_url" and prepared["url"] == SEED:
            doc = _doc("index", SEED, kind="index", links=[{"url": ARTICLE, "title": "fixture"}])
        else:
            doc = _doc("article", ARTICLE)
        return {"http_attempts": 1, "document": doc, "latency_seconds": .01,
                "run_id": run_id, "request_id": request_id}


class Factory:
    def __init__(self, direction, contract_supports=None):
        self.direction = direction
        self.contract_supports = contract_supports or [direction] if direction in {
            "home_positive", "home_negative", "away_positive", "away_negative", "mixed"
        } else ["home_positive"]

    def __call__(self, workspace, protocol):
        factory = self

        class P:
            def configuration(self):
                return {"provider": "directional_test", "protocol": protocol.version}

            def decide(self, public):
                if protocol.version == CandidateDiscoveryProtocol.version:
                    response = {"schema": protocol.version, "queries": []}
                elif protocol.version in {ProposalProtocol.version, DirectionalProposalProtocol.version}:
                    supports = list(factory.contract_supports)
                    opposites = {"home_positive": "home_negative", "home_negative": "home_positive",
                                 "away_positive": "away_negative", "away_negative": "away_positive"}
                    refutes = [opposites[item] for item in supports if item in opposites]
                    if not refutes:
                        # The fixture does not exercise mixed-proposition
                        # semantics; keep a single legal opposite bucket.
                        refutes = ["home_positive"]
                    response = {"schema": protocol.version,
                        "hypotheses": [
                            {"id": "ready", "description": "主力可用", "prior": .5,
                             "outcome_probabilities": [.8, .15, .05]},
                            {"id": "limited", "description": "主力受限", "prior": .5,
                             "outcome_probabilities": [.15, .4, .45]}],
                        "actions": [{"action_id": c["action_id"],
                            "question": "本场主力可用并对主队产生明确影响吗？",
                            "observation_group": "fixture-effect",
                            **({"direction_contract": {"supports": supports,
                                "refutes": refutes}} if protocol.version == DirectionalProposalProtocol.version else {}),
                            "likelihoods": [
                                {"hypothesis_id": "ready", "probabilities": [.8, .1, .1]},
                                {"hypothesis_id": "limited", "probabilities": [.1, .8, .1]}]}
                            for c in public["candidates"]],
                        "assumptions": ["测试构造情景。"]}
                elif protocol.version == DirectionalObservationProtocol.version:
                    # Numerical forecasts must not lead the source classifier.
                    assert set(public) == {"input_contract", "match", "action", "documents"}
                    assert set(public["action"]) == {"action_id", "question", "direction_contract"}
                    assert public["input_contract"] == "semantic-observation-evidence-only-v2"
                    doc = public["documents"][0]
                    response = {"schema": protocol.version,
                        "action_id": public["action"]["action_id"], "outcome": "supports",
                        "direction": factory.direction, "subject": "home" if factory.direction != "unknown" else "unknown",
                        "direction_basis": "explicit_effect" if factory.direction != "unknown" else "availability_only",
                        "directional_claim": "正文明确说明本场影响。" if factory.direction != "unknown" else "正文只说明状态，未说明方向。",
                        "evidence": [{"document_id": doc["document_id"], "quote": doc["text"]}],
                        "applicability": "match_specific", "reason": "测试分类。"}
                else:
                    raise AssertionError("unexpected protocol " + protocol.version)
                return {"response": response, "receipt": {"usage": {}, "latency_seconds": .01}}
        return P()


def run(tmp_path, direction, contract_supports=None):
    runtime = SemanticScenarioRuntime(provider_factory=Factory(direction, contract_supports), transport=Transport(),
        output=tmp_path / "run", case=deepcopy(CASE), seed_urls=[SEED], bootstrap_url=SEED,
        request_budget=2, semantic_call_budget=5, max_extensions=0, max_rediscoveries=0,
        require_directional_evidence=True, clock=lambda: NOW)
    return runtime.run()


def test_availability_only_answer_is_inert(tmp_path):
    result = run(tmp_path, "unknown")
    assert result["observations"][0]["outcome"] == "no_answer"
    assert result["observations"][0]["directional_gate"]["status"] == "inert"
    assert result["final_prediction"] == pytest.approx(BASELINE, abs=1e-12)


def test_explicit_effect_can_update(tmp_path):
    result = run(tmp_path, "home_positive")
    assert result["observations"][0]["outcome"] == "supports"
    assert result["observations"][0]["directional_gate"]["status"] == "accepted"
    assert result["final_prediction"][0] > BASELINE[0]


def test_opposite_direction_is_inert(tmp_path):
    result = run(tmp_path, "home_positive", contract_supports=["home_negative"])
    observation = result["observations"][0]
    assert observation["outcome"] == "no_answer"
    assert observation["directional_gate"]["reason"] == "direction_contract_mismatch"
    assert result["final_prediction"] == pytest.approx(BASELINE, abs=1e-12)


@pytest.mark.parametrize("direction", ["unknown", "home_positive"])
def test_evidence_loop_acceptance_checks_actual_runtime_transitions(tmp_path, direction):
    from tools.check_semantic_evidence_loop import audit_evidence_loop
    result = run(tmp_path, direction)
    before = deepcopy(result)
    audit = audit_evidence_loop(result)
    assert audit["status"] == "passed", audit["failures"]
    assert result == before
    assert audit["coverage"]["accepted_body_classifications"] == 1
    assert audit["coverage"]["no_answer_transitions"] == (direction == "unknown")
    assert audit["network_calls"] == audit["semantic_calls"] == 0
    if direction == "home_positive":
        assert "no_answer_transition" in audit["coverage"]["unexercised"]
    else:
        assert "informative_directional_update" in audit["coverage"]["unexercised"]
    assert "reselection_after_no_answer" in audit["coverage"]["unexercised"]


@pytest.mark.parametrize("corruption", ["baseline", "body", "effective_outcome", "final_probability", "incomplete"])
def test_evidence_loop_acceptance_rejects_leakage_and_broken_chain(tmp_path, corruption):
    from tools.check_semantic_evidence_loop import audit_evidence_loop
    result = run(tmp_path, "unknown")
    observation_call = next(c for c in result["semantic_calls"] if c["protocol"] == DirectionalObservationProtocol.version)
    if corruption == "baseline":
        observation_call["input"]["baseline"] = {"probabilities": BASELINE}
    elif corruption == "body":
        observation_call["input"]["documents"][0]["text"] = "A different unacquired article."
    elif corruption == "effective_outcome":
        result["observations"][0]["outcome"] = "supports"
    elif corruption == "final_probability":
        result["final_prediction"] = [.9, .05, .05]
    else:
        result["status"] = "incomplete"
    audit = audit_evidence_loop(result)
    assert audit["status"] == "failed" and audit["failures"]


def test_evidence_loop_resource_stop_is_a_valid_stop_but_not_body_coverage(tmp_path):
    from tools.check_semantic_evidence_loop import audit_evidence_loop
    base_factory, repaired = Factory("unknown"), []

    def factory(workspace, protocol):
        provider = base_factory(workspace, protocol)
        original_decide = provider.decide

        def decide(public):
            returned = original_decide(public)
            if protocol.version == CandidateDiscoveryProtocol.version and not repaired:
                returned["response"]["schema"] = "malformed-test-schema"
                repaired.append(True)
            return returned

        provider.decide = decide
        return provider

    runtime = SemanticScenarioRuntime(provider_factory=factory, transport=Transport(),
        output=tmp_path/"run", case=deepcopy(CASE), seed_urls=[SEED], bootstrap_url=SEED,
        request_budget=2, semantic_call_budget=3, max_extensions=0, max_rediscoveries=0,
        require_directional_evidence=True, clock=lambda: NOW)
    result = runtime.run()
    assert result["status"] == "completed_with_resource_stop"
    audit = audit_evidence_loop(result)
    assert audit["status"] == "passed", audit["failures"]
    assert audit["coverage"]["accepted_body_classifications"] == 0
    assert "accepted_body_classification" in audit["coverage"]["unexercised"]
    result["terminal"]["stop_reason"] = "request_budget_exhausted"
    assert audit_evidence_loop(result)["status"] == "failed"
