"""Constructed integration: metadata nomination precedes actual numerical VoI."""
from copy import deepcopy

import pytest

from football_agent.semantic_candidate_discovery import CatalogDiscoveryProtocol
from football_agent.semantic_scenario_protocol import ProposalProtocol
from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime, replay_math
from tests.agent.test_semantic_scenario_runtime import (
    BASELINE, CASE, NOW, SEED, STRONG, WEAK, FakeProviderFactory, FakeTransport,
)


class CatalogFactory(FakeProviderFactory):
    def __init__(self, nominations, **kwargs):
        super().__init__(**kwargs)
        self.nominations = nominations

    def __call__(self, workspace, protocol):
        if protocol.version != CatalogDiscoveryProtocol.version:
            return super().__call__(workspace, protocol)
        factory = self

        class Provider:
            def configuration(self):
                return {"provider": "controlled_test_semantics", "protocol": protocol.version}

            def decide(self, public):
                factory.inputs.append({"protocol": protocol.version, "public": deepcopy(public)})
                index = sum(item["protocol"] == protocol.version for item in factory.inputs)-1
                urls = factory.nominations[min(index, len(factory.nominations)-1)]
                by_url = {item["url"]: item for item in public["observed_articles"]}
                response = {"schema": protocol.version, "queries": [], "article_candidates": [
                    {"action_id": by_url[url]["action_id"], "information_need": "仅提名本场相关的待核实信息。"}
                    for url in urls]}
                return {"response": response, "receipt": {"real_provider_call": False, "latency_seconds": .01}}

        return Provider()


def execute(tmp_path, factory, transport, **kwargs):
    runtime = SemanticScenarioRuntime(provider_factory=factory, transport=transport,
        output=tmp_path/"run", case=CASE, seed_urls=[SEED], bootstrap_url=SEED,
        candidate_discovery_version=2, clock=lambda: NOW, **kwargs)
    return runtime.run()


def test_late_visible_article_can_be_nominated_and_selected_without_prefetch(tmp_path):
    earlier = tuple(f"https://example.org/en/news/unrelated-{i}" for i in range(99))
    transport = FakeTransport(source_links=earlier+(STRONG,))
    factory = CatalogFactory([[STRONG]])
    result = execute(tmp_path, factory, transport, request_budget=3)
    assert result["status"] == "completed"
    first = factory.inputs[0]["public"]
    assert len(first["observed_articles"]) == 100
    assert first["observed_articles"][-1]["url"] == STRONG
    assert all("text" not in item for item in first["observed_articles"])
    assert first["documents"][0]["text"] == ""
    assert [item["url"] for item in transport.executions] == [SEED, STRONG]
    assert result["candidate_discovery"]["unmodeled_article_candidates"] == 99
    assert len(result["cycles"][0]["decision"]["candidates"]) == 1
    assert result["final_prediction"][0] > BASELINE[0]
    assert replay_math(result)["status"] == "passed"


def test_semantic_nomination_cannot_force_acquisition_when_numerical_value_is_nonpositive(tmp_path):
    factory, transport = CatalogFactory([[WEAK]]), FakeTransport()
    result = execute(tmp_path, factory, transport, request_budget=3)
    assert result["status"] == "completed"
    assert result["terminal"]["stop_reason"] == "no_positive_declared_lower_net_value"
    assert result["final_prediction"] == pytest.approx(BASELINE)
    assert [item["url"] for item in transport.executions] == [SEED]
    assert not result["terminal"]["global_stop_certificate"]
    proposal = next(item["public"] for item in factory.inputs if item["protocol"] == ProposalProtocol.version)
    assert [item["url"] for item in proposal["candidates"]] == [WEAK]


def test_empty_nomination_is_incomplete_and_does_not_fall_back_to_dom_order(tmp_path):
    factory, transport = CatalogFactory([[]]), FakeTransport()
    result = execute(tmp_path, factory, transport)
    assert result["status"] == "incomplete"
    assert result["candidate_discovery"]["menu"] == []
    assert result["cycles"] == []
    assert "no_executable_visible_candidates" in result["terminal"]["error"]
    assert len(transport.executions) == 1


def test_after_no_answer_new_nomination_sees_history_and_excludes_existing_actions(tmp_path):
    class InformativeFactory(CatalogFactory):
        @staticmethod
        def action(candidate):
            return FakeProviderFactory.action({**candidate, "tool": "search_then_read"})

    factory = InformativeFactory([[STRONG], [WEAK]], answer="no_answer")
    transport = FakeTransport()
    result = execute(tmp_path, factory, transport, request_budget=3, max_rediscoveries=1)
    assert result["status"] == "completed"
    discoveries = [item["public"] for item in factory.inputs if item["protocol"] == CatalogDiscoveryProtocol.version]
    assert len(discoveries) == 2
    assert discoveries[1]["actual_observations"][0]["outcome"] == "no_answer"
    assert [item["url"] for item in discoveries[1]["observed_articles"]] == [WEAK]
    assert [item["url"] for item in transport.executions] == [SEED, STRONG, WEAK]
    assert len(result["extensions"]) == 1
    assert result["proposal"]["hypotheses"] == result["cycles"][0]["model"]["proposal"]["hypotheses"]
    assert replay_math(result)["status"] == "passed"


def test_v2_initialization_resume_reuses_the_validated_nomination(tmp_path):
    first_factory = CatalogFactory([[STRONG]], fail_protocol=ProposalProtocol.version)
    first = execute(tmp_path/"original", first_factory, FakeTransport(), request_budget=3)
    assert first["status"] == "incomplete"
    factory, transport = CatalogFactory([[]]), FakeTransport()
    runtime = SemanticScenarioRuntime(provider_factory=factory, transport=transport,
        output=tmp_path/"continued", case=CASE, seed_urls=[SEED], bootstrap_url=SEED,
        candidate_discovery_version=2, clock=lambda: NOW, request_budget=3,
        resume_from=tmp_path/"original/run/run.json")
    result = runtime.run()
    assert result["status"] == "completed"
    assert all(item["protocol"] != CatalogDiscoveryProtocol.version for item in factory.inputs)
    assert [item["url"] for item in transport.executions] == [STRONG]
    assert result["resources"]["http_requests"] == 2
    assert result["resume_provenance"]["cumulative_budget_no_refund"] is True
    assert result["semantic_calls"][1]["status"] == "failed"
    assert replay_math(result)["status"] == "passed"
