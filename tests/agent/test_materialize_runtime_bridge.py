"""The explicit case materializer must feed the canonical semantic runtime."""
from datetime import datetime, timezone

from football_agent.autonomous_match_contract import build_match_bundle
from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime, replay_math
from tests.agent.test_semantic_scenario_runtime import FakeProviderFactory, FakeTransport
from tools.materialize_footballmind_case import materialize


NOW = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)


def test_materialized_case_runs_and_replays_without_cross_fixture_defaults(tmp_path):
    bundle = build_match_bundle({
        "match_id": "bridge:test",
        "home": "Home FC",
        "away": "Away FC",
        "competition": "Synthetic League",
        "kickoff": "2026-10-10T12:00:00Z",
        "prediction_cutoff": "2026-10-09T12:00:00Z",
    }, now=NOW)
    profile = {
        "status": "verified",
        "teams": [{"side": "home", "canonical_name": "Home FC", "input_name": "Home FC"},
                  {"side": "away", "canonical_name": "Away FC", "input_name": "Away FC"}],
        "allowed_hosts": ["example.org"],
        "seed_urls": ["https://example.org/en"],
        "bootstrap_url": "https://example.org/en",
    }
    baseline = {"probabilities": [.475, .275, .25], "source": "synthetic-bridge",
                "match_id": "bridge:test", "prediction_cutoff": "2026-10-09T12:00:00Z"}
    case = materialize(bundle, profile, baseline)
    transport = FakeTransport()
    runtime = SemanticScenarioRuntime(
        provider_factory=FakeProviderFactory(), transport=transport,
        output=tmp_path / "run", case=case, seed_urls=case["source_profile"]["seed_urls"],
        bootstrap_url=case["source_profile"]["bootstrap_url"], request_budget=3,
        planning_horizon=2, clock=lambda: NOW,
    )
    result = runtime.run()
    assert result["status"] == "completed"
    assert result["configuration"]["bootstrap_url"] == "https://example.org/en"
    assert "leedsunited" not in result["configuration"]["bootstrap_url"]
    assert replay_math(result)["status"] == "passed"
