"""A failed initialization continues one budget; it is not a fresh experiment."""
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json

import pytest

from football_agent.semantic_scenario_protocol import CandidateDiscoveryProtocol, ProposalProtocol
from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime, replay_math
from tests.agent.test_semantic_scenario_runtime import (
    BASELINE, CASE, NOW, SEED, STRONG, FakeProviderFactory, FakeTransport,
)


def prefix(tmp_path, *, stage=CandidateDiscoveryProtocol.version, calls=6):
    transport = FakeTransport()
    runtime = SemanticScenarioRuntime(provider_factory=FakeProviderFactory(fail_protocol=stage),
        transport=transport, output=tmp_path/"source", case=CASE, seed_urls=[SEED], bootstrap_url=SEED,
        request_budget=3, semantic_call_budget=calls, max_extensions=0, clock=lambda: NOW)
    result = runtime.run()
    assert result["status"] == "incomplete"
    return tmp_path/"source/run.json", result


def continued(tmp_path, source, *, factory=None, transport=None, **changes):
    kwargs = dict(provider_factory=factory or FakeProviderFactory(), transport=transport or FakeTransport(),
        output=tmp_path/"continued", case=CASE, seed_urls=[SEED], bootstrap_url=SEED,
        request_budget=3, semantic_call_budget=6, max_extensions=0,
        clock=lambda: NOW+timedelta(seconds=1), resume_from=source)
    kwargs.update(changes)
    return SemanticScenarioRuntime(**kwargs)


@pytest.mark.parametrize("stage", [CandidateDiscoveryProtocol.version, ProposalProtocol.version])
def test_resume_preserves_original_and_cumulative_calls_without_bootstrap_repeat(tmp_path, stage):
    source, original = prefix(tmp_path, stage=stage)
    before = {str(p): sha256(p.read_bytes()).hexdigest() for p in source.parent.rglob("*") if p.is_file()}
    factory, transport = FakeProviderFactory(), FakeTransport()
    result = continued(tmp_path, source, factory=factory, transport=transport).run()
    assert result["status"] == "completed"
    assert [item["url"] for item in transport.executions] == [STRONG]
    assert result["resources"]["http_requests"] == result["resources"]["request_budget_reserved"] == 2
    assert result["semantic_calls"][:len(original["semantic_calls"])] == original["semantic_calls"]
    new_discovery_calls = sum(item["protocol"] == CandidateDiscoveryProtocol.version for item in factory.inputs)
    assert new_discovery_calls == int(stage == CandidateDiscoveryProtocol.version)
    provenance = result["resume_provenance"]
    assert provenance["source_run_sha256"] == before[str(source)]
    assert provenance["retained_failed_semantic_calls"] == 1
    assert provenance["cumulative_budget_no_refund"] is True
    assert before == {str(p): sha256(p.read_bytes()).hexdigest() for p in source.parent.rglob("*") if p.is_file()}
    assert replay_math(result)["status"] == "passed"


def test_failed_call_is_not_refunded_at_semantic_budget_stop(tmp_path):
    source, _ = prefix(tmp_path, calls=3)
    transport = FakeTransport()
    result = continued(tmp_path, source, semantic_call_budget=3, transport=transport).run()
    assert result["status"] == "completed_with_resource_stop"
    assert result["terminal"]["stop_reason"] == "semantic_call_budget_exhausted"
    assert result["resources"]["semantic_calls"] == 3
    assert result["resources"]["http_requests"] == 1
    assert transport.executions == []
    assert result["final_prediction"] == pytest.approx(BASELINE)


@pytest.mark.parametrize("change", [
    {"request_budget": 4}, {"semantic_call_budget": 7}, {"planning_horizon": 2},
    {"request_loss_cost": .0}, {"candidate_discovery_version": 2},
])
def test_resume_rejects_changed_policy_or_refunded_budget_before_acquisition(tmp_path, change):
    source, _ = prefix(tmp_path)
    transport = FakeTransport()
    with pytest.raises(ValueError, match="mismatch|incompatible"):
        continued(tmp_path, source, transport=transport, **change)
    assert transport.executions == []


@pytest.mark.parametrize("mutation", ["legacy", "public_input", "accounting", "completed", "evidence"])
def test_resume_rejects_inconsistent_or_unsupported_saved_prefix(tmp_path, mutation):
    source, result = prefix(tmp_path)
    if mutation == "legacy":
        result["configuration"].pop("initialization_contract")
    elif mutation == "public_input":
        result["semantic_calls"][0]["input"]["baseline"]["probabilities"] = [.8, .1, .1]
    elif mutation == "accounting":
        result["resources"]["request_budget_reserved"] = 0
    elif mutation == "completed":
        result["status"] = "completed"
    else:
        result["observations"] = [{"outcome": "supports"}]
    source.write_text(json.dumps(result), encoding="utf-8")
    transport = FakeTransport()
    with pytest.raises(ValueError, match="Resume"):
        continued(tmp_path, source, transport=transport)
    assert transport.executions == []


@pytest.mark.parametrize("clock", [lambda: NOW-timedelta(seconds=1), lambda: NOW+timedelta(days=3)])
def test_resume_refuses_backward_time_or_post_cutoff_network(tmp_path, clock):
    source, _ = prefix(tmp_path)
    transport = FakeTransport()
    with pytest.raises(ValueError, match="time|cutoff"):
        continued(tmp_path, source, transport=transport, clock=clock)
    assert transport.executions == []


def test_cli_exposes_new_defaults_and_passes_explicit_resume_compatibility_flags(tmp_path, monkeypatch):
    from tools import run_footballmind_semantic_case as cli

    case = deepcopy(CASE)
    case["source_profile"].update(seed_urls=[SEED], bootstrap_url=SEED, allowed_hosts=["example.org"])
    (tmp_path/"case.json").write_text(json.dumps(case), encoding="utf-8")
    seen = {}

    def transport(*args, **kwargs):
        seen["projection"] = kwargs["projection_version"]
        return FakeTransport()

    class Runtime:
        def __init__(self, **kwargs):
            seen["runtime"] = kwargs

        def run(self):
            return {"status": "completed", "cycles": [], "initial_prediction": BASELINE,
                    "final_prediction": BASELINE, "resources": {}, "terminal": {}, "failures": []}

    monkeypatch.setattr(cli, "NewsDiscoveryTools", transport)
    monkeypatch.setattr(cli, "SemanticScenarioRuntime", Runtime)
    monkeypatch.setattr(cli, "write_semantic_report", lambda *args: None)
    args = ["run", "--case-dir", str(tmp_path), "--output", str(tmp_path/"out")]
    monkeypatch.setattr(cli.sys, "argv", args)
    cli.main()
    assert seen["projection"] == 8
    assert seen["runtime"]["candidate_discovery_version"] == 2
    assert seen["runtime"]["resume_from"] is None
    monkeypatch.setattr(cli.sys, "argv", args+["--projection-version", "7", "--candidate-discovery-version", "1",
                                             "--resume-from", str(tmp_path/"prior/run.json")])
    cli.main()
    assert seen["projection"] == 7
    assert seen["runtime"]["candidate_discovery_version"] == 1
    assert seen["runtime"]["resume_from"] == tmp_path/"prior/run.json"
