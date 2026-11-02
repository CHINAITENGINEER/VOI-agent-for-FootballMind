"""CLI wiring uses declared sources and rejects constructed live text models.

The runtime is spied here to test construction before network I/O. Text models
are deliberately stubbed; these checks do not assert real training provenance.
"""
from copy import deepcopy
from hashlib import sha256
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from football_agent import learned_voi_cli as cli
from football_agent.inquiry_transport import RoutedOpenWebTools
from football_agent.source_recovery import SourceRoutedOpenWebTools
from football_research.acquisition import text_history


@pytest.fixture
def launching(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze.json"
    action = {"action_id": "xg_view", "group_ids": ["xg"], "available": True,
              "cost_status": "simulated", "cost": {"money": .03, "requests": 1.},
              "source": "fixture_archive", "tool": "feature", "version": "test"}
    freeze.write_text(json.dumps({"actions": [action], "lambda_money": 2.,
                                 "limits": {"money": .05, "requests": 2.}}), encoding="utf-8")
    row = {"match_id": "fixture-test", "competition_id": "EPL", "home_team_id": "home",
           "away_team_id": "away", "kickoff": "2026-10-10T11:30:00Z",
           "prediction_cutoff": "2026-10-09T11:30:00Z",
           "features_available_at": "2026-10-06T08:00:00Z", "information_mode": "operational"}
    report = {"match_id": row["match_id"], "match": "Arsenal vs Leeds",
              "kickoff": row["kickoff"], "prediction_cutoff": row["prediction_cutoff"],
              "export_manifest_sha256": "fixture-export-sha"}
    captured = {}
    monkeypatch.setattr(cli, "prepare_fixture", lambda *_args, **_kw: (deepcopy(row), ["Arsenal", "Leeds"], report))
    monkeypatch.setattr(cli, "load_frozen_models", lambda *_args, **_kw: (None, SimpleNamespace(model_version="frozen-base")))
    monkeypatch.setattr(cli, "LearnedJointVOIPolicy", lambda model, **_kw: SimpleNamespace(model=model))

    class RuntimeSpy:
        def __init__(self, policy, **kwargs):
            captured["policy"] = policy

        def run(self, initial, **kwargs):
            captured.update(initial=initial, **kwargs)
            return {"run_id": kwargs["run_id"], "attempts": [], "exploration_actions": 0,
                    "documents": [], "measured_resources": {"requests": 0},
                    "terminal_selection": {"decision": "defer", "reason": "test_spy"}}

    monkeypatch.setattr(cli, "LearnedVOIRuntime", RuntimeSpy)
    monkeypatch.setattr("football_agent.match_analysis_report.write_analysis_report",
                        lambda *_args, **_kw: {"status": "test_spy"})
    args = {"bundle": "fixture-export", "output": tmp_path / "run", "freeze_path": freeze,
            "match": "Arsenal vs Leeds", "exploration_actions": 0}
    return args, captured, row


def test_default_route_keeps_original_tool_and_no_source_route_configuration(launching):
    args, captured, _ = launching
    cli.launch(**args)
    assert type(captured["execute"].tools) is RoutedOpenWebTools
    config = captured["run_metadata"]["transport"]
    assert "source_route_policy" not in config
    assert config["transport_policy"]["route"] == "environment"
    assert captured["propose"].numeric_actions


def test_explicit_article_evidence_version_is_bound_to_runtime(launching):
    from football_agent.match_evidence import ARTICLES_VERSION
    args, captured, _ = launching
    cli.launch(**args, evidence_version=ARTICLES_VERSION)
    assert captured["evidence_version"] == ARTICLES_VERSION
    assert captured["evidence_match"]["match_id"] == "fixture-test"


@pytest.mark.parametrize("options", [
    {"evidence_version": "unsupported"},
    {"evidence_version": "match-evidence-extraction-v3", "source_profile": "understat"},
])
def test_invalid_evidence_mode_rejected_before_loading_or_io(launching, monkeypatch, options):
    args, captured, _ = launching
    monkeypatch.setattr(cli, "load_frozen_models", lambda *_a, **_kw: pytest.fail("Invalid evidence mode reached model loading"))
    with pytest.raises(ValueError):
        cli.launch(**args, **options)
    assert not captured and not args["output"].exists()


def test_cli_forwards_explicit_article_evidence_version(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "launch", lambda **kw: calls.append(kw) or {})
    cli.main(["--evidence-version", "match-evidence-extraction-v3"])
    assert calls[0]["evidence_version"] == "match-evidence-extraction-v3"


def test_source_routes_bind_actual_transport_configuration_before_runtime(launching):
    args, captured, _ = launching
    cli.launch(**args, source_routes={"www.bing.com": "direct"})
    tools = captured["execute"].tools
    assert type(tools) is SourceRoutedOpenWebTools
    assert captured["run_metadata"]["transport"] == tools.configuration()
    assert tools.configuration()["source_route_policy"]["route_overrides"] == {"www.bing.com": "direct"}
    saved = json.loads((args["output"] / "source-configuration.json").read_text(encoding="utf-8"))
    assert saved["transport"] == tools.configuration()
    assert captured["provider_version"].endswith(cli.digest(tools.configuration()))


def test_unknown_source_route_is_rejected_before_runtime_and_io(launching):
    args, captured, _ = launching
    with pytest.raises(ValueError, match="registered"):
        cli.launch(**args, source_routes={"unregistered.example": "direct"})
    assert not captured and not args["output"].exists()


@pytest.mark.parametrize("option", ["missing-equals", "www.bing.com=anything", "https://www.bing.com=direct", "=direct"])
def test_cli_rejects_malformed_source_route_before_launch(monkeypatch, option):
    monkeypatch.setattr(cli, "launch", lambda **_: pytest.fail("Invalid CLI option reached launch"))
    with pytest.raises(SystemExit) as error:
        cli.main(["--source-route", option])
    assert error.value.code == 2


def test_repeated_cli_options_forward_distinct_hosts_and_reject_duplicate(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "launch", lambda **kwargs: received.append(kwargs) or {})
    cli.main(["--source-route", "www.bing.com=direct", "--source-route", "www.leedsunited.com=environment",
              "--text-model", "frozen-text.json"])
    assert received[0]["source_routes"] == {"www.bing.com": "direct", "www.leedsunited.com": "environment"}
    assert str(received[0]["text_model_path"]) == "frozen-text.json"
    with pytest.raises(SystemExit) as error:
        cli.main(["--source-route", "www.bing.com=direct", "--source-route", "www.bing.com=environment"])
    assert error.value.code == 2 and len(received) == 1


def test_constructed_text_model_rejected_before_runtime(launching, monkeypatch):
    args, captured, _ = launching
    monkeypatch.setattr(text_history, "load_text_model", lambda _: SimpleNamespace(metadata={"evidence_origin": "constructed_mechanism"}))
    with pytest.raises(ValueError, match="real_observational"):
        cli.launch(**args, text_model_path="constructed.json")
    assert not captured and not args["output"].exists()


def test_historical_initialization_cannot_enter_operational_text_mode(launching, monkeypatch):
    args, captured, row = launching
    row["information_mode"] = "historical_reconstruction"
    monkeypatch.setattr(text_history, "load_text_model", lambda _: SimpleNamespace(metadata={"evidence_origin": "real_observational"}))
    with pytest.raises(ValueError, match="historical_reconstruction"):
        cli.launch(**args, text_model_path="not-a-real-model-stub.json")
    assert not captured and not args["output"].exists()


def test_text_mode_requires_evidence_source_profile(launching, monkeypatch):
    args, captured, _ = launching
    monkeypatch.setattr(text_history, "load_text_model", lambda _: pytest.fail("Wrong source profile reached text model loader"))
    with pytest.raises(ValueError, match="club-news"):
        cli.launch(**args, source_profile="understat", text_model_path="unused.json")
    assert not captured and not args["output"].exists()


def test_text_mode_disables_numeric_acquisition_and_binds_artifact(launching, monkeypatch, tmp_path):
    args, captured, _ = launching
    path = tmp_path / "stub-text.json"
    path.write_text("unit-test-only model stub", encoding="utf-8")
    model = SimpleNamespace(metadata={"evidence_origin": "real_observational"}, model_version="test-text-version")
    monkeypatch.setattr(text_history, "load_text_model", lambda _: model)
    # This is only constructor wiring; real policy/model behavior is tested by
    # the response-model and runtime suites, using their explicit provenance.
    module = ModuleType("football_agent.text_voi_policy")
    module.TextVOIPolicy = lambda base, text: SimpleNamespace(base=base, text_model=text)
    monkeypatch.setitem(sys.modules, "football_agent.text_voi_policy", module)
    cli.launch(**args, text_model_path=path)
    assert captured["policy"].text_model is model
    assert captured["propose"].numeric_actions == []
    assert captured["execute"].numeric_ids == set()
    assert captured["evidence_match"]["match_id"] == "fixture-test"
    bindings = captured["run_metadata"]["input_bindings"]
    assert bindings["text_response_model_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert bindings["text_response_model_version"] == "test-text-version"
    assert bindings["numeric_acquisition_disabled"] is True
    assert bindings["initialization_mode"] == "operational"


def test_discovery_model_requires_its_bound_text_model_before_loading_or_io(launching, monkeypatch):
    args, captured, _ = launching
    monkeypatch.setattr(cli, "load_frozen_models", lambda *_a, **_kw: pytest.fail("Invalid mode reached model loading"))
    with pytest.raises(ValueError, match="requires its jointly bound text response model"):
        cli.launch(**args, discovery_model_path="unbound.json")
    assert not captured and not args["output"].exists()


def test_discovery_cli_forwards_both_frozen_models(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "launch", lambda **kw: calls.append(kw) or {})
    cli.main(["--text-model", "text.json", "--discovery-model", "discovery.json"])
    assert str(calls[0]["text_model_path"]) == "text.json"
    assert str(calls[0]["discovery_model_path"]) == "discovery.json"


@pytest.mark.parametrize("paths", [
    {"text_risk_model_path": "risk.json"},
    {"text_risk_model_path": "risk.json", "text_model_path": "text.json", "discovery_model_path": "discovery.json"},
])
def test_invalid_empirical_mode_rejected_before_model_loading(launching, monkeypatch, paths):
    args, captured, _ = launching
    monkeypatch.setattr(cli, "load_frozen_models", lambda *_a, **_kw: pytest.fail("Invalid mode reached loader"))
    with pytest.raises(ValueError, match="empirical text risk model"):
        cli.launch(**args, **paths)
    assert not captured and not args["output"].exists()


def test_synthetic_risk_supervision_cannot_enter_live_execution(launching, monkeypatch):
    from football_research.acquisition import neighborhood_text_risk
    args, captured, _ = launching
    monkeypatch.setattr(text_history, "load_text_model", lambda _: SimpleNamespace(metadata={"evidence_origin": "real_observational"}))
    monkeypatch.setattr(neighborhood_text_risk, "load_text_risk_model", lambda *a, **kw: SimpleNamespace(manifest={"evidence_origin": "constructed_mechanism"}))
    with pytest.raises(ValueError, match="real_observational supervision"):
        cli.launch(**args, text_model_path="text.json", text_risk_model_path="risk.json")
    assert not captured and not args["output"].exists()


def test_empirical_cli_forwards_frozen_artifact(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "launch", lambda **kw: calls.append(kw) or {})
    cli.main(["--text-model", "text.json", "--text-risk-model", "risk.json"])
    assert str(calls[0]["text_risk_model_path"]) == "risk.json"
