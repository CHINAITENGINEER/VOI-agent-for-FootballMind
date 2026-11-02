"""Deployment boundaries and the match-only operator workflow, without HTTP."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
import requests

from football_agent import footballmind as cli
from football_research.data_adapters.types import ResearchMatch, ResearchOddsSnapshot, json_record


NOW = datetime(2026, 10, 6, 13, tzinfo=timezone.utc)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("Unexpected network access"))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "_now", lambda: NOW)
    # Raw projection itself is tested in prospective/verified-fixture integration;
    # this test boundary supplies a confirmed identity but preserves exact hashes.
    monkeypatch.setattr(cli, "verify_identity_binding", lambda binding, **kw: binding)
    monkeypatch.setattr(cli, "verify_prior_archive", lambda *a, **kw: {"status": "verified"})
    monkeypatch.setattr(cli, "verify_run_identity", lambda *a, **kw: {"status": "verified"})
    for name in ("prior-archive", "identity-archive"):
        (tmp_path / name).mkdir()
    match = ResearchMatch("m1", "league", None, "home", "away", NOW + timedelta(hours=2),
        fixture_available_at=NOW - timedelta(minutes=2), source="spftll", input_manifest=("observation:fixture",),
        status="scheduled", home_team_name="主队", away_team_name="客队")
    quote = ResearchOddsSnapshot("m1", NOW - timedelta(minutes=1), 2., 3., 4., input_manifest=("observation:quote",))
    policy = cli.MarketPriorPolicy(json_record(match), json_record(quote), frozen_at=NOW,
                                   prediction_cutoff=match.kickoff - timedelta(minutes=1))
    binding = {"schema": "raw-backed-fixture-identity-v1", "status": "CONFIRMED", "match_id": "m1",
        "initialization_manifest": policy.manifest, "canonical": {"source_event_id": "uefa:match:1",
            "home_name": "Home United", "away_name": "Away City"},
        "aliases": {"home": ["主队", "Home United"], "away": ["客队", "Away City"]}}
    write(tmp_path / "initializations.json", {"schema": "prospective-text-initializations-v1",
        "created_at": NOW.isoformat(), "selected": [policy.manifest]})
    write(tmp_path / "identity.json", {"schema": "raw-backed-fixture-identity-bundle-v1", "bindings": {"m1": binding}})
    run_config = {"schema": "learned-voi-open-execution-v2", "policy_manifest": policy.manifest,
        "initial_state": {"match_id": "m1"}, "supervisor_metadata": {"verified_fixture_identity": binding}}
    write(tmp_path / "run/configuration.json", run_config)
    write(tmp_path / "run/run.json", {"configuration_digest": cli.digest(run_config)})
    write(tmp_path / "run/prior-bindings.json", {policy.model.model_version: policy.prior_binding()})
    config = {"schema": cli.VERSION, "scope": cli.REAL_SCOPE,
        "initializations": {"path": "initializations.json", "sha256": cli._hash(tmp_path / "initializations.json")},
        "identity_bundle": {"path": "identity.json", "sha256": cli._hash(tmp_path / "identity.json")},
        "prior_archive": "prior-archive", "identity_archive": "identity-archive",
        "runs": [{"match_id": "m1", "path": "run", "run_sha256": cli._hash(tmp_path / "run/run.json"),
            "configuration_sha256": cli._hash(tmp_path / "run/configuration.json"),
            "prior_bindings_sha256": cli._hash(tmp_path / "run/prior-bindings.json")}],
        "collection": {"requests_per_match": 3, "seed": 180, "route": "environment",
            "news_provider": "bing_rss", "allowed_hosts": ["www.uefa.com"], "seed_urls": []},
        "settlement": {"route": "direct", "max_http_requests": 4}}
    write(tmp_path / "config/footballmind.json", config)
    return tmp_path, config, policy, binding


def completed_collection(root, plan_path, output):
    """Persist the collector's contract for CLI selection/error unit tests."""
    output.mkdir()
    folder = output / "match-m1"
    config = cli._json(root / "run/configuration.json")
    config["created_at"] = NOW.isoformat()
    config["supervisor_metadata"]["input_bindings"] = {"initialization_plan_sha256": cli._hash(plan_path)}
    write(folder / "configuration.json", config)
    write(folder / "run.json", {"configuration_digest": cli.digest(config)})
    write(folder / "prior-bindings.json", cli._json(root / "run/prior-bindings.json"))
    write(folder / "analysis.json", {"status": "defer"})
    (folder / "report.md").write_text("No supported new probability update.\n", encoding="utf-8")
    batch = {"http_requests": 2, "independent_labels": 0, "unique_matches": 1,
             "runs": [{"match_id": "m1", "folder": str(folder)}]}
    write(output / "batch-report.json", batch)
    return batch


def test_matches_reports_real_frozen_clock_scope_and_unknown_status(deployment, capsys):
    assert cli.main(["matches", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["network_requests"] == 0
    assert result["scope"] == cli.REAL_SCOPE
    row = result["matches"][0]
    assert row["frozen_at"] == NOW.isoformat()
    assert row["prediction_cutoff"] < row["kickoff"]
    assert row["analyze_available"] is True
    assert row["numeric_voi_status"].startswith("UNKNOWN")


@pytest.mark.parametrize("query", ["m1", "uefa:match:1", "主队vs客队", "主队 vs 客队", "HOME UNITED vs away city", "主队对阵客队"])
def test_exact_identity_aliases_support_chinese_and_multiword_english(deployment, query):
    root = deployment[0]
    data = cli.load_deployment(root / "config/footballmind.json", root=root, now=NOW)
    assert cli.resolve_match(data["matches"], query)["match_id"] == "m1"


@pytest.mark.parametrize("query", ["主队", "客队 vs 主队", "Home vs Away", "Arsenal vs Leeds"])
def test_unknown_reverse_or_partial_names_never_start_collection(deployment, monkeypatch, query):
    root = deployment[0]
    monkeypatch.setattr(cli, "collect_prepared_batch", lambda *a, **k: pytest.fail("Must reject before I/O"))
    assert cli.main(["analyze", "--match", query, "--output", str(root / "out")]) == 2
    assert not (root / "out").exists()


def test_ambiguous_names_require_id(deployment):
    root = deployment[0]
    data = cli.load_deployment(root / "config/footballmind.json", root=root, now=NOW)
    other = deepcopy(data["matches"][0])
    other["match_id"] = "m2"
    with pytest.raises(cli.DeploymentError, match="歧义"):
        cli.resolve_match(data["matches"] + [other], "主队 vs 客队")


@pytest.mark.parametrize("offset", [timedelta(hours=3), timedelta(seconds=-1)])
def test_expired_or_future_plan_rejected_before_creating_output(deployment, monkeypatch, offset):
    root = deployment[0]
    monkeypatch.setattr(cli, "_now", lambda: NOW + offset)
    monkeypatch.setattr(cli, "collect_prepared_batch", lambda *a, **k: pytest.fail("Must reject before I/O"))
    assert cli.main(["analyze", "--match", "m1", "--output", str(root / "out")]) == 2
    assert not (root / "out").exists()


def test_analyze_only_supplies_one_frozen_match_and_deployment_capabilities(deployment, monkeypatch, capsys):
    root, config, policy, binding = deployment
    calls = []
    def collect(plan_path, **kwargs):
        calls.append((cli._json(plan_path), kwargs))
        assert cli._json(kwargs["identity_bundle"])["bindings"] == {"m1": binding}
        assert not kwargs["output"].exists()
        return completed_collection(root, plan_path, kwargs["output"])
    monkeypatch.setattr(cli, "collect_prepared_batch", collect)
    assert cli.main(["analyze", "--match", "主队 vs 客队", "--output", str(root / "out")]) == 0
    plan, kwargs = calls[0]
    assert plan["selected"] == [policy.manifest]
    for key, value in config["collection"].items():
        assert kwargs[key] == value
    assert "question" not in kwargs and "text_model" not in kwargs
    result = cli._json(root / "out/result.json")
    assert result["mode"] == "live_public_acquisition"
    assert result["open_world_policy_benefit_proven"] is False
    assert Path(result["artifacts"]["report_markdown"]).is_file()
    assert "分析报告：" in capsys.readouterr().out
    assert result["run_artifacts_sha256"] == cli._run_files(root / "out/analysis/match-m1")
    assert cli.main(["analyze", "--match", "m1", "--output", str(root / "out")]) == 2
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["initializations.json", "identity.json", "run/configuration.json", "run/run.json", "run/prior-bindings.json"])
def test_modified_bound_inputs_fail_without_replay(deployment, monkeypatch, name):
    root = deployment[0]
    path = root / name
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(cli, "replay_saved_run", lambda *a, **kw: pytest.fail("Unverified inputs"))
    assert cli.main(["replay", "--match", "m1", "--output", str(root / "out")]) == 2
    assert not (root / "out").exists()


def test_real_entry_rejects_synthetic_policy_even_with_rebound_file_hash(deployment):
    root, config, _, _ = deployment
    initial = cli._json(root / "initializations.json")
    initial["selected"][0]["schema"] = "learned-joint-strict-voi-v1"
    write(root / "initializations.json", initial)
    config["initializations"]["sha256"] = cli._hash(root / "initializations.json")
    write(root / "config/footballmind.json", config)
    assert cli.main(["analyze", "--match", "m1", "--output", str(root / "out")]) == 2


def test_replay_stays_allowed_after_cutoff_and_never_collects(deployment, monkeypatch):
    root = deployment[0]
    monkeypatch.setattr(cli, "_now", lambda: NOW + timedelta(days=3))
    monkeypatch.setattr(cli, "collect_prepared_batch", lambda *a, **k: pytest.fail("No collection from replay"))
    calls = []
    def replay(folder, **kwargs):
        calls.append((folder, kwargs))
        kwargs["output"].mkdir()
        write(kwargs["output"] / "analysis.json", {"status": "defer"})
        (kwargs["output"] / "report.md").write_text("Archived analysis", encoding="utf-8")
        return {"replay_network_requests": 0, "identical_replay": True}
    monkeypatch.setattr(cli, "replay_saved_run", replay)
    assert cli.main(["replay", "--match", "m1", "--output", str(root / "out")]) == 0
    result = cli._json(root / "out/result.json")
    assert result["mode"] == "archived_real_replay" and result["new_prediction"] is False
    assert result["match"]["availability"] == "expired"
    assert calls[0][1] == {"output": root / "out/replay", "project_root": root}


def test_settle_passes_raw_archive_to_audit_and_does_not_supply_labels(deployment, monkeypatch):
    root = deployment[0]
    def capture(argv):
        assert argv[0] == "capture-results" and argv[-2:] == ["--max-http-requests", "4"]
        output = Path(argv[argv.index("--output") + 1])
        write(output / "verification.json", {"reports": {"m1": {"status": "PENDING"}}, "settlement_rows": []})
        return 0
    def audit(folders, **kwargs):
        assert folders == [root / "run"]
        assert kwargs["settlements"] == []
        assert kwargs["result_archive"] == root / "out/results"
        return {"dataset": {"audit": {"labels": 0}}, "training_records": []}
    monkeypatch.setattr(cli, "capture_results", capture)
    monkeypatch.setattr(cli, "join_settlements", audit)
    assert cli.main(["settle", "--output", str(root / "out")]) == 0
    result = cli._json(root / "out/result.json")
    assert result["verified_settlement_rows"] == 0
    assert result["fixture_status"] == {"m1": "PENDING"}
    assert result["models_refitted"] is False and result["labels_supplied_by_user"] is False


def test_deployment_paths_cannot_escape_root(deployment):
    root, config, _, _ = deployment
    config["prior_archive"] = ".."
    write(root / "config/footballmind.json", config)
    assert cli.main(["matches"]) == 2


def test_failed_replay_does_not_write_a_success_result(deployment, monkeypatch):
    root = deployment[0]
    monkeypatch.setattr(cli, "replay_saved_run", lambda *a, **kw: {"replay_network_requests": 0, "identical_replay": False})
    assert cli.main(["replay", "--match", "m1", "--output", str(root / "out")]) == 2
    assert not (root / "out/result.json").exists()


def test_generic_replay_rejects_synthetic_base_for_real_market_archives(deployment):
    root = deployment[0]
    with pytest.raises(ValueError, match="cannot override a real prospective run"):
        cli.replay_saved_run(root / "run", output=root / "out", project_root=root,
                             base_model={"path": "never-read", "sha256": "not-a-real-hash"})
    assert not (root / "out").exists()


@pytest.fixture
def mechanism_deployment(deployment):
    root, config, _, _ = deployment
    paths = ["mechanism/run/run.json", "mechanism/run/configuration.json", "mechanism/run/text-response-model.json",
             "mechanism/project/var/parallel/agent/round175/frozen-next-period.json"]
    for path in paths:
        write(root / path, {})
    write(root / "mechanism/base.json", {"frozen": True})
    ref = lambda path: {"path": path, "sha256": cli._hash(root / path)}
    write(root / "mechanism/recovery.json", {"status": "passed", "scope": "synthetic_mechanism_only",
        "base_model_sha256": cli._hash(root / "mechanism/base.json"),
        "cases": {"two-step": {"base_manifest_exact": True}}})
    config["mechanism"] = {"scope": "synthetic_mechanism_only", "base_model": ref("mechanism/base.json"),
        "recovery_report": ref("mechanism/recovery.json"), "cases": {"two-step": {
            "run_path": "mechanism/run", "project_root": "mechanism/project", "files": [ref(p) for p in paths]}}}
    write(root / "config/footballmind.json", config)
    return root, config


def test_mechanism_is_separate_from_real_initialization_and_never_fits(mechanism_deployment, monkeypatch):
    root, config = mechanism_deployment
    (root / "initializations.json").unlink()
    def replay(folder, **kwargs):
        assert folder == root / "mechanism/run"
        assert kwargs["project_root"] == root / "mechanism/project"
        assert kwargs["base_model"] == {"path": str(root / "mechanism/base.json"),
                                       "sha256": config["mechanism"]["base_model"]["sha256"]}
        write(kwargs["output"] / "run.json", {"attempts": [{}, {}], "initial_probabilities": [1 / 3] * 3,
            "final_probabilities": [.8, .1, .1], "reserved_cost": {"requests": 2}, "terminal_selection": {}})
        return {"identical_replay": True, "replay_network_requests": 0}
    monkeypatch.setattr(cli, "replay_saved_run", replay)
    assert cli.main(["mechanism", "--case", "two-step", "--output", str(root / "out")]) == 0
    result = cli._json(root / "out/result.json")
    assert result["scope"] == "synthetic_mechanism_only"
    assert result["models_refitted"] is False and result["real_football_benefit_proven"] is False
    assert cli._json(root / "out/invocation.json")["scope"] == "synthetic_mechanism_only"


@pytest.mark.parametrize("path", ["mechanism/base.json", "mechanism/recovery.json", "mechanism/run/run.json"])
def test_modified_mechanism_artifacts_rejected_before_replay(mechanism_deployment, monkeypatch, path):
    root, _ = mechanism_deployment
    (root / path).write_bytes((root / path).read_bytes() + b" ")
    monkeypatch.setattr(cli, "replay_saved_run", lambda *a, **kw: pytest.fail("No unbound mechanism"))
    assert cli.main(["mechanism", "--case", "two-step", "--output", str(root / "out")]) == 2
    assert not (root / "out").exists()


@pytest.mark.parametrize("bad", ["wrong_hash", "wrong_origin"])
def test_generic_synthetic_replay_requires_provenance_and_pinned_base(deployment, bad):
    root = deployment[0]
    write(root / "var/parallel/agent/round175/frozen-next-period.json", {"old": "synthetic placeholder"})
    base = root / "base.json"
    write(base, {"not": "an injected live object"})
    config = {"schema": "learned-voi-open-execution-v3", "policy_manifest": {"base_policy": {"schema": "learned-joint-strict-voi-v1"}},
        "provider_version": "synthetic-archive-discovery", "text_response_model": {"evidence_origin": "constructed_mechanism"},
        "supervisor_metadata": {"input_bindings": {"frozen_model_config_sha256":
            cli._hash(root / "var/parallel/agent/round175/frozen-next-period.json")}}}
    if bad == "wrong_origin":
        config["text_response_model"]["evidence_origin"] = "real_news"
    write(root / "synthetic/configuration.json", config)
    write(root / "synthetic/run.json", {"configuration_digest": cli.digest(config)})
    binding = {"path": str(base), "sha256": "mismatch" if bad == "wrong_hash" else cli._hash(base)}
    with pytest.raises(ValueError, match="SHA-256 differs|restricted to"):
        cli.replay_saved_run(root / "synthetic", output=root / "out", project_root=root, base_model=binding)
    assert not (root / "out").exists()


def test_analyze_incomplete_collection_preserves_partial_output_without_success(deployment, monkeypatch):
    root = deployment[0]
    calls = []
    def expired_after_preflight(plan, **kwargs):
        calls.append(plan)
        raise ValueError("Initialization is from the future or its prematch horizon has expired")
    monkeypatch.setattr(cli, "collect_prepared_batch", expired_after_preflight)
    args = ["analyze", "--match", "m1", "--output", str(root / "out")]
    assert cli.main(args) == 2
    assert (root / "out/initializations.json").is_file()
    assert not (root / "out/result.json").exists()
    assert cli.main(args) == 2
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["foreign_deployment", "changed_plan", "changed_run", "not_analyze"])
def test_explicit_analyze_run_rejects_unbound_or_wrong_wrapper(deployment, monkeypatch, kind):
    root = deployment[0]
    monkeypatch.setattr(cli, "collect_prepared_batch", lambda plan, **kw: completed_collection(root, plan, kw["output"]))
    assert cli.main(["analyze", "--match", "m1", "--output", str(root / "collected")]) == 0
    path = root / "collected" / ({"foreign_deployment": "deployment.json", "changed_plan": "initializations.json",
                                "changed_run": "analysis/match-m1/run.json", "not_analyze": "result.json"}[kind])
    item = cli._json(path)
    if kind == "not_analyze":
        item["command"] = "mechanism"
    elif kind == "changed_plan":
        item["selected"] = []
    elif kind == "changed_run":
        item["configuration_digest"] = "changed"
    else:
        item["collection"]["seed"] = 999
    write(path, item)
    monkeypatch.setattr(cli, "replay_saved_run", lambda *a, **kw: pytest.fail("Cannot replay unbound new run"))
    monkeypatch.setattr(cli, "capture_results", lambda *a, **kw: pytest.fail("Cannot settle unbound new run"))
    for command in ("replay", "settle"):
        assert cli.main([command, "--run", str(root / "collected"), "--output", str(root / command)]) == 2
        assert not (root / command).exists()


def test_replay_match_and_run_are_mutually_exclusive(deployment):
    with pytest.raises(SystemExit) as raised:
        cli.main(["replay", "--match", "m1", "--run", "some-output", "--output", "not-created"])
    assert raised.value.code == 2


def test_cli_actual_collector_accepts_selected_plan_and_new_run_replays_and_settles(tmp_path, monkeypatch, capsys):
    """Use real source validation, collector, runtime, report and replay.

    The existing verified-collection fixture creates independent original
    market and official identity bytes. Only the HTTP execution boundary is
    replaced; a request that escapes that boundary makes the test fail.
    """
    import runpy
    from football_agent.news_discovery import NewsDiscoveryTools
    from football_research.data_adapters import prospective_text
    helper = Path(__file__).resolve().parents[1] / "research/acquisition/test_verified_collection.py"
    inputs = runpy.run_path(str(helper))["verified_inputs"].__wrapped__(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **kw: pytest.fail("Actual network is forbidden"))
    assert cli.collect_prepared_batch is prospective_text.collect_prepared_batch
    binding, mid = inputs["binding"], inputs["binding"]["match_id"]
    write(tmp_path / "identity-bundle.json", {"schema": "raw-backed-fixture-identity-bundle-v1", "bindings": {mid: binding}})
    ref = lambda p: {"path": p, "sha256": cli._hash(tmp_path / p)}
    config = {"schema": cli.VERSION, "scope": cli.REAL_SCOPE, "initializations": ref("plan.json"),
        "identity_bundle": ref("identity-bundle.json"), "prior_archive": "market", "identity_archive": "identity",
        # Explicit --run must not fall back to the deployment's old archive.
        "runs": [{"match_id": mid, "path": "old-archive-not-selected"}],
        "collection": {"requests_per_match": 2, "seed": 180, "route": "environment",
            "news_provider": "bing_rss", "allowed_hosts": ["www.uefa.com"], "seed_urls": []},
        "settlement": {"route": "direct", "max_http_requests": 4}}
    write(tmp_path / "config/footballmind.json", config)
    calls = []
    def archived_response(self, prepared, *, run_id, request_id):
        calls.append(deepcopy(prepared))
        now = datetime.now(timezone.utc)
        body = b"<rss><channel></channel></rss>" if prepared["tool"] == "search_web" else inputs["body"]
        observation = self.archive.save(prepared["source"], prepared["endpoint"], prepared["params"], body,
            status=200, observed_at=now, context={"requested_at": now.isoformat()})
        return {"document": self.project(observation, prepared), "observation": observation.model_dump(mode="json"),
                "http_attempts": 1, "money": None, "money_status": "not_measured", "latency_seconds": .01}
    monkeypatch.setattr(NewsDiscoveryTools, "execute", archived_response)
    output = tmp_path / "collected"
    assert cli.main(["analyze", "--match", mid, "--output", str(output)]) == 0
    result = cli._json(output / "result.json")
    assert cli._json(output / "initializations.json")["selected"] == inputs["plan"]["selected"]
    assert cli._json(output / "identity-bundle.json")["bindings"] == {mid: binding}
    assert result["batch_report"]["unique_matches"] == 1
    assert result["batch_report"]["http_requests"] == len(calls) == 2
    assert result["run_artifacts_sha256"] == cli._run_files(output / "analysis/match-101")
    assert str(output / "analysis/match-101/report.md") in capsys.readouterr().out
    assert cli.main(["replay", "--run", str(output), "--output", str(tmp_path / "replayed")]) == 0
    replay = cli._json(tmp_path / "replayed/result.json")
    assert replay["source_run"] == str(output / "analysis/match-101")
    assert replay["run_selection"] == "explicit_analyze_output"
    assert replay["acceptance"]["identical_replay"] and replay["network_requests"] == 0
    assert replay["acceptance"]["raw_responses_reprojected"] == 2
    assert len(calls) == 2
    # Round181 wrappers lacked the explicit new sidecar hashes. Their existing
    # full runtime/plan/raw bindings remain sufficient for verified replay.
    del result["run_artifacts_sha256"]
    write(output / "result.json", result)
    assert cli.main(["replay", "--run", str(output), "--output", str(tmp_path / "legacy-replayed")]) == 0
    def capture(argv):
        assert argv[0] == "capture-results"
        selected_bundle = cli._json(Path(argv[argv.index("--identity-bundle") + 1]))
        assert selected_bundle["bindings"] == {mid: binding}
        target = Path(argv[argv.index("--output") + 1])
        write(target / "verification.json", {"reports": {mid: {"status": "PENDING"}}, "settlement_rows": []})
        return 0
    monkeypatch.setattr(cli, "capture_results", capture)
    assert cli.main(["settle", "--run", str(output), "--output", str(tmp_path / "settled")]) == 0
    settled = cli._json(tmp_path / "settled/result.json")
    assert settled["source_runs"] == [str(output / "analysis/match-101")]
    assert settled["fixture_status"] == {mid: "PENDING"}
    assert settled["training_records"] == 0 and settled["models_refitted"] is False
    audit = cli._json(tmp_path / "settled/settlement-audit.json")
    assert audit["independent_result_verification"][0]["match_id"] == mid
    assert len(calls) == 2
