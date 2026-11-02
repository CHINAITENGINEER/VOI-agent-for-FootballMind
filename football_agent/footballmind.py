"""Match-only entry point for FootballMind historical and live backends.

Real collection uses the observed market prior and bounded UNKNOWN-VoI
exploration. Archived replay is a different, network-free operation. Source
capabilities and budgets belong to the deployment, not a user's research topic.
Historical analysis actively reruns a frozen learned policy on archived data;
it does not require a future result or a live deployment to be available.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import unicodedata

from football_research.acquisition.state import parse_time
from football_research.data_adapters.prospective_text import (
    MarketPriorPolicy, collect_prepared_batch, join_settlements, verify_prior_archive,
    verify_run_identity,
)
from football_research.data_adapters.verified_fixtures import (
    main as capture_results, verify_identity_binding,
)
from .learned_voi_replay import replay_saved_run
from .learned_voi_runtime import digest

ROOT = Path(__file__).resolve().parents[1]
VERSION = "footballmind-deployment-v1"
RESULT_VERSION = "footballmind-command-result-v1"
REAL_SCOPE = "verified_three_fixture_deployment_not_arbitrary_matches"
MECHANISM_CASES = ("two-step", "one-step", "one-request", "actual-no-link")


class DeploymentError(ValueError):
    """A deployment or command precondition failed before external execution."""


def _now():
    return datetime.now(timezone.utc)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    # No overwrite, including an incomplete artifact left by an interrupted run.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _path(root, relative, *, directory=False):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DeploymentError("部署路径必须是项目目录内的相对路径。")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise DeploymentError("部署路径越出项目目录。")
    if not (result.is_dir() if directory else result.is_file()):
        raise DeploymentError(f"部署文件或目录缺失：{relative}")
    return result


def _pinned(root, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise DeploymentError("部署文件必须同时绑定 path 和 SHA-256。")
    path = _path(root, reference["path"])
    if _hash(path) != reference["sha256"]:
        raise DeploymentError(f"部署文件 SHA-256 不一致：{reference['path']}")
    return path


def load_deployment(path, *, root, now):
    """Validate the declared scope, frozen initial states and raw identities."""
    path, root = Path(path), Path(root).resolve()
    config = _json(path)
    if config.get("schema") != VERSION or config.get("scope") != REAL_SCOPE:
        raise DeploymentError("不支持的部署版本或比赛范围；不能自动换用其他模型。")
    initial_path = _pinned(root, config["initializations"])
    identity_path = _pinned(root, config["identity_bundle"])
    initial, identity = _json(initial_path), _json(identity_path)
    if initial.get("schema") != "prospective-text-initializations-v1":
        raise DeploymentError("赛前初态文件版本不受支持。")
    if identity.get("schema") != "raw-backed-fixture-identity-bundle-v1":
        raise DeploymentError("比赛身份归档版本不受支持。")
    prior_archive = _path(root, config["prior_archive"], directory=True)
    identity_archive = _path(root, config["identity_archive"], directory=True)
    selected, bindings = initial.get("selected", []), identity.get("bindings", {})
    if not isinstance(selected, list) or not selected or not isinstance(bindings, dict):
        raise DeploymentError("部署缺少赛前初态或比赛身份。")
    matches = []
    for manifest in selected:
        # This prevents a synthetic response model from entering the real CLI.
        policy = MarketPriorPolicy.from_manifest(manifest)
        mid = policy.model.fixture["match_id"]
        binding = bindings.get(mid)
        if (not isinstance(binding, dict) or binding.get("match_id") != mid
                or binding.get("initialization_manifest") != manifest):
            raise DeploymentError(f"比赛 {mid} 的身份与完整赛前初态不一致。")
        checked = verify_identity_binding(binding, archive_root=identity_archive, as_of=now)
        if checked.get("status") != "CONFIRMED":
            raise DeploymentError(f"比赛 {mid} 尚未独立确认，不能运行。")
        verify_prior_archive(manifest, prior_archive)
        matches.append({"match_id": mid, "manifest": manifest, "binding": binding})
    ids = [item["match_id"] for item in matches]
    runs = config.get("runs", [])
    if (len(set(ids)) != len(ids) or set(ids) != set(bindings)
            or not isinstance(runs, list) or len(runs) != len(ids)
            or {row.get("match_id") for row in runs} != set(ids)):
        raise DeploymentError("部署比赛、身份和归档记录必须一一对应，不能重复计数。")
    collection = config["collection"]
    settlement = config["settlement"]
    if (type(collection.get("requests_per_match")) is not int
            or not 1 <= collection["requests_per_match"] <= 4
            or type(collection.get("seed")) is not int
            or collection.get("route") not in {"environment", "direct"}
            or collection.get("news_provider") not in {"bing_rss", "bing_news_rss"}
            or not isinstance(collection.get("allowed_hosts"), list)
            or not collection["allowed_hosts"] or not isinstance(collection.get("seed_urls"), list)
            or settlement.get("route") not in {"environment", "direct"}
            or type(settlement.get("max_http_requests")) is not int
            or not len(ids) + 1 <= settlement["max_http_requests"] <= 32):
        raise DeploymentError("部署的来源、请求预算或网络路由无效。")
    return {"config": config, "config_path": path, "config_sha256": _hash(path),
            "root": root, "matches": matches, "initial": initial, "identity": identity,
            "prior_archive": prior_archive, "identity_archive": identity_archive}


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def resolve_match(matches, query):
    """Exact identity or ordered alias pair; never fuzzy fixture selection."""
    needle, found = _normal(query), []
    for item in matches:
        binding = item["binding"]
        names = {item["match_id"], binding["canonical"]["source_event_id"]}
        for home in binding["aliases"]["home"]:
            for away in binding["aliases"]["away"]:
                for separator in (" vs ", "vs", " v ", " 对阵 ", "对阵", " - ", "-", " ", " / "):
                    names.add(home + separator + away)
        if needle in {_normal(name) for name in names}:
            found.append(item)
    if len(found) > 1:
        raise DeploymentError("比赛名称存在歧义，请使用完整比赛 ID：" + "、".join(x["match_id"] for x in found))
    if not found:
        raise DeploymentError("未找到已部署比赛。请运行 matches，使用完整 ID 或精确的主队 vs 客队；当前不支持任意比赛。")
    return found[0]


def _match_view(item, now):
    prior = item["manifest"]["market_initialization"]
    fixture, canonical = prior["fixture"], item["binding"]["canonical"]
    frozen, cutoff = parse_time(prior["frozen_at"]), parse_time(prior["prediction_cutoff"])
    return {"match_id": item["match_id"], "home": fixture["home_team_name"],
            "away": fixture["away_team_name"], "home_english": canonical["home_name"],
            "away_english": canonical["away_name"], "kickoff": fixture["kickoff"],
            "prediction_cutoff": prior["prediction_cutoff"], "frozen_at": prior["frozen_at"],
            "analyze_available": frozen <= now < cutoff,
            "availability": "not_yet_available" if now < frozen else "expired" if now >= cutoff else "prematch",
            "initial_probabilities": prior["probabilities"],
            "probability_basis": "observed_market_snapshot", "scope": REAL_SCOPE,
            "numeric_voi_status": "UNKNOWN_pending_real_response_training"}


def _saved_run(deployment, item):
    entry = next(row for row in deployment["config"]["runs"] if row["match_id"] == item["match_id"])
    folder = _path(deployment["root"], entry["path"], directory=True)
    for name, key in (("run.json", "run_sha256"), ("configuration.json", "configuration_sha256"),
                      ("prior-bindings.json", "prior_bindings_sha256")):
        if _hash(folder / name) != entry.get(key):
            raise DeploymentError(f"归档文件 SHA-256 不一致：{entry['path']}/{name}")
    _check_run_binding(folder, item)
    return folder


def _check_run_binding(folder, item):
    run, config = _json(folder / "run.json"), _json(folder / "configuration.json")
    if (run.get("configuration_digest") != digest(config)
            or config.get("policy_manifest") != item["manifest"]
            or config.get("initial_state", {}).get("match_id") != item["match_id"]
            or config.get("supervisor_metadata", {}).get("verified_fixture_identity") != item["binding"]):
        raise DeploymentError("归档运行与部署初态、原始身份或运行配置不一致。")
    return config


def _run_files(folder):
    return {name: _hash(folder / name) for name in ("run.json", "configuration.json", "prior-bindings.json")}


def _report_paths(folder):
    paths = {"report_markdown": folder / "report.md", "analysis_json": folder / "analysis.json"}
    if any(not path.is_file() for path in paths.values()):
        raise DeploymentError("运行未生成完整分析报告；保留当前目录，不能标记为完成。")
    return {key: str(path.resolve()) for key, path in paths.items()}


def _analyze_archive(deployment, output):
    """Select exactly a completed analyze output, never an implicit latest run.

    New outputs bind the run files explicitly. Round181 outputs additionally
    undergo the existing wrapper/configuration/raw-prior/identity checks; they
    need not be rewritten to be selected. Absolute paths inside old reports are
    descriptive only, so a copied archive remains portable.
    """
    output = Path(output).resolve()
    result, invocation = _json(output / "result.json"), _json(output / "invocation.json")
    if (result.get("schema") != RESULT_VERSION or result.get("command") != "analyze"
            or result.get("mode") != "live_public_acquisition" or result.get("scope") != REAL_SCOPE
            or invocation.get("schema") != RESULT_VERSION or invocation.get("command") != "analyze"
            or invocation.get("scope") != REAL_SCOPE
            or invocation.get("deployment_sha256") != deployment["config_sha256"]
            or _json(output / "deployment.json") != deployment["config"]):
        raise DeploymentError("--run 必须指向同一部署下已完成的 analyze 输出目录；不能选择回放或合成记录。")
    item = resolve_match(deployment["matches"], result["match"]["match_id"])
    started = parse_time(invocation["started_at"])
    if result["match"] != _match_view(item, started) or not result["match"]["analyze_available"]:
        raise DeploymentError("新运行的比赛、原始初态时间或赛前窗口不一致。")
    plan, bundle = _json(output / "initializations.json"), _json(output / "identity-bundle.json")
    if (plan.get("schema") != "prospective-text-initializations-v1"
            or plan.get("selected") != [item["manifest"]]
            or plan.get("parent_plan_sha256") != deployment["config"]["initializations"]["sha256"]
            or bundle.get("bindings") != {item["match_id"]: item["binding"]}):
        raise DeploymentError("新运行的单场计划或身份快照与部署不一致。")
    batch = _json(output / "analysis/batch-report.json")
    if (batch != result.get("batch_report") or batch.get("unique_matches") != 1
            or len(batch.get("runs", [])) != 1 or batch["runs"][0].get("match_id") != item["match_id"]):
        raise DeploymentError("新运行未完成单场采集或批次结果与入口记录不一致。")
    folder = (output / "analysis" / ("match-" + item["match_id"].rsplit(":", 1)[-1])).resolve()
    if not folder.is_relative_to(output):
        raise DeploymentError("归档路径越出所选运行目录。")
    config = _check_run_binding(folder, item)
    created = parse_time(config["created_at"])
    if (not started <= created < parse_time(item["manifest"]["market_initialization"]["prediction_cutoff"])
            or config["supervisor_metadata"]["input_bindings"].get("initialization_plan_sha256")
            != _hash(output / "initializations.json")):
        raise DeploymentError("新运行的实际创建时间或计划原始绑定不一致。")
    if "run_artifacts_sha256" in result and result["run_artifacts_sha256"] != _run_files(folder):
        raise DeploymentError("新运行的归档文件 SHA-256 不一致。")
    verify_prior_archive(item["manifest"], folder / "prior-archive")
    verify_run_identity(folder, config)
    return item, folder


def _new_output(output, deployment, command, now, *, scope=REAL_SCOPE):
    output = Path(output).resolve()
    if output.exists():
        raise DeploymentError(f"输出已存在，必须使用新的目录；已有运行不会覆盖：{output}")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "deployment.json", deployment["config"])
    _write(output / "invocation.json", {"schema": RESULT_VERSION, "command": command,
        "started_at": now.isoformat(), "deployment_sha256": deployment["config_sha256"],
        "scope": scope, "output": str(output)})
    return output


def _display_match(row):
    print(f"{row['match_id']}  {row['home']} vs {row['away']}  ({row['home_english']} vs {row['away_english']})")
    print(f"  初态冻结：{row['frozen_at']}；开球：{row['kickoff']}；赛前截止：{row['prediction_cutoff']}")
    print("  可发起赛前采集。" if row["analyze_available"] else "  当前不可发起赛前采集；可使用 replay 查看已存归档。")


def _analyze(deployment, item, output, now):
    row = _match_view(item, now)
    if not row["analyze_available"]:
        raise DeploymentError("赛前计划已过期或初态尚未可用；请使用 replay，不能把归档回放当作新预测。")
    output = _new_output(output, deployment, "analyze", now)
    plan = {"schema": "prospective-text-initializations-v1", "selected": [item["manifest"]],
            "created_at": deployment["initial"].get("created_at"),
            "parent_plan_sha256": deployment["config"]["initializations"]["sha256"]}
    bundle = {**deployment["identity"], "bindings": {item["match_id"]: item["binding"]}}
    _write(output / "initializations.json", plan)
    _write(output / "identity-bundle.json", bundle)
    _display_match(row)
    print("正在实际联网获取信息；当前采用未知 VoI 下的有限探索，未部署合成训练模型。")
    report = collect_prepared_batch(output / "initializations.json", output=output / "analysis",
        archive_root=deployment["prior_archive"], identity_bundle=output / "identity-bundle.json",
        identity_archive=deployment["identity_archive"], **deployment["config"]["collection"])
    if (report.get("unique_matches") != 1 or len(report.get("runs", [])) != 1
            or report["runs"][0].get("match_id") != item["match_id"]):
        raise DeploymentError("采集器未完成所选单场比赛，不能生成成功结果。")
    folder = output / "analysis" / ("match-" + item["match_id"].rsplit(":", 1)[-1])
    _check_run_binding(folder, item)
    return {"schema": RESULT_VERSION, "command": "analyze", "mode": "live_public_acquisition",
            "match": row, "scope": REAL_SCOPE, "batch_report": report,
            "run_directory": str(folder), "artifacts": _report_paths(folder),
            "run_artifacts_sha256": _run_files(folder),
            "numeric_voi_status": row["numeric_voi_status"], "open_world_policy_benefit_proven": False}


def _replay(deployment, item, output, now, *, folder=None):
    explicit_run = folder is not None
    folder = _saved_run(deployment, item) if folder is None else folder
    output = _new_output(output, deployment, "replay", now)
    print("正在零网络重放真实归档；这是原运行的复现，不是新的赛前预测。")
    report = replay_saved_run(folder, output=output / "replay", project_root=deployment["root"])
    if report.get("replay_network_requests") != 0 or report.get("identical_replay") is not True:
        raise DeploymentError("归档未通过零网络精确回放验收。")
    return {"schema": RESULT_VERSION, "command": "replay", "mode": "archived_real_replay",
            "scope": REAL_SCOPE, "match": _match_view(item, now), "network_requests": 0,
            "acceptance": report, "source_run": str(folder), "new_prediction": False,
            "run_selection": "explicit_analyze_output" if explicit_run else "frozen_deployment_archive",
            "artifacts": _report_paths(output / "replay")}


def _settle(deployment, output, now, *, selected_run=None):
    selected = [selected_run] if selected_run is not None else [
        (item, _saved_run(deployment, item)) for item in deployment["matches"]]
    folders = [folder for _, folder in selected]
    proofs = {}
    for folder in folders:
        for version, proof in _json(folder / "prior-bindings.json").items():
            if version in proofs and proofs[version] != proof:
                raise DeploymentError("不同归档的先验绑定冲突。")
            proofs[version] = proof
    output = _new_output(output, deployment, "settle", now)
    bundle = {**deployment["identity"], "bindings": {item["match_id"]: item["binding"] for item, _ in selected}}
    _write(output / "identity-bundle.json", bundle)
    settings = deployment["config"]["settlement"]
    print("正在联网核验官方赛果并审计原始字节；未完场、冲突或不明确的比分会保留待定。")
    code = capture_results(["capture-results", "--identity-bundle", str(output / "identity-bundle.json"),
        "--identity-archive", str(deployment["identity_archive"]), "--output", str(output / "results"),
        "--route", settings["route"], "--max-http-requests", str(settings["max_http_requests"])])
    if code != 0:
        raise DeploymentError("独立赛果采集未完成，已保留当前目录，不能生成标签。")
    verification = _json(output / "results" / "verification.json")
    audit = join_settlements(folders, settlements=[], prior_bindings=proofs,
                             as_of=_now(), result_archive=output / "results")
    _write(output / "settlement-audit.json", audit)
    return {"schema": RESULT_VERSION, "command": "settle", "mode": "live_independent_result_audit",
            "scope": REAL_SCOPE, "verified_settlement_rows": len(verification["settlement_rows"]),
            "source_runs": [str(folder) for folder in folders],
            "run_selection": "explicit_analyze_output" if selected_run is not None else "frozen_deployment_archive",
            "fixture_status": {key: value["status"] for key, value in verification["reports"].items()},
            "training_records": len(audit["training_records"]),
            "audit": audit["dataset"]["audit"], "labels_supplied_by_user": False,
            "models_refitted": False, "open_world_policy_benefit_proven": False}


def _mechanism(config_path, case, output, now):
    config = _json(config_path)
    if config.get("schema") != VERSION:
        raise DeploymentError("不支持的部署版本。")
    settings = config.get("mechanism", {})
    if settings.get("scope") != "synthetic_mechanism_only":
        raise DeploymentError("部署未绑定可独立恢复的合成机制归档。")
    base = _pinned(ROOT, settings["base_model"])
    recovery = _json(_pinned(ROOT, settings["recovery_report"]))
    for reference in settings.get("recovery_inputs", []):
        _pinned(ROOT, reference)
    if (recovery.get("status") != "passed" or recovery.get("scope") != "synthetic_mechanism_only"
            or recovery.get("base_model_sha256") != settings["base_model"]["sha256"]
            or recovery.get("cases", {}).get(case, {}).get("base_manifest_exact") is not True):
        raise DeploymentError("合成初态恢复未通过原策略清单的一致性验收。")
    entry = settings["cases"][case]
    folder = _path(ROOT, entry["run_path"], directory=True)
    project = _path(ROOT, entry["project_root"], directory=True)
    for reference in entry["files"]:
        _pinned(ROOT, reference)
    needed = {folder / "run.json", folder / "configuration.json", folder / "text-response-model.json",
              project / "var/parallel/agent/round175/frozen-next-period.json"}
    if (folder / "discovery-response-model.json").exists():
        needed.add(folder / "discovery-response-model.json")
    if {(_path(ROOT, ref["path"])) for ref in entry["files"]} != needed:
        raise DeploymentError("合成机制缺少必要文件的冻结绑定。")
    deployment = {"config": config, "config_sha256": _hash(config_path)}
    output = _new_output(output, deployment, "mechanism", now, scope="synthetic_mechanism_only")
    print("正在零网络重放合成机制：用于检验两步信息价值和预算约束，不能视为真实比赛收益。")
    acceptance = replay_saved_run(folder, output=output / "replay", project_root=project,
        base_model={"path": str(base), "sha256": settings["base_model"]["sha256"]})
    if acceptance.get("replay_network_requests") != 0 or acceptance.get("identical_replay") is not True:
        raise DeploymentError("合成机制未通过零网络精确回放。")
    run = _json(output / "replay" / "run.json")
    return {"schema": RESULT_VERSION, "command": "mechanism", "case": case,
            "scope": "synthetic_mechanism_only", "network_requests": 0,
            "acceptance": acceptance, "source_run": str(folder), "models_refitted": False,
            "attempts": len(run["attempts"]), "initial_probabilities": run["initial_probabilities"],
            "final_probabilities": run["final_probabilities"], "reserved_cost": run["reserved_cost"],
            "terminal_selection": run["terminal_selection"], "real_football_benefit_proven": False}


def _load_semantic_submission_checker():
    """Load the repository-local offline checker without spawning a process."""
    path = ROOT / "tools/check_footballmind_semantic_submission.py"
    spec = importlib.util.spec_from_file_location("footballmind_semantic_submission_checker", path)
    if spec is None or spec.loader is None:
        raise DeploymentError("无法加载语义提交验收器。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relative_artifacts(root, paths):
    root = Path(root).resolve()
    result = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            continue
        result.append({"path": path.relative_to(root).as_posix(),
                       "sha256": _hash(path)})
    return result


def _demo(output, config_path, now):
    """Build one deterministic, zero-network AIC demonstration bundle.

    This command is intentionally an evidence packager, not a new model run:
    the synthetic mechanism, the frozen semantic acceptance, and the archived
    real replay retain separate claim boundaries.
    """
    output = Path(output).resolve()
    if output.exists():
        raise DeploymentError(f"输出已存在，必须使用新的目录；已有演示包不会覆盖：{output}")
    output.mkdir(parents=True, exist_ok=False)
    generated_at = now.isoformat()
    items = []

    def record(item_id, evidence_type, claims_boundary, runner):
        item_root = output / item_id
        try:
            result, artifacts = runner(item_root)
            item = {"id": item_id, "status": "passed", "evidence_type": evidence_type,
                    "claims_boundary": claims_boundary,
                    "artifacts": _relative_artifacts(output, artifacts),
                    "result": result}
        except Exception as error:  # preserve diagnostics from earlier phases
            item = {"id": item_id, "status": "failed", "evidence_type": evidence_type,
                    "claims_boundary": claims_boundary,
                    "artifacts": _relative_artifacts(output, item_root.rglob("*")),
                    "error": {"type": type(error).__name__, "message": str(error)}}
        items.append(item)

    def run_contract(item_root):
        item_root.mkdir(parents=True, exist_ok=False)
        from .autonomous_mechanism_acceptance import run_autonomous_mechanism_acceptance
        result = run_autonomous_mechanism_acceptance()
        _write(item_root / "acceptance.json", result)
        if result.get("status") != "passed":
            raise DeploymentError("自主机制验收未通过。")
        return result, [item_root / "acceptance.json"]

    def run_submission(item_root):
        item_root.mkdir(parents=True, exist_ok=False)
        result = _load_semantic_submission_checker().run_acceptance(ROOT)
        _write(item_root / "acceptance.json", result)
        if result.get("status") != "passed":
            raise DeploymentError("冻结语义提交验收未通过。")
        return result, [item_root / "acceptance.json"]

    def run_mechanism(item_root):
        result = _mechanism(config_path, "two-step", item_root, now)
        _write(item_root / "result.json", result)
        return result, [item_root / "result.json", item_root / "replay/report.md",
                        item_root / "replay/acceptance.json"]

    def run_real_replay(item_root):
        deployment = load_deployment(config_path, root=ROOT, now=now)
        item = resolve_match(deployment["matches"], "克罗地亚 vs 西班牙")
        result = _replay(deployment, item, item_root, now)
        _write(item_root / "result.json", result)
        return result, [item_root / "result.json", item_root / "replay/report.md",
                        item_root / "replay/acceptance.json"]

    record("autonomous-contract", "autonomous_contract",
           "8个场景和67项离线边界检查；不证明真实收益或g已训练。", run_contract)
    record("submission-check", "historical_semantic_submission",
           "冻结历史语义机制的数学与证据链；不证明足球预测准确率。", run_submission)
    record("mechanism", "synthetic_mechanism",
           "合成两步VoI机制回放；不代表真实比赛结果、真实收益或已训练g。", run_mechanism)
    record("real-replay", "archived_real_replay",
           "真实公开来源归档的零网络精确回放；不产生新的赛前预测。", run_real_replay)

    passed = all(item["status"] == "passed" for item in items)
    manifest = {
        "schema": "footballmind-aic-demo-bundle-v1",
        "status": "passed" if passed else "failed",
        "generated_at": generated_at,
        "command": "python -m football_agent.footballmind demo --output <new-dir>",
        "network_calls": 0,
        "semantic_calls": 0,
        "items": items,
        "claims_boundary": [
            "合成机制、历史提交验收和真实归档回放严格分开。",
            "g的真实经验校准仍为deferred；本包不把合成响应当作真实收益。",
            "真实回放是可审计执行复现，不是对未来比赛的重新预测。",
        ],
    }
    _write(output / "demo-manifest.json", manifest)
    lines = [
        "# FootballMind AIC 一键演示包",
        "",
        f"生成时间（UTC）：`{generated_at}`",
        "",
        "本包是零网络、零语义模型调用的封版演示入口。它把自主机制边界、冻结证据验收、合成两步 VoI 和真实归档回放放在同一个可审计目录中；四类证据不合并解释。",
        "",
        "## 三分钟阅读顺序",
        "",
        "1. 先看 `demo-manifest.json`，确认每一项的状态、证据类型和 claims boundary。",
        "2. 看 `mechanism/replay/report.md`：目录查询本身不划算，但可能带来后续正文，Agent 因此按两步 VoI 选择并重新评估。",
        "3. 看 `real-replay/replay/report.md`：真实归档能复现获取和停止轨迹，但当前真实响应价值未校准，所以系统保留 `defer`，没有伪造概率更新。",
        "4. 看 `autonomous-contract/acceptance.json` 和 `submission-check/acceptance.json`，核对离线验收结果。",
        "",
        "## 运行边界",
        "",
        "- `network_calls=0`、`semantic_calls=0`；不会触发外部网页或LLM。",
        "- 合成机制用于说明“为什么先查什么、何时停止”的算法链条，不能当作足球准确率或真实收益。",
        "- `g` 的真实经验训练和开放世界收益评估仍待时间戳信息流；本包不声称已完成。",
        "",
        "## 单命令复现",
        "",
        "```powershell",
        "python -m football_agent.footballmind demo --output var/submission/aic-demo-bundle-<date>",
        "```",
        "",
        f"总体状态：`{manifest['status']}`。详细机器可读结果见 [`demo-manifest.json`](demo-manifest.json)。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="FootballMind：只指定比赛，由系统选择可获取的信息。")
    parser.add_argument("--config", type=Path, default=ROOT / "config/footballmind.json", help="部署配置（默认已冻结范围）")
    commands = parser.add_subparsers(dest="command", required=True)
    historical = commands.add_parser("historical-matches", help="列出可立即体验自主信息选择的历史比赛（零网络）")
    historical.add_argument("--json", action="store_true", help="输出结构化比赛列表")
    historical.add_argument("--limit", type=int, help="仅显示前 N 场；省略显示全部")
    historical_run = commands.add_parser("historical-analyze", help="只输入比赛，重新运行历史数值 VoI 选择、更新和停止")
    historical_run.add_argument("--match", required=True, help="历史比赛 ID 或列表中的精确主队 vs 客队")
    historical_run.add_argument("--json", action="store_true", help="输出完整结构化结果；默认显示摘要和报告位置")
    historical_run.add_argument("--output", type=Path, help="新的输出目录；省略时自动创建")
    historical_run.add_argument("--max-queries", type=int, help="可选资源上限；省略使用冻结实验预算")
    historical_run.add_argument("--lambda-money", type=float, help="可选模拟成本权重；更改会标记为参数试验")
    open_matches = commands.add_parser("open-matches", help="列出可零网络重算的真实公开来源采集比赛")
    open_matches.add_argument("--json", action="store_true")
    open_replay = commands.add_parser("open-replay", help="只输入比赛，重算真实归档的开放候选发现与选择过程")
    open_replay.add_argument("--match", required=True)
    open_replay.add_argument("--json", action="store_true", help="输出完整结构化结果")
    open_replay.add_argument("--output", type=Path, help="新的输出目录；省略时自动创建")
    matches = commands.add_parser("matches", help="列出已确认的部署比赛和真实时间边界（零网络）")
    matches.add_argument("--json", action="store_true", help="输出结构化列表")
    for name, help_text in (("analyze", "实际联网分析一场尚未过期的部署比赛"),
                            ("replay", "零网络精确重放已保存的真实运行")):
        sub = commands.add_parser(name, help=help_text)
        selector = sub.add_mutually_exclusive_group(required=True) if name == "replay" else sub
        selector.add_argument("--match", required=name == "analyze",
            help="完整比赛 ID 或精确主客队；replay 此选项重放部署自带的冻结归档")
        if name == "replay":
            selector.add_argument("--run", type=Path, help="精确指定之前 analyze 的输出目录（不会猜测最新运行）")
        sub.add_argument("--output", type=Path, required=True, help="新的、尚不存在的输出目录")
    settle = commands.add_parser("settle", help="联网采集独立赛果并审计，保留未确认结果")
    settle.add_argument("--output", type=Path, required=True)
    settle.add_argument("--run", type=Path, help="只结算所指定的 analyze 输出；省略则审计部署自带的冻结归档")
    mechanism = commands.add_parser("mechanism", help="零网络重放冻结的合成信息价值机制（不代表真实收益）")
    mechanism.add_argument("--case", choices=MECHANISM_CASES, default="two-step")
    mechanism.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo", help="生成零网络、可审计的一键 AIC 演示包")
    demo.add_argument("--output", type=Path, required=True,
                      help="新的、尚不存在的演示包目录")
    start = commands.add_parser("start", help="只输入新比赛，自动准备、登记并运行采集任务")
    start.add_argument("--match", required=True, help="spftll 比赛ID或精确的主队 vs 客队")
    start.add_argument("--date", help="球队名称有歧义时的比赛日期 YYYY-MM-DD（北京时间）")
    start.add_argument("--output", type=Path, required=True, help="新的持久任务目录")
    start.add_argument("--settings", type=Path, help="后台作业配置，省略使用项目默认配置")
    for command in ("task-status", "task-resume", "task-replay"):
        task = commands.add_parser(command, help="查看、继续或零网回放持久比赛任务")
        task.add_argument("--task", type=Path, required=True)
        if command == "task-resume":
            task.add_argument("--settle", action="store_true", help="如已到期，同时执行有界官方结果核验")
        if command == "task-replay":
            task.add_argument("--output", type=Path, required=True)
    tick = commands.add_parser("task-tick", help="推进目录中当前到期的任务一次；不启动隐藏后台服务")
    tick.add_argument("--task-root", type=Path, default=ROOT / "var/live-tasks")
    tick.add_argument("--limit", type=int, default=20)
    batch = commands.add_parser("start-batch", help="冻结比赛清单并逐个运行新比赛任务（最多20场）")
    batch.add_argument("--match", action="append", required=True)
    batch.add_argument("--date")
    batch.add_argument("--settings", type=Path)
    batch.add_argument("--output", type=Path, required=True)
    batch_resume = commands.add_parser("batch-resume", help="继续已有批次，复用已经完成的任务")
    batch_resume.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in {"open-matches", "open-replay"}:
            from .open_archive_analysis import load_open_catalog, replay_open_match
            if args.command == "open-matches":
                rows = load_open_catalog(root=ROOT)
                if args.json:
                    print(json.dumps({"mode": "archived_open_information", "matches": rows,
                        "network_requests": 0}, ensure_ascii=False, indent=2))
                else:
                    print(f"真实公开来源归档：{len(rows)} 场。此入口重算候选发现和有限探索，不重新请求网页。")
                    for row in rows:
                        print(json.dumps(row, ensure_ascii=False))
                return 0
            if args.output is None:
                from uuid import uuid4
                args.output = ROOT / "var/submission/open-runs" / (_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
            result = replay_open_match(args.match, output=args.output, root=ROOT)
            display = result if args.json else {key: result[key] for key in
                ("status", "match", "summary", "terminal_selection", "replay_verification") if key in result}
            print(json.dumps(display, ensure_ascii=False, indent=2, allow_nan=False))
            print(f"开放信息报告：{args.output.resolve() / 'report.md'}")
            return 0
        if args.command in {"historical-matches", "historical-analyze"}:
            from .historical_analysis import load_catalog, analyze_match
            if args.command == "historical-matches":
                if args.limit is not None and args.limit < 1:
                    raise ValueError("--limit 必须是正整数。")
                catalog = load_catalog(root=ROOT)
                rows = catalog if args.limit is None else catalog[:args.limit]
                if args.json:
                    print(json.dumps({"mode": "historical_reconstruction", "total": len(catalog),
                        "network_requests": 0, "matches": rows}, ensure_ascii=False, indent=2))
                else:
                    print(f"历史自主分析：共 {len(catalog)} 场，显示 {len(rows)} 场。只需选择比赛，信息由算法选择。")
                    for row in rows:
                        print(json.dumps(row, ensure_ascii=False))
                    print("历史数据按需揭示，成本为模拟资源；没有等待未来赛果或调用联网接口。")
                return 0
            if args.output is None:
                from uuid import uuid4
                args.output = ROOT / "var/submission/runs" / (_now().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
            result = analyze_match(args.match, output=args.output, root=ROOT,
                max_queries=args.max_queries, lambda_money=args.lambda_money)
            display = result if args.json else {key: result[key] for key in
                ("status", "match", "initial_probabilities", "actions", "final_probabilities",
                 "cost", "terminal_selection", "network_requests") if key in result}
            print(json.dumps(display, ensure_ascii=False, indent=2, allow_nan=False))
            print(f"分析报告：{args.output.resolve() / 'report.md'}")
            return 0
        if args.command in {"start", "task-status", "task-resume", "task-replay", "task-tick", "start-batch", "batch-resume"}:
            from .live_match_workflow import start_match, status_match, resume_match, replay_match, tick_tasks, start_batch, resume_batch
            if args.command == "start":
                result = start_match(args.match, args.output, settings_path=args.settings, match_date=args.date)
            elif args.command == "task-status":
                result = status_match(args.task)
            elif args.command == "task-resume":
                result = resume_match(args.task, settle=args.settle)
            elif args.command == "task-replay":
                result = replay_match(args.task, args.output)
            elif args.command == "task-tick":
                result = tick_tasks(args.task_root, limit=args.limit)
            elif args.command == "start-batch":
                result = start_batch(args.match, args.output, settings_path=args.settings, match_date=args.date)
            else:
                result = resume_batch(args.batch)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("status") not in {"unsupported", "not_found", "ambiguous", "not_prematch",
                "unavailable", "budget_exhausted", "expired", "reconciliation_required", "duplicate_fixture", "incomplete"} else 2
        now = _now()
        if args.command == "mechanism":
            result = _mechanism(args.config, args.case, args.output, now)
            _write(args.output.resolve() / "result.json", result)
            print(f"已完成合成机制回放。结果：{args.output.resolve() / 'result.json'}")
            return 0
        if args.command == "demo":
            result = _demo(args.output, args.config, now)
            print(json.dumps({"status": result["status"], "network_calls": 0,
                              "semantic_calls": 0, "output": str(args.output.resolve()),
                              "manifest": str(args.output.resolve() / "demo-manifest.json"),
                              "report": str(args.output.resolve() / "report.md")},
                             ensure_ascii=False, indent=2))
            return 0 if result["status"] == "passed" else 2
        deployment = load_deployment(args.config, root=ROOT, now=now)
        if args.command == "matches":
            rows = [_match_view(item, now) for item in deployment["matches"]]
            if args.json:
                print(json.dumps({"schema": RESULT_VERSION, "scope": REAL_SCOPE,
                    "deployment_sha256": deployment["config_sha256"], "as_of": now.isoformat(),
                    "network_requests": 0, "matches": rows}, ensure_ascii=False, indent=2))
            else:
                print(f"当前部署范围：{len(rows)} 场已独立核实的比赛；不支持任意比赛。所有时间均明确标注 UTC 偏移。")
                print("真实路径为未知 VoI 下的有限探索，初态来自原始市场报价；尚无已验证的真实文本收益。")
                for row in rows:
                    _display_match(row)
            return 0
        if args.command == "settle":
            selected = _analyze_archive(deployment, args.run) if args.run is not None else None
            result = _settle(deployment, args.output, now, selected_run=selected)
        elif args.command == "replay" and args.run is not None:
            item, folder = _analyze_archive(deployment, args.run)
            result = _replay(deployment, item, args.output, now, folder=folder)
        else:
            item = resolve_match(deployment["matches"], args.match)
            handler = _analyze if args.command == "analyze" else _replay
            result = handler(deployment, item, args.output, now)
        _write(args.output.resolve() / "result.json", result)
        print(f"已完成 {args.command}。结果：{args.output.resolve() / 'result.json'}")
        for name, label in (("report_markdown", "分析报告"), ("analysis_json", "结构化报告")):
            if name in result.get("artifacts", {}):
                print(f"{label}：{result['artifacts'][name]}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"无法执行：{error}\n未覆盖任何已有运行；若已产生部分产物，请保留目录用于排查。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
