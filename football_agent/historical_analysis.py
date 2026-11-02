"""Match-only entry for the existing frozen, real historical acquisition study.

The owner loads archived numeric features; the policy receives only revealed
states. This does not relabel reconstructed availability as live observation,
and never fits on a requested match or waits for future results.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import unicodedata

from football_research.acquisition.contextual_joint import ContextualJointReturnPredictor
from football_research.acquisition.data import load_episodes
from football_research.acquisition.joint_voi import LearnedJointVOIPolicy
from football_research.acquisition.state import parse_time
from football_research.data_adapters.platform_history import verify_completed_export
from football_research.evaluation.learned_voi_replay import run_learned_voi_episode

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("var/parallel/baselines/round8/joint-returns-v2")
STUDY = Path("var/parallel/agent/round177/evaluation/results")
SCHEMA = "footballmind-historical-analysis-v1"
ACTION_NAMES = {"results_view": "球队实力与近期战绩", "xg_view": "近期预期进球表现"}
REASONS = {
    "best_positive_net_voi": "当前候选中预期净信息价值最高且为正",
    "candidate_generation_incomplete": "当前已声明候选没有继续获取的正净收益；开放信息范围仍未穷尽",
    "candidate_value_incomplete": "剩余候选的价值无法可靠估计，保留未知",
    "physical_budget_exhausted": "可用查询预算已耗尽",
    "already_attempted": "本次分析已获取过该信息",
    "action_unavailable": "该数据源当前不可用，跳过",
    "response_model_unavailable": "没有适用的返回模型，价值未知",
    "insufficient_conditional_return_support": "相同可见条件下历史支持不足，价值未知",
    "unknown_fixture_identity": "冻结模型未覆盖该比赛的球队身份，价值未知",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _check(path, expected):
    if not expected or _digest(path) != expected:
        raise ValueError(f"历史交付文件与冻结记录不一致：{path}")


def _inside(root, value):
    path = (root / Path(str(value).replace("\\", "/"))).resolve()
    if not path.is_relative_to(root):
        raise ValueError("历史交付引用必须位于项目目录内")
    return path


def _export_path(root, value):
    # Old reports contain absolute workspace paths. Resolve the same immutable
    # relative export after moving the project; never look outside this root.
    parts = str(value).replace("\\", "/").split("/")
    if "var" not in parts:
        raise ValueError("无法定位冻结历史数据目录")
    return _inside(root, "/".join(parts[parts.index("var"):]))


def _eligibility(row, model, fold):
    forbidden = set(model.metadata.get("training_match_ids", []))
    forbidden.update(model.metadata.get("base_training_match_ids", []))
    forbidden.update(model.selection_report.get("match_ids", []))
    forbidden.update(fold.get("training_match_ids", []))
    forbidden.update(fold.get("validation_match_ids", []))
    if row["match_id"] in forbidden:
        return "overlaps_model_training_or_selection"
    if row.get("information_mode") != "historical_reconstruction":
        return "unsupported_information_mode"
    try:
        cutoff = parse_time(row["prediction_cutoff"])
        earliest = max(parse_time(model.metadata["training_as_of"]),
                       parse_time(model.selection_report["as_of"]))
        if not earliest <= cutoff < parse_time(row["kickoff"]) < parse_time(row["result_available_at"]):
            return "incompatible_model_or_match_chronology"
        if any(parse_time(row[key]) > cutoff for key in ("fixture_available_at", "features_available_at")):
            return "input_unavailable_at_reconstructed_cutoff"
    except (KeyError, TypeError, ValueError):
        return "missing_or_invalid_time_metadata"
    return None


def _load(root):
    root = Path(root).resolve()
    source, study = root / SOURCE, root / STUDY
    source_report, report = _read(source / "report.json"), _read(study / "report.json")
    design = _read(study / "design.json")
    _check(study / "design.json", report["design_sha256"])
    _check(source / "report.json", design["source_report_sha256"])
    _check(source / "catalog.json", design["source_catalog_sha256"])
    audit = _read(source / "data-audit.json")
    export = _export_path(root, audit["current"]["dataset_audit"]["export_directory"])
    verified = verify_completed_export(export)
    if verified.manifest_sha256 != audit["current"]["platform_manifest_sha256"]:
        raise ValueError("历史数据导出版本与冻结实验不一致")
    # The design independently binds this input, rather than trusting only the
    # older loader audit when resolving a relocated export.
    if verified.manifest_sha256 != design["source_data"]["current"]["manifest_sha256"]:
        raise ValueError("历史数据导出版本与研究设计不一致")
    rows, loader = load_episodes(export)
    if loader["platform_manifest_sha256"] != verified.manifest_sha256:
        raise ValueError("读取期间历史数据版本发生变化")
    raw = {r["match_id"]: r for r in verified.rows("matches")}
    artifacts = {k.replace("\\", "/"): v for k, v in report["artifacts_sha256"].items()}
    models, assignment, parameters = {}, {}, {}
    for fold in source_report["folds"]:
        ident = fold["fold"]
        relative = f"models/fold-{ident}-contextual.json"
        _check(study / relative, artifacts.get(relative))
        model = ContextualJointReturnPredictor.load(study / relative)
        metadata = next(m for m in report["models"] if m["fold"] == ident)
        if model.model_version != metadata["models"]["contextual-strength"]["model_version"]:
            raise ValueError("历史上下文模型版本与实验登记不一致")
        models[ident], parameters[ident] = model, metadata
        for mid in fold["test_match_ids"]:
            if mid in assignment:
                raise ValueError("同一比赛被重复登记到历史留出分组")
            assignment[mid] = fold
    retained, skipped, catalog = {}, list(loader.get("exclusions", [])), []
    for row in rows:
        mid = row["match_id"]
        fold = assignment.get(mid)
        reason = "outside_registered_historical_test_matches" if fold is None else _eligibility(row, models[fold["fold"]], fold)
        if reason:
            skipped.append({"match_id": mid, "reason": reason})
            continue
        fixture = raw[mid]
        item = {"match_id": mid, "source_id": fixture.get("source_id"),
                "home_team": fixture["home_team_name"], "away_team": fixture["away_team_name"],
                "competition": fixture["competition_name"], "kickoff": row["kickoff"],
                "prediction_cutoff": row["prediction_cutoff"], "fold": fold["fold"],
                "information_mode": "historical_reconstruction"}
        retained[mid] = row
        catalog.append(item)
    catalog.sort(key=lambda r: (parse_time(r["kickoff"]), r["match_id"]))
    actions = _read(source / "catalog.json")["actions"]
    limits = next(b for b in design["budgets"] if b["requests"] == 2)
    return {"catalog": catalog, "rows": retained, "models": models, "parameters": parameters,
            "actions": actions, "limits": limits, "skipped": skipped,
            "source_observed_at": verified.manifest["source_observed_at"],
            "assumptions": verified.manifest.get("assumptions", {}),
            "manifest_sha256": verified.manifest_sha256,
            "study_sha256": _digest(study / "report.json"), "source_match_count": loader["source_match_count"]}


def load_catalog(root=ROOT):
    """Return eligible match identities, names and clocks, never outcomes."""
    return HistoricalAnalysisSession(root=root).load_catalog()


def _normal(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _resolve(match, catalog):
    if isinstance(match, dict):
        mid = match.get("match_id") or match.get("id")
        if mid is not None:
            matches = [r for r in catalog if str(mid) in (r["match_id"], str(r["source_id"]))]
        else:
            home, away = match.get("home_team"), match.get("away_team")
            if not home or not away:
                raise ValueError("请提供比赛 ID，或明确的主队与客队")
            matches = [r for r in catalog if (_normal(r["home_team"]), _normal(r["away_team"])) == (_normal(home), _normal(away))]
        date = match.get("date")
        if date:
            matches = [r for r in matches if r["kickoff"][:10] == str(date)]
    elif isinstance(match, str) and match.strip():
        query = _normal(match)
        matches = [r for r in catalog if query in (_normal(r["match_id"]), _normal(r["source_id"]))]
        if not matches:
            # Explicit separators avoid guessing team substrings or reversing
            # home and away. Names are those in the source-verified catalog.
            parts = re.split(r"\s+(?:vs\.?|v\.?|对阵|对|-)\s+|\s*[—–]\s*", query)
            if len(parts) == 2:
                matches = [r for r in catalog if (_normal(r["home_team"]), _normal(r["away_team"])) == tuple(parts)]
    else:
        raise ValueError("比赛输入必须是比赛 ID、‘主队 vs 客队’，或比赛身份对象")
    if not matches:
        raise ValueError("该比赛不在当前可运行的历史目录中；请先列出比赛，不会替换为其他比赛")
    if len(matches) != 1:
        raise ValueError("比赛身份有歧义，请使用目录中的精确比赛 ID 或补充日期")
    return matches[0]


def _number(value):
    return "未知" if value is None else f"{value:.6f}"


def _probabilities(values):
    return " / ".join(f"{label} {value:.2%}" for label, value in zip(("主胜", "平局", "客胜"), values))


def _report(report):
    match = report["match"]
    lines = [f"# {match['home_team']} vs {match['away_team']}", "",
             f"比赛：{match['competition']}，{match['kickoff']}；ID：{match['match_id']}。", "",
             "用户只指定比赛；查询对象、先后顺序和结束时点由冻结模型根据当前可见信息决定。", "",
             "运行已完成，不存在等待执行的后台任务。原始策略的 defer 只表示没有对开放世界的信息充分性作保证。", "",
             "数据模式：真实历史数值数据的受控逐次揭示；不是当年的在线采集，不包含已验证的新闻价值模型。",
             "查询成本使用冻结实验的模拟单位。时间可用性为声明的历史重建假设，未将后来抓取时间伪装为赛前观察。", "",
             f"初始预测：{_probabilities(report['initial_probabilities'])}", "",
             f"配置：最多 {report['configuration']['limits']['requests']:g} 次查询；模拟费用上限 {report['configuration']['limits']['money']:g}；成本换算权重 {report['configuration']['lambda_money']:g}。"]
    for cycle in report["cycles"]:
        decision = cycle["decision"]
        lines += ["", f"## 决策 {cycle['step'] + 1}", "",
                  "候选信息 | 预期信息价值 | 换算成本 | 预期净价值 | 本步处理", "--- | ---: | ---: | ---: | ---"]
        selected = decision["selection"].get("candidate_id")
        for candidate in decision["candidates"]:
            value = candidate.get("value") or {}
            reason = candidate.get("reason") or value.get("reason")
            if candidate["action_id"] == selected:
                disposition = "选择：" + REASONS.get(decision["selection"]["reason"], decision["selection"]["reason"])
            elif reason and reason != "computed_coherent_evsi":
                disposition = REASONS.get(reason, reason)
            elif value.get("net_voi") is not None and value["net_voi"] <= 0:
                disposition = "跳过：预期净价值不为正"
            else:
                disposition = "本步未选择：比较后的优先级较低"
            lines.append(f"{ACTION_NAMES.get(candidate['action_id'], candidate['action_id'])} | {_number(value.get('gross_evsi'))} | {_number(value.get('cost'))} | {_number(value.get('net_voi'))} | {disposition}")
        if cycle["update"]:
            update = cycle["update"]
            lines += ["", f"获取后预测：{_probabilities(update['prediction_after'])}"]
            missing = [name for name, group in cycle["state_after"]["groups"].items() if group["status"] == "source_missing"]
            if missing:
                lines += [f"缺失数据：{', '.join(missing)}；保留缺失状态，没有补造数据。"]
        else:
            reason = decision["selection"]["reason"]
            lines += ["", "结束原因：" + REASONS.get(reason, reason) + "。"]
    lines += ["", f"最终预测：{_probabilities(report['final_probabilities'])}", "",
              f"实际获取：{report['cost']['requests']:g} 次，模拟费用 {report['cost']['money']:g}。", "",
              "该次运行展示有限工具范围内的自主信息选择机制；不声明开放互联网的信息已经充分，也不以历史胜负分数作为机制是否完成的条件。",
              "未提供阵容、伤停、天气或新闻时不生成这些事实；这些能力的真实价值学习仍属后续数据工作。"]
    return "\n".join(lines) + "\n"


def _validate_run(output, max_queries, lambda_money):
    if max_queries is not None and (type(max_queries) is not int or not 0 <= max_queries <= 2):
        raise ValueError("max_queries 必须是 0、1 或 2；当前仅有两个真实历史信息动作")
    if lambda_money is not None and (isinstance(lambda_money, bool) or not isinstance(lambda_money, (int, float)) or not math.isfinite(lambda_money) or lambda_money < 0):
        raise ValueError("lambda_money 必须是有限非负数")
    output = Path(output).resolve()
    names = ("analysis.json", "trajectory.json", "report.md")
    if any((output / name).exists() for name in names):
        raise FileExistsError("输出目录已包含分析记录，请选择新目录以保留旧结果")
    return output, names


def _analyze_loaded(loaded, match, *, output, max_queries=None, lambda_money=None):
    output, names = _validate_run(output, max_queries, lambda_money)
    fixture = _resolve(match, loaded["catalog"])
    fold = fixture["fold"]
    original_lambda = loaded["parameters"][fold]["lambda_money"]
    weight = original_lambda if lambda_money is None else float(lambda_money)
    limits = deepcopy(loaded["limits"])
    if max_queries is not None:
        limits["requests"] = max_queries
    policy = LearnedJointVOIPolicy(loaded["models"][fold], lambda_cost={"money": weight})
    trajectory = run_learned_voi_episode(loaded["rows"][fixture["match_id"]], policy, loaded["actions"],
                                         limits=limits, candidate_generation_complete=False)
    report = {"schema": SCHEMA, "status": "completed", "execution_completed": True,
              "awaiting_background_execution": False, "match": fixture,
              "configuration": {"limits": limits, "lambda_money": weight,
                                "original_limits": loaded["limits"], "original_lambda_money": original_lambda,
                                "explicit_sensitivity_override": max_queries is not None or lambda_money is not None,
                                "parameter_refit": False, "candidate_generation_complete": False},
              **{key: trajectory[key] for key in ("initial_probabilities", "final_probabilities", "actions", "cost", "cycles", "terminal_selection", "model_version", "policy_version")},
              "trajectory": trajectory,
              "data_scope": {"mode": "historical_reconstruction", "source": "understat",
                             "source_match_count": loaded["source_match_count"], "eligible_matches": len(loaded["catalog"]),
                             "source_observed_at": loaded["source_observed_at"], "availability_assumptions": loaded["assumptions"],
                             "manifest_sha256": loaded["manifest_sha256"], "study_report_sha256": loaded["study_sha256"],
                             "input_binding": "verified_in_memory_session_snapshot",
                             "real_historical_data": True, "live_news_value_model": False,
                             "costs": "simulated_reveal_not_actual_provider_bills", "prospective_collection": False,
                             "real_world_policy_superiority_proven": False,
                             "missing_data_policy": "retain_source_missing_or_unknown_no_fabrication",
                             "unknown_action_policy": "retain_unknown_no_unmodeled_tool_execution"},
              "skipped_matches": loaded["skipped"],
              "skipped_reason_counts": dict(Counter(r["reason"] for r in loaded["skipped"])),
              "network_requests": 0, "model_refitted": False,
              "artifacts": {name: str(output / name) for name in names}}
    # Returned reports and catalogs must not expose mutable references into
    # the session; each actor already owns a new ControlledEpisode and budget.
    report = deepcopy(report)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("trajectory.json", trajectory), ("analysis.json", report)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(report), encoding="utf-8")
    return report


class HistoricalAnalysisSession:
    """One explicitly verified snapshot reused across a bounded match batch.

    There is no process-global cache. Construct a new session to observe new
    artifacts; completed sessions retain precisely their initially verified
    model/data bytes. Each match has an independent policy, state and budget.
    """

    def __init__(self, root=ROOT):
        self._loaded = _load(root)

    def load_catalog(self):
        return deepcopy(self._loaded["catalog"])

    def analyze_match(self, match, *, output, max_queries=None, lambda_money=None):
        return _analyze_loaded(self._loaded, match, output=output,
                               max_queries=max_queries, lambda_money=lambda_money)


def analyze_match(match, *, output, root=ROOT, max_queries=None, lambda_money=None):
    """Run one match with a freshly verified, temporally eligible snapshot.

    Optional resource overrides are explicitly sensitivity runs, not newly
    tuned benchmark results. Use HistoricalAnalysisSession for a match batch.
    """
    _validate_run(output, max_queries, lambda_money)
    return HistoricalAnalysisSession(root=root).analyze_match(
        match, output=output, max_queries=max_queries, lambda_money=lambda_money)
