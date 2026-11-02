"""Match-only access to real archived open-information execution, without network.

The catalog is deliberately bounded by the existing R202 inventory. Replaying
recomputes candidates, exploration and evidence from original raw returns; it
does not attach the historical two-tool response model to unrelated news.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .learned_voi_replay import replay_saved_run
from .learned_voi_runtime import _save, digest

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = Path("var/parallel/agent/round202/training-inputs.json")
INVENTORY_SHA256 = "b669f45958113d0d6d35522d4d353fe2959798ce83ba6ee645e99ab4c219ad02"
_ROUNDS = {"round200", "round202"}
_SCHEMAS = {"learned-voi-open-execution-v5", "learned-voi-open-execution-v6"}
_BASIS = {
    "bounded_unknown_information_exploration": "UNKNOWN：在明确预算内探索，尚无数值收益估计",
    "computed_positive_voi": "计算得到正净信息价值",
    "computed_positive_empirical_gain": "计算得到正预期净收益",
    "no_executable_selection": "本轮结束：没有可执行选择",
}
_STOPS = {
    "physical_budget_exhausted": "查询预算已用完，本次分析结束；不表示其余信息没有价值。",
    "candidate_generation_incomplete": "本次分析结束；已发现信息不足以证明所有未知信息都无价值。",
    "candidate_value_incomplete": "本次有限探索结束；剩余候选价值未知。",
    "all_computed_net_voi_nonpositive": "已计算的可行候选没有正净收益，本次分析结束。",
    "forecast_cutoff_elapsed": "已到预测截止时间，本次分析结束。",
    "step_limit": "已达到部署的有限步骤上限，本次分析结束。",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inventory_paths(root):
    path = root / INVENTORY
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != INVENTORY_SHA256:
        raise ValueError("真实开放归档清单与冻结版本不一致")
    for stored in json.loads(raw)["run_folders"]:
        # The old inventory records Windows absolute paths. Relocate only its
        # fixed workspace-relative artifact subtree, never arbitrary paths.
        parts = str(stored).replace("\\", "/").split("/")
        try:
            tail = parts[parts.index("var"):]
        except ValueError:
            raise ValueError("归档清单缺少受支持的项目路径") from None
        if (tail[:3] != ["var", "parallel", "agent"] or ".." in tail):
            raise ValueError("归档路径超出项目范围")
        if len(tail) < 6 or tail[3] not in _ROUNDS or tail[4] != "live-cohort":
            continue
        yield root.joinpath(*tail)


def load_open_catalog(root=ROOT):
    """Return six real R200/R202 fixtures; no outcomes or fabricated cases."""
    root = Path(root).resolve()
    catalog = []
    for folder in _inventory_paths(root):
        config = _read(folder / "configuration.json")
        run = _read(folder / "run.json")
        if config.get("schema") not in _SCHEMAS:
            raise ValueError("目录中的归档不是受支持的真实开放运行")
        if config.get("initial_state", {}).get("information_mode") != "operational":
            raise ValueError("开放目录不能混入构造或历史特征模拟")
        if run.get("configuration_digest") != digest(config):
            raise ValueError("开放运行配置校验失败")
        identity = config["evidence_projection"]["match"]
        catalog.append({
            "match_id": identity["match_id"],
            "match": f'{identity["home_team"]} vs {identity["away_team"]}',
            "home_team": identity["home_team"], "away_team": identity["away_team"],
            "kickoff": identity["kickoff_at"],
            "archive_created_at": config["created_at"],
            "run_directory": folder.relative_to(root).as_posix(),
            "mode": "real_open_archive_replay",
            "prior_available": run.get("initial_probabilities") is not None,
        })
    return sorted(catalog, key=lambda item: (item["kickoff"], item["match_id"]))


def _action(action):
    if action is None:
        return None
    return {key: action.get(key) for key in ("action_id", "tool", "url", "query")}


def _quotes(public):
    evidence = public.get("evidence") or {}
    return {item["evidence_id"] for key in ("evidence", "unscoped_source_quotes")
            for item in evidence.get(key, []) if item.get("evidence_id")}


def _summarize_steps(run, replay_dir):
    seen, steps = set(), []
    for cycle in run["cycles"]:
        candidates = cycle["generated_actions"]
        keys = {a["information_key"] for a in candidates}
        new = [a for a in candidates if a["information_key"] not in seen]
        newly_discovered = new if steps else []
        seen.update(keys)
        selected = cycle["selected_action"]
        receipt = (_read(replay_dir / f'receipt-{cycle["step"]:03}.json')
                   if selected is not None else {})
        document = receipt.get("document") or {}
        candidate_values = cycle["strict_decision"]["candidates"]
        selected_value = next((row.get("value") for row in candidate_values
                               if selected and row["action_id"] == selected["action_id"]), None)
        before, after = cycle["prediction_before"], cycle.get("prediction_after")
        new_quotes = (_quotes(cycle.get("public_after") or {}) -
                      _quotes(cycle["public_before"]))
        after_evidence = (cycle.get("public_after") or {}).get("evidence") or {}
        quote_preview = [
            {"text": item.get("statement") or item.get("anchor", {}).get("quote"),
             "url": item.get("anchor", {}).get("url"),
             "applicability": item.get("applicability")}
            for key in ("evidence", "unscoped_source_quotes")
            for item in after_evidence.get(key, [])
            if item.get("evidence_id") in new_quotes
        ][:2]
        steps.append({
            "step": cycle["step"], "candidate_count": len(candidates),
            "feasible_candidate_count": sum(bool(c["feasible"]) for c in candidate_values),
            "initial_candidate_count": len(new) if not steps else 0,
            "newly_discovered_count": len(newly_discovered),
            "newly_discovered_candidates": [_action(a) for a in newly_discovered],
            "selected_action": _action(selected),
            "selection_basis": cycle["selection_basis"],
            "selection_explanation_zh": _BASIS.get(cycle["selection_basis"], cycle["selection_basis"]),
            "estimated_net_voi": (selected_value or {}).get("net_voi"),
            "execution_status": cycle["execution_status"],
            "return_status": receipt.get("status"), "http_status": document.get("http_status"),
            "returned_url": document.get("url"), "document_kind": document.get("kind"),
            "new_source_quote_count": len(new_quotes),
            "new_source_quote_preview": quote_preview,
            "admission": cycle.get("admission"),
            "prediction_before": before, "prediction_after": after,
            "probability_changed": before is not None and after is not None and before != after,
        })
    return steps


def _render(analysis):
    summary = analysis["summary"]
    lines = [f'# 真实开放信息归档体验：{analysis["match"]["match"]}', "",
        "本次分析已结束。系统从保存的真实网页返回重新计算候选发现、选择和停止；本次不联网、不等待赛果。", "",
        f'原始采集时间：{analysis["match"]["archive_created_at"]}。', "",
        f'最初 {summary["initial_candidate_count"]} 个候选；执行中发现 {summary["newly_discovered_count"]} 个新候选；'
        f'重新核验 {summary["archived_return_count"]} 个原始返回；当前网络请求为 0。', "",
        "这些归档展示开放来源发现和预算内自主探索。UNKNOWN 是未知收益，不是零收益；"
        "它们没有接上已校准的新闻数值 VoI，不能与历史战绩/xG 模型合并声称同一项已验证能力。", ""]
    for step in analysis["steps"]:
        selected = step["selected_action"]
        lines.extend([f'第 {step["step"] + 1} 轮：候选 {step["candidate_count"]} 个，'
                      f'新发现 {step["newly_discovered_count"]} 个。', "",
                      step["selection_explanation_zh"] + "。", ""])
        if selected:
            lines.extend([f'选择：{selected["url"] or selected["query"] or selected["action_id"]}', "",
                f'原始返回：{step["return_status"]}；HTTP {step["http_status"]}；'
                f'新增原文引述 {step["new_source_quote_count"]} 段。引述数量不等于与本场相关的信息数量。', ""])
            for quote in step["new_source_quote_preview"]:
                lines.extend(["原文片段（相关性未确认）：", "", str(quote["text"]), ""])
    lines.extend([analysis["stop_explanation_zh"], "",
        ("初始和最终概率均不可用，保持缺失，没有用新闻虚构胜平负变化。"
         if analysis["initial_probabilities"] is None and analysis["final_probabilities"] is None
         else f'初始概率：{analysis["initial_probabilities"]}；最终概率：{analysis["final_probabilities"]}。'), "",
        "用户无需提出调查问题或指定获取哪些信息。原始轨迹和核验结果位于 replay/。", ""])
    return "\n".join(lines)


def replay_open_match(match, *, output, root=ROOT):
    """Resolve a fixture and genuinely replay its archived open execution."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not isinstance(match, str) or not match.strip():
        raise ValueError("请提供比赛 ID 或完整主队 vs 客队")
    target = match.strip().casefold()
    matches = [item for item in load_open_catalog(root)
               if target in {item["match_id"].casefold(), item["match"].casefold()}]
    if len(matches) != 1:
        raise ValueError("比赛未在真实开放归档目录中唯一匹配，请先运行 open-matches")
    selected = matches[0]
    output.mkdir(parents=True, exist_ok=False)
    replay_dir = output / "replay"
    verified = replay_saved_run(root / selected["run_directory"], output=replay_dir, project_root=root)
    if not verified.get("identical_replay") or verified.get("replay_network_requests") != 0:
        raise ValueError("开放归档未通过零网络精确重算")
    run = _read(replay_dir / "run.json")
    steps = _summarize_steps(run, replay_dir)
    terminal = run["terminal_selection"]
    analysis = {
        "schema": "footballmind-open-archive-analysis-v1", "status": "completed",
        "match": selected, "mode": "real_open_archive_replay", "steps": steps,
        "initial_probabilities": run["initial_probabilities"],
        "final_probabilities": run["final_probabilities"],
        "terminal_selection": terminal,
        "stop_explanation_zh": _STOPS.get(terminal["reason"], f'本次分析已结束，原因：{terminal["reason"]}。'),
        "summary": {
            "initial_candidate_count": steps[0]["candidate_count"] if steps else 0,
            "newly_discovered_count": sum(s["newly_discovered_count"] for s in steps),
            "archived_return_count": verified["returns"], "replay_network_requests": 0,
            "new_source_quote_count": sum(s["new_source_quote_count"] for s in steps),
            "probability_changed": any(s["probability_changed"] for s in steps),
            "computed_selections": sum(s["selection_basis"].startswith("computed_positive") for s in steps),
            "unknown_exploration_selections": sum(s["selection_basis"] == "bounded_unknown_information_exploration" for s in steps),
        },
        "replay_verification": verified,
        "open_text_numeric_voi_validated": False,
        "historical_two_tool_model_attached": False,
        "source_relevance_established": False,
    }
    _save(output / "analysis.json", analysis)
    (output / "report.md").write_text(_render(analysis), encoding="utf-8")
    return analysis
