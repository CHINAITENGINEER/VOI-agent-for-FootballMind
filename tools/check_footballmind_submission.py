"""Offline acceptance of match-only autonomous information acquisition.

This checks executable selection, revelation, probability updates and stopping;
it deliberately does not select cases by match outcome or compute leaderboard
scores. Run from the repository with a NEW --output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VERSION = "footballmind-submission-acceptance-v1"
LABEL_KEYS = {"label", "outcome", "outcome_index", "home_goals", "away_goals",
              "home_score", "away_score", "result", "result_available_at"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _probabilities(values, name):
    _require(isinstance(values, (list, tuple)) and len(values) == 3
             and all(_number(x) and 0 <= x <= 1 for x in values)
             and abs(sum(values) - 1) <= 1e-8, name + ": invalid probability vector")


def _equal_probabilities(left, right, name):
    _probabilities(left, name)
    _probabilities(right, name)
    _require(max(abs(x-y) for x, y in zip(left, right)) <= 1e-8, name + ": probability mismatch")


def _label_free(value, name):
    if isinstance(value, dict):
        _require(not LABEL_KEYS.intersection(value), name + ": result label entered public input")
        for key, item in value.items():
            _label_free(item, name + "." + str(key))
    elif isinstance(value, list):
        for item in value:
            _label_free(item, name)


def _public_state(state):
    _require(isinstance(state, dict), "missing public state")
    _label_free(state, "public_state")
    groups = state.get("groups")
    _require(isinstance(groups, dict) and bool(groups), "missing public information groups")
    for group in groups.values():
        _require(isinstance(group, dict) and isinstance(group.get("status"), str), "invalid group status")
        if group["status"] != "observed":
            _require(group.get("values") in (None, {}), "unread or unavailable group exposed values")


@contextmanager
def offline_guard():
    """Bound guard lifetime to this invocation; do not poison later test runs."""
    attempts = []

    def denied(*_args, **_kwargs):
        attempts.append("network_access_blocked")
        raise RuntimeError("Submission acceptance must not access the network")

    with ExitStack() as stack:
        for obj, name in ((socket, "create_connection"), (socket, "getaddrinfo"),
                          (socket.socket, "connect"), (socket.socket, "connect_ex"),
                          (socket.socket, "sendto")):
            stack.enter_context(patch.object(obj, name, denied))
        yield attempts


def validate_analysis(result, expected_match):
    """Validate the actual historical runner's public trajectory, without labels."""
    _require(result.get("match", {}).get("match_id") == expected_match["match_id"], "match identity changed")
    _label_free(result["match"], "fixture")
    trajectory = result.get("trajectory")
    _require(isinstance(trajectory, dict), "missing trajectory")
    _require(trajectory.get("actual_external_requests") == 0, "external request reported")
    _require(trajectory.get("match_id") == expected_match["match_id"], "trajectory identity changed")
    cycles = result.get("cycles")
    _require(isinstance(cycles, list) and bool(cycles) and cycles == trajectory.get("cycles"), "missing or divergent cycles")
    limits = result.get("configuration", {}).get("limits")
    _require(isinstance(limits, dict) and bool(limits), "missing physical budget")
    _require(all(_number(x) and x >= 0 for x in limits.values()), "invalid physical budget")
    _require(limits == trajectory.get("limits"), "trajectory budget differs")
    spent, selected, previous_after = {}, [], None
    unsupported = Counter()
    for index, cycle in enumerate(cycles):
        before, after = cycle.get("state_before"), cycle.get("state_after")
        _public_state(before)
        _require(before.get("match_id") == expected_match["match_id"], "public match identity changed")
        if previous_after is not None:
            _require(before == previous_after, "next decision did not consume the preceding acquired state")
        decision = cycle.get("decision", {})
        prediction = decision.get("prediction", {}).get("probabilities")
        _probabilities(prediction, "decision prediction")
        physical = decision.get("physical_budget", {})
        _require(physical.get("limits") == limits, "decision budget changed")
        actual_spent = physical.get("spent", {})
        _require(isinstance(actual_spent, dict) and all(_number(v) and v >= 0 for v in actual_spent.values()), "invalid spent vector")
        _require(all(abs(actual_spent.get(unit, 0)-spent.get(unit, 0)) <= 1e-8
                     for unit in set(actual_spent) | set(spent)), "decision spent vector differs from receipts")
        selection = decision.get("selection", {})
        candidates = decision.get("candidates", [])
        _require(isinstance(candidates, list), "invalid candidate menu")
        for candidate in candidates:
            value = candidate.get("value")
            if isinstance(value, dict) and value.get("status") == "UNKNOWN":
                unsupported[str(value.get("reason", "unknown"))] += 1
        if selection.get("decision") != "acquire":
            _require(index == len(cycles)-1, "execution continued after terminal decision")
            _require(selection.get("decision") in {"stop", "defer", "halt"}
                     and isinstance(selection.get("reason"), str) and bool(selection["reason"]), "missing explicit stop reason")
            _require(after is None and cycle.get("update") is None and cycle.get("charged_cost") is None,
                     "terminal decision acquired or charged information")
            continue
        _require(index < len(cycles)-1, "missing terminal decision after acquisition")
        action = selection.get("candidate_id")
        _require(action not in selected and isinstance(action, str), "duplicate or invalid selected action")
        matches = [c for c in candidates if c.get("action_id") == action]
        _require(len(matches) == 1 and matches[0].get("feasible") is True, "selected action absent or infeasible")
        candidate, cost = matches[0], cycle.get("charged_cost")
        value = candidate.get("value", {})
        _require(value.get("status") == "COMPUTED" and _number(value.get("net_voi")) and value["net_voi"] > 0,
                 "selection has no computed positive net information value")
        alternatives = [c["value"]["net_voi"] for c in candidates if c.get("feasible")
                        and isinstance(c.get("value"), dict) and c["value"].get("status") == "COMPUTED"
                        and _number(c["value"].get("net_voi"))]
        _require(value["net_voi"] >= max(alternatives)-1e-8, "did not select maximum computed net value")
        _require(isinstance(cost, dict) and cost == candidate.get("cost")
                 and all(_number(v) and v >= 0 for v in cost.values()), "invalid acquisition cost")
        for unit, amount in cost.items():
            spent[unit] = spent.get(unit, 0) + amount
            _require(unit not in limits or spent[unit] <= limits[unit]+1e-8, "physical budget exceeded")
        _public_state(after)
        groups = candidate.get("group_ids", [])
        _require(bool(groups), "selected action has no declared evidence groups")
        for name, group in before["groups"].items():
            if name in groups:
                _require(group["status"] == "not_acquired" and after["groups"][name]["status"] in {"observed", "source_missing"},
                         "selected information was not newly acquired")
            else:
                _require(group == after["groups"].get(name), "unselected evidence changed")
        update = cycle.get("update", {})
        _require(update.get("action_id") == action, "update action differs")
        _equal_probabilities(prediction, update.get("prediction_before"), "pre-query probability")
        _equal_probabilities(update.get("prediction_after"), cycles[index+1].get("decision", {}).get("prediction", {}).get("probabilities"), "post-query probability")
        selected.append(action)
        previous_after = after
    _require(selected == result.get("actions") == trajectory.get("actions"), "action sequence differs")
    _require(result.get("terminal_selection") == cycles[-1]["decision"]["selection"] == trajectory.get("terminal_selection"), "terminal selection differs")
    for name, expected in (("initial_probabilities", cycles[0]["decision"]["prediction"]["probabilities"]),
                           ("final_probabilities", cycles[-1]["decision"]["prediction"]["probabilities"])):
        _equal_probabilities(result.get(name), expected, name)
        _equal_probabilities(trajectory.get(name), expected, "trajectory " + name)
    for vector in (result.get("cost"), trajectory.get("cost")):
        _require(isinstance(vector, dict) and all(_number(x) and x >= 0 for x in vector.values()), "invalid final cost")
        _require(all(abs(vector.get(u, 0)-spent.get(u, 0)) <= 1e-8 for u in set(vector) | set(spent)), "final cost differs from receipts")
    return {"match_id": expected_match["match_id"], "status": "passed", "actions": selected,
            "terminal_decision": result["terminal_selection"]["decision"],
            "stop_reason": result["terminal_selection"]["reason"], "cost": result["cost"],
            "decision_cycles": len(cycles), "unsupported_candidates": dict(unsupported)}


def _backend():
    from football_agent.historical_analysis import HistoricalAnalysisSession
    session = None
    session_root = None

    def catalog(*, root):
        nonlocal session, session_root
        session_root = Path(root).resolve()
        session = HistoricalAnalysisSession(root=session_root)
        return session.load_catalog()

    def analyze(match, *, output, root):
        _require(session is not None and Path(root).resolve() == session_root,
                 "batch analysis must use the verified catalog snapshot")
        return session.analyze_match(match, output=output)

    return catalog, analyze


def _write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_acceptance(output, *, root=ROOT, limit=None):
    _require(limit is None or (type(limit) is int and limit > 0), "limit must be a positive integer")
    output, root = Path(output).resolve(), Path(root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, cases, failures, sequence_counts = [], [], [], Counter()
    total = 0
    with offline_guard() as network_attempts:
        try:
            load_catalog, analyze_match = _backend()
            catalog = load_catalog(root=root)
            _require(isinstance(catalog, list) and bool(catalog), "empty historical catalog")
            _label_free(catalog, "catalog")
            ids = [item["match_id"] for item in catalog]
            _require(len(ids) == len(set(ids)), "duplicate historical match identity")
            total = len(catalog)
            for index, item in enumerate(catalog if limit is None else catalog[:limit]):
                folder = output / "matches" / f"{index+1:04d}"
                try:
                    result = analyze_match(item["match_id"], output=folder, root=root)
                    for name in ("analysis.json", "trajectory.json", "report.md"):
                        _require((folder/name).is_file(), "missing analysis artifact: " + name)
                    stored = json.loads((folder/"trajectory.json").read_text(encoding="utf-8"))
                    _require(stored == result["trajectory"], "saved trajectory differs from executed result")
                    record = validate_analysis(result, item)
                    record["artifact_directory"] = str(folder)
                    key = tuple(record["actions"])
                    sequence_counts[key] += 1
                    if sequence_counts[key] == 1:
                        cases.append({"actions": list(key), "match_id": item["match_id"],
                                      "artifact_directory": str(folder), "selection_basis": "first_occurrence_in_catalog_order"})
                except Exception as error:
                    record = {"match_id": item["match_id"], "status": "failed", "error_type": type(error).__name__, "error": str(error)}
                    failures.append(record)
                records.append(record)
                print(f"[{index+1}/{min(total, limit) if limit is not None else total}] {item['match_id']}: {record['status']}", flush=True)
        except Exception as error:
            failures.append({"stage": "catalog", "error_type": type(error).__name__, "error": str(error)})
    summary = {"schema": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "passed" if not failures and not network_attempts else "failed",
               "catalog_matches": total, "executed_matches": len(records), "explicit_limit": limit,
               "coverage": "full_catalog" if total and len(records) == total else "explicit_subset" if limit else "incomplete",
               "passed_matches": sum(r["status"] == "passed" for r in records),
               "failed_matches": sum(r["status"] == "failed" for r in records),
               "network_attempts": len(network_attempts), "independent_outcomes_required": False,
               "forecast_scores": {"status": "skipped", "reason": "acceptance_checks_autonomous_acquisition_not_prediction_ranking"},
               "scope": "real_historical_reconstruction_with_simulated_reveal_cost_not_live_open_world_validation",
               "selection_sequences": [{"actions": list(k), "matches": v} for k, v in sequence_counts.items()],
               "first_cases": cases, "failures": failures, "matches": records}
    _write(output/"summary.json", summary)
    lines = ["# FootballMind 历史人工体验验收", "", f"结果：{summary['status']}。历史目录 {total} 场，执行 {len(records)} 场，通过 {summary['passed_matches']} 场，失败 {summary['failed_matches']} 场。",
             f"范围：{'完整目录' if summary['coverage']=='full_catalog' else '部分目录或未完成'}；显式数量限制：{limit if limit is not None else '无'}。网络尝试：{len(network_attempts)}。", "",
             "验收关注机器自主比较信息价值、读取所选信息、更新概率、遵守预算与明确停止。历史特征为受控揭示，成本为模拟揭示成本；本报告不声称在线收益或独立预测优势。预测分数按本次交付目标跳过。", "", "## 实际选择路径", ""]
    for row in summary["selection_sequences"]:
        lines.append(f"- {' → '.join(row['actions']) if row['actions'] else '直接停止或暂缓'}：{row['matches']} 场。")
    lines += ["", "## 可人工查看的案例", "", "每种动作路径只取目录顺序中的第一次出现；没有按比分、预测分数或输赢挑选。", ""]
    for case in cases:
        lines.append(f"- {case['match_id']}：{case['artifact_directory']}\\report.md")
    if failures:
        lines += ["", "## 需修复的执行或数据错误", ""]
        for failure in failures:
            lines.append(f"- {failure.get('match_id', failure.get('stage'))}：{failure['error_type']} — {failure['error']}")
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新的验收目录，不覆盖已有结果")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int, help="显式限制比赛数量；省略则运行全部历史目录")
    args = parser.parse_args(argv)
    try:
        summary = run_acceptance(args.output, root=args.root, limit=args.limit)
    except (ValueError, OSError) as error:
        print(f"无法验收：{error}", file=sys.stderr)
        return 2
    print(json.dumps({k: summary[k] for k in ("status", "catalog_matches", "executed_matches", "passed_matches", "failed_matches", "network_attempts")}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
