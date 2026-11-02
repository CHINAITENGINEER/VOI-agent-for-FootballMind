"""Frozen-state horizon ablations; no counterfactual source acquisition."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.decision_value import VALUE_TOLERANCE
from football_agent.semantic_scenario_model import SemanticScenarioModel
from football_agent.semantic_scenario_runtime import digest, replay_math


def _identity(run):
    match = run.get("configuration", {}).get("match", {})
    if not isinstance(match, dict):
        raise ValueError("Missing fixture identity")
    identity = {k: match.get(k) for k in
                ("match_id", "target_key", "match", "competition", "kickoff", "prediction_cutoff")}
    if not any(identity[k] for k in ("match_id", "target_key", "match")) or not identity["prediction_cutoff"]:
        raise ValueError("Missing fixture identity or cutoff")
    return identity


def _state(cycle):
    # Include semantic assumptions, costs, evidence, capabilities and budget.
    # Two independent LLM proposals are NOT a controlled horizon experiment.
    manifest = deepcopy(cycle["model"])
    manifest.pop("planning_horizon", None)
    return {"model": manifest, "observations": cycle["observations"],
            "remaining_requests": cycle["remaining_requests"],
            "candidate_bindings": cycle.get("candidate_bindings", {})}


def _summary(decision):
    selected = decision["selected_action_id"]
    c = next((r for r in decision["candidates"] if r["action_id"] == selected), None)
    return {"selected_action_id": selected, "stop_reason": decision["stop_reason"],
            "prediction": decision["prediction_nominal"],
            "immediate_net": c.get("net") if c else None,
            "planning_net": c.get("planning_net", c.get("net")) if c else None,
            "planning_lower": c.get("planning_lower", c.get("lower")) if c else None,
            "continuation_actions": c.get("continuation_actions", []) if c else []}


def _paired(index, first, second):
    one, two = _summary(first), _summary(second)
    changed = one["selected_action_id"] != two["selected_action_id"]
    previous = next((r for r in second["candidates"]
                     if r["action_id"] == one["selected_action_id"]), None)
    previous_value = previous.get("planning_lower", previous.get("lower")) if previous else None
    delta = (two["planning_lower"] - previous_value
             if two["planning_lower"] is not None and previous_value is not None else None)
    near_tie = bool(changed and delta is not None and abs(delta) <= VALUE_TOLERANCE)
    enabled = bool(one["selected_action_id"] is None and two["selected_action_id"] is not None
                   and two["planning_lower"] is not None and two["planning_lower"] > VALUE_TOLERANCE)
    by_id = {r["action_id"]: r for r in first["candidates"]}
    rows = []
    for r in second["candidates"]:
        before = by_id[r["action_id"]]
        rows.append({"action_id": r["action_id"], "question": r["question"],
                     "feasible": r["feasible"], "reason": r["reason"],
                     "immediate_net": before.get("net"), "immediate_lower": before.get("lower"),
                     "planning_net": r.get("planning_net", r.get("net")),
                     "planning_lower": r.get("planning_lower", r.get("lower")),
                     "continuation_actions": r.get("continuation_actions", [])})
    return {"cycle_index": index, "h1": one, "h2": two,
            "choice_changed": changed, "changed_choice_is_near_tie": near_tie,
            "h2_value_delta_vs_h1_action_under_same_h2_policy": delta,
            "value_tolerance": VALUE_TOLERANCE,
            "h2_enables_acquisition_where_h1_stops": enabled, "candidates": rows}


def analyze_frozen_run(run):
    """Local ablation at actual observed states; never fabricate an h2 rollout."""
    identity = _identity(run)
    if not run.get("cycles"):
        return {"schema": "footballmind-frozen-horizon-analysis-v1", "status": "not_available",
                "fixture": identity, "reason": "No accepted frozen decision state exists",
                "network_calls": 0, "semantic_calls": 0, "comparisons": []}
    verified = replay_math(run)
    rows = []
    for cycle in run["cycles"]:
        original = SemanticScenarioModel.from_manifest(cycle["model"])
        # Explicitly use the CURRENT planner while retaining all scientific
        # inputs and group semantics from the saved model. Old replay is above.
        model = SemanticScenarioModel(original.proposal, original.baseline,
            request_loss_cost=original.request_loss_cost,
            contamination_levels=original.contamination_levels,
            action_request_costs=original.action_request_costs,
            observation_group_policy=original.observation_group_policy,
            planning_horizon=2)
        h1 = model.evaluate(cycle["observations"], cycle["remaining_requests"], horizon=1)
        h2 = model.evaluate(cycle["observations"], cycle["remaining_requests"], horizon=2)
        if h1["prediction_nominal"] != h2["prediction_nominal"]:
            raise AssertionError("Changing horizon changed current belief")
        row = _paired(cycle["index"], h1, h2)
        row.update(state_sha256=digest(_state(cycle)), frozen_manifest_sha256=digest(cycle["model"]),
                   analysis_manifest_sha256=digest(model.manifest()),
                   remaining_requests=cycle["remaining_requests"],
                   observations=deepcopy(cycle["observations"]))
        rows.append(row)
    return {"schema": "footballmind-frozen-horizon-analysis-v1", "status": "completed",
            "scope": "same_frozen_state_planning_ablation_not_counterfactual_rollout",
            "fixture": identity, "source_run_id": run.get("run_id"), "source_run_status": run.get("status"),
            "replay_verification": verified, "network_calls": 0, "semantic_calls": 0,
            "changed_choices": sum(r["choice_changed"] for r in rows),
            "near_tied_changed_choices": sum(r["changed_choice_is_near_tie"] for r in rows),
            "h2_acquisition_where_h1_stops": sum(r["h2_enables_acquisition_where_h1_stops"] for r in rows),
            "comparisons": rows,
            "claims_boundary": "Uncalibrated frozen assumptions; modeled continuation only. No unseen bodies, "
                "outcome accuracy, empirical g, realized policy gain or global stop certificate."}


def compare(h1, h2):
    identity = _identity(h1)
    if identity != _identity(h2):
        raise ValueError("h1 and h2 are not bound to the same fixture/cutoff")
    if (h1.get("configuration", {}).get("planning_horizon") != 1
            or h2.get("configuration", {}).get("planning_horizon") != 2):
        raise ValueError("expected planning horizons 1 and 2")
    result = {"schema": "footballmind-horizon-comparison-v2", "status": "not_comparable", "fixture": identity,
              "claims_boundary": "Only shared frozen states support planning-depth comparison; "
                  "separate live trajectories are not a controlled policy-effect experiment.",
              "runs": {"h1": {"status": h1.get("status")}, "h2": {"status": h2.get("status")}}}
    if h1.get("status") != "completed" or h2.get("status") != "completed":
        result["reason"] = "both runs must complete before action or planning-value comparison"
        return result
    if not h1.get("cycles") or not h2.get("cycles"):
        result["reason"] = "missing frozen decision states"
        return result
    result["replay_verification"] = {"h1": replay_math(h1), "h2": replay_math(h2)}
    if digest(_state(h1["cycles"][0])) != digest(_state(h2["cycles"][0])):
        result["reason"] = "initial model, observations, candidate bindings or budget differ"
        return result
    states = {digest(_state(c)): c for c in h1["cycles"]}
    rows = []
    for c in h2["cycles"]:
        shared = states.get(digest(_state(c)))
        if shared is not None:
            row = _paired(c["index"], shared["decision"], c["decision"])
            row["h1_cycle_index"] = shared["index"]
            rows.append(row)
    result.update(status="comparable", scope="shared_frozen_states_only", comparisons=rows,
                  h2_positive_continuation_evidence=[r for r in rows if r["h2_enables_acquisition_where_h1_stops"]],
                  unmatched_h2_states=len(h2["cycles"])-len(rows))
    return result


def render_report(report):
    lines = ["FootballMind：相同冻结信息下的一步／二步 VoI 对照", "",
             f"状态：{report['status']}；新增 HTTP：0；新增语义调用：0。",
             "情景、概率、观察历史、候选、成本和预算固定；只改变规划深度。",
             "每个周期是原轨迹上的局部重算，不拼接为另一个策略的真实运行。", ""]
    if report.get("reason"):
        lines.append(report["reason"])
    for r in report.get("comparisons", []):
        lines.append(f"周期 {r['cycle_index']}：h1={r['h1']['selected_action_id'] or 'STOP'}；"
                     f"h2={r['h2']['selected_action_id'] or 'STOP'}")
        if r.get("changed_choice_is_near_tie"):
            lines.append("  动作不同但二步价值在数值容差内并列；不能将顺序变化作为优势证据。")
        for c in r["candidates"]:
            if c["feasible"]:
                lines.append(f"  {c['action_id']}：一步净值下包络={c['immediate_lower']:.10g}；"
                             f"二步规划净值下包络={c['planning_lower']:.10g}；{c['question']}")
        lines.append(f"  一步停止而二步值得获取：{r['h2_enables_acquisition_where_h1_stops']}")
    lines.extend(["", "结果条件于未经校准的冻结假设；未获取反事实正文，不证明真实预测改善或全局信息充分。"])
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="compute both horizons on each state of ONE frozen run")
    parser.add_argument("--h1", type=Path, help="completed horizon=1 run.json")
    parser.add_argument("--h2", type=Path, help="completed horizon=2 run.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.run) == bool(args.h1 or args.h2) or (not args.run and not (args.h1 and args.h2)):
        parser.error("Use either --run or both --h1 and --h2")
    if args.output.exists() or args.output.with_suffix(".txt").exists():
        parser.error("Choose new output paths; existing artifacts are immutable")
    paths = [args.run] if args.run else [args.h1, args.h2]
    try:
        raws = [p.read_bytes() for p in paths]
        runs = [json.loads(raw) for raw in raws]
        report = analyze_frozen_run(runs[0]) if args.run else compare(*runs)
        report["source_files"] = [{"path": str(p.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
                                  for p, raw in zip(paths, raws)]
        if any(p.read_bytes() != raw for p, raw in zip(paths, raws)):
            raise ValueError("Source run changed during analysis")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    args.output.with_suffix(".txt").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "compared_states": len(report.get("comparisons", [])),
                      "changed_choices": report.get("changed_choices"),
                      "h2_acquisition_where_h1_stops": report.get("h2_acquisition_where_h1_stops")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
