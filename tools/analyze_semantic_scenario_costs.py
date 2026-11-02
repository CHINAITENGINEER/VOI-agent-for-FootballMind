"""Post-hoc cost sensitivity of a frozen state; never an acquisition rollout."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.semantic_scenario_model import SemanticScenarioModel
from football_agent.semantic_scenario_runtime import digest, replay_math


DEFAULT_COSTS = (0., .0001, .001, .01)


def analyze_run(run, costs=DEFAULT_COSTS, *, cycle_index=0):
    """Recompute one saved frontier with immutable laws, observations and budget."""
    verified = replay_math(run)
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int) or not 0 <= cycle_index < len(run["cycles"]):
        raise ValueError("cycle_index must select an existing saved cycle")
    costs = tuple(costs)
    if not costs or any(isinstance(c, bool) or not isinstance(c, (int, float)) or
                        not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError("costs must be nonnegative finite numbers")
    if len(set(costs)) != len(costs):
        raise ValueError("costs must be unique")
    cycle = run["cycles"][cycle_index]
    original = SemanticScenarioModel.from_manifest(cycle["model"])
    zero_manifest = deepcopy(cycle["model"])
    zero_manifest["request_loss_cost"] = 0.
    zero_model = SemanticScenarioModel.from_manifest(zero_manifest)
    zero = zero_model.evaluate(cycle["observations"], cycle["remaining_requests"])
    thresholds = []
    for candidate in zero["candidates"]:
        binding = cycle.get("candidate_bindings", {}).get(candidate["action_id"], {})
        row = {"action_id": candidate["action_id"], "question": candidate["question"],
               "url": binding.get("url"), "query": binding.get("query"),
               "request_cost": candidate["request_cost"], "feasible": candidate["feasible"],
               "gross_nominal": candidate["gross"], "gross_lower": candidate["gross_lower"],
               "gross_upper": candidate["gross_upper"],
               "nominal_break_even_cost_per_request": None,
               "robust_break_even_cost_per_request": None,
               "upper_break_even_cost_per_request": None,
               "model_gross_and_thresholds": {}}
        if candidate["feasible"]:
            requests = candidate["request_cost"]
            row.update(nominal_break_even_cost_per_request=candidate["gross"] / requests,
                       robust_break_even_cost_per_request=candidate["gross_lower"] / requests,
                       upper_break_even_cost_per_request=candidate["gross_upper"] / requests,
                       model_gross_and_thresholds={key: {"gross": value["gross"],
                           "break_even_cost_per_request": value["gross"] / requests}
                           for key, value in candidate["model_values"].items()})
        thresholds.append(row)
    variants = []
    for cost in costs:
        manifest = deepcopy(cycle["model"])
        manifest["request_loss_cost"] = float(cost)
        model = SemanticScenarioModel.from_manifest(manifest)
        decision = model.evaluate(cycle["observations"], cycle["remaining_requests"])
        if decision["prediction_nominal"] != zero["prediction_nominal"]:
            raise AssertionError("Changing acquisition cost must not change the frozen posterior")
        by_id = {row["action_id"]: row for row in zero["candidates"]}
        for row in decision["candidates"]:
            reference = by_id[row["action_id"]]
            if row["feasible"] != reference["feasible"] or row["gross"] != reference["gross"]:
                raise AssertionError("Changing cost altered request feasibility or gross information value")
        variants.append({"request_loss_cost": cost, "selected_action_id": decision["selected_action_id"],
                         "stop_reason": decision["stop_reason"], "prediction_nominal": decision["prediction_nominal"],
                         "candidates": decision["candidates"]})
    return {"schema": "semantic-scenario-posthoc-cost-sensitivity-v1",
            "scope": "posthoc_same_frozen_state_sensitivity_not_cost_selection_or_policy_evaluation",
            "source_run_id": run.get("run_id"), "cycle_index": cycle_index,
            "frozen_model_id": original.model_id, "frozen_manifest_sha256": digest(cycle["model"]),
            "frozen_proposal_sha256": digest(cycle["model"]["proposal"]),
            "original_request_loss_cost": original.request_loss_cost,
            "original_selected_action_id": cycle["decision"]["selected_action_id"],
            "original_stop_reason": cycle["decision"]["stop_reason"],
            "remaining_requests": cycle["remaining_requests"],
            "observations": deepcopy(cycle["observations"]),
            "baseline": list(original.baseline), "contamination_levels": list(original.contamination_levels),
            "action_request_costs": dict(original.action_request_costs),
            "action_thresholds": thresholds, "cost_variants": variants,
            "replay_verification": verified, "http_requests": 0, "semantic_calls": 0,
            "threshold_semantics": "zero_net_cost_gross_divided_by_requests_in_expected_log_loss_nats; selection_requires_positive_net_above_solver_tolerance",
            "restrictions": ["Post-hoc sensitivity does not choose or replace the main-run cost.",
                "Hypotheses, priors, conditional outcomes, likelihoods, observations, candidate menu and request budget are frozen.",
                "Observation laws are uncalibrated semantic assumptions; bounds are not confidence intervals.",
                "No counterfactual article bodies, future results or later policy outcomes are produced.",
                "Zero-cost acquisition does not demonstrate an advantage after real resource costs."]}


def render_report(result):
    lines = ["冻结主运行的事后成本敏感性分析", "",
        "本分析仅重新计算同一个已保存状态，不获取网页、不调用语义模型，也不修改主运行。",
        "它不用于选择新的成本或把主运行的停止改写为成功。所有情景概率与观察律仍是未经统计校准的语义假设。",
        f"主运行：{result['source_run_id']}；周期：{result['cycle_index']}；剩余请求：{result['remaining_requests']}",
        f"原成本：{result['original_request_loss_cost']:.8g} nats/请求；原动作：{result['original_selected_action_id'] or 'STOP'}；原因：{result['original_stop_reason']}",
        "", "相同冻结状态下的重算："]
    for variant in result["cost_variants"]:
        selected = variant["selected_action_id"]
        candidate = next((row for row in variant["candidates"] if row["action_id"] == selected), None)
        suffix = f"；净价值下包络 {candidate['lower']:.12g}" if candidate else "；" + variant["stop_reason"]
        lines.append(f"单位成本 {variant['request_loss_cost']:.8g} → {selected or 'STOP'}{suffix}")
    lines.extend(["", "各动作的零净价值单位成本阈值（gross / 该动作请求数；数值单位为 nats/请求）：",
                  "阈值是声明模型下的盈亏平衡点，不是估计置信区间；达到阈值不满足严格正净价值选择规则。"])
    for row in result["action_thresholds"]:
        lines.append(f"- {row['action_id']}（最多 {row['request_cost']} 请求）")
        lines.append("  " + row["question"])
        if row["feasible"]:
            lines.append(f"  名义阈值 {row['nominal_break_even_cost_per_request']:.12g}；污染模型下阈值 {row['robust_break_even_cost_per_request']:.12g}")
            lines.append(f"  名义总信息价值 {row['gross_nominal']:.12g}；总信息价值下包络 {row['gross_lower']:.12g}")
        else:
            lines.append("  本状态不可执行，未计算阈值。")
    lines.extend(["", "解释边界：", "- 停止与获取的变化仅说明决策对所声明成本敏感，不能证明查询有效或预测改善。",
                  "- 0 成本情况下的读取只能检验执行、观察和重新计算机制，不能算作主策略成功。",
                  "- 本次没有执行反事实读取，没有生成未读取正文，也没有计算未来赛果收益。",
                  f"- 原冻结运行数学回放：{result['replay_verification']['status']}；{result['replay_verification']['cycles']} 个周期；0 HTTP / 0 语义调用。"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycle", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new directory; sensitivity artifacts are immutable")
    raw = args.run.read_bytes()
    result = analyze_run(json.loads(raw), cycle_index=args.cycle)
    result["source_run_path"] = str(args.run.resolve())
    result["source_run_sha256"] = sha256(raw).hexdigest()
    if args.run.read_bytes() != raw:
        raise RuntimeError("Source run changed during analysis")
    args.output.mkdir(parents=True)
    (args.output / "sensitivity.json").write_text(json.dumps(result, ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "report.txt").write_text(render_report(result), encoding="utf-8")
    (args.output / "replay.json").write_text(json.dumps(result["replay_verification"],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "replay": result["replay_verification"],
                      "decisions": [{"cost": row["request_loss_cost"], "selected": row["selected_action_id"]}
                                    for row in result["cost_variants"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
