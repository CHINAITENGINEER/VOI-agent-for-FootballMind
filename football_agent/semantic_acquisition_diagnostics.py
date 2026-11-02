"""Finite-grid assumption checks at saved decision states, without new evidence.

These are local sensitivity calculations, not a new policy, a confidence
interval, a calibration estimate, or a counterfactual rollout. Every variant
uses the existing numerical solver with one explicitly changed assumption.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math

from .decision_value import VALUE_TOLERANCE
from .semantic_scenario_model import SemanticScenarioModel
from .semantic_scenario_runtime import digest, replay_math


VERSION = "semantic-acquisition-sensitivity-v1"
GRID = {
    "prior_odds": (.5, 2.),
    "outcome_contrast": (.5, 0.),
    "observation_signal": (.75, .5, 0.),
    "answer_retention": (.75, .5),
    "cost_multiplier": (.5, 2., 10.),
}


def vary_manifest(manifest, specification):
    """Change exactly one declared assumption and re-establish joint coherence.

    Signal retention mixes one action's law with its original prior-predictive
    marginal. Answer retention transfers supports/refutes mass to no_answer,
    equally across hypotheses. Both preserve uninformative missingness.
    Outcome contrast mixes ALIGNED conditional outcomes with the fixed baseline.
    Prior odds changes re-align the Y marginal to that same external baseline.
    """
    family, amount = specification["family"], specification["amount"]
    if family not in GRID or type(amount) not in (int, float) or amount not in GRID[family]:
        raise ValueError("Unsupported sensitivity grid point")
    original = SemanticScenarioModel.from_manifest(manifest)
    changed = deepcopy(manifest)
    proposal = changed["proposal"]
    if family == "cost_multiplier":
        changed["request_loss_cost"] *= amount
    elif family == "prior_odds":
        target = specification["hypothesis_id"]
        if target not in original.hypothesis_ids:
            raise ValueError("Unknown sensitivity hypothesis")
        weights = [h["prior"] * (amount if h["id"] == target else 1.) for h in proposal["hypotheses"]]
        total = math.fsum(weights)
        for h, weight in zip(proposal["hypotheses"], weights):
            h["prior"] = weight / total
    elif family == "outcome_contrast":
        for h, row in zip(proposal["hypotheses"], original.aligned_outcome_probabilities):
            h["outcome_probabilities"] = [amount*p + (1.-amount)*b for p, b in zip(row, original.baseline)]
    else:
        target = specification["action_id"]
        if target not in original.actions:
            raise ValueError("Unknown sensitivity action")
        action = next(a for a in proposal["actions"] if a["action_id"] == target)
        law = original.likelihoods[target]
        marginal = [math.fsum(w*law[h][z] for h, w in zip(original.hypothesis_ids, original.prior))
                    for z in range(3)]
        no_answer = next(iter(law.values()))[2]
        for row in action["likelihoods"]:
            p = law[row["hypothesis_id"]]
            if family == "observation_signal":
                row["probabilities"] = [amount*p[z] + (1.-amount)*marginal[z] for z in range(2)] + [no_answer]
            else:
                row["probabilities"] = [amount*p[0], amount*p[1], 1.-amount*(1.-no_answer)]
    # Rebuild through the actual validator/solver; never patch candidate values.
    return SemanticScenarioModel.from_manifest(changed).manifest()


def grid_for_state(model, decision):
    for family in ("cost_multiplier", "outcome_contrast"):
        for amount in GRID[family]:
            yield {"family": family, "amount": amount}
    for hypothesis_id in model.hypothesis_ids:
        for amount in GRID["prior_odds"]:
            yield {"family": "prior_odds", "hypothesis_id": hypothesis_id, "amount": amount}
    # Never modify an attempted observation's law in these action-specific
    # checks. The frozen evidence and current belief must remain unchanged.
    for candidate in decision["candidates"]:
        if candidate["feasible"]:
            for family in ("observation_signal", "answer_retention"):
                for amount in GRID[family]:
                    yield {"family": family, "action_id": candidate["action_id"], "amount": amount}


def decision_summary(decision):
    values = {row["action_id"]: row.get("planning_lower", row["lower"])
              for row in decision["candidates"] if row["feasible"]}
    selected = decision["selected_action_id"]
    best = max([0., *values.values()])
    competitors = [0.] + [v for action, v in values.items() if action != selected]
    tied = sorted(action for action, value in values.items() if best-value <= VALUE_TOLERANCE)
    if best <= VALUE_TOLERANCE:
        tied.append("STOP")
    return {"selected_action_id": selected, "stop_reason": decision["stop_reason"],
            "prediction": decision["prediction_nominal"], "candidate_lower_values": values,
            "near_optimal_actions_within_tolerance": tied,
            "selected_lower_value": values[selected] if selected is not None else 0.,
            "margin_to_next_action_or_stop": best-max(competitors) if selected is not None else
                (-max(values.values()) if values else None)}


def _variant_result(original, decision, spec, model):
    summary = decision_summary(decision)
    original_action = original["selected_action_id"]
    original_value = 0. if original_action is None else summary["candidate_lower_values"][original_action]
    shortfall = max(0., summary["selected_lower_value"] - original_value)
    changed = original_action != summary["selected_action_id"]
    prediction_delta = max(abs(a-b) for a, b in zip(original["prediction"], summary["prediction"]))
    return {"specification": spec, "model_sha256": digest(model.manifest()), **summary,
            "choice_changed": changed,
            "changed_beyond_value_tolerance": changed and shortfall > VALUE_TOLERANCE,
            "original_action_lower_value_under_variant": original_value,
            "modeled_shortfall_of_original_action": shortfall,
            "current_prediction_max_abs_change": prediction_delta}


def semantic_call_profile(run):
    """Attribute measured overhead; no assumed speed-up for unexecuted code."""
    by_protocol, invalid = {}, []
    for call in run.get("semantic_calls", []):
        protocol = call.get("protocol", "unknown")
        group = by_protocol.setdefault(protocol, {"calls": 0, "statuses": Counter(),
            "reported_seconds": 0., "calls_without_latency": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "calls_without_token_usage": 0,
            "input_utf8_bytes": 0})
        group["calls"] += 1
        group["statuses"][call.get("status", "pending")] += 1
        group["input_utf8_bytes"] += len(json.dumps(call.get("input", {}), ensure_ascii=False,
                                                   sort_keys=True).encode("utf-8"))
        receipt = call.get("receipt") or {}
        seconds = receipt.get("latency_seconds")
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            group["reported_seconds"] += seconds
        else:
            group["calls_without_latency"] += 1
        usage = receipt.get("usage") or []
        if isinstance(usage, dict):
            usage = [usage]
        reported = False
        for item in usage:
            if not isinstance(item, dict):
                continue
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                n = item.get(key)
                if type(n) is int and n >= 0:
                    group[key] += n
                    reported = True
        if not reported:
            group["calls_without_token_usage"] += 1
        if call.get("status") in {"invalid", "failed"}:
            invalid.append({"index": call.get("index"), "protocol": protocol,
                            "status": call.get("status"), "reported_seconds": seconds})
    return {"by_protocol": by_protocol, "unsuccessful_calls": invalid,
            "unsuccessful_reported_seconds": math.fsum(
                c["reported_seconds"] for c in invalid if type(c["reported_seconds"]) in (int, float)
                and math.isfinite(c["reported_seconds"]) and c["reported_seconds"] >= 0),
            "scope": "recorded_overhead_not_estimated_savings_or_policy_utility"}


def analyze_acquisition(run):
    identity = run.get("configuration", {}).get("match", {})
    if not identity.get("prediction_cutoff") or not any(identity.get(k) for k in ("match_id", "target_key", "match")):
        raise ValueError("Missing fixture identity or prediction cutoff")
    verified = replay_math(run)
    rows = []
    for cycle in run["cycles"]:
        original_model = SemanticScenarioModel.from_manifest(cycle["model"])
        original_decision = cycle["decision"]
        original = decision_summary(original_decision)
        entry = {"cycle_index": cycle["index"], "state_sha256": digest(cycle),
                 "remaining_requests": cycle["remaining_requests"], "original": original,
                 "variations": [], "status": "analyzed"}
        if not any(c["feasible"] for c in original_decision["candidates"]):
            entry.update(status="not_actionable", reason=original_decision["stop_reason"])
            rows.append(entry)
            continue
        for spec in grid_for_state(original_model, original_decision):
            manifest = vary_manifest(cycle["model"], spec)
            model = SemanticScenarioModel.from_manifest(manifest)
            decision = model.evaluate(cycle["observations"], cycle["remaining_requests"],
                                      horizon=original_decision["horizon"])
            result = _variant_result(original, decision, spec, model)
            if spec["family"] in {"observation_signal", "answer_retention", "cost_multiplier"}:
                if result["current_prediction_max_abs_change"] > 1e-10:
                    raise ArithmeticError("Prospective acquisition/cost perturbation changed current belief")
            entry["variations"].append(result)
        changed = [v for v in entry["variations"] if v["changed_beyond_value_tolerance"]]
        entry.update(grid_points=len(entry["variations"]), meaningful_choice_changes=len(changed),
                     choice_counts=dict(Counter(v["selected_action_id"] or "STOP" for v in entry["variations"])),
                     maximum_modeled_shortfall=max(v["modeled_shortfall_of_original_action"] for v in entry["variations"]))
        rows.append(entry)
    return {"schema": VERSION, "status": "completed" if rows else "not_available",
            "scope": "one_factor_finite_grid_at_saved_states_not_a_counterfactual_rollout",
            "source_run_id": run.get("run_id"), "source_run_status": run.get("status"),
            "fixture": {k: deepcopy(identity[k]) for k in
                        ("match_id", "match", "home_name", "away_name", "kickoff", "prediction_cutoff") if k in identity},
            "replay_verification": verified, "grid": deepcopy(GRID), "value_tolerance": VALUE_TOLERANCE,
            "grid_definitions": {
                "prior_odds": "multiply one hypothesis prior by factor, renormalize, re-align baseline",
                "outcome_contrast": "factor * aligned q(y|theta) + (1-factor) * external baseline",
                "observation_signal": "factor * selected q(z|theta) + (1-factor) * its original prior-predictive q(z)",
                "answer_retention": "multiply supports/refutes probabilities by factor, transfer remaining mass to no_answer",
                "cost_multiplier": "multiply loss-equivalent per-request cost; integer request budget is fixed"},
            "network_calls": 0, "semantic_calls": 0, "used_for_action_selection": False,
            "cycles": rows, "semantic_call_profile": semantic_call_profile(run),
            "claims_boundary": "Grid counts are not confidence/probability estimates. No target outcomes, "
                "unseen bodies, empirical g, realized improvement or global robustness certificate. "
                "Each point re-solves its declared-model shared continuation; values do not constitute "
                "one policy robust across the entire sensitivity grid. Baseline remains fixed; "
                "prior/outcome variants may change the posterior at states with informative history."}


def render_acquisition_report(report):
    lines = ["FootballMind：信息选择的假设敏感性", "",
             "冻结同一比赛、已见证据、候选身份与请求预算；逐项改变声明假设并重新求解。",
             "新增网络和模型调用均为0。以下格点没有概率权重，翻转比例不能解释为置信度。",
             "每个状态独立重算，不把不同假设的动作拼接成一次实际运行。", ""]
    for state in report["cycles"]:
        original = state["original"]
        lines += [f"周期 {state['cycle_index']}：原选择 {original['selected_action_id'] or 'STOP'}；"
                  f"剩余请求 {state['remaining_requests']}"]
        if state["status"] == "not_actionable":
            lines += [f"  无可执行动作：{state['reason']}；不将预算停止算作稳定性证据。", ""]
            continue
        lines += [f"  {state['grid_points']} 个预设格点，{state['meaningful_choice_changes']} 个发生非并列动作变化。",
                  f"  原选择相对下一候选/停止的净值差：{original['margin_to_next_action_or_stop']:.9g} nats。",
                  "  会改变选择的假设："]
        if len(original["near_optimal_actions_within_tolerance"]) > 1:
            lines.insert(len(lines)-1, "  原状态存在数值并列，不能声称首选唯一优于其他动作：" +
                         ", ".join(original["near_optimal_actions_within_tolerance"]))
        changed = [v for v in state["variations"] if v["changed_beyond_value_tolerance"]]
        for v in changed:
            s = v["specification"]
            target = s.get("action_id", s.get("hypothesis_id", "all"))
            lines.append(f"    {s['family']} / {target} / {s['amount']:g} → "
                         f"{v['selected_action_id'] or 'STOP'}；原动作的模型内净值差 "
                         f"{v['modeled_shortfall_of_original_action']:.9g} nats")
        if not changed:
            lines.append("    所检查格点未发现非并列变化；未证明格点之外稳定。")
        lines.append("")
    lines += ["调用开销（已记录值，非预计优化收益）："]
    for protocol, value in report["semantic_call_profile"]["by_protocol"].items():
        lines.append(f"  {protocol}：{value['calls']}次，{value['reported_seconds']:.2f}秒；"
                     f"输入token={value['prompt_tokens']}，输出token={value['completion_tokens']}；"
                     f"缺时延记录={value['calls_without_latency']}，缺token记录={value['calls_without_token_usage']}")
    lines += ["", "限制：没有训练或验证经验g，没有重判正文，也没有检验真实预测收益；"
              "诊断不参与本轮动作选择。"]
    return "\n".join(lines)+"\n"
