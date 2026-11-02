"""Independent calibrated continuation arithmetic and complete policy replay."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .analyze_round7 import match_means, paired
from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows
from .verify_round6 import KEYS, verify_aggregate_metrics
from .verify_round7 import verify_episode_coverage
from .verify_round8 import close, require
from .verify_round9 import check_estimate, reference_summary
from .verify_round13 import ContentReference, digest, verify_numbers as verify_marginal_numbers

MODES = {"calibrated-adaptive-two-step": "adaptive", "calibrated-commit-two-step": "commit"}
REFERENCES = {"adaptive": "frozen-version-mean", "commit": "content-marginal-mean"}
KIND = "calibrated-continuation-planner-v1"
DESIGN_HASH = "9739f3f33abdb7fcdefd532b2f63f2f027a93ced692cb785ef01114e4e8eefc7"
SOURCE_REPORT_HASH = "63a157974e7e4c5f20ee6ad1a779041f11607b1c3c6a1dcb2c4c8267f2df8592"
UNITS = ("money", "requests", "latency_ms", "tokens")


def continuation_terms(rows, action, *, mode, eligible=True):
    """Finite conditional policy sum; all costs below are projected, not spent.

    No current outcome or chosen actual first-return category is an argument.
    Gain estimates are supplied from the independent old residual calculation.
    """
    require(mode in {"adaptive", "commit"}, "Unknown continuation mode")
    if rows:
        require(all(type(r["probability"]) in (int, float) and math.isfinite(r["probability"])
                    and r["probability"] > 0 for r in rows), "Invalid positive branch mass")
        close(math.fsum(r["probability"] for r in rows), 1., "Continuation mass")
    costs = {u: float(action["cost"].get(u, 0.)) for u in UNITS} if action else dict.fromkeys(UNITS, 0.)
    penalty = .5*costs["money"]
    supported = bool(rows) and all(r["expected_gain"] is not None for r in rows)
    enabled = eligible and supported
    average = math.fsum(r["probability"]*r["expected_gain"] for r in rows) if supported else None
    adaptive_bonus = math.fsum(r["probability"]*(r["expected_gain"]-penalty)
        for r in rows if round(r["expected_gain"]-penalty, 12) > 0) if enabled else 0.
    commit_query = enabled and round(average-penalty, 12) > 0
    commit_bonus = average-penalty if commit_query else 0.
    require(adaptive_bonus >= commit_bonus-1e-12, "Adaptive opportunity lost more than rounding tolerance")
    branches = []
    for row in rows:
        gain = row["expected_gain"]
        selected = enabled and (round(gain-penalty, 12) > 0 if mode == "adaptive" else commit_query)
        branches.append({**row, "continuation_action": action["action_id"] if selected else None,
                         "conditional_net_gain": gain-penalty if gain is not None else None})
    future_gain = math.fsum(b["probability"]*b["expected_gain"] for b in branches if b["continuation_action"])
    future_cost = {u: math.fsum(b["probability"]*costs[u] for b in branches if b["continuation_action"]) for u in UNITS}
    future_penalty = .5*future_cost["money"]
    return {"continuation_bonus": future_gain-future_penalty,
            "expected_continuation_gain": future_gain, "expected_continuation_cost": future_cost,
            "expected_continuation_cost_penalty": future_penalty,
            "adaptive_bonus": adaptive_bonus, "commit_bonus": commit_bonus,
            "surrogate_option_gap": adaptive_bonus-commit_bonus,
            "branches": branches, "continuation_supported": supported,
            "continuation_enabled": bool(enabled), "marginal_continuation_gain": average}


def select(candidates):
    eligible = [(r["selection_score"], r["action_id"]) for r in candidates
                if r["feasible"] and r["selection_score"] is not None and r["selection_score"] > 0]
    eligible.sort(key=lambda row: (-row[0], row[1]))
    return eligible[0][1] if eligible else None


class ContinuationReference(ContentReference):
    def __init__(self, *args):
        super().__init__(*args)
        self.plans = {}

    def current_estimate(self, fold, mid, aid, acquired, mode):
        acquired = tuple(sorted(acquired))
        if not acquired:
            branch = self.estimate(fold, mid, aid, acquired)["branches"][0]
            return {**branch["estimate"], "model_evsi": branch["model_evsi"]}
        if mode == "commit":
            return self.estimate(fold, mid, aid, acquired)
        _, _, values = self.public.state(fold, mid, acquired)
        evsi = max(0., values[aid]["evsi"])
        return {**reference_summary(self.parents[fold][aid, acquired], evsi), "model_evsi": evsi}

    def plan(self, fold, mid, acquired, budget, mode):
        acquired = tuple(sorted(acquired))
        key = fold, mid, acquired, budget, mode
        if key in self.plans:
            return self.plans[key]
        spent = {u: math.fsum(a["cost"].get(u, 0.) for a in self.actions if a["action_id"] in acquired) for u in UNITS}
        limits = {"money": budget, "requests": 2.}
        candidates = []
        for action in sorted(self.actions, key=lambda a: a["action_id"]):
            aid = action["action_id"]
            if aid in acquired:
                continue
            cost = {u: float(action["cost"].get(u, 0.)) for u in UNITS}
            if not action.get("available", True):
                candidates.append({"action_id": aid, "feasible": False, "reason": "action_unavailable"})
                continue
            if any(spent[u]+cost[u] > limit+1e-12 for u, limit in limits.items()):
                candidates.append({"action_id": aid, "feasible": False, "reason": "physical_budget_exhausted"})
                continue
            estimate = self.current_estimate(fold, mid, aid, acquired, mode)
            gain = estimate["expected_gain"]
            penalty = .5*cost["money"]
            second = next((a for a in self.actions if a["action_id"] != aid and a["action_id"] not in acquired), None)
            rows, eligible = [], False
            if second:
                evidence = action.get("evidence_id")
                eligible = (second.get("available", True) and not (evidence is not None and evidence == second.get("evidence_id"))
                    and all(spent[u]+cost[u]+second["cost"].get(u, 0.) <= cap+1e-12 for u, cap in limits.items()))
                if eligible:
                    rows = self.estimate(fold, mid, second["action_id"], (aid,))["branches"]
            continuation = continuation_terms(rows, second, mode=mode, eligible=bool(eligible and gain is not None))
            total_gain = gain+continuation["expected_continuation_gain"] if gain is not None else None
            total_penalty = penalty+continuation["expected_continuation_cost_penalty"]
            net = gain-penalty+continuation["continuation_bonus"] if gain is not None else None
            candidates.append({"action_id": aid, "feasible": gain is not None,
                "acquired_action_ids": list(acquired), "immediate_estimate": estimate,
                "immediate_expected_gain": gain, "immediate_cost_penalty": penalty,
                "immediate_net_gain": gain-penalty if gain is not None else None,
                "expected_gain": total_gain, "cost_penalty": total_penalty, "mean_net_gain": net,
                "net_gain": net, "selection_score": round(net, 12) if net is not None else None,
                "expected_plan_cost_vector": {u: cost[u]+continuation["expected_continuation_cost"][u] for u in UNITS},
                **continuation})
        result = {"action_id": select(candidates), "candidates": candidates, "spent": spent}
        self.plans[key] = result
        return result


def verify_numbers(actual, expected, mode):
    require(actual.get("estimator_kind") == KIND, "Wrong continuation estimator kind")
    for key in ("model_evsi", "lower_gain", "standard_error"):
        require(actual.get(key) is None, "Invented plan-level EVSI or uncertainty")
    require(actual["acquired_action_ids"] == expected["acquired_action_ids"], "Acquired set differs")
    for key in ("immediate_expected_gain", "immediate_cost_penalty", "immediate_net_gain", "expected_gain", "cost_penalty",
                "mean_net_gain", "selection_score", "continuation_bonus", "expected_continuation_gain",
                "expected_continuation_cost_penalty", "adaptive_bonus", "commit_bonus", "surrogate_option_gap"):
        if expected[key] is None:
            require(actual[key] is None, "Unsupported planner value became numeric")
        else:
            close(actual[key], expected[key], "Independent continuation "+key)
    for name in ("expected_continuation_cost", "expected_plan_cost_vector"):
        for unit in UNITS:
            close(actual[name][unit], expected[name][unit], "Expected future/full physical resource "+unit)
    immediate = expected["immediate_estimate"]
    if mode == "commit" and expected["acquired_action_ids"]:
        verify_marginal_numbers(actual["immediate_estimate"], immediate)
    else:
        evsi = actual["immediate_estimate"]["model_evsi"]
        # For the initial state source records were independently integrated by
        # ContentReference; use its branch EVSI stored alongside the summary.
        if "model_evsi" in immediate:
            close(evsi, immediate["model_evsi"], "Immediate conditional EVSI")
        check_estimate(actual["immediate_estimate"], immediate, evsi)
    branches = {r["observation"]: r for r in actual["branches"]}
    wanted = {r["observation"]: r for r in expected["branches"]}
    require(len(branches) == len(actual["branches"]) and branches.keys() == wanted.keys(), "Continuation branch coverage differs")
    for token, branch in wanted.items():
        other = branches[token]
        close(other["probability"], branch["probability"], "Conditional return probability")
        close(other["model_evsi"], branch["model_evsi"], "Second-query branch EVSI")
        check_estimate(other["estimate"], branch["estimate"], branch["model_evsi"])
        if branch["expected_gain"] is None:
            require(other["expected_gain"] is None, "Unsupported future gain fabricated")
        else:
            close(other["expected_gain"], branch["expected_gain"], "Second-query calibrated mean")
        chosen = other["continuation_action_id"] if other["continuation_decision"] == "acquire" else None
        require(chosen == branch["continuation_action"], "Content-specific continuation differs")
    return len(branches)


def verify_trace(row, trace, source, actions, reference, estimator, mode):
    env, spent, counts = ControlledEpisode(source, actions), dict.fromkeys(UNITS, 0.), Counter()
    fold, mid = row["fold"], row["match_id"]
    events = trace["events"]
    require(events and row["lambda_money"] == .5, "Invalid continuation trajectory")
    _, initial, _ = reference.public.state(fold, mid, ())
    require(row["initial_probabilities"] == initial["probabilities"], "Free prior changed")
    for step, event in enumerate(events):
        acquired = tuple(env.performed_actions)
        state = {**env.visible_state(), "action_history": [{"action_id": aid, "status": "acquired_or_source_missing"} for aid in acquired]}
        require(event["step"] == step and event["visible_state"] == state, "Wrong actual state or hidden return access")
        _, prediction, _ = reference.public.state(fold, mid, acquired)
        require(event["prediction"]["probabilities"] == prediction["probabilities"], "Real posterior differs")
        wanted = reference.plan(fold, mid, acquired, row["budget_money"], mode)
        decision = event["decision"]
        require(decision["action_id"] == wanted["action_id"] and decision["decision"] == ("acquire" if wanted["action_id"] else "stop"),
                "Independent calibrated lookahead action/STOP differs")
        require(decision["risk_level"] == "mean" and decision["horizon"] == 2 and decision["estimator_version"] == estimator.version,
                "Wrong horizon/risk/planner version")
        for unit in UNITS:
            close(decision["physical_budget"]["spent"][unit], spent[unit], "Actual charge before planning")
        close(decision["physical_budget"]["limits"]["money"], row["budget_money"], "Planner money limit")
        close(decision["physical_budget"]["limits"]["requests"], 2., "Planner request limit")
        actual = {r["action_id"]: r for r in decision["candidates"]}
        candidates = {r["action_id"]: r for r in wanted["candidates"]}
        require(len(actual) == len(decision["candidates"]) and actual.keys() == candidates.keys(), "Wrong candidate coverage")
        for aid, expected in candidates.items():
            candidate = actual[aid]
            require(candidate["feasible"] == expected["feasible"], "Planner support/physical budget differs")
            if "expected_gain" not in expected:
                require(candidate["reason"] == expected["reason"], "Wrong infeasibility reason")
                continue
            counts["model_only_continuation_branches"] += verify_numbers(candidate, expected, mode)
            counts["candidate_values"] += 1
        selected = wanted["action_id"]
        if selected is None:
            require(step == len(events)-1 and "charged_cost" not in event and row["stop_reason"] == decision["reason"], "Activity after STOP")
            break
        action = next(a for a in env.legal_actions() if a["action_id"] == selected)
        require(event["charged_cost"] == action["cost"], "Projected costs were charged as actual")
        counts["negative_immediate_net_acquisitions"] += int(candidates[selected]["immediate_net_gain"] < 0)
        for unit in UNITS:
            spent[unit] += action["cost"].get(unit, 0.)
        env.reveal(selected)
        counts["charges"] += 1
    require(events[-1]["decision"]["decision"] == "stop", "No final STOP")
    require(row["actions"] == list(env.performed_actions) and row["final_mask"] == prediction["mask"], "Final actual evidence differs")
    require(row["final_probabilities"] == prediction["probabilities"], "Wrong final prediction")
    for unit in UNITS:
        close(row["cost"][unit], spent[unit], "Only executed calls charged")
    for key, value in losses(prediction["probabilities"], outcome(source)).items():
        close(row[key], value, "Actual-outcome score")
    counts["events"] = len(events)
    return counts


def load_reference(run, bundle, earlier_bundle):
    from .calibrated_continuation import CalibratedContinuationPlanner
    run = Path(run)
    report = json.loads((run/"report.json").read_bytes())
    sources = {r["match_id"]: r for path in (bundle, earlier_bundle) for r in load_episodes(path, lead_hours=24)[0]}
    actions = json.loads((run/"catalog.json").read_bytes())["actions"]
    models, calibrators, planners = {}, {}, {}
    for fold in report["folds"]:
        n = fold["fold"]
        jp, cp = run/fold["joint_model"]["path"], run/fold["source_calibrator"]["path"]
        require(digest(jp) == fold["joint_model"]["sha256"] and digest(cp) == fold["source_calibrator"]["sha256"], "Changed frozen planning input")
        model = models[n] = JointReturnPredictor.load(jp)
        cal = calibrators[n] = ResidualEVSICalibrator.load(cp, model=model, model_sha256=digest(jp))
        for mode, info in fold["planners"].items():
            require(digest(run/info["path"]) == info["sha256"], "Changed planning wrapper")
            planner = planners[n, mode] = CalibratedContinuationPlanner.load(run/info["path"], model=model,
                model_sha256=digest(jp), calibrator=cal, calibrator_sha256=digest(cp))
            require(planner.mode == mode and planner.version == info["version"] and planner.manifest == info["manifest"], "Wrong planning identity")
    return ContinuationReference(sources, actions, models, calibrators), planners, report


def verify(run, bundle, earlier_bundle):
    run = Path(run)
    read = lambda name: json.loads((run/name).read_bytes())
    report, design = read("report.json"), read("design.json")
    require(report["status"] == "OK", "Incomplete planning experiment")
    roots = {i: Path(design[f"source_round{i}"]) for i in (11, 12, 13)}
    source = roots[11]
    original = json.loads((source/"report.json").read_bytes())
    require(digest(source/"report.json") == SOURCE_REPORT_HASH, "Altered original reference release")
    require(digest(design["design_note_path"]) == design["design_note_sha256"] == DESIGN_HASH, "Altered pre-run design")
    require(design["new_strategies"] == MODES and design["target_folds"] == [2, 3, 4, 5, 6], "Unregistered planning intervention")
    require(design["fixed_lambda_money"] == design["validation_utility_money_weight"] == .5, "Changed objective/cost conversion")
    require(design["money_budgets"] == [0., .02, .03, .05] and design["request_budget"] == 2, "Changed physical budgets")
    require(design["bootstrap_samples"] == 2000 and design["bootstrap_seed"] == 1729, "Changed interval settings")
    require(report["implementation_sha256"] == read("implementation.json")
            and report["source_artifacts_sha256"] == read("source-artifacts.json"), "Changed source manifest")
    for name, h in report["source_artifacts_sha256"].items():
        root, file = roots[11], name
        for n in (12, 13):
            if name.startswith(f"round{n}/"):
                root, file = roots[n], name.split("/", 1)[1]
        require(digest(root/file) == h, "Changed reference artifact: "+name)
    for name, h in report["implementation_sha256"].items():
        require(digest(Path(__file__).with_name(name)) == h, "Changed research implementation: "+name)
    for item in report["model_artifacts"]:
        require(digest(run/item["path"]) == item["sha256"], "Changed model/wrapper")
    require(report["new_outcome_predictors"] == report["new_fitted_estimators"] == report["new_independent_test_matches"] == 0,
            "Unexpected fitting or inflated independent sample count")
    reference, planners, _ = load_reference(run, bundle, earlier_bundle)
    require(len(reference.sources) == report["input_matches"], "Changed data identities")
    audit = read("data-audit.json")
    for name, path in (("current", bundle), ("earlier", earlier_bundle)):
        require(load_episodes(path, lead_hours=24)[1]["platform_manifest_sha256"] == audit[name]["platform_manifest_sha256"], "Changed data snapshot")
    require(digest(run/"catalog.json") == digest(source/"catalog.json"), "Changed tool menu/costs")
    folds = {f["fold"]: f for f in report["folds"]}
    olds = {f["fold"]: f for f in original["folds"]}
    require(len(folds) == len(report["folds"]) == 5 and folds.keys() == olds.keys(), "Changed fold coverage")
    counts = Counter()
    for n, fold in folds.items():
        old = olds[n]
        for key in ("reference_fold", "split", "training_match_ids", "calibration_match_ids", "validation_match_ids", "test_match_ids"):
            require(fold[key] == old[key], "Changed fold partition: "+key)
        require(fold["joint_model"]["sha256"] == old["joint_model"]["sha256"]
                and fold["source_calibrator"]["sha256"] == old["calibrators"]["same_version"]["sha256"], "Planning used a different frozen model")
        require(set(fold["planners"]) == {"adaptive", "commit"}, "Missing planning mode")
        boundary = min(parse_time(reference.sources[mid]["prediction_cutoff"]) for mid in fold["validation_match_ids"]+fold["test_match_ids"])
        require(all(parse_time(r["result_available_at"]) < boundary for r in reference.calibrators[n].records), "Evaluation-time label used in fitting")
        require(set(fold["calibration_match_ids"]).isdisjoint(fold["validation_match_ids"]+fold["test_match_ids"]), "Own outcome in value fitting")
    all_new = {}
    for partition, prefix in (("test", ""), ("validation", "validation-")):
        rows = list(_rows(run/(prefix+"episodes.jsonl")))
        old_rows = [r for root in roots.values() for r in _rows(root/(prefix+"episodes.jsonl"))]
        verify_episode_coverage(rows, {n: set(f[partition+"_match_ids"]) for n, f in folds.items()}, list(MODES), design["money_budgets"], [0])
        require(len(rows) == report["episode_runs" if partition == "test" else "validation_episode_runs"], "Wrong new episode count")
        combined = old_rows+rows
        metrics = report["metrics" if partition == "test" else "validation_metrics_diagnostic_only"]
        verify_aggregate_metrics(combined, metrics)
        for value in metrics.values():
            close(value["utility_loss"], value["log_loss"]+.5*value["money"], "Utility aggregation")
        indexed = {tuple(r[k] for k in KEYS): r for r in rows}
        all_new.update({(partition, *key): row for key, row in indexed.items()})
        previous = {(r["fold"], r["match_id"], r["budget_money"], r["strategy"]): r for r in old_rows}
        seen = set()
        for trace in _rows(run/(prefix+"trajectories.jsonl")):
            key = tuple(trace[k] for k in KEYS)
            require(key in indexed and key not in seen, "Unknown/duplicate planning trajectory")
            seen.add(key)
            row = indexed[key]
            n, mid, mode = row["fold"], row["match_id"], MODES[row["strategy"]]
            require(row["partition"] == trace["partition"] == partition and row["reference_fold"] == trace["reference_fold"] == n-2, "Wrong trajectory split")
            require(row["mode"] == mode and row["gain_version"] == planners[n, mode].version and row["estimator_kind"] == KIND, "Wrong episode planner")
            require(row["model_version"] == reference.models[n].model_version
                    and row["source_calibrator_version"] == reference.calibrators[n].version, "Wrong source model version")
            require(row["actual_external_requests"] == row["actual_llm_tokens"] == 0, "Unexpected external calls")
            result = verify_trace(row, trace, reference.sources[mid], reference.actions, reference, planners[n, mode], mode)
            counts.update({partition+"_"+name: value for name, value in result.items()})
            old = previous[n, mid, row["budget_money"], REFERENCES[mode]]
            if row["budget_money"] < .05:
                require(all(row[k] == old[k] for k in ("actions", "cost", "final_probabilities", "log_loss", "brier_score")), "Unaffordable continuation changed path")
                counts[partition+"_low_budget_exact_one_step_paths"] += 1
        require(seen == indexed.keys(), "Missing planning trajectory")
        counts[partition+"_episodes"] = len(rows)
        if partition == "test":
            means = match_means(combined, .5)
            for label, expected in report["paired_comparisons"].items():
                pair, budget = label.rsplit("/money=", 1)
                a, b = pair.split("-vs-")
                calculated = {metric: paired(means[a, float(budget)], means[b, float(budget)], metric, 2000, 1729)
                              for metric in ("log_loss", "money", "utility_loss")}
                require(calculated == expected, "Changed paired planning comparison")
                counts["paired_comparisons"] += 1
        print(f"round14 independent audit: {partition} {len(rows)} planning trajectories verified", flush=True)
    # Compare every initial candidate/mode at the full budget against a fresh
    # production plan; values and future policies are independently calculated.
    for fold in folds.values():
        n = fold["fold"]
        for partition in ("test", "validation"):
            for mid in fold[partition+"_match_ids"]:
                state, _, _ = reference.public.state(n, mid, ())
                for mode in MODES.values():
                    actual = planners[n, mode].plan(state, reference.actions,
                        {"limits": {"money": .05, "requests": 2}, "spent": {}}, risk_level="mean", lambda_cost={"money": .5})
                    wanted = reference.plan(n, mid, (), .05, mode)
                    require(actual["action_id"] == wanted["action_id"], "Common initial-state policy differs")
                    expected = {c["action_id"]: c for c in wanted["candidates"]}
                    for candidate in actual["candidates"]:
                        counts["matched_initial_model_branches"] += verify_numbers(candidate, expected[candidate["action_id"]], mode)
                        counts["matched_initial_candidate_values"] += 1
    names = ("report.json", "design.json", "catalog.json", "data-audit.json", "episodes.jsonl", "trajectories.jsonl",
             "validation-episodes.jsonl", "validation-trajectories.jsonl", "realized-transitions.jsonl", "path-invariants.json",
             "implementation.json", "source-artifacts.json")
    return {"status": "VERIFIED", "round": 14, "counts": dict(counts),
            "independent_free_joint_cubes": len(reference.cubes), "independent_actual_public_states": len(reference.public.states),
            "artifacts": {name: digest(run/name) for name in names}, "verifier_sha256": digest(__file__),
            "new_independent_matches": 0, "external_requests": 0,
            "scope": "Independently calibrated finite conditional and committed continuation, all new actual paths, physical charges and paired metrics; no Bayes-optimality claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "bundle", "earlier-bundle", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), "Verification output exists")
    result = verify(args.run, args.bundle, args.earlier_bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
