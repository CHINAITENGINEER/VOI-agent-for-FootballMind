"""Independent finite-sum reference for the acquired-content ablation.

Reconstruct the joint table from frozen training records, slice numpy tensors,
and rebuild calibrated residual means from historical rows. Production branch
enumeration and the new estimator are not used by the numerical reference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .analyze_round7 import match_means, paired
from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome, state_after
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows
from .verify_round6 import KEYS, verify_aggregate_metrics
from .verify_round7 import verify_episode_coverage
from .verify_round8 import AXES, branches, close, marginal, probability_checks, rebuild_joint, require
from .verify_round9 import IndependentEvaluator, check_estimate, reference_summary, verify_value_cells

STRATEGY = "content-marginal-mean"
CONDITIONAL = "frozen-version-mean"
KIND = "content-marginal-gain-v1"
DESIGN_HASH = "dc26af59df5fc7533130c48a2767e643a7d86835f8b8610c16c71ac724309054"
SOURCE_REPORT_HASH = "63a157974e7e4c5f20ee6ad1a779041f11607b1c3c6a1dcb2c4c8267f2df8592"
UNITS = ("money", "requests", "latency_ms", "tokens")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def reference_marginal(cube, action_id, acquired, parents, categories):
    """Integrate *calibrated gains*, not EVSI before its nonlinear correction.

    cube is always the free-context joint, not the observed posterior. No match
    outcome or actual acquired return is an argument to this function.
    """
    acquired = tuple(sorted(acquired))
    require(action_id in AXES and len(acquired) <= 1 and set(acquired) <= set(AXES)
            and action_id not in acquired, "Invalid marginalization context")
    cube = np.asarray(cube, dtype=float)
    probability_checks(cube, ())
    if acquired:
        first = acquired[0]
        require(len(categories[first]) == cube.shape[AXES[first]], "Wrong category vocabulary")
        options = [(categories[first][i], mass, posterior) for i, mass, posterior in branches(cube, first)]
    else:
        options = [(None, 1., cube)]
    records = parents.get((action_id, acquired), [])
    rows = []
    for token, mass, posterior in options:
        evsi = max(0., probability_checks(posterior, (action_id,))[action_id]["evsi"])
        estimate = reference_summary(records, evsi)
        rows.append({"observation": token, "probability": mass,
                     "posterior_probabilities": marginal(posterior).tolist(),
                     "model_evsi": evsi, "expected_gain": estimate["expected_gain"],
                     "support": estimate["support"], "estimate": estimate})
    close(math.fsum(r["probability"] for r in rows), 1., "Content integration mass")
    supported = all(r["estimate"]["supported"] and r["expected_gain"] is not None for r in rows)
    gain = math.fsum(r["probability"]*r["expected_gain"] for r in rows) if supported else None
    return {"action_id": action_id, "acquired_action_ids": list(acquired), "supported": supported,
            "expected_gain": gain, "marginal_model_evsi": math.fsum(r["probability"]*r["model_evsi"] for r in rows),
            "support": min(r["support"] for r in rows),
            "parent_support": min(r["estimate"]["parent_support"] for r in rows), "branches": rows}


def select(candidates, *, money, requests):
    usable = []
    for candidate in candidates:
        action, gain = candidate["action"], candidate["expected_gain"]
        if (not action.get("available", True) or gain is None
                or action["cost"]["money"] > money+1e-12
                or action["cost"]["requests"] > requests+1e-12):
            continue
        usable.append((round(gain-.5*action["cost"]["money"], 12), action["action_id"]))
    usable.sort(key=lambda row: (-row[0], row[1]))
    return usable[0][1] if usable and usable[0][0] > 0 else None


class ContentReference:
    def __init__(self, sources, actions, models, calibrators):
        self.sources, self.actions, self.models = sources, actions, models
        self.calibrators = calibrators
        self.parents = {n: verify_value_cells(cal)[0] for n, cal in calibrators.items()}
        self.public = IndependentEvaluator(sources, actions, models, {}, {})
        self.cubes, self.values = {}, {}

    def base_cube(self, fold, mid):
        key = fold, mid
        if key not in self.cubes:
            model = self.models[fold]
            state = state_after(self.sources[mid], self.actions, ())
            cube, observed = rebuild_joint(model, state)
            require(not observed, "Content integration started from acquired evidence")
            self.cubes[key] = cube
        return self.cubes[key]

    def estimate(self, fold, mid, action_id, acquired):
        key = fold, mid, action_id, tuple(sorted(acquired))
        if key not in self.values:
            model = self.models[fold]
            categories = {"results_view": model.metadata["results_categories"],
                          "xg_view": model.metadata["xg_categories"]}
            self.values[key] = reference_marginal(self.base_cube(fold, mid), action_id,
                    acquired, self.parents[fold], categories)
        return self.values[key]

    def candidates(self, fold, mid, acquired):
        return [{**self.estimate(fold, mid, action["action_id"], acquired), "action": action}
                for action in self.actions if action["action_id"] not in acquired]


def verify_numbers(actual, expected):
    require(actual["estimator_kind"] == KIND, "Incorrect marginal estimator kind")
    for key in ("model_evsi", "evsi_bin", "lower_gain", "standard_error"):
        require(actual[key] is None, "Invented conditional value or aggregate uncertainty: "+key)
    require(actual["action_id"] == expected["action_id"]
            and actual["acquired_action_ids"] == expected["acquired_action_ids"], "Marginal candidate identity differs")
    for key in ("expected_gain", "g_hat", "decision_gain"):
        if expected["expected_gain"] is None:
            require(actual[key] is None, "Unsupported branch was silently excluded")
        else:
            close(actual[key], expected["expected_gain"], "Independent weighted "+key)
    close(actual["marginal_model_evsi"], expected["marginal_model_evsi"], "Expected conditional model EVSI")
    require(actual["support"] == expected["support"] and actual["parent_support"] == expected["parent_support"],
            "Branch support was counted as independent data")
    got = {r["observation"]: r for r in actual["branches"]}
    want = {r["observation"]: r for r in expected["branches"]}
    require(len(got) == len(actual["branches"]) and got.keys() == want.keys(), "Content category coverage differs")
    for token, target in want.items():
        branch = got[token]
        close(branch["probability"], target["probability"], "Free-prior category probability")
        close(branch["model_evsi"], target["model_evsi"], "Conditional branch EVSI")
        check_estimate(branch["estimate"], target["estimate"], target["model_evsi"])
        if target["expected_gain"] is None:
            require(branch["expected_gain"] is None, "Unsupported branch gained a fabricated mean")
        else:
            close(branch["expected_gain"], target["expected_gain"], "Calibrated before averaging")
        require(branch["support"] == target["support"], "Branch support differs")
    return len(want)


def verify_trace(row, trace, source, actions, reference, estimator):
    env, spent = ControlledEpisode(source, actions), dict.fromkeys(UNITS, 0.)
    events, counts = trace["events"], Counter()
    require(bool(events) and row["lambda_money"] == .5, "Invalid trajectory or execution coefficient")
    fold, mid = row["fold"], row["match_id"]
    _, initial, _ = reference.public.state(fold, mid, ())
    require(row["initial_probabilities"] == initial["probabilities"], "Different initial outcome prediction")
    for step, event in enumerate(events):
        acquired = tuple(env.performed_actions)
        state = {**env.visible_state(), "action_history": [
            {"action_id": aid, "status": "acquired_or_source_missing"} for aid in acquired]}
        require(event["step"] == step and event["visible_state"] == state, "Wrong public state or hidden information access")
        _, prediction, _ = reference.public.state(fold, mid, acquired)
        require(event["prediction"]["probabilities"] == prediction["probabilities"], "Actual posterior was erased")
        decision = event["decision"]
        candidates = reference.candidates(fold, mid, acquired)
        remaining_money, remaining_requests = row["budget_money"]-spent["money"], 2-spent["requests"]
        wanted = select(candidates, money=remaining_money, requests=remaining_requests)
        require(decision["action_id"] == wanted and decision["decision"] == ("acquire" if wanted else "stop"),
                "Independent content-marginal query/STOP differs")
        require(decision["risk_level"] == "mean" and decision["horizon"] == 1
                and decision["estimator_version"] == estimator.version, "Wrong marginal estimator or policy mode")
        for unit in UNITS:
            close(decision["physical_budget"]["spent"][unit], spent[unit], "Running query charge")
        close(decision["physical_budget"]["limits"]["money"], row["budget_money"], "Money budget")
        close(decision["physical_budget"]["limits"]["requests"], 2., "Request budget")
        eligible = {a["action_id"]: a for a in env.legal_actions()}
        actual = {c["action_id"]: c for c in decision["candidates"]}
        require(len(actual) == len(decision["candidates"]) and actual.keys() == eligible.keys(), "Wrong candidate menu")
        for aid, action in eligible.items():
            candidate = actual[aid]
            affordable = action["cost"]["money"] <= remaining_money+1e-12 and action["cost"]["requests"] <= remaining_requests+1e-12
            if not affordable:
                require(not candidate["feasible"] and candidate["reason"] == "physical_budget_exhausted", "Incorrect budget gate")
                continue
            expected = reference.estimate(fold, mid, aid, acquired)
            counts["integrated_branches"] += verify_numbers(candidate, expected)
            require(candidate["feasible"] == expected["supported"], "Incorrect supported-mean gate")
            penalty = .5*action["cost"]["money"]
            close(candidate["cost_penalty"], penalty, "Acquisition cost conversion")
            if expected["supported"]:
                net = expected["expected_gain"]-penalty
                close(candidate["net_gain"], net, "Marginal net gain")
                close(candidate["selection_score"], round(net, 12), "Rounded decision value")
            counts["candidates"] += 1
        if wanted is None:
            require(step == len(events)-1 and "charged_cost" not in event, "Activity after STOP")
            require(row["stop_reason"] == decision["reason"], "Wrong final stop reason")
            break
        action = eligible[wanted]
        require(event["charged_cost"] == action["cost"], "Wrong tool charge")
        for unit in UNITS:
            spent[unit] += action["cost"].get(unit, 0.)
        env.reveal(wanted)
        counts["charges"] += 1
    require(events[-1]["decision"]["decision"] == "stop", "No terminal STOP")
    require(row["actions"] == list(env.performed_actions) and row["final_mask"] == prediction["mask"], "Wrong final evidence")
    require(row["final_probabilities"] == prediction["probabilities"], "Wrong final real posterior")
    for unit in UNITS:
        close(row["cost"][unit], spent[unit], "Total query cost")
    for key, value in losses(prediction["probabilities"], outcome(source)).items():
        close(row[key], value, "Real observed outcome score")
    counts["events"] = len(events)
    return counts


def verify(run, bundle, earlier_bundle):
    from .analyze_round13 import observed_policy_pairs
    from .content_marginal import ContentMarginalGainEstimator

    run = Path(run)
    read = lambda name: json.loads((run/name).read_bytes())
    report, design = read("report.json"), read("design.json")
    require(report["status"] == "OK", "Incomplete content-ablation run")
    source, source12 = Path(design["source_round11"]), Path(design["source_round12"])
    old = json.loads((source/"report.json").read_bytes())
    require(digest(source/"report.json") == SOURCE_REPORT_HASH, "Frozen joint experiment changed")
    require(digest(design["design_note_path"]) == design["design_note_sha256"] == DESIGN_HASH, "Prespecified content ablation changed")
    require(design["new_strategies"] == [STRATEGY] and design["target_folds"] == [2,3,4,5,6], "Unregistered strategy/fold")
    require(design["fixed_lambda_money"] == design["validation_utility_money_weight"] == .5, "Wrong cost or utility coefficient")
    require(design["money_budgets"] == [0., .02, .03, .05] and design["request_budget"] == 2, "Wrong fixed budgets")
    require(design["bootstrap_samples"] == 2000 and design["bootstrap_seed"] == 1729, "Changed interval settings")
    require(report["implementation_sha256"] == read("implementation.json")
            and report["source_artifacts_sha256"] == read("source-artifacts.json"), "Incorrect artifact manifest")
    for name, h in report["source_artifacts_sha256"].items():
        root, file = (source12, name[len("round12/"):]) if name.startswith("round12/") else (source, name)
        require(digest(root/file) == h, "Frozen reference changed: "+name)
    for name, h in report["implementation_sha256"].items():
        require(digest(Path(__file__).with_name(name)) == h, "Implementation changed: "+name)
    for item in report["model_artifacts"]:
        require(digest(run/item["path"]) == item["sha256"], "Model or wrapper changed")
    require(report["new_outcome_predictors"] == report["new_fitted_estimators"] == report["new_independent_test_matches"] == 0,
            "Unexpected fitting or inflated evidence count")
    current, ca = load_episodes(bundle, lead_hours=24)
    earlier, ea = load_episodes(earlier_bundle, lead_hours=24)
    sources = {r["match_id"]: r for r in current+earlier}
    require(len(sources) == len(current+earlier) == report["input_matches"], "Snapshot identities changed")
    audit = read("data-audit.json")
    for name, actual in (("current", ca), ("earlier", ea)):
        require(actual["platform_manifest_sha256"] == audit[name]["platform_manifest_sha256"], "Snapshot hash changed")
    require(digest(run/"catalog.json") == digest(source/"catalog.json"), "Tool menu or costs changed")
    actions = read("catalog.json")["actions"]
    original_folds, folds = {f["fold"]: f for f in old["folds"]}, {f["fold"]: f for f in report["folds"]}
    require(len(folds) == len(report["folds"]) == 5 and folds.keys() == original_folds.keys(), "Wrong target-fold coverage")
    models, calibrators, estimators, summaries = {}, {}, {}, []
    counts = Counter()
    for n, fold in sorted(folds.items()):
        previous = original_folds[n]
        for key in ("reference_fold", "split", "training_match_ids", "calibration_match_ids", "validation_match_ids", "test_match_ids"):
            require(fold[key] == previous[key], "Changed source fold: "+key)
        joint_path, cal_path = run/fold["joint_model"]["path"], run/fold["source_calibrator"]["path"]
        jhash, chash = digest(joint_path), digest(cal_path)
        require(jhash == fold["joint_model"]["sha256"] == previous["joint_model"]["sha256"]
                == digest(source/previous["joint_model"]["path"]), "Altered reference predictor")
        require(chash == fold["source_calibrator"]["sha256"] == previous["calibrators"]["same_version"]["sha256"]
                == digest(source/previous["calibrators"]["same_version"]["path"]), "Altered residual calibrator")
        model = models[n] = JointReturnPredictor.load(joint_path)
        cal = calibrators[n] = ResidualEVSICalibrator.load(cal_path, model=model, model_sha256=jhash)
        artifact = fold["content_marginal"]
        estimator = estimators[n] = ContentMarginalGainEstimator.load(run/artifact["path"], model=model,
                model_sha256=jhash, calibrator=cal, calibrator_sha256=chash)
        require(estimator.version == artifact["version"] and estimator.manifest == artifact["manifest"], "Altered wrapper binding")
        require(estimator.manifest["source_calibrator_sha256"] == chash
                and estimator.manifest["source_calibrator_version"] == cal.version, "Wrong wrapper source identity")
        require(set(estimator.manifest["training_match_ids"]) == set(fold["calibration_match_ids"]), "Changed calibration membership")
        require(parse_time(estimator.manifest["trained_as_of"]) == parse_time(fold["split"]["train_cutoff"]), "Changed calibration time")
        boundary = min(parse_time(sources[mid]["prediction_cutoff"]) for mid in fold["validation_match_ids"]+fold["test_match_ids"])
        require(all(parse_time(r["result_available_at"]) < boundary for r in cal.records), "Evaluation label entered calibration")
        require(set(fold["calibration_match_ids"]).isdisjoint(fold["validation_match_ids"]+fold["test_match_ids"]), "Own outcome used in gain calibration")
        summaries.append({"fold": n, "joint_sha256": jhash, "source_calibrator_sha256": chash,
                          "wrapper_sha256": digest(run/artifact["path"]), "source_calibration_matches": len(fold["calibration_match_ids"]),
                          "test_matches": len(fold["test_match_ids"]), "validation_matches": len(fold["validation_match_ids"])})
        counts["bound_frozen_predictors"] += 1
        counts["bound_frozen_calibrators"] += 1
        counts["wrappers"] += 1
    reference = ContentReference(sources, actions, models, calibrators)
    for partition, prefix in (("test", ""), ("validation", "validation-")):
        rows = list(_rows(run/(prefix+"episodes.jsonl")))
        old_rows = list(_rows(source/(prefix+"episodes.jsonl")))
        prior_rows = list(_rows(source12/(prefix+"episodes.jsonl")))
        verify_episode_coverage(rows, {n: set(f[partition+"_match_ids"]) for n, f in folds.items()}, [STRATEGY], design["money_budgets"], [0])
        require(len(rows) == report["episode_runs" if partition == "test" else "validation_episode_runs"], "Wrong new episode count")
        combined = old_rows+prior_rows+rows
        metrics = report["metrics" if partition == "test" else "validation_metrics_diagnostic_only"]
        verify_aggregate_metrics(combined, metrics)
        for item in metrics.values():
            close(item["utility_loss"], item["log_loss"]+.5*item["money"], "Combined utility score")
        identities = {tuple(r[k] for k in KEYS): r for r in rows}
        seen = set()
        for trace in _rows(run/(prefix+"trajectories.jsonl")):
            key = tuple(trace[k] for k in KEYS)
            require(key in identities and key not in seen, "Unknown/duplicate trajectory")
            seen.add(key)
            row = identities[key]
            n, mid = row["fold"], row["match_id"]
            require(row["partition"] == trace["partition"] == partition and row["reference_fold"] == trace["reference_fold"] == n-2,
                    "Wrong trajectory partition")
            require(row["gain_version"] == estimators[n].version and row["estimator_kind"] == KIND, "Wrong episode estimator")
            require(row["source_calibrator_version"] == calibrators[n].version
                    and row["model_version"] == models[n].model_version
                    and row["base_model_version"] == models[n].base_predictor.model_version, "Wrong source versions")
            require(row["actual_external_requests"] == row["actual_llm_tokens"] == 0, "Unexpected external execution")
            result = verify_trace(row, trace, sources[mid], actions, reference, estimators[n])
            counts.update({partition+"_"+name: value for name, value in result.items()})
        require(seen == identities.keys(), "Missing trajectory")
        counts[partition+"_episodes"] = len(rows)
        pairs = observed_policy_pairs(rows, old_rows)
        counts[partition+"_initial_action_equal"] = len(pairs)
        counts[partition+"_path_disagreements"] = sum(r["path_differs"] for r in pairs)
        counts[partition+"_low_budget_identical_paths"] = sum(r["budget_money"] < .05 for r in pairs)
        invariants = report["path_invariants"][partition]
        require(invariants["initial_action_equal"] == len(pairs)
                and invariants["low_budget_identical_paths_costs_probabilities"] == counts[partition+"_low_budget_identical_paths"], "Incorrect reported path invariant")
        require(len(invariants["changed_path_ids"]) == counts[partition+"_path_disagreements"], "Wrong changed-path count")
        if partition == "test":
            means = match_means(combined, .5)
            for label, expected in report["paired_comparisons"].items():
                pair, budget = label.rsplit("/money=", 1)
                a, b = pair.split("-vs-")
                calculated = {name: paired(means[a, float(budget)], means[b, float(budget)], name, 2000, 1729)
                              for name in ("log_loss", "money", "utility_loss")}
                require(calculated == expected, "Incorrect paired comparison")
                counts["paired_comparisons"] += 1
        print(f"round13 independent audit: {partition} {len(rows)} trajectories verified", flush=True)
    # Check every actual-state candidate, including unvisited first-action paths.
    # Finite integration uses the free cube, despite varying actual return content.
    for row in _rows(source/"transition-diagnostics.jsonl"):
        n, mid, aid, acquired = row["fold"], row["match_id"], row["action_id"], row["acquired_action_ids"]
        state, prediction, _ = reference.public.state(n, mid, acquired)
        action = next(a for a in actions if a["action_id"] == aid)
        actual = estimators[n].estimate_value(state, [action], prediction=prediction)[0]
        expected = reference.estimate(n, mid, aid, acquired)
        counts["common_state_integrated_branches"] += verify_numbers(actual, expected)
        counts["common_state_estimates"] += 1
        if not acquired:
            # The production empty-state gain is literally reused, not a new fit.
            require(actual["expected_gain"] == row["estimates"]["same_version"]["expected_gain"], "Initial native gain differs")
            counts["identical_initial_candidate_gains"] += 1
    names = ("report.json", "design.json", "catalog.json", "data-audit.json", "episodes.jsonl", "trajectories.jsonl",
             "validation-episodes.jsonl", "validation-trajectories.jsonl", "path-invariants.json", "implementation.json", "source-artifacts.json")
    return {"status": "VERIFIED", "round": 13, "counts": dict(counts), "frozen_model_bindings": summaries,
            "independent_free_joint_cubes": len(reference.cubes), "independent_actual_public_states": len(reference.public.states),
            "artifacts": {name: digest(run/name) for name in names}, "verifier_sha256": digest(__file__),
            "new_independent_matches": 0, "external_requests": 0,
            "scope": "Independent joint tensor slicing, historical residual correction before marginalization, all new actual policy paths and paired metrics; branch simulations are not actual queries"}


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
