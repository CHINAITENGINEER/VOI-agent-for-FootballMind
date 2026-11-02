"""Independent matched-supervision and autonomous-action audit for round eleven.

This reads outcome-bearing material only after the experiment. Runtime decisions
are reconstructed from public states, past residuals, budgets and frozen models.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .data import ControlledEpisode, load_episodes
from .experiment import losses, outcome
from .joint_returns import JointReturnPredictor
from .round9_residual import ResidualEVSICalibrator
from .state import parse_time
from .verify_round4 import _rows
from .verify_round6 import KEYS, verify_aggregate_metrics
from .verify_round7 import verify_episode_coverage, verify_heuristic_trace, verify_oof_sample
from .verify_round8 import close, require, rebuild_joint, reference_plan, verify_plan_fields
from .verify_round9 import (IndependentEvaluator, check_estimate, fitted_identity,
    transition_identity, verify_calibrated_trace, verify_value_cells)

DESIGN_HASH = "37e769df19bef0c83fd94efe81ed4cf94aaa0a2591576852758f4e2a8f283663"
STRATEGIES = ("base-only", "all-available", "random", "entropy", "evsi-one-step",
    "matched-mixed-mean", "matched-mixed-lcb", "frozen-version-mean", "frozen-version-lcb")
ARMS = ("same_version", "matched_mixed")
CALIBRATED = {"matched-mixed-mean": ("matched_mixed", "calibrated-mean"),
    "matched-mixed-lcb": ("matched_mixed", "calibrated-lcb"),
    "frozen-version-mean": ("same_version", "calibrated-mean"),
    "frozen-version-lcb": ("same_version", "calibrated-lcb")}
LEGAL = {((), "results_view"), ((), "xg_view"),
    (("results_view",), "xg_view"), (("xg_view",), "results_view")}
UNITS = ("money", "requests", "latency_ms", "tokens")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def matched_ids(sources, target, reference, mixed):
    """Selection uses original membership and dates, never the value of a gain."""
    by_match = defaultdict(set)
    seen = set()
    for sample in mixed:
        mid, aid, acquired = transition_identity(sample)
        require((mid, aid, acquired) not in seen, "Duplicate mixed transition")
        require((acquired, aid) in LEGAL, "Illegal mixed transition")
        seen.add((mid, aid, acquired))
        by_match[mid].add((acquired, aid))
    reference_time = parse_time(reference.metadata["training_as_of"])
    cutoff = parse_time(target["split"]["train_cutoff"])
    evaluation = set(target["validation_match_ids"]) | set(target["test_match_ids"])
    boundary = min(parse_time(sources[mid]["prediction_cutoff"]) for mid in evaluation)
    require(reference_time < cutoff <= boundary, "Invalid frozen calibration interval")
    prior_ids = set(reference.base_predictor.metadata["training_match_ids"])
    ids = set()
    for mid in target["training_match_ids"]:
        source = sources[mid]
        if (reference_time < parse_time(source["prediction_cutoff"])
                and parse_time(source["result_available_at"]) <= cutoff
                and parse_time(source["result_available_at"]) < boundary
                and mid not in prior_ids and mid not in evaluation
                and by_match.get(mid) == LEGAL):
            ids.add(mid)
    return ids, boundary


def verify_fit(calibrator, samples, selected, *, cutoff, boundary, arm, model, model_hash):
    expected = {transition_identity(s): s for s in samples}
    actual = {fitted_identity(r): r for r in calibrator.records}
    require(len(expected) == len(samples) == len(selected)*4, "Supervision multiplicity differs")
    require(len(actual) == len(calibrator.records) and actual.keys() == expected.keys(), "Fitted transition coverage differs")
    manifest = calibrator.manifest
    require(set(manifest["training_match_ids"]) == selected, "Fitted match coverage differs")
    require(manifest["calibration_match_ids"] == [] and calibrator.calibration["records"] == [], "Unexpected inner tail")
    require(parse_time(manifest["trained_as_of"]) == cutoff <= boundary, "Calibration fitting time differs")
    require(parse_time(manifest["diagnostics_as_of"]) == cutoff, "Diagnostic declaration differs")
    require(calibrator.min_support == 12 and calibrator.shrinkage == 20 and calibrator.risk_multiplier == 1.64,
            "Frozen residual hyperparameters changed")
    metadata = manifest["metadata"]
    require(metadata["supervision_arm"] == arm, "Arm provenance differs")
    expected_kind = "same_version_heldout_fixed_predictor" if arm == "same_version" else "matched_mixed_past_forward_oof"
    require(metadata["supervision_kind"] == expected_kind, "Generating-version declaration differs")
    require(metadata["reference_model_version"] == model.model_version, "Deployment model version differs")
    for identity, sample in expected.items():
        row = actual[identity]
        for name in ("kickoff", "result_available_at", "oof_train_cutoff", "oof_manifest_ref", "oof_predictor_version"):
            require(row[name] == sample[name], "Fitted provenance changed: " + name)
        require(set(row["oof_train_ids"]) == set(sample["oof_train_ids"]), "Fitted generator membership changed")
        require(row["prediction_cutoff"] == sample["state"]["prediction_cutoff"], "Fitted target cutoff changed")
        require(parse_time(row["result_available_at"]) <= cutoff and parse_time(row["result_available_at"]) < boundary,
                "Calibration outcome reaches evaluation period")
        require(row["match_id"] not in row["oof_train_ids"], "In-training calibration target")
        close(row["gain"], sample["delta"], "Fitted realized update gain")
        close(row["model_evsi"], max(0., sample["model_evsi"]), "Fitted public model EVSI")
        if arm == "same_version":
            require(row["oof_predictor_version"] == model.model_version and row["oof_manifest_ref"] == "sha256:"+model_hash,
                    "Same-version label generator changed")
    return verify_value_cells(calibrator)[0]


def verify_raw_trace(row, trace, source, actions, evaluator, design, plans):
    """Independent heuristic or finite-joint EVSI action, then every real reveal."""
    env, spent = ControlledEpisode(source, actions), dict.fromkeys(UNITS, 0.)
    numerical = row["strategy"] == "evsi-one-step"
    events, charges = trace["events"], 0
    require(bool(events), "Empty raw trajectory")
    for step, event in enumerate(events):
        require(event["step"] == step, "Noncontiguous raw trajectory")
        state = {**env.visible_state(), "action_history": [
            {"action_id": a, "status": "acquired_or_source_missing"} for a in env.performed_actions]}
        require(state == event["visible_state"], "Raw trajectory exposes unpurchased evidence")
        _, prediction, _ = evaluator.state(row["fold"], row["match_id"], env.performed_actions)
        require(prediction["probabilities"] == event["prediction"]["probabilities"], "Raw prediction changed")
        decision = event["decision"]
        if numerical and env.legal_actions():
            key = row["fold"], row["match_id"], tuple(sorted(env.performed_actions)), row["budget_money"]
            if key not in plans:
                cube, observed = rebuild_joint(evaluator.models[row["fold"]], state)
                plans[key] = reference_plan(cube, actions, observed=observed,
                    money=max(0., row["budget_money"]-spent["money"]),
                    requests=design["request_budget"]-spent["requests"], lambda_money=.5, horizon=1)
            model = evaluator.models[row["fold"]]
            verify_plan_fields(decision, plans[key], {"results_view": model.metadata["results_categories"],
                "xg_view": model.metadata["xg_categories"]})
            for unit in UNITS:
                close(decision["physical_budget"]["spent"][unit], spent[unit], "Raw resource accounting")
            close(decision["physical_budget"]["limits"]["money"], row["budget_money"], "Raw money ceiling")
            close(decision["physical_budget"]["limits"]["requests"], 2, "Raw request ceiling")
        if decision["decision"] == "stop":
            require(step == len(events)-1 and "charged_cost" not in event, "Action or charge after STOP")
            require(row["stop_reason"] == decision["reason"], "Raw STOP reason differs")
            break
        aid = decision["action_id"]
        action = next((a for a in env.legal_actions() if a["action_id"] == aid), None)
        require(action is not None and event["charged_cost"] == action["cost"], "Invalid raw purchase")
        for unit in UNITS:
            spent[unit] += action["cost"].get(unit, 0.)
        require(spent["money"] <= row["budget_money"]+1e-12 and spent["requests"] <= 2, "Raw budget exceeded")
        env.reveal(aid)
        charges += 1
    require(events[-1]["decision"]["decision"] == "stop", "Raw trajectory lacks STOP")
    require(row["actions"] == list(env.performed_actions) and row["final_mask"] == prediction["mask"], "Raw final evidence differs")
    require(row["final_probabilities"] == prediction["probabilities"], "Raw final probabilities differ")
    for unit in UNITS:
        close(row["cost"][unit], spent[unit], "Raw final cost")
    for name, value in losses(prediction["probabilities"], outcome(source)).items():
        close(row[name], value, "Raw held-out score")
    if not numerical:
        verify_heuristic_trace(row, trace, source, actions, design)
    return {"events": len(events), "charges": charges, "candidates": 0}


def verify(run, bundle, earlier_bundle):
    run = Path(run)
    read = lambda name: json.loads((run/name).read_bytes())
    report, design = read("report.json"), read("design.json")
    require(report["status"] == "OK", "Incomplete research run")
    source = Path(design["source_round9"])
    old = json.loads((source/"report.json").read_bytes())
    old_folds = {r["fold"]: r for r in old["folds"]}
    require(digest(source/"report.json") == "39b18c7f61ed08f381ab69b035affa7836761e368b67258ccbe28f87080c4633", "Frozen round-nine report changed")
    require(design["design_note_sha256"] == digest(design["design_note_path"]) == DESIGN_HASH, "Pre-run written design changed")
    require(design["target_folds"] == [2,3,4,5,6] and tuple(design["strategies"]) == STRATEGIES, "Prespecified fold/strategy selection changed")
    require(design["fixed_lambda_money"] == design["validation_utility_money_weight"] == .5, "Cost target changed")
    require(design["money_budgets"] == [0.,.02,.03,.05] and design["request_budget"] == 2, "Physical budgets changed")
    require(design["random_seeds"] == [7,19,41,73,101] and design["bootstrap_samples"] == 2000 and design["bootstrap_seed"] == 1729,
            "Prespecified repeat/uncertainty design changed")
    require(read("source-artifacts.json") == report["source_artifacts_sha256"], "Upstream manifest differs")
    require(read("implementation.json") == report["implementation_sha256"], "Implementation manifest differs")
    for name, h in report["source_artifacts_sha256"].items():
        require(digest(source/name) == h, "Upstream content changed: "+name)
    old_verified = json.loads(source.with_name(source.name+"-verification.json").read_bytes())
    require(digest(source/"oof-supervision.jsonl") == old_verified["artifacts"]["oof-supervision.jsonl"], "Previously verified mixed supervision changed")
    for name, h in report["implementation_sha256"].items():
        require(digest(Path(__file__).with_name(name)) == h, "Implementation changed: "+name)
    for artifact in report["model_artifacts"] + report["supervision_artifacts"]:
        require(digest(run/artifact["path"]) == artifact["sha256"], "Experiment artifact changed")
    current, ca = load_episodes(bundle, lead_hours=design["lead_hours"])
    earlier, ea = load_episodes(earlier_bundle, lead_hours=design["lead_hours"])
    audit = read("data-audit.json")
    require(ca["platform_manifest_sha256"] == audit["current"]["platform_manifest_sha256"] and
            ea["platform_manifest_sha256"] == audit["earlier"]["platform_manifest_sha256"], "Data snapshot changed")
    sources = {r["match_id"]:r for r in earlier+current}
    require(len(sources) == len(earlier+current) == report["input_matches"], "Source coverage differs")
    require(digest(run/"catalog.json") == digest(source/"catalog.json"), "Action catalog changed")
    actions = read("catalog.json")["actions"]
    mixed = list(_rows(source/"oof-supervision.jsonl"))
    mixed_index = {transition_identity(r):r for r in mixed}
    models, calibrators, parents = {}, {a:{} for a in ARMS}, {a:{} for a in ARMS}
    counts, fold_summaries = Counter(), []
    folds = {f["fold"]:f for f in report["folds"]}
    require(set(folds) == {2,3,4,5,6} and len(folds) == len(report["folds"]), "Fold report coverage differs")
    for number, fold in sorted(folds.items()):
        target, reference = old_folds[number], old_folds[number-2]
        require(fold["reference_fold"] == number-2 and fold["split"] == target["split"], "Frozen reference or partition changed")
        for name in ("training_match_ids", "validation_match_ids", "test_match_ids"):
            require(fold[name] == target[name], "Original target membership changed")
        path = run/fold["joint_model"]["path"]
        model_hash = digest(path)
        require(model_hash == digest(source/reference["joint_model"]["path"]) == reference["joint_model"]["sha256"] == fold["joint_model"]["sha256"],
                "Reference model is not byte-identical")
        model = models[number] = JointReturnPredictor.load(path)
        selected, boundary = matched_ids(sources, target, model, mixed)
        require(selected == set(fold["calibration_match_ids"]), "Matched calibration selection differs")
        selection = json.loads((run/fold["selection"]["path"]).read_bytes())
        require(selected == set(selection["calibration_match_ids"]), "Selection manifest differs")
        excluded = {r["match_id"]:r for r in selection["excluded"]}
        candidates = set(target["training_match_ids"]) | set(selection["nominal_window_match_ids"])
        require(not selected.intersection(excluded) and selected | set(excluded) == candidates, "Selection/exclusion ledger incomplete")
        require(all(r["reasons"] for r in excluded.values()), "Unexplained calibration exclusion")
        cutoff = parse_time(target["split"]["train_cutoff"])
        for mid in set(target["validation_match_ids"]) | set(target["test_match_ids"]):
            prediction_at = parse_time(sources[mid]["prediction_cutoff"])
            require(cutoff <= prediction_at, "Calibration declaration exceeds evaluation time")
            require(all(parse_time(sources[tid]["result_available_at"]) < prediction_at
                        for tid in set(model.base_predictor.metadata["training_match_ids"]) | selected), "Training label reaches evaluation prediction")
        for arm in ARMS:
            samples = list(_rows(run/fold["supervision"][arm]["path"]))
            require({r["match_id"] for r in samples} == selected, "Arm memberships differ")
            for sample in samples:
                identity = transition_identity(sample)
                require((identity[2], identity[1]) in LEGAL, "Illegal matched transition")
                if arm == "matched_mixed":
                    original = mixed_index[identity]
                    require({k:sample[k] for k in original} == original, "Mixed arm changed a frozen source label")
                    require(sample["supervision_kind"] == "matched_mixed_past_forward_oof", "Mixed semantics changed")
                else:
                    verify_oof_sample(sample, sources[sample["match_id"]], model, actions,
                        model.base_predictor.metadata["training_match_ids"], model.metadata["training_as_of"])
                    require(sample["supervision_kind"] == "same_version_heldout_fixed_predictor", "Fixed-version semantics changed")
                counts[arm+"_supervision"] += 1
            artifact = fold["calibrators"][arm]
            cal = calibrators[arm][number] = ResidualEVSICalibrator.load(run/artifact["path"], model=model, model_sha256=model_hash)
            require(cal.version == artifact["version"] and cal.manifest == artifact["manifest"], "Calibrator report differs")
            parents[arm][number] = verify_fit(cal, samples, selected, cutoff=cutoff, boundary=boundary,
                arm=arm, model=model, model_hash=model_hash)
        fold_summaries.append({"fold":number,"reference_fold":number-2,"calibration_matches":len(selected),
            "validation_matches":len(target["validation_match_ids"]),"test_matches":len(target["test_match_ids"]),
            "reference_version":model.model_version})
        print(f"round11 audit fold {number}: matched labels and residual cells verified", flush=True)
    evaluators = {a:IndependentEvaluator(sources, actions, models, calibrators[a], parents[a]) for a in ARMS}
    evaluator = evaluators["same_version"]
    evaluators["matched_mixed"].states = evaluator.states
    # Independently rebuild EVSI from the joint tables for every new held-out label.
    for number, fold in folds.items():
        for sample in _rows(run/fold["supervision"]["same_version"]["path"]):
            _, _, values = evaluator.state(number, sample["match_id"], sample["acquired_action_ids"])
            close(sample["model_evsi"], max(0.,values[sample["action"]["action_id"]]["evsi"]), "Held-out independent EVSI")
            close(sample["residual"], sample["delta"]-sample["model_evsi"], "Held-out residual")
    plans = {}
    for partition, prefix in (("test", ""), ("validation", "validation-")):
        rows = list(_rows(run/(prefix+"episodes.jsonl")))
        fold_ids = {n:set(f[partition+"_match_ids"]) for n,f in folds.items()}
        verify_episode_coverage(rows, fold_ids, STRATEGIES, design["money_budgets"], design["random_seeds"])
        require(len(rows) == report["episode_runs" if partition == "test" else "validation_episode_runs"], "Episode count differs")
        metrics = report["metrics" if partition == "test" else "validation_metrics_diagnostic_only"]
        verify_aggregate_metrics(rows, metrics)
        indexed = {tuple(r[k] for k in KEYS):r for r in rows}
        seen = set()
        for trace in _rows(run/(prefix+"trajectories.jsonl")):
            key = tuple(trace[k] for k in KEYS)
            require(key in indexed and key not in seen, "Unknown/duplicate trajectory")
            seen.add(key)
            row = indexed[key]
            number, mid = row["fold"], row["match_id"]
            require(row["partition"] == trace["partition"] == partition and row["reference_fold"] == trace["reference_fold"] == number-2,
                    "Episode design identity differs")
            require(row["model_version"] == models[number].model_version and row["base_model_version"] == models[number].base_predictor.model_version,
                    "Deployed predictor differs across policies")
            require(row["actual_external_requests"] == row["actual_llm_tokens"] == 0, "Unexpected real external use")
            require(row["kickoff"] == sources[mid]["kickoff"] and row["prediction_cutoff"] == sources[mid]["prediction_cutoff"], "Episode time differs")
            _, p0, _ = evaluator.state(number, mid, ())
            require(row["initial_probabilities"] == p0["probabilities"], "Policies receive unequal free predictions")
            if row["strategy"] in CALIBRATED:
                arm, name = CALIBRATED[row["strategy"]]
                require(row["calibrator_arm"] == arm and row["gain_version"] == calibrators[arm][number].version,
                        "Policy value artifact differs")
                require(row["lambda_money"] == .5, "Numerical policy cost coefficient changed")
                checked = verify_calibrated_trace({**row,"strategy":name}, trace, sources[mid], actions, evaluators[arm], 2)
            else:
                require(row["lambda_money"] == (.5 if row["strategy"] == "evsi-one-step" else 0.), "Raw policy coefficient changed")
                checked = verify_raw_trace(row, trace, sources[mid], actions, evaluator, design, plans)
            counts.update({partition+"_"+k:v for k,v in checked.items()})
        require(seen == set(indexed), "Missing experiment trajectory")
        counts[partition+"_episodes"] = len(rows)
        print(f"round11 audit {partition}: {len(rows)} trajectories independently verified", flush=True)
    diagnostics = list(_rows(run/"transition-diagnostics.jsonl"))
    expected = {(n,p,mid,acq,aid) for n,f in folds.items() for p in ("validation","test") for mid in f[p+"_match_ids"] for acq,aid in LEGAL}
    seen = set()
    for row in diagnostics:
        n, mid, aid, acq = row["fold"], row["match_id"], row["action_id"], tuple(row["acquired_action_ids"])
        key = n,row["partition"],mid,acq,aid
        require(key in expected and key not in seen, "Unknown/duplicate diagnostic transition")
        seen.add(key)
        _, before, values = evaluator.state(n,mid,acq)
        _, after, _ = evaluator.state(n,mid,(*acq,aid))
        require(row["prediction_before"] == before["probabilities"] and row["prediction_after"] == after["probabilities"], "Diagnostic update differs")
        require(row["joint_model_version"] == models[n].model_version, "Diagnostic predictor differs")
        actual_gain = losses(before["probabilities"],outcome(sources[mid]))["log_loss"]-losses(after["probabilities"],outcome(sources[mid]))["log_loss"]
        close(row["actual_gain"], actual_gain, "Offline realized gain")
        close(row["model_evsi"], max(0.,values[aid]["evsi"]), "Offline public EVSI")
        for arm in ARMS:
            candidate = next(c for c in evaluators[arm].candidates(n,mid,acq) if c["action"]["action_id"] == aid)
            check_estimate(row["estimates"][arm], candidate, candidate["model_evsi"])
    require(seen == expected, "Incomplete diagnostic coverage")
    names = ("report.json","design.json","catalog.json","data-audit.json","implementation.json","source-artifacts.json",
        "episodes.jsonl","validation-episodes.jsonl","trajectories.jsonl","validation-trajectories.jsonl","transition-diagnostics.jsonl")
    return {"status":"VERIFIED","round":11,"folds":fold_summaries,"counts":dict(counts),
        "test_matches":len(set().union(*(set(f["test_match_ids"]) for f in folds.values()))),
        "shared_diagnostic_transitions":len(diagnostics),"unique_public_joint_states":len(evaluator.states),
        "independent_raw_evsi_decisions":len(plans),"new_independent_matches":0,"external_requests":0,
        "artifacts":{n:digest(run/n) for n in names},"verifier_sha256":digest(__file__),
        "scope":"Independent matching, same-version label generation, residual cells, decisions, charges, public-state and STOP replay; exploratory reused-history evidence"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run","bundle","earlier-bundle","output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    result = verify(args.run,args.bundle,args.earlier_bundle)
    output = Path(args.output)
    require(not output.exists(), "Verification output exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","counts","test_matches","shared_diagnostic_transitions")}))
