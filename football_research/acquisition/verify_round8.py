"""Independent probability and finite-horizon accounting reference for round 8.

The reference integrates the joint tensor directly; it never invokes the
production planner or uses its hypothetical-state construction.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path

import numpy as np


AXES = {"results_view": 1, "xg_view": 2}
UNITS = ("money", "requests", "latency_ms", "tokens")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, tolerance=1e-9):
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance,
            f"Mismatch in {name}: {actual} vs {expected}")


def entropy(probabilities):
    return -math.fsum(float(p) * math.log(float(p)) for p in probabilities if p > 0)


def marginal(cube):
    return cube.sum(axis=(1, 2))


def branches(cube, action):
    axis = AXES[action]
    for category in range(cube.shape[axis]):
        index = [slice(None)] * 3
        index[axis] = category
        mass = float(cube[tuple(index)].sum())
        if mass <= 0:
            continue
        posterior = np.zeros_like(cube)
        posterior[tuple(index)] = cube[tuple(index)] / mass
        yield category, mass, posterior


def probability_checks(cube, available_actions):
    cube = np.asarray(cube, dtype=float)
    require(cube.ndim == 3 and cube.shape[0] == 3, "Expected Y/results/xg tensor")
    require(np.isfinite(cube).all() and (cube >= 0).all(), "Invalid joint mass")
    close(cube.sum(), 1, "joint normalization")
    before = marginal(cube)
    diagnostics = {}
    for action in available_actions:
        axis = AXES[action]
        expected, after_risk = np.zeros(3), 0.0
        mutual_information = 0.0
        # Compute MI directly from the two-dimensional marginal, independently
        # of the entropy-difference calculation below.
        pair = cube.sum(axis=2 if axis == 1 else 1)
        z_probabilities = pair.sum(axis=0)
        for y in range(3):
            for z in range(pair.shape[1]):
                value = pair[y, z]
                if value > 0:
                    mutual_information += value * math.log(value / (before[y] * z_probabilities[z]))
        mass_sum = 0.0
        for _, mass, posterior in branches(cube, action):
            predicted = marginal(posterior)
            expected += mass * predicted
            after_risk += mass * entropy(predicted)
            mass_sum += mass
        close(mass_sum, 1, "return probabilities sum")
        deviation = float(np.max(np.abs(expected - before)))
        require(deviation < 1e-10, "Posterior expectation differs from current prediction")
        gain = entropy(before) - after_risk
        close(gain, mutual_information, "EVSI versus target mutual information")
        require(gain >= -1e-10, "Negative model-internal Bayes EVSI")
        diagnostics[action] = {"evsi": gain, "target_mutual_information": mutual_information,
                               "posterior_expectation_max_error": deviation}
    return diagnostics


def reference_plan(cube, actions, *, observed=(), money, requests, lambda_money, horizon):
    """Enumerate STOP and affordable choices with an independent joint-table DP."""
    require(type(horizon) is int and 0 <= horizon <= 2, "Invalid reference horizon")
    require(all(math.isfinite(float(x)) and x >= 0 for x in (money, requests, lambda_money)), "Invalid budget/price")
    cube = np.asarray(cube, dtype=float)
    probability_checks(cube, ())
    catalog = {a["action_id"]: a for a in actions}
    require(len(catalog) == len(actions) and set(catalog) <= set(AXES), "Invalid action catalog")
    observed = set(observed)
    require(observed <= set(catalog), "Unknown observed action")
    risk = entropy(marginal(cube))
    best = {"decision": "stop", "action_id": None, "objective": risk,
            "selection_score": 0.0, "expected_terminal_risk": risk,
            "expected_plan_cost_vector": dict.fromkeys(UNITS, 0.0)}
    candidates = {}
    if horizon:
        for aid, action in sorted(catalog.items()):
            cost = {u: float(action["cost"].get(u, 0)) for u in UNITS}
            require(all(math.isfinite(x) and x >= 0 for x in cost.values()), "Invalid resource cost")
            if aid in observed or not action.get("available", True):
                continue
            if cost["money"] > money + 1e-12 or cost["requests"] > requests + 1e-12:
                continue
            expected_cost = cost.copy()
            terminal, conditional_entropy = 0.0, 0.0
            continuation = []
            for category, mass, posterior in branches(cube, aid):
                later = reference_plan(posterior, actions, observed=observed | {aid},
                         money=max(0., money-cost["money"]), requests=max(0., requests-cost["requests"]),
                         lambda_money=lambda_money, horizon=horizon-1)
                terminal += mass * later["expected_terminal_risk"]
                conditional_entropy += mass * entropy(marginal(posterior))
                for unit in UNITS:
                    expected_cost[unit] += mass * later["expected_plan_cost_vector"][unit]
                continuation.append({"category": category, "probability": mass,
                                     "posterior_probabilities": marginal(posterior).tolist(),
                                     "decision": later["decision"], "action_id": later["action_id"],
                                     "expected_terminal_risk": later["expected_terminal_risk"],
                                     "expected_plan_cost_vector": later["expected_plan_cost_vector"],
                                     "objective": later["objective"]})
            objective = terminal + lambda_money * expected_cost["money"]
            candidate = {"decision": "action", "action_id": aid, "objective": objective,
                         "immediate_evsi": risk-conditional_entropy,
                         "expected_terminal_risk": terminal, "expected_plan_cost_vector": expected_cost,
                         "expected_loss_improvement": risk-terminal,
                         "expected_plan_penalty": lambda_money*expected_cost["money"],
                         "planning_net_gain": risk-objective, "selection_score": round(risk-objective, 12),
                         "branches": continuation}
            candidates[aid] = candidate
            if candidate["selection_score"] > best["selection_score"]:
                best = candidate.copy()
    return {**best, "current_risk": risk, "candidates_by_action": candidates}


def source_summary(row):
    """Independently specified, prescheduled lossy return summaries."""
    fields = {"elo_difference": ("home_elo", "away_elo"),
              "form_difference": ("home_rolling_points", "away_rolling_points"),
              "net_xg_difference": ("home_rolling_xg_for", "home_rolling_xg_against",
                                    "away_rolling_xg_for", "away_rolling_xg_against")}
    output = {}
    for name, members in fields.items():
        values = [row.get(f) for f in members]
        if any(v is None for v in values):
            output[name] = None
        else:
            require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "Invalid source feature")
            output[name] = values[0] - values[1] if len(values) == 2 else values[0] - values[1] - values[2] + values[3]
    return output


def source_categories(row, quantiles):
    summary = source_summary(row)
    bucket = lambda name: int(np.searchsorted(quantiles[name], summary[name], side="right"))
    result = ("missing" if summary["elo_difference"] is None or summary["form_difference"] is None
              else f"elo{bucket('elo_difference')}_form{bucket('form_difference')}")
    xg = "missing" if summary["net_xg_difference"] is None else f"xg{bucket('net_xg_difference')}"
    return {"results_view": result, "xg_view": xg}


def rebuild_joint(model, state):
    """Recompute fitted q from the saved training records, not model._q()."""
    from .state import GROUP_FIELDS, sanitize_state

    public = sanitize_state(state)
    free = deepcopy(public)
    free["groups"] = {g: {"status": "not_acquired", "values": {}} for g in GROUP_FIELDS}
    p0 = np.asarray(model.base_predictor.predict_partial(free)["probabilities"])
    # The production model normalizes to remove sub-ulp summation drift.
    p0 = p0 / p0.sum()
    results = model.metadata["results_categories"]
    xgs = model.metadata["xg_categories"]
    likelihood = np.full((3, len(results), len(xgs)), model.dirichlet_mass / (len(results) * len(xgs)))
    for record in model.records:
        distance = sum((p - q) ** 2 for p, q in zip(p0, record["probabilities"]))
        weight = math.exp(-distance / (2 * model.bandwidth ** 2))
        likelihood[record["outcome"], results.index(record["results_category"]), xgs.index(record["xg_category"])] += weight
    for y in range(3):
        likelihood[y] /= likelihood[y].sum()
    joint = p0[:, None, None] * likelihood
    values = {name: value for group in public["groups"].values() for name, value in group["values"].items()}
    categories = source_categories(values, model.quantiles)
    observed = {}
    for aid, groups in (("results_view", ("elo", "recent_form")), ("xg_view", ("xg",))):
        statuses = [public["groups"][g]["status"] for g in groups]
        if all(s in ("observed", "source_missing") for s in statuses):
            observed[aid] = categories[aid]
            vocabulary = results if aid == "results_view" else xgs
            keep = vocabulary.index(categories[aid])
            index = [slice(None)] * 3
            index[AXES[aid]] = np.arange(len(vocabulary)) != keep
            joint[tuple(index)] = 0
    joint /= joint.sum()
    return joint, observed


def verify_plan_fields(decision, reference, categories):
    """Check every feasible action, including branches the policy did not take."""
    expected = "acquire" if reference["action_id"] is not None else "stop"
    require(decision["decision"] == expected and decision["action_id"] == reference["action_id"], "Independent Bellman decision mismatch")
    for name in ("current_risk", "expected_terminal_risk"):
        close(decision[name], reference[name], "selected " + name)
    close(decision["expected_total_risk"], reference["objective"], "selected risk plus cost")
    for unit in UNITS:
        close(decision["expected_plan_cost_vector"][unit], reference["expected_plan_cost_vector"][unit], "selected expected " + unit)
    actual = {r["action_id"]: r for r in decision["candidates"] if r["feasible"]}
    require(set(actual) == set(reference["candidates_by_action"]), "Feasible candidate set differs")
    branch_count = 0
    for aid, wanted in reference["candidates_by_action"].items():
        recorded = actual[aid]
        for name in ("immediate_evsi", "expected_terminal_risk", "expected_loss_improvement",
                     "expected_plan_penalty", "planning_net_gain", "selection_score"):
            close(recorded[name], wanted[name], aid + " " + name)
        for unit in UNITS:
            close(recorded["expected_plan_cost_vector"][unit], wanted["expected_plan_cost_vector"][unit], aid + " planned " + unit)
        options = {branch["observation"]: branch for branch in recorded["branches"]}
        require(len(options) == len(recorded["branches"]) == len(wanted["branches"]), "Return branch coverage mismatch")
        for branch in wanted["branches"]:
            token = categories[aid][branch["category"]]
            require(token in options, "Missing return category")
            other = options[token]
            close(other["probability"], branch["probability"], "branch probability")
            require(other["continuation_action"] == branch["action_id"], "Branch contingent continuation differs")
            require(other["hypothetical"] is True, "Branch must be explicitly hypothetical")
            for x, y in zip(other["posterior_probabilities"], branch["posterior_probabilities"]):
                close(x, y, "branch posterior")
            close(other["continuation_expected_terminal_risk"], branch["expected_terminal_risk"], "branch terminal risk")
            close(other["continuation_expected_total_risk"], branch["objective"], "branch risk plus cost")
            for unit in UNITS:
                close(other["continuation_expected_cost_vector"][unit], branch["expected_plan_cost_vector"][unit], "branch expected " + unit)
            branch_count += 1
    return branch_count


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_training(model, fold, sources, contexts, cutoff):
    from .experiment import outcome
    from .state import parse_time

    expected_ids = set(fold["training_match_ids"]) & set(contexts)
    records = {r["match_id"]: r for r in model.records}
    require(len(records) == len(model.records), "Duplicate joint training match")
    require(set(records) == expected_ids == set(model.metadata["training_match_ids"]), "Joint training membership differs")
    require(expected_ids == set(fold["joint_training_match_ids"]), "Fold joint membership differs")
    require(set(model.base_predictor.metadata["training_match_ids"]) == set(fold["training_match_ids"]), "Base training split mismatch")
    require(parse_time(model.metadata["training_as_of"]) == cutoff, "Joint training cutoff mismatch")
    require(parse_time(model.base_predictor.metadata["training_as_of"]) == cutoff, "Base training cutoff mismatch")
    summaries = [source_summary(sources[mid]) for mid in expected_ids]
    for field, boundaries in model.quantiles.items():
        expected = np.quantile([s[field] for s in summaries if s[field] is not None], [1/3, 2/3])
        require(len(boundaries) == 2, "Tertile boundary count")
        for a, b in zip(boundaries, expected):
            close(a, b, "train-only quantile " + field)
    for mid, record in records.items():
        row, context = sources[mid], contexts[mid]
        require(parse_time(row["result_available_at"]) <= cutoff, "Joint label available after training")
        require(parse_time(context["oof_train_cutoff"]) < parse_time(row["prediction_cutoff"]) < parse_time(row["kickoff"]), "OOF temporal leakage")
        require(mid not in context["oof_train_ids"], "OOF self training")
        for field in ("features_available_at", "fixture_available_at"):
            require(parse_time(row[field]) <= parse_time(row["prediction_cutoff"]), "Historical information cutoff violation")
        for field in ("kickoff", "prediction_cutoff", "result_available_at"):
            require(parse_time(record[field]) == parse_time(row[field]), "Training source time differs")
        for field in ("oof_train_cutoff", "oof_manifest_ref", "oof_predictor_version"):
            require(record[field] == context[field], "Training OOF provenance differs")
        require(record["outcome"] == outcome(row), "Joint training outcome differs")
        expected = source_categories(row, model.quantiles)
        require(record["results_category"] == expected["results_view"] and record["xg_category"] == expected["xg_view"], "Training return category differs")
        for a, b in zip(record["probabilities"], context["probabilities"]):
            close(a, b, "Training OOF prior")
    return len(records)


def verify(run, bundle, earlier_bundle):
    from football_research.evaluation.splits import rolling_origin_splits
    from .contextual_partial import ContextualPartialPredictor
    from .data import ControlledEpisode, load_episodes
    from .experiment import losses, outcome, state_after
    from .joint_returns import JointReturnPredictor, extract_oof_contexts
    from .state import parse_time
    from .verify_round4 import _rows
    from .verify_round6 import KEYS, verify_aggregate_metrics
    from .verify_round7 import verify_episode_coverage, verify_heuristic_trace

    run = Path(run)
    read = lambda name: json.loads((run/name).read_text(encoding="utf-8"))
    report, design = read("report.json"), read("design.json")
    require(report["implementation_sha256"] == read("implementation.json"), "Code manifest differs")
    for name, value in read("implementation.json").items():
        require(digest(Path(__file__).parent/name) == value, "Source changed: " + name)
    for artifact in report["model_artifacts"]:
        require(digest(run/artifact["path"]) == artifact["sha256"], "Model hash differs")
    prior = Path(design["source_round7"])
    require(read("source-artifacts.json") == report["source_artifacts_sha256"], "Prior manifest differs")
    for name, value in read("source-artifacts.json").items():
        require(digest(prior/name) == value, "Frozen round-seven source changed: " + name)
    require(digest(run/"catalog.json") == digest(prior/"catalog.json"), "Tool catalog changed")
    actions = read("catalog.json")["actions"]
    current, ca = load_episodes(bundle, lead_hours=design["lead_hours"])
    earlier, ea = load_episodes(earlier_bundle, lead_hours=design["lead_hours"])
    for name, audit in (("current", ca), ("earlier", ea)):
        require(read("data-audit.json")[name]["platform_manifest_sha256"] == audit["platform_manifest_sha256"], "Input export changed")
    all_rows = earlier + current
    sources = {r["match_id"]: r for r in all_rows}
    require(len(sources) == len(all_rows) == report["input_matches"], "Input coverage mismatch")

    # The only reused OOF targets are empty-context forecasts; verify those
    # against their original forward model and source, without using old deltas.
    contexts_list = list(_rows(run/"training-oof-contexts.jsonl"))
    contexts = {r["match_id"]: r for r in contexts_list}
    require(len(contexts) == len(contexts_list), "Duplicate OOF context")
    require(contexts == extract_oof_contexts(_rows(prior/"oof-supervision.jsonl")), "OOF prior extraction mismatch")
    oof = json.loads((prior/"oof-manifest.json").read_text(encoding="utf-8"))
    checked_contexts = set()
    for entry in oof["folds"]:
        path = Path(entry["artifact"])
        if not path.is_file():
            path = prior/path
        require(digest(path) == entry["sha256"], "OOF model changed")
        predictor = ContextualPartialPredictor.load(path)
        train = set(entry["training_match_ids"])
        require(train == set(predictor.metadata["training_match_ids"]), "OOF train IDs differ")
        for mid in train:
            require(parse_time(sources[mid]["result_available_at"]) <= parse_time(entry["cutoff"]), "OOF future label")
        for mid in entry["prediction_match_ids"]:
            context = contexts[mid]
            require(mid not in checked_contexts and mid not in train, "OOF target overlaps")
            checked_contexts.add(mid)
            require(train == set(context["oof_train_ids"]), "OOF context train IDs differ")
            require(context["oof_train_cutoff"] == entry["cutoff"], "OOF context cutoff differs")
            require(parse_time(entry["cutoff"]) < parse_time(sources[mid]["prediction_cutoff"]), "OOF lookahead")
            require(predictor.model_version == context["oof_predictor_version"] == entry["model_version"], "OOF version differs")
            expected = predictor.predict_partial(state_after(sources[mid], actions, ())) ["probabilities"]
            for a, b in zip(context["probabilities"], expected):
                close(a, b, "Recomputed original OOF prior")
    require(checked_contexts == set(contexts), "OOF target coverage differs")

    prior_report = json.loads((prior/"report.json").read_text(encoding="utf-8"))
    prior_folds = {f["fold"]: f for f in prior_report["folds"]}
    splits = list(rolling_origin_splits(all_rows, initial_validation_start=design["validation_start"],
                  validation_window=timedelta(days=30), test_window=timedelta(days=30)))
    models, fold_ids, training_records = {}, {}, 0
    for fold in report["folds"]:
        number = fold["fold"]
        split = splits[number]
        require(fold["split"] == split.manifest(), "Recomputed outer split differs")
        for key, records in (("training_match_ids", split.train), ("validation_match_ids", split.validation), ("test_match_ids", split.test)):
            require(set(fold[key]) == set(prior_folds[number][key]) == {r["match_id"] for r in records}, "Frozen partition differs")
        fold_ids[number] = set(fold["test_match_ids"])
        base_path = run/fold["predictor"]["path"]
        require(digest(base_path) == digest(prior/"models"/f"fold-{number}-predictor.json") == fold["predictor"]["sha256"], "Frozen base predictor differs")
        joint_path = run/fold["joint_model"]["path"]
        require(digest(joint_path) == fold["joint_model"]["sha256"], "Joint fold hash mismatch")
        model = JointReturnPredictor.load(joint_path)
        require(model.base_predictor.to_state() == ContextualPartialPredictor.load(base_path).to_state(), "Embedded base differs")
        require(model.model_version == fold["joint_model"]["version"], "Joint version mismatch")
        require(model.bandwidth == design["joint_model"]["bandwidth"] and model.dirichlet_mass == design["joint_model"]["dirichlet_mass"], "Prescheduled parameters differ")
        training_records += verify_training(model, fold, sources, contexts, split.train_cutoff)
        models[number] = model

    cube_cache, plan_cache, prediction_cache = {}, {}, {}
    max_martingale_error = 0.0

    def state_data(number, mid, acquired):
        nonlocal max_martingale_error
        key = number, mid, tuple(sorted(acquired))
        if key not in cube_cache:
            state = state_after(sources[mid], actions, acquired)
            cube, observed = rebuild_joint(models[number], state)
            production = models[number].joint_distribution(state)
            require(np.max(np.abs(cube-np.asarray(production["probabilities"]))) < 1e-10, "Independent fitted joint differs")
            diagnostics = probability_checks(cube, [a for a in AXES if a not in observed])
            if diagnostics:
                max_martingale_error = max(max_martingale_error, *(d["posterior_expectation_max_error"] for d in diagnostics.values()))
            cube_cache[key] = cube, observed
            prediction_cache[key] = models[number].predict_partial(state)
        return cube_cache[key], prediction_cache[key]

    def plan_for(number, mid, acquired, money, requests, lam, horizon):
        key = number, mid, tuple(sorted(acquired)), round(money, 12), requests, lam, horizon
        if key not in plan_cache:
            (cube, observed), _ = state_data(number, mid, acquired)
            plan_cache[key] = reference_plan(cube, actions, observed=observed,
                money=max(0., money), requests=max(0., requests), lambda_money=lam, horizon=horizon)
        return plan_cache[key]

    validation_replays = 0
    for fold in report["folds"]:
        number = fold["fold"]
        for key, selection in fold["lambda_selection"].items():
            strategy, budget_text = key.split("/")
            budget, horizon = float(budget_text), (2 if strategy == "evsi-two-step" else 1)
            require(set(selection["validation_match_ids"]) == set(fold["validation_match_ids"]), "Lambda uses nonvalidation rows")
            require({r["lambda_money"] for r in selection["candidates"]} == set(design["lambda_money_grid"]), "Lambda grid differs")
            for candidate in selection["candidates"]:
                utilities = []
                for mid in selection["validation_match_ids"]:
                    acquired, spent = [], 0.0
                    for _ in range(len(actions)+1):
                        decision = plan_for(number, mid, acquired, budget-spent, design["request_budget"]-len(acquired), candidate["lambda_money"], horizon)
                        if decision["action_id"] is None:
                            break
                        aid = decision["action_id"]
                        acquired.append(aid)
                        spent += next(a["cost"]["money"] for a in actions if a["action_id"] == aid)
                    (_, _), prediction = state_data(number, mid, acquired)
                    utilities.append(losses(prediction["probabilities"], outcome(sources[mid]))["log_loss"] + design["validation_utility_money_weight"]*spent)
                close(candidate["validation_utility"], np.mean(utilities), "Independent validation utility")
                require(candidate["validation_matches"] == len(utilities), "Validation count differs")
                validation_replays += len(utilities)
            close(selection["selected"], min(selection["candidates"], key=lambda c: (c["validation_utility"], -c["lambda_money"]))["lambda_money"], "Selected lambda")
        print(f"round8 audit fold {number}: training and independent validation verified", flush=True)

    rows = list(_rows(run/"episodes.jsonl"))
    require(len(rows) == report["episode_runs"], "Episode count differs")
    verify_episode_coverage(rows, fold_ids, design["strategies"], design["money_budgets"], design["random_seeds"])
    verify_aggregate_metrics(rows, report["metrics"])
    old_base = {(r["match_id"], r["budget_money"]): r["final_probabilities"] for r in _rows(prior/"episodes.jsonl") if r["strategy"] == "base-only"}
    keyed = {tuple(r[k] for k in KEYS): r for r in rows}
    seen, events_count, charges, numerical_count, branch_count, base_parity = set(), 0, 0, 0, 0, 0
    fold_lookup = {f["fold"]: f for f in report["folds"]}
    for trace in _rows(run/"trajectories.jsonl"):
        key = tuple(trace[k] for k in KEYS)
        require(key in keyed and key not in seen, "Duplicate or unknown trajectory")
        seen.add(key)
        row = keyed[key]
        number, mid = row["fold"], row["match_id"]
        source, model = sources[mid], models[number]
        require(row["kickoff"] == source["kickoff"] and row["prediction_cutoff"] == source["prediction_cutoff"], "Episode source time differs")
        require(row["model_version"] == model.model_version and row["base_model_version"] == model.base_predictor.model_version, "Episode model differs")
        learned = row["strategy"].startswith("evsi-")
        numerical_count += int(learned)
        if learned:
            selected = fold_lookup[number]["lambda_selection"][f"{row['strategy']}/{row['budget_money']:g}"]["selected"]
            close(row["lambda_money"], selected, "Test lambda")
        env, spent = ControlledEpisode(source, actions), dict.fromkeys(UNITS, 0.0)
        events = trace["events"]
        require(bool(events), "Empty trajectory")
        for step, event in enumerate(events):
            require(event["step"] == step, "Noncontiguous step")
            expected_state = {**env.visible_state(), "action_history": [{"action_id": aid, "status": "acquired_or_source_missing"} for aid in env.performed_actions]}
            require(event["visible_state"] == expected_state, "Hidden data or invalid visible state")
            (cube, observed), prediction = state_data(number, mid, env.performed_actions)
            require(len(event["prediction"]["probabilities"]) == 3, "Prediction length differs")
            for a, b, c in zip(event["prediction"]["probabilities"], prediction["probabilities"], marginal(cube)):
                close(a, b, "Model prediction")
                close(a, c, "Independent joint marginal")
            decision = event["decision"]
            legal = env.legal_actions()
            if learned and legal:
                reference = plan_for(number, mid, env.performed_actions, row["budget_money"]-spent["money"], design["request_budget"]-spent["requests"], row["lambda_money"], 2 if row["strategy"] == "evsi-two-step" else 1)
                branch_count += verify_plan_fields(decision, reference, {"results_view": model.metadata["results_categories"], "xg_view": model.metadata["xg_categories"]})
                for unit in UNITS:
                    close(decision["physical_budget"]["spent"][unit], spent[unit], "Physical resource spent")
                close(decision["physical_budget"]["limits"]["money"], row["budget_money"], "Physical money limit")
                close(decision["physical_budget"]["limits"]["requests"], design["request_budget"], "Physical request limit")
            elif learned:
                require(decision == {"decision": "stop", "action_id": None, "reason": "no_legal_action"}, "No-action stop differs")
            if decision["decision"] == "stop":
                require(step == len(events)-1 and "charged_cost" not in event, "Action after stop or charged stop")
                require(row["stop_reason"] == decision["reason"], "Terminal reason differs")
                break
            require(decision["decision"] in ("action", "acquire"), "Unknown decision")
            selected = next((a for a in legal if a["action_id"] == decision["action_id"]), None)
            require(selected is not None, "Repeated or unavailable action")
            require(event["charged_cost"] == selected["cost"], "Wrong immediate charge")
            for unit in UNITS:
                spent[unit] += selected["cost"].get(unit, 0)
            require(spent["money"] <= row["budget_money"]+1e-10 and spent["requests"] <= design["request_budget"], "Physical budget violation")
            env.reveal(selected["action_id"])
            charges += 1
        require(events[-1]["decision"]["decision"] == "stop", "No terminal stop")
        require(row["actions"] == list(env.performed_actions) and row["final_mask"] == prediction["mask"], "Final observed mask differs")
        for a, b in zip(row["final_probabilities"], prediction["probabilities"]):
            close(a, b, "Final probability")
        for unit in UNITS:
            close(row["cost"][unit], spent[unit], "Final accumulated cost")
        for name, value in losses(prediction["probabilities"], outcome(source)).items():
            close(row[name], value, "Recomputed held-out " + name)
        require(row["actual_external_requests"] == row["actual_llm_tokens"] == 0, "Unexpected external calls")
        if not learned:
            verify_heuristic_trace(row, trace, source, actions, design)
        if row["strategy"] == "base-only":
            require(row["final_probabilities"] == old_base[mid, row["budget_money"]], "Exact round-seven p0 parity fails")
            base_parity += 1
        events_count += len(events)
    require(seen == set(keyed), "Missing trajectory")
    return {"status": "VERIFIED", "round": 8, "test_matches": len({r["match_id"] for r in rows}),
            "episode_runs": len(rows), "oof_empty_priors_recomputed": len(contexts),
            "joint_training_records_recomputed": training_records, "model_artifacts": len(report["model_artifacts"]),
            "visible_states_replayed": events_count, "unique_joint_states_independently_rebuilt": len(cube_cache),
            "independent_bellman_decisions": len(plan_cache), "contingent_branches_verified": branch_count,
            "max_posterior_expectation_error": max_martingale_error,
            "acquisition_charges_verified": charges, "numerical_policy_episodes": numerical_count,
            "heuristic_policy_episodes": len(rows)-numerical_count, "validation_episodes_recomputed": validation_replays,
            "exact_round7_base_rows": base_parity, "external_requests": 0, "llm_tokens": 0,
            "artifacts": {n: digest(run/n) for n in ("report.json", "design.json", "episodes.jsonl", "trajectories.jsonl", "training-oof-contexts.jsonl", "return-diagnostics.jsonl", "initial-return-diagnostics.jsonl")},
            "verifier_sha256": digest(__file__),
            "scope": "Independent joint-table/Bellman/trace integrity; no claim of real-world predictive benefit"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("run", "bundle", "earlier-bundle", "output"):
        parser.add_argument("--"+argument, required=True)
    args = parser.parse_args()
    result = verify(args.run, args.bundle, args.earlier_bundle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "Verification output already exists")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
