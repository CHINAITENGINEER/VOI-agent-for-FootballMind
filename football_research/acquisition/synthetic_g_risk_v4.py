"""Synthetic g with a separately calibrated negative-expected-net risk head.

G is the shared-world teacher's *expected* fixed-f log-loss reduction. The
risk target is 1{G-cost<0}, not 1{one realized outcome worsens the prediction}.
Only observable v3 features enter g; the risk head additionally sees cost.
No empirical gate, acquisition runtime, or historical artifact is changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from . import synthetic_g_observable_v3 as v3
from . import synthetic_g_shared_world as teacher

SCHEMA = "synthetic-g-risk-aware-v4"
RISK_TARGET = "probability_teacher_expected_net_negative_given_observables_and_cost"
PROVENANCE = {
    **v3.PROVENANCE,
    "evidence_origin": "constructed_shared_world_observable_mean_and_risk_heads",
    "risk_calibration_scope": "synthetic_heldout_match_groups_only",
}
DEFAULT_SEED = 20261021
MARGINS = (0., .002, .005, .01)
RISK_LIMITS = (.10, .20, .30, .40, .50)
VALIDATION_MAX_HARM = .30
MIN_VALIDATION_QUERIES = 20
MIN_VALIDATION_QUERY_GROUPS = 15


def cost_value(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0.):
        raise ValueError("Explicit finite nonnegative loss-unit cost required")
    return float(value)


def group_split(rows, seed):
    ids = sorted({row["physical_match_id"] for row in rows})
    if len(ids) < 40:
        raise ValueError("At least 40 groups for four-way synthetic evaluation")
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    boundaries = (0, n // 2, int(.7 * n), int(.85 * n), n)
    return {name: sorted(ids[boundaries[i]:boundaries[i + 1]])
            for i, name in enumerate(("train", "calibration", "validation", "test"))}


def _matrix(rows, *, risk=False):
    x = np.vstack([v3.feature_vector(row["observable_features"]) for row in rows])
    return np.column_stack((x, [cost_value(row["cost_units"]) for row in rows])) if risk else x


def choose_action(candidates, rule):
    """Choose using predictions only; a safer second candidate may still win."""
    if rule["mode"] == "stop_only":
        return None
    eligible = [candidate for candidate in candidates
                if candidate["predicted_net"] > rule["stop_margin"]
                and (rule["mode"] == "mean_only"
                     or candidate["predicted_risk"] <= rule["risk_limit"])]
    return min(eligible, key=lambda item: (-item["predicted_net"], item["action_id"])) if eligible else None


def evaluate(rows, gain_predictions, risk_predictions, rule):
    if (len(rows) != len(gain_predictions) or len(rows) != len(risk_predictions)
            or not np.isfinite(gain_predictions).all() or not np.isfinite(risk_predictions).all()
            or np.any(np.asarray(risk_predictions) < 0.) or np.any(np.asarray(risk_predictions) > 1.)):
        raise ValueError("Finite gain and risk predictions required for every candidate")
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["state_id"], []).append({
            "index": i, "action_id": row["action_id"],
            "predicted_net": float(gain_predictions[i] - row["cost_units"]),
            "predicted_risk": float(risk_predictions[i]),
        })
    traces = []
    for state_id, candidates in sorted(groups.items()):
        selected = choose_action(candidates, rule)
        oracle = min(candidates, key=lambda item: (-rows[item["index"]]["teacher_net_value"], item["action_id"]))
        oracle_row = rows[oracle["index"]]
        oracle_net = max(0., oracle_row["teacher_net_value"])
        row = rows[selected["index"]] if selected else None
        net = row["teacher_net_value"] if row else 0.
        traces.append({
            "state_id": state_id, "physical_match_id": rows[candidates[0]["index"]]["physical_match_id"],
            "selected": selected["action_id"] if selected else "STOP",
            "selected_row_index": selected["index"] if selected else None,
            "predicted_net": selected["predicted_net"] if selected else None,
            "predicted_risk": selected["predicted_risk"] if selected else None,
            "net_value": net, "queried": row is not None,
            "oracle": oracle["action_id"] if oracle_net > 0. else "STOP",
            "oracle_net_value": oracle_net, "oracle_queried": oracle_net > 0.,
            "regret": oracle_net - net,
        })
    queried = [t for t in traces if t["queried"]]
    oracle_queries = [t for t in traces if t["oracle_queried"]]
    return {"summary": {
        "states": len(traces), "matches": len({t["physical_match_id"] for t in traces}),
        "queried_states": len(queried),
        "queried_matches": len({t["physical_match_id"] for t in queried}),
        "negative_expected_net_queries": sum(t["net_value"] < 0. for t in queried),
        "negative_expected_net_given_query": float(np.mean([t["net_value"] < 0. for t in queried])) if queried else None,
        "positive_net_precision": float(np.mean([t["net_value"] > 0. for t in queried])) if queried else None,
        "query_rate": len(queried) / len(traces),
        "mean_teacher_net_value": float(np.mean([t["net_value"] for t in traces])),
        "mean_regret": float(np.mean([t["regret"] for t in traces])),
        "top1_including_stop": float(np.mean([t["selected"] == t["oracle"] for t in traces])),
        "oracle_query_states": len(oracle_queries),
        "oracle_query_recall": float(np.mean([t["queried"] for t in oracle_queries])) if oracle_queries else None,
        "selected_risk_brier": float(np.mean([(t["predicted_risk"] - (t["net_value"] < 0.)) ** 2 for t in queried])) if queried else None,
        "selected_mean_predicted_risk": float(np.mean([t["predicted_risk"] for t in queried])) if queried else None,
        "mean_negative_net_magnitude_per_query": float(np.mean([max(0., -t["net_value"]) for t in queried])) if queried else None,
    }, "traces": traces}


def select_rules(rows, means, risks):
    mean_search, risk_search = [], []
    for margin in MARGINS:
        rule = {"mode": "mean_only", "stop_margin": margin, "risk_limit": None}
        mean_search.append({"rule": rule, **evaluate(rows, means, risks, rule)["summary"]})
        for limit in RISK_LIMITS:
            rule = {"mode": "risk_filtered", "stop_margin": margin, "risk_limit": limit}
            result = {"rule": rule, **evaluate(rows, means, risks, rule)["summary"]}
            result["validation_feasible"] = (
                result["queried_states"] >= MIN_VALIDATION_QUERIES
                and result["queried_matches"] >= MIN_VALIDATION_QUERY_GROUPS
                and result["negative_expected_net_given_query"] <= VALIDATION_MAX_HARM
                and result["mean_teacher_net_value"] > 0.)
            risk_search.append(result)
    mean_winner = max(mean_search, key=lambda item: (item["mean_teacher_net_value"], -item["query_rate"]))
    feasible = [item for item in risk_search if item["validation_feasible"]]
    risk_winner = max(feasible, key=lambda item: (item["mean_teacher_net_value"], -item["query_rate"])) if feasible else None
    return {"mean_rule": mean_winner["rule"],
            "risk_rule": risk_winner["rule"] if risk_winner else {
                "mode": "stop_only", "stop_margin": 0., "risk_limit": None},
            "status": "nontrivial_validation_rule_found" if risk_winner else "no_feasible_nontrivial_rule",
            "mean_search": mean_search, "risk_search": risk_search}


def _forest_predict(trees, matrix):
    x = np.asarray(matrix, dtype=np.float32)
    result = np.zeros(len(x))
    for tree in trees:
        stack = [(0, np.arange(len(x)))]
        while stack:
            node, indices = stack.pop()
            if not len(indices):
                continue
            if tree["left"][node] == -1:
                result[indices] += tree["value"][node]
            else:
                left = x[indices, tree["feature"][node]] <= tree["threshold"][node]
                stack.append((tree["left"][node], indices[left]))
                stack.append((tree["right"][node], indices[~left]))
    return result / len(trees)


def _validate_forest(trees, dimensions, *, probability=False):
    if not isinstance(trees, list) or not trees:
        raise ValueError("Nonempty portable forest required")
    for tree in trees:
        if not isinstance(tree, dict) or set(tree) != {"left", "right", "feature", "threshold", "value"}:
            raise ValueError("Malformed portable tree")
        n = len(tree["left"])
        if not n or any(len(values) != n for values in tree.values()):
            raise ValueError("Inconsistent tree lengths")
        for i in range(n):
            left, right, feature = tree["left"][i], tree["right"][i], tree["feature"][i]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (left, right, feature)):
                raise ValueError("Tree indexes must be integers")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in (tree["threshold"][i], tree["value"][i])):
                raise ValueError("Finite tree values required")
            if probability and not 0. <= tree["value"][i] <= 1.:
                raise ValueError("Risk tree values must be probabilities")
            if left == right == -1:
                continue
            if not (i < left < n and i < right < n and 0 <= feature < dimensions):
                raise ValueError("Invalid or cyclic tree")


class RiskAwareSyntheticGModel:
    """Portable synthetic-only mean/risk estimator; no execution authority."""

    def __init__(self, payload):
        if (payload.get("schema") != SCHEMA or payload.get("feature_schema") != v3.FEATURE_SCHEMA
                or payload.get("mean_feature_names") != list(v3.OBSERVABLE_NAMES)
                or payload.get("risk_feature_names") != [*v3.OBSERVABLE_NAMES, "cost_units"]
                or payload.get("risk_target") != RISK_TARGET or payload.get("provenance") != PROVENANCE):
            raise ValueError("Exact synthetic risk v4 model contract required")
        _validate_forest(payload["mean_trees"], len(v3.OBSERVABLE_NAMES))
        _validate_forest(payload["risk_trees"], len(v3.OBSERVABLE_NAMES) + 1, probability=True)
        calibration = payload["calibration"]
        x, y = calibration["x"], calibration["y"]
        if (not x or len(x) != len(y)
                or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                       or not math.isfinite(v) or not 0. <= v <= 1. for v in [*x, *y])
                or any(a >= b for a, b in zip(x, x[1:]))
                or any(a > b for a, b in zip(y, y[1:]))):
            raise ValueError("Monotone finite isotonic knots required")
        for name in ("mean_rule", "risk_rule"):
            rule = payload[name]
            if rule["mode"] not in {"mean_only", "risk_filtered", "stop_only"}:
                raise ValueError("Unknown selection rule")
            cost_value(rule["stop_margin"])
            if rule["mode"] == "risk_filtered":
                v3._unit(rule["risk_limit"], "risk_limit")
        self.payload = deepcopy(payload)
        self.model_sha256 = v3._digest(payload)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, features, costs):
        if len(features) != len(costs):
            raise ValueError("An explicit cost is required for every feature vector")
        if not features:
            return {"gain": np.array([]), "risk_raw": np.array([]), "risk": np.array([])}
        x = np.vstack([v3.feature_vector(f) for f in features])
        costs = [cost_value(c) for c in costs]
        gain = _forest_predict(self.payload["mean_trees"], x)
        raw = _forest_predict(self.payload["risk_trees"], np.column_stack((x, costs)))
        calibration = self.payload["calibration"]
        risk = np.interp(raw, calibration["x"], calibration["y"])
        return {"gain": gain, "risk_raw": raw, "risk": risk}

    def estimate(self, features, *, cost_units):
        predicted = self.predict([features], [cost_units])
        gain, risk = float(predicted["gain"][0]), float(predicted["risk"][0])
        return {"model_schema": SCHEMA, "model_sha256": self.model_sha256,
                "expected_signed_gain": gain, "expected_net_voi": gain - cost_value(cost_units),
                "probability_teacher_expected_net_negative": risk, "risk_target": RISK_TARGET,
                "cost_units": cost_value(cost_units), **PROVENANCE}

    def rank_shadow(self, candidates):
        if (not isinstance(candidates, list) or not candidates
                or any(not isinstance(c, dict) or not isinstance(c.get("action_id"), str)
                       or not c["action_id"] or "features" not in c or "cost_units" not in c for c in candidates)
                or len({c["action_id"] for c in candidates}) != len(candidates)):
            raise ValueError("A complete explicit menu with unique action IDs is required")
        rows = [{"action_id": c["action_id"], **self.estimate(c["features"], cost_units=c["cost_units"])}
                for c in candidates]
        inputs = [{"action_id": r["action_id"], "predicted_net": r["expected_net_voi"],
                   "predicted_risk": r["probability_teacher_expected_net_negative"]} for r in rows]
        selected = choose_action(inputs, self.payload["risk_rule"])
        return {"schema": "synthetic-g-risk-aware-shadow-v4", "candidates": rows,
                "shadow_proposal": selected["action_id"] if selected else "STOP",
                "rule": self.payload["risk_rule"], "risk_target": RISK_TARGET,
                "selection_allowed": False, "request_executed": False, **PROVENANCE}


def calibration_report(target, probabilities):
    y, p = np.asarray(target), np.asarray(probabilities)
    bins = []
    for i in range(10):
        mask = (p >= i / 10.) & (p < (i + 1) / 10. if i < 9 else p <= 1.)
        bins.append({"lower": i / 10., "upper": (i + 1) / 10., "count": int(mask.sum()),
                     "mean_prediction": float(p[mask].mean()) if mask.any() else None,
                     "observed_frequency": float(y[mask].mean()) if mask.any() else None})
    return {"brier": float(np.mean((p - y) ** 2)), "base_rate": float(y.mean()), "bins": bins}


def grouped_comparison(first, second, seed):
    ids = sorted({t["physical_match_id"] for t in first["traces"]})
    def aggregates(evaluation):
        return np.asarray([[np.mean([t["net_value"] for t in evaluation["traces"] if t["physical_match_id"] == mid]),
                            sum(t["queried"] for t in evaluation["traces"] if t["physical_match_id"] == mid),
                            sum(t["net_value"] < 0. for t in evaluation["traces"] if t["physical_match_id"] == mid)]
                           for mid in ids], dtype=float)
    a, b = aggregates(first), aggregates(second)
    indices = np.random.default_rng(seed).integers(0, len(ids), size=(2000, len(ids)))
    delta = (a[:, 0] - b[:, 0])[indices].mean(axis=1)
    ac, bc = a[indices, 1:].sum(axis=1), b[indices, 1:].sum(axis=1)
    valid = (ac[:, 0] > 0) & (bc[:, 0] > 0)
    harm_delta = ac[valid, 1] / ac[valid, 0] - bc[valid, 1] / bc[valid, 0]
    return {"mean_net_difference": float(np.mean(a[:, 0] - b[:, 0])),
            "mean_net_difference_cluster_bootstrap_95": np.quantile(delta, [.025, .975]).tolist(),
            "conditional_negative_net_difference_cluster_bootstrap_95": np.quantile(harm_delta, [.025, .975]).tolist() if len(harm_delta) else None,
            "conditional_bootstrap_valid_replicates": int(valid.sum()),
            "scope": "synthetic_match_groups_not_real_football_or_postselection_guarantee"}


def build_artifact(output, *, seed=DEFAULT_SEED, matches=2000, states_per_match=3):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a new empty output directory; existing artifacts are immutable")
    source = teacher.generate_dataset(seed=seed, matches=matches, states_per_match=states_per_match)
    # Namespace the simulator's positional IDs: a new seed is a new synthetic cohort.
    for row in source["rows"]:
        row["physical_match_id"] = row["physical_match_id"].replace("synthetic-v2:", f"synthetic-risk-v4:{seed}:")
        row["state_id"] = row["state_id"].replace("synthetic-v2:", f"synthetic-risk-v4:{seed}:")
    for world in source["teacher_only_worlds"]:
        world["state_id"] = world["state_id"].replace("synthetic-v2:", f"synthetic-risk-v4:{seed}:")
    rows = [{"physical_match_id": row["physical_match_id"], "state_id": row["state_id"],
             "action_id": row["action_id"], "observable_features": v3.make_observable_features(row["features"]),
             "cost_units": row["cost_units"],
             "teacher_expected_signed_gain": row["teacher"]["expected_signed_gain"],
             "teacher_net_value": row["teacher"]["net_value"],
             "negative_expected_net": int(row["teacher"]["net_value"] < 0.)} for row in source["rows"]]
    split = group_split(rows, seed + 1)
    id_sets = {key: set(ids) for key, ids in split.items()}
    partitions = {key: [row for row in rows if row["physical_match_id"] in ids] for key, ids in id_sets.items()}
    protocol = {
        "schema": SCHEMA, "seed": seed, "matches": matches, "states_per_match": states_per_match,
        "split": split, "feature_schema": v3.FEATURE_SCHEMA,
        "mean_target": "teacher_expected_signed_fixed_f_logloss_reduction_before_cost",
        "risk_target": RISK_TARGET, "risk_is_realized_outcome_harm_probability": False,
        "teacher": teacher.SCHEMA, "updater": teacher.UPDATER,
        "mean_learner": {"n_estimators": 96, "max_depth": 14, "min_samples_leaf": 8},
        "risk_learner": {"n_estimators": 128, "max_depth": 14, "min_samples_leaf": 12},
        "calibrator": "isotonic_on_separate_calibration_matches",
        "margins": MARGINS, "risk_limits": RISK_LIMITS,
        "validation_constraint": {"max_observed_negative_expected_net_given_query": VALIDATION_MAX_HARM,
            "min_queries": MIN_VALIDATION_QUERIES, "min_queried_matches": MIN_VALIDATION_QUERY_GROUPS,
            "positive_mean_net_required": True,
            "meaning": "research_operating_point_not_a_statistical_safety_guarantee"},
        "selection": "maximize_validation_mean_net_among_feasible_rules_else_STOP",
        "all_stop_does_not_count_as_risk_success": True,
        "test_retuning_allowed": False, "cost_basis": "simulated_logloss_penalty_not_dollars",
        "implementation_sha256": {Path(path).name: sha256(Path(path).read_bytes()).hexdigest()
                                  for path in (__file__, v3.__file__, teacher.__file__)},
        "provenance": PROVENANCE,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "protocol.json").write_bytes(v3._canonical(protocol))
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.isotonic import IsotonicRegression
    train, cal, val, test = (partitions[key] for key in ("train", "calibration", "validation", "test"))
    y = np.asarray([r["teacher_expected_signed_gain"] for r in train])
    labels = np.asarray([r["negative_expected_net"] for r in train])
    mean_fit = ExtraTreesRegressor(**protocol["mean_learner"], max_features=1., random_state=seed + 2, n_jobs=2).fit(_matrix(train), y)
    risk_fit = ExtraTreesRegressor(**protocol["risk_learner"], max_features=1., random_state=seed + 3, n_jobs=2).fit(_matrix(train, risk=True), labels)
    calibrator = IsotonicRegression(y_min=0., y_max=1., out_of_bounds="clip").fit(
        risk_fit.predict(_matrix(cal, risk=True)), [r["negative_expected_net"] for r in cal])
    val_mean = mean_fit.predict(_matrix(val))
    val_risk = calibrator.predict(risk_fit.predict(_matrix(val, risk=True)))
    search = select_rules(val, val_mean, val_risk)
    constant_search = [{"rule": {"mode": "mean_only", "stop_margin": margin, "risk_limit": None},
                        **evaluate(val, np.full(len(val), float(y.mean())), val_risk,
                                   {"mode": "mean_only", "stop_margin": margin, "risk_limit": None})["summary"]}
                       for margin in MARGINS]
    constant_rule = max(constant_search, key=lambda r: (r["mean_teacher_net_value"], -r["query_rate"]))["rule"]
    model = {"schema": SCHEMA, "feature_schema": v3.FEATURE_SCHEMA,
             "mean_feature_names": list(v3.OBSERVABLE_NAMES),
             "risk_feature_names": [*v3.OBSERVABLE_NAMES, "cost_units"], "risk_target": RISK_TARGET,
             "mean_trees": [v3._tree_payload(tree) for tree in mean_fit.estimators_],
             "risk_trees": [v3._tree_payload(tree) for tree in risk_fit.estimators_],
             "calibration": {"x": calibrator.X_thresholds_.tolist(), "y": calibrator.y_thresholds_.tolist()},
             "mean_rule": search["mean_rule"], "risk_rule": search["risk_rule"],
             "validation_status": search["status"], "constant_gain": float(y.mean()),
             "split": split, "protocol_sha256": v3._digest(protocol), "provenance": PROVENANCE}
    portable = RiskAwareSyntheticGModel(model)
    (folder / "model.json").write_bytes(v3._canonical(model))
    (folder / "validation-selection.json").write_bytes(v3._canonical({**search, "constant_search": constant_search}))
    # Test targets are first consumed below, after fitting, calibration and rule freeze.
    predicted = portable.predict([r["observable_features"] for r in test], [r["cost_units"] for r in test])
    reference_mean = mean_fit.predict(_matrix(test))
    reference_risk = calibrator.predict(risk_fit.predict(_matrix(test, risk=True)))
    mean_error = float(np.max(np.abs(predicted["gain"] - reference_mean)))
    risk_error = float(np.max(np.abs(predicted["risk"] - reference_risk)))
    if mean_error > 1e-10 or risk_error > 1e-10:
        raise AssertionError("Portable inference differs from training inference")
    results = {
        "mean_only": evaluate(test, predicted["gain"], predicted["risk"], model["mean_rule"]),
        "risk_filtered": evaluate(test, predicted["gain"], predicted["risk"], model["risk_rule"]),
        "constant": evaluate(test, np.full(len(test), float(y.mean())), predicted["risk"], constant_rule),
        "stop_only": evaluate(test, np.zeros(len(test)), predicted["risk"], {"mode": "stop_only"}),
    }
    truth = np.asarray([r["teacher_expected_signed_gain"] for r in test])
    risk_truth = np.asarray([r["negative_expected_net"] for r in test])
    report = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": v3._digest(model), "protocol_sha256": v3._digest(protocol),
        "counts": {"matches": matches, "rows": len(rows), **{k + "_matches": len(v) for k, v in split.items()}},
        "validation_status": search["status"], "mean_rule": model["mean_rule"], "risk_rule": model["risk_rule"],
        "test": {key: result["summary"] for key, result in results.items()},
        "test_mean_mse": float(np.mean((predicted["gain"] - truth) ** 2)),
        "test_constant_mse": float(np.mean((float(y.mean()) - truth) ** 2)),
        "risk_calibration": {"raw": calibration_report(risk_truth, predicted["risk_raw"]),
                             "calibrated": calibration_report(risk_truth, predicted["risk"]),
                             "train_prevalence_baseline_brier": float(np.mean((float(labels.mean()) - risk_truth) ** 2))},
        "paired_risk_minus_mean": grouped_comparison(results["risk_filtered"], results["mean_only"], seed + 4),
        "paired_risk_minus_stop": grouped_comparison(results["risk_filtered"], results["stop_only"], seed + 5),
        "portable_inference_max_error": {"gain": mean_error, "risk": risk_error},
        "learner_hidden_fields": [], "network_requests": 0, "empirical_training_rows": 0,
        "formal_runtime_modified": False, "provenance": PROVENANCE,
        "limitations": ["Teacher labels and capabilities are constructed, not football observations",
                        "Risk is uncertainty over teacher expected net, not realized Y harm",
                        "Validation harm constraint is not a finite-sample guarantee",
                        "Same-state one-step policy; no learned multistep stopping claim",
                        "New-seed IID synthetic holdout does not establish domain transfer",
                        "No promotion of synthetic estimates into empirical UNKNOWN/selection gates"]}
    outputs = {"report.json": report, "test-decisions.json": results,
               "learner-dataset.json": {"schema": SCHEMA, "split": split, "rows": rows, "provenance": PROVENANCE},
               "teacher-audit-dataset.json": source}
    for name, value in outputs.items():
        (folder / name).write_bytes(v3._canonical(value))
    files = {p.name: {"sha256": sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in folder.iterdir() if p.is_file()}
    (folder / "manifest.json").write_bytes(v3._canonical({"schema": SCHEMA, "files": files}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--matches", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(build_artifact(args.output, matches=args.matches, seed=args.seed), indent=2))
