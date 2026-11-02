"""Observable-only synthetic g v3.

This is a research ablation over the shared-world v2 teacher.  The teacher
still uses the constructed latent world to produce labels, but the learner is
allowed to see only explicit pre-query declarations that could be supplied by
an online caller.  Latent probability and effect logits are intentionally
excluded from the learner feature vector.  This artifact never changes the
empirical UNKNOWN gate or formal runtime.
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

from . import synthetic_g_shared_world as v2

SCHEMA = "synthetic-g-observable-only-v3"
FEATURE_SCHEMA = "footballmind-g-observable-features-v3"
OBSERVABLE_NAMES = (
    "p_home", "p_draw", "p_away", "normalized_entropy",
    "source_reliability", "freshness", "redundancy", "response_probability",
    "updater_trust", "time_to_kickoff_hours",
    *("action_" + action for action in v2.ACTIONS),
)
PROVENANCE = {
    "synthetic": True,
    "evidence_origin": "constructed_shared_world_teacher_observable_learner",
    "support_status": "synthetic_research_only",
    "production_ready": False,
    "calibration_status": "not_established",
    "empirical_selection_allowed": False,
    "hidden_teacher_features_excluded": ["latent_probability", "effect_logits"],
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _unit(value, name, low=0., high=1.):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be finite in [{low},{high}]")
    return float(value)


def make_observable_features(full_features):
    """Project v2 rows without reading hidden mechanism fields."""
    if not isinstance(full_features, dict):
        raise ValueError("Explicit v2 feature mapping required")
    required = {"probabilities", "action", "source_reliability", "freshness",
                "redundancy", "response_probability", "updater_trust",
                "time_to_kickoff_hours"}
    if not required <= set(full_features):
        raise ValueError("Missing observable pre-query declarations")
    p = full_features["probabilities"]
    if not isinstance(p, list) or len(p) != 3:
        raise ValueError("Three public probabilities are required")
    p = [_unit(value, "probability") for value in p]
    if not math.isclose(sum(p), 1., abs_tol=1e-8, rel_tol=0.):
        raise ValueError("Probabilities must sum to one")
    action = full_features["action"]
    if action not in v2.ACTIONS:
        raise ValueError("Unknown action")
    entropy = -sum(value * math.log(value) for value in p if value) / math.log(3)
    values = [*p, entropy]
    values += [_unit(full_features[name], name, .5 if name == "updater_trust" else 0.)
               for name in ("source_reliability", "freshness", "redundancy",
                            "response_probability", "updater_trust")]
    values.append(_unit(full_features["time_to_kickoff_hours"],
                        "time_to_kickoff_hours", 0., 168.))
    values += [float(action == candidate) for candidate in v2.ACTIONS]
    return {"schema": FEATURE_SCHEMA, "values": values,
            "action": action, "probabilities": p}


def feature_vector(features):
    if (not isinstance(features, dict) or features.get("schema") != FEATURE_SCHEMA
            or set(features) != {"schema", "values", "action", "probabilities"}):
        raise ValueError("Exact observable-only feature contract required")
    values = features["values"]
    if not isinstance(values, list) or len(values) != len(OBSERVABLE_NAMES):
        raise ValueError("Observable vector has the wrong length")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) for value in values):
        raise ValueError("Observable vector must be finite")
    if features["action"] not in v2.ACTIONS:
        raise ValueError("Unknown action")
    p = features["probabilities"]
    if (not isinstance(p, list) or len(p) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0. <= value <= 1. for value in p)
            or not math.isclose(sum(p), 1., abs_tol=1e-8, rel_tol=0.)):
        raise ValueError("Invalid public probabilities")
    if not np.allclose(values[:3], p, rtol=0., atol=1e-8):
        raise ValueError("Observable vector probabilities must match public probabilities")
    expected_entropy = -sum(value * math.log(value) for value in p if value) / math.log(3)
    if not 0. <= values[3] <= 1. or not math.isclose(values[3], expected_entropy,
                                                         rel_tol=0., abs_tol=1e-8):
        raise ValueError("normalized_entropy must be in [0,1]")
    if any(not 0. <= values[index] <= 1. for index in range(4, 9)):
        raise ValueError("observable reliability fields must be in [0,1]")
    if not .5 <= values[8] <= 1.:
        raise ValueError("updater_trust must be in [0.5,1]")
    if not 0. <= values[9] <= 168.:
        raise ValueError("time_to_kickoff_hours must be in [0,168]")
    if not all(value in (0., 1.) for value in values[10:]):
        raise ValueError("action indicator fields must be binary")
    if sum(values[10:]) != 1.:
        raise ValueError("exactly one action indicator is required")
    action_index = v2.ACTIONS.index(features["action"])
    if values[10 + action_index] != 1.:
        raise ValueError("action indicator does not match declared action")
    return np.asarray(values, dtype=float)


def _matrix(rows):
    return np.vstack([feature_vector(row["observable_features"]) for row in rows])


def _tree_payload(estimator):
    tree = estimator.tree_
    return {"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist()}


def _portable_predict(payload, features):
    x = np.asarray([feature_vector(f) for f in features], dtype=np.float32)
    if len(x) == 0:
        return np.array([], dtype=float)
    result = np.zeros(len(x))
    for tree in payload["trees"]:
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
    return result / len(payload["trees"])


def _conditional_metrics(evaluation):
    traces = evaluation["traces"]
    queried = [trace for trace in traces if trace["queried"]]
    positive_oracle = [trace for trace in traces if trace["oracle_queried"]]
    return {
        "harmful_query_given_query": float(np.mean([trace["net_value"] < 0 for trace in queried])) if queried else None,
        "positive_net_precision": float(np.mean([trace["net_value"] > 0 for trace in queried])) if queried else None,
        "oracle_query_recall": float(np.mean([trace["queried"] for trace in positive_oracle])) if positive_oracle else None,
        "queried_states": len(queried),
        "oracle_query_states": len(positive_oracle),
    }


class ObservableOnlyGModel:
    """Portable diagnostic g whose input is the observable v3 contract only."""

    def __init__(self, payload):
        if (payload.get("schema") != SCHEMA
                or payload.get("feature_schema") != FEATURE_SCHEMA
                or payload.get("feature_names") != list(OBSERVABLE_NAMES)
                or payload.get("provenance") != PROVENANCE):
            raise ValueError("An explicitly observable-only synthetic v3 model is required")
        trees = payload.get("trees")
        if not isinstance(trees, list) or not trees:
            raise ValueError("Nonempty portable forest required")
        margin = payload.get("stop_margin")
        if (isinstance(margin, bool) or not isinstance(margin, (int, float))
                or not math.isfinite(margin) or margin < 0.):
            raise ValueError("stop_margin must be finite and nonnegative")
        for tree in trees:
            n = len(tree.get("left", []))
            if (not n or set(tree) != {"left", "right", "feature", "threshold", "value"}
                    or any(len(values) != n for values in tree.values())):
                raise ValueError("Invalid portable tree")
            for index in range(n):
                if not math.isfinite(tree["threshold"][index]) or not math.isfinite(tree["value"][index]):
                    raise ValueError("Nonfinite portable tree")
                left, right, feature = (tree["left"][index], tree["right"][index],
                                        tree["feature"][index])
                if any(isinstance(value, bool) or not isinstance(value, int)
                       for value in (left, right, feature)):
                    raise ValueError("Portable tree child and feature indexes must be integers")
                if left == right == -1:
                    continue
                if not (index < left < n and index < right < n
                        and 0 <= feature < len(OBSERVABLE_NAMES)):
                    raise ValueError("Invalid/cyclic portable tree")
        self.payload = deepcopy(payload)
        self.model_sha256 = _digest(self.payload)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, features):
        return _portable_predict(self.payload, features)

    def estimate(self, features, *, cost_units):
        if (isinstance(cost_units, bool) or not isinstance(cost_units, (int, float))
                or not math.isfinite(cost_units) or cost_units < 0.):
            raise ValueError("Explicit finite nonnegative loss-unit cost required")
        gain = float(self.predict([features])[0])
        return {"model_schema": SCHEMA, "model_sha256": self.model_sha256,
                "expected_signed_gain": gain, "expected_net_voi": gain - float(cost_units),
                "cost_units": float(cost_units), **PROVENANCE}

    def rank_shadow(self, candidates):
        if (not candidates
                or any(not isinstance(candidate, dict) for candidate in candidates)
                or len({candidate.get("action_id") for candidate in candidates}) != len(candidates)):
            raise ValueError("Nonempty menu with unique action_ids required")
        for candidate in candidates:
            if not isinstance(candidate.get("action_id"), str) or not candidate["action_id"]:
                raise ValueError("Every candidate needs a nonempty action_id")
            if "features" not in candidate or "cost_units" not in candidate:
                raise ValueError("Every candidate needs observable features and cost_units")
        rows = [{"action_id": candidate["action_id"],
                 **self.estimate(candidate["features"], cost_units=candidate["cost_units"])}
                for candidate in candidates]
        rows.sort(key=lambda row: (-row["expected_net_voi"], row["action_id"]))
        winner = rows[0]
        choice = (winner["action_id"]
                  if winner["expected_net_voi"] > self.payload["stop_margin"] else "STOP")
        return {"schema": "synthetic-g-observable-only-shadow-v3", "candidates": rows,
                "shadow_proposal": choice, "stop_margin": self.payload["stop_margin"],
                "selection_allowed": False, "request_executed": False, **PROVENANCE}


def build_artifact(output, *, seed=20261010, matches=600, states_per_match=3):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a new empty artifact directory; frozen outputs are never overwritten")
    dataset = v2.generate_dataset(seed=seed, matches=matches, states_per_match=states_per_match)
    rows = []
    for row in dataset["rows"]:
        item = deepcopy(row)
        item["observable_features"] = make_observable_features(row["features"])
        rows.append(item)
    split = v2.group_split(rows, seed=seed + 1)
    partitions = {key: [row for row in rows if row["physical_match_id"] in set(ids)]
                  for key, ids in split.items()}
    protocol = {
        "schema": SCHEMA,
        "seed": seed,
        "split": split,
        "learner_feature_schema": FEATURE_SCHEMA,
        "excluded_hidden_teacher_features": ["latent_probability", "effect_logits"],
        "teacher_schema": v2.SCHEMA,
        "teacher_target": "expected_signed_fixed_f_log_loss_reduction_before_cost",
        "learner": {"type": "ExtraTreesRegressor", "n_estimators": 96,
                     "max_depth": 14, "min_samples_leaf": [3, 8],
                     "max_features": 1.0, "random_state": seed + 2},
        "updater": "shared-world-fixed-likelihood-update-v2",
        "learner_dataset": {
            "file": "learner-dataset.json",
            "hidden_teacher_fields_in_learner_rows": False,
            "labels_are_synthetic_teacher_targets": True,
        },
        "selection_metric": "validation_mean_net_decision_regret_including_STOP",
        "stop_margin_candidates": [0., .002, .005, .01],
        "provenance": PROVENANCE,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "protocol.json").write_bytes(_canonical(protocol))
    from sklearn.ensemble import ExtraTreesRegressor
    train, validation, test = partitions["train"], partitions["validation"], partitions["test"]
    x, xv = _matrix(train), _matrix(validation)
    y = np.asarray([row["teacher"]["expected_signed_gain"] for row in train])
    choices, fits = [], {}
    for leaf in (3, 8):
        fitted = ExtraTreesRegressor(n_estimators=96, max_depth=14, min_samples_leaf=leaf,
                                     max_features=1., random_state=seed + 2, n_jobs=2).fit(x, y)
        fits[leaf] = fitted
        validation_pred = fitted.predict(xv)
        for threshold in (0., .002, .005, .01):
            validation_evaluation = v2.evaluate_decisions(
                validation, validation_pred, threshold=threshold)
            metrics = {**validation_evaluation["summary"],
                       **_conditional_metrics(validation_evaluation)}
            choices.append({"min_samples_leaf": leaf, "stop_margin": threshold, **metrics})
    winner = min(choices, key=lambda choice: (choice["mean_regret"], choice["mse"],
                                               choice["stop_margin"], choice["min_samples_leaf"]))
    fitted = fits[winner["min_samples_leaf"]]
    model = {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(OBSERVABLE_NAMES),
        "target": "expected_signed_fixed_f_log_loss_reduction_before_cost",
        "learner": "ExtraTreesRegressor_portable_json",
        "seed": seed,
        "stop_margin": winner["stop_margin"],
        "selected_min_samples_leaf": winner["min_samples_leaf"],
        "train_match_ids": sorted({row["physical_match_id"] for row in train}),
        "validation_match_ids": sorted({row["physical_match_id"] for row in validation}),
        "test_match_ids": sorted({row["physical_match_id"] for row in test}),
        "validation_search": choices,
        "validation_winner": winner,
        "trees": [_tree_payload(tree) for tree in fitted.estimators_],
        "constant_baseline": float(y.mean()),
        "provenance": PROVENANCE,
        "usage": "shadow_only_no_transport_or_empirical_authority",
    }
    (folder / "model.json").write_bytes(_canonical(model))
    test_features = [row["observable_features"] for row in test]
    prediction = fitted.predict(_matrix(test))
    evaluated = v2.evaluate_decisions(test, prediction, threshold=model["stop_margin"])
    baseline = v2.evaluate_decisions(test, np.full(len(test), model["constant_baseline"]),
                                     threshold=model["stop_margin"])
    stop = v2.evaluate_decisions(test, np.zeros(len(test)), threshold=model["stop_margin"])
    risk_frontier = []
    for threshold in (0., .002, .005, .01):
        frontier_evaluation = v2.evaluate_decisions(test, prediction, threshold=threshold)
        risk_frontier.append({"stop_margin": threshold,
                              **frontier_evaluation["summary"],
                              **_conditional_metrics(frontier_evaluation)})
    portable_prediction = _portable_predict(model, test_features)
    if not np.allclose(portable_prediction, prediction, rtol=1e-10, atol=1e-12):
        raise AssertionError("Portable and training inference differ")
    per_match_delta = []
    for match_id in split["test"]:
        learned = [trace["net_value"] for trace in evaluated["traces"] if trace["physical_match_id"] == match_id]
        constant = [trace["net_value"] for trace in baseline["traces"] if trace["physical_match_id"] == match_id]
        per_match_delta.append(float(np.mean(learned) - np.mean(constant)))
    rng = np.random.default_rng(seed + 3)
    bootstrap = np.mean(rng.choice(per_match_delta, size=(2000, len(per_match_delta)), replace=True), axis=1)
    errors = prediction - np.asarray([row["teacher"]["expected_signed_gain"] for row in test])
    dataset_payload = {**dataset, "rows": rows}
    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": _digest(model),
        "protocol_sha256": _digest(protocol),
        "dataset_sha256": _digest(dataset_payload),
        "teacher_dataset_sha256": _digest(dataset),
        "counts": {"matches": matches, "rows": len(rows),
                    **{key + "_matches": len(ids) for key, ids in split.items()}},
        "test": {**evaluated["summary"], **_conditional_metrics(evaluated),
                 "mae": float(np.mean(np.abs(errors))),
                 "median_absolute_error": float(np.median(np.abs(errors)))},
        "test_risk_frontier": risk_frontier,
        "test_constant": baseline["summary"],
        "test_stop_only": stop["summary"],
        "net_improvement_over_constant": float(np.mean(per_match_delta)),
        "group_bootstrap_95pct_net_improvement": np.quantile(bootstrap, [.025, .975]).tolist(),
        "bootstrap_scope": "synthetic_physical_match_groups_only_not_real_world_confidence",
        "negative_teacher_gross_rows": sum(row["teacher"]["expected_signed_gain"] < -1e-12 for row in rows),
        "max_marginal_coherence_error": max(row["teacher"]["marginal_coherence_error"] for row in rows),
        "portable_inference_max_error": float(np.max(np.abs(portable_prediction - prediction))),
        "hidden_teacher_features_excluded_from_learner": ["latent_probability", "effect_logits"],
        "formal_runtime_modified": False,
        "network_requests": 0,
        "provenance": PROVENANCE,
        "limitations": [
            "Teacher labels still come from a constructed shared-world mechanism",
            "Observable declarations are not independently verified football facts",
            "No empirical calibration or production selection authority",
            "One-step state decisions; no claim of optimal multi-step policy",
            "The synthetic prior cannot be used to replace empirical UNKNOWN gating",
        ],
    }
    learner_payload = {
        "schema": "synthetic-g-observable-only-learner-dataset-v3",
        "feature_schema": FEATURE_SCHEMA,
        "seed": seed,
        "split": split,
        "rows": [{
            "physical_match_id": row["physical_match_id"],
            "state_id": row["state_id"],
            "action_id": row["action_id"],
            "observable_features": row["observable_features"],
            "cost_units": row["cost_units"],
            "teacher_expected_signed_gain": row["teacher"]["expected_signed_gain"],
            "teacher_net_value": row["teacher"]["net_value"],
        } for row in rows],
        "provenance": {
            **PROVENANCE,
            "hidden_teacher_fields_included": False,
            "labels_are_synthetic_teacher_targets": True,
        },
    }
    for name, content in (("dataset.json", dataset_payload),
                          ("learner-dataset.json", learner_payload),
                          ("report.json", report), ("test-decisions.json", evaluated),
                          ("test-constant-decisions.json", baseline)):
        (folder / name).write_bytes(_canonical(content))
    files = {path.name: {"sha256": sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
             for path in folder.iterdir() if path.is_file()}
    (folder / "manifest.json").write_bytes(_canonical({"schema": SCHEMA, "files": files}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--matches", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(json.dumps(build_artifact(args.output, matches=args.matches, seed=args.seed), indent=2))
