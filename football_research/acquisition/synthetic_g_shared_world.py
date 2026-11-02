"""Synthetic g v2: shared worlds, fixed-f gains, net decisions and STOP.

This is a research simulator and a portable value regressor, NOT a calibrated
football model. Every candidate in a state observes the same latent world.
Source returns affect the fixed predictor, never the world's match outcome.
Old v1 models, adapters and empirical selection authority are left untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "synthetic-g-shared-world-v2"
FEATURE_SCHEMA = "footballmind-g-shared-world-features-v2"
UPDATER = "shared-world-fixed-likelihood-update-v2"
ACTIONS = ("lineup", "injury", "motivation", "weather")
BITS = np.asarray(list(product((0., 1.), repeat=4)))
FEATURE_NAMES = (
    "p_home", "p_draw", "p_away", "normalized_entropy", "latent_probability",
    "effect_home", "effect_draw", "effect_away", "source_reliability",
    "freshness", "redundancy", "response_probability", "updater_trust",
    "time_to_kickoff_hours", *("action_" + a for a in ACTIONS),
)
PROVENANCE = dict(synthetic=True, evidence_origin="constructed_shared_world",
                  support_status="synthetic_research_only", production_ready=False,
                  calibration_status="not_established", empirical_selection_allowed=False)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _softmax(x):
    a = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return a / a.sum(axis=-1, keepdims=True)


def _unit(value, name, low=0., high=1.):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be finite in [{low},{high}]")
    return float(value)


def feature_vector(features):
    """Validate exact pre-query declarations; no labels/returns are accepted."""
    keys = {"schema", "probabilities", "action", "latent_probability", "effect_logits",
            "source_reliability", "freshness", "redundancy", "response_probability",
            "updater_trust", "time_to_kickoff_hours"}
    if not isinstance(features, dict) or set(features) != keys:
        raise ValueError("Complete explicit v2 pre-query features required, without extra fields")
    if features["schema"] != FEATURE_SCHEMA or features["action"] not in ACTIONS:
        raise ValueError("Unknown feature schema/action")
    p, effects = features["probabilities"], features["effect_logits"]
    if not isinstance(p, list) or len(p) != 3 or not isinstance(effects, list) or len(effects) != 3:
        raise ValueError("Three probabilities and three declared logit effects required")
    p = [_unit(v, "probability") for v in p]
    if not math.isclose(sum(p), 1., abs_tol=1e-8, rel_tol=0.):
        raise ValueError("Probabilities must sum to one")
    effects = [_unit(v, "effect", -2., 2.) for v in effects]
    values = [*p, -sum(v * math.log(v) for v in p if v) / math.log(3),
              _unit(features["latent_probability"], "latent_probability"), *effects]
    values += [_unit(features[n], n, .5 if n in {"source_reliability", "updater_trust"} else 0.)
               for n in ("source_reliability", "freshness", "redundancy",
                         "response_probability", "updater_trust")]
    values += [_unit(features["time_to_kickoff_hours"], "time_to_kickoff_hours", 0., 168.)]
    values += [float(features["action"] == a) for a in ACTIONS]
    return np.asarray(values)


def score_channel(weights, outcome_by_world, bit_index, *, actual_reliability,
                  updater_trust, response_probability):
    """Exact E[log f_after(Y)-log f_before(Y)] over one common joint law.

    The predictor trusts a declared signal likelihood. The environment may be
    less reliable. Thus gross value may be negative even before request cost.
    Failure/NO_DATA leaves f unchanged. Returned signal != future outcome.
    """
    w, q = np.asarray(weights, dtype=float), np.asarray(outcome_by_world, dtype=float)
    if (w.shape != (16,) or q.shape != (16, 3) or not np.isfinite(w).all()
            or not np.isfinite(q).all() or np.any(w < 0) or np.any(q <= 0)
            or not np.isclose(w.sum(), 1.) or not np.allclose(q.sum(axis=1), 1.)
            or bit_index not in range(4)):
        raise ValueError("A normalized shared 16-world law is required")
    r = _unit(actual_reliability, "actual_reliability")
    trust = _unit(updater_trust, "updater_trust", .5, .999999)
    response = _unit(response_probability, "response_probability")
    before = w @ q
    expected, second, harm_probability, joint_marginal = 0., 0., 0., np.zeros(3)
    branches = []
    for signal in (0., 1.):
        correct = BITS[:, bit_index] == signal
        # All actions share w and q. Only their observation channel changes.
        joint = (w * np.where(correct, r, 1. - r)) @ q
        assumed = w * np.where(correct, trust, 1. - trust)
        after = (assumed / assumed.sum()) @ q
        signed = np.log(after) - np.log(before)
        expected += float(joint @ signed)
        second += float(joint @ (signed ** 2))
        harm_probability += float(joint[signed < -1e-12].sum())
        joint_marginal += joint
        branches.append({"signal": int(signal), "signal_probability": float(joint.sum()),
                         "joint_outcome_probability": joint.tolist(), "f_after": after.tolist(),
                         "signed_gain_by_outcome": signed.tolist()})
    return {"expected_signed_gain": response * expected,
            "second_moment": response * second,
            "probability_of_negative_realized_gain": response * harm_probability,
            "p_before": before.tolist(), "branches": branches,
            "no_data_probability": 1. - response,
            "marginal_coherence_error": float(np.max(np.abs(joint_marginal - before)))}


def generate_dataset(*, seed=20261010, matches=600, states_per_match=3):
    """Generate teacher labels; publish only features to the learner.

    Effects and latent probabilities are declared mechanism priors. They are
    not observed football facts. Real integration must supply them before any
    candidate body is fetched, or abstain; this module invents no live values.
    """
    if matches < 30 or states_per_match < 1:
        raise ValueError("At least 30 synthetic matches and one state required")
    rng = np.random.default_rng(seed)
    rows, worlds = [], []
    for mid in range(matches):
        # A single outcome mechanism for all states/actions of this match.
        base = rng.dirichlet(np.full(3, 1.6))
        directions = np.array([[1., -.25, -.75], [-.8, .15, .65],
                               [-.45, .9, -.45], [-.35, .7, -.35]])
        magnitudes = rng.uniform(.01, 1.3, 4)
        magnitudes[rng.integers(4)] *= 1.4
        effects = np.clip(directions * magnitudes[:, None] * rng.choice([-1, 1], (4, 1)), -2, 2)
        q = _softmax(np.log(base)[None, :] + (2 * BITS - 1) @ effects)
        advertised = rng.uniform(.55, .98, 4)
        for sid in range(states_per_match):
            state_id = f"synthetic-v2:{mid:04d}:s{sid}"
            # Current visible state can already resolve some latent factors.
            u = rng.beta(.8, .8, 4)
            if sid:
                u[rng.integers(4)] = rng.choice([.001, .999])
            w = np.prod(np.where(BITS == 1, u, 1 - u), axis=1)
            w /= w.sum()
            before = (w @ q).tolist()
            world_id = _digest({"weights": w.tolist(), "outcomes": q.tolist()})
            worlds.append({"state_id": state_id, "world_sha256": world_id,
                           "weights": w.tolist(), "outcome_by_world": q.tolist()})
            freshness = rng.uniform(.05, 1., 4)
            redundancy = rng.choice([0., .1, .6, 1.], 4, p=[.50, .15, .20, .15])
            response_probability = rng.uniform(.45, 1., 4)
            trust = rng.uniform(.55, .98, 4)
            # Hypothetical source correctness, independent of query choice.
            actual = .5 + (advertised - .5) * freshness * (1 - redundancy)
            hours = float(rng.uniform(.05, 72.))
            for ai, action in enumerate(ACTIONS):
                features = {"schema": FEATURE_SCHEMA, "probabilities": before,
                    "action": action, "latent_probability": float(u[ai]),
                    "effect_logits": effects[ai].tolist(),
                    "source_reliability": float(advertised[ai]),
                    "freshness": float(freshness[ai]), "redundancy": float(redundancy[ai]),
                    "response_probability": float(response_probability[ai]),
                    "updater_trust": float(trust[ai]), "time_to_kickoff_hours": hours}
                feature_vector(features)
                score = score_channel(w, q, ai, actual_reliability=float(actual[ai]),
                    updater_trust=float(trust[ai]), response_probability=float(response_probability[ai]))
                cost = float(rng.uniform(.001, .025))
                rows.append({"physical_match_id": f"synthetic-v2:{mid:04d}",
                    "state_id": state_id, "action_id": f"query.{action}",
                    "world_sha256": world_id, "features": features,
                    "cost_units": cost, "cost_basis": "simulated_log_loss_penalty_not_dollars",
                    "teacher": {**score, "net_value": score["expected_signed_gain"] - cost},
                    "provenance": PROVENANCE})
    return {"schema": SCHEMA, "seed": seed, "matches": matches,
            "states_per_match": states_per_match, "rows": rows,
            "teacher_only_worlds": worlds, "provenance": PROVENANCE,
            "updater_version": UPDATER}


def group_split(rows, *, seed):
    ids = sorted({r["physical_match_id"] for r in rows})
    if len(ids) < 30:
        raise ValueError("At least 30 match groups for three-way synthetic evaluation")
    np.random.default_rng(seed).shuffle(ids)
    n = len(ids)
    return {"train": sorted(ids[:int(.6 * n)]),
            "validation": sorted(ids[int(.6 * n):int(.8 * n)]),
            "test": sorted(ids[int(.8 * n):])}


def _matrix(rows):
    return np.vstack([feature_vector(r["features"]) for r in rows])


def evaluate_decisions(rows, estimates, *, threshold=0.):
    """The same net ranking and STOP action are used in every comparison."""
    if len(rows) != len(estimates) or not np.isfinite(estimates).all():
        raise ValueError("A finite estimate for every row required")
    grouped = {}
    for row, estimate in zip(rows, estimates):
        grouped.setdefault(row["state_id"], []).append((row, float(estimate)))
    traces = []
    for sid, group in sorted(grouped.items()):
        best = min(group, key=lambda pair: (-(pair[1] - pair[0]["cost_units"]), pair[0]["action_id"]))
        selected = best[0] if best[1] - best[0]["cost_units"] > threshold else None
        oracle = min(group, key=lambda pair: (-pair[0]["teacher"]["net_value"], pair[0]["action_id"]))[0]
        if oracle["teacher"]["net_value"] <= 0:
            oracle = None
        actual = selected["teacher"]["net_value"] if selected else 0.
        optimal = oracle["teacher"]["net_value"] if oracle else 0.
        traces.append({"state_id": sid, "physical_match_id": group[0][0]["physical_match_id"],
            "selected": selected["action_id"] if selected else "STOP",
            "oracle": oracle["action_id"] if oracle else "STOP",
            "net_value": actual, "oracle_net_value": optimal, "regret": optimal - actual,
            "cost": selected["cost_units"] if selected else 0.,
            "queried": selected is not None, "oracle_queried": oracle is not None})
    mean = lambda xs: float(np.mean(list(xs)))
    oracle_stops = [x for x in traces if not x["oracle_queried"]]
    oracle_queries = [x for x in traces if x["oracle_queried"]]
    summary = {"states": len(traces), "matches": len({r["physical_match_id"] for r in rows}),
        "mse": mean((float(p) - r["teacher"]["expected_signed_gain"]) ** 2 for r, p in zip(rows, estimates)),
        "mean_teacher_net_value": mean(t["net_value"] for t in traces),
        "mean_oracle_net_value": mean(t["oracle_net_value"] for t in traces),
        "mean_regret": mean(t["regret"] for t in traces),
        "top1_including_stop": mean(t["selected"] == t["oracle"] for t in traces),
        "query_rate": mean(t["queried"] for t in traces),
        "mean_cost": mean(t["cost"] for t in traces),
        "harmful_query_rate": mean(t["net_value"] < 0 for t in traces),
        "oracle_stop_count": len(oracle_stops),
        "false_query_given_oracle_stop": mean(t["queried"] for t in oracle_stops) if oracle_stops else None,
        "missed_query_given_oracle_query": mean(not t["queried"] for t in oracle_queries) if oracle_queries else None}
    return {"summary": summary, "traces": traces}


def _tree_payload(estimator):
    tree = estimator.tree_
    return {"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist()}


def train_model(train, validation, *, seed):
    """Test rows cannot enter this API. Tune decisions on validation only."""
    from sklearn.ensemble import ExtraTreesRegressor
    tids = {r["physical_match_id"] for r in train}
    vids = {r["physical_match_id"] for r in validation}
    if not train or not validation or tids & vids:
        raise ValueError("Disjoint nonempty match groups required")
    x, xv = _matrix(train), _matrix(validation)
    y = np.array([r["teacher"]["expected_signed_gain"] for r in train])
    choices, fits = [], {}
    # Frozen small search space. No further search based on test performance.
    for leaf in (3, 8):
        fitted = ExtraTreesRegressor(n_estimators=96, max_depth=14, min_samples_leaf=leaf,
                                    max_features=1., random_state=seed, n_jobs=2).fit(x, y)
        fits[leaf] = fitted
        pred = fitted.predict(xv)
        for threshold in (0., .002, .005, .01):
            metrics = evaluate_decisions(validation, pred, threshold=threshold)["summary"]
            choices.append({"min_samples_leaf": leaf, "stop_margin": threshold, **metrics})
    winner = min(choices, key=lambda v: (v["mean_regret"], v["mse"], v["stop_margin"], v["min_samples_leaf"]))
    fitted = fits[winner["min_samples_leaf"]]
    model = {"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES), "updater_version": UPDATER,
        "target": "expected_signed_fixed_f_log_loss_reduction",
        "selection_rule": "argmax(gross_minus_cost, STOP_zero)_with_validation_stop_margin",
        "learner": "ExtraTreesRegressor_portable_json", "seed": seed,
        "cost_in_gross_regression": False,
        "stop_margin": winner["stop_margin"], "selected_min_samples_leaf": winner["min_samples_leaf"],
        "train_match_ids": sorted(tids), "validation_match_ids": sorted(vids),
        "validation_search": choices, "validation_winner": winner,
        "trees": [_tree_payload(t) for t in fitted.estimators_],
        "constant_baseline": float(y.mean()), "provenance": PROVENANCE,
        "usage": "shadow_only_no_transport_or_empirical_authority"}
    return model, fitted


class SharedWorldGModel:
    """Portable research g. No sklearn/pickle required at prediction time."""
    def __init__(self, payload):
        if (payload.get("schema") != SCHEMA or payload.get("feature_schema") != FEATURE_SCHEMA
                or payload.get("feature_names") != list(FEATURE_NAMES)
                or payload.get("provenance") != PROVENANCE):
            raise ValueError("An explicitly synthetic shared-world v2 model is required")
        self.payload = deepcopy(payload)
        trees = self.payload.get("trees")
        if not isinstance(trees, list) or not trees:
            raise ValueError("Nonempty portable forest required")
        for t in trees:
            n = len(t.get("left", []))
            if not n or set(t) != {"left", "right", "feature", "threshold", "value"} or any(len(v) != n for v in t.values()):
                raise ValueError("Invalid portable tree")
            for i in range(n):
                if not math.isfinite(t["value"][i]) or not math.isfinite(t["threshold"][i]):
                    raise ValueError("Nonfinite portable tree")
                l, r, f = t["left"][i], t["right"][i], t["feature"][i]
                if l == r == -1:
                    continue
                if not (i < l < n and i < r < n and 0 <= f < len(FEATURE_NAMES)):
                    raise ValueError("Invalid/cyclic tree child reference")
        self.model_sha256 = _digest(self.payload)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def predict(self, features):
        x = np.asarray([feature_vector(f) for f in features], dtype=np.float32)
        if len(x) == 0:
            return np.array([], dtype=float)
        result = np.zeros(len(x))
        for tree in self.payload["trees"]:
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
        return result / len(self.payload["trees"])

    def estimate(self, features, *, cost_units):
        if isinstance(cost_units, bool) or not isinstance(cost_units, (int, float)) or not math.isfinite(cost_units) or cost_units < 0:
            raise ValueError("Explicit finite nonnegative loss-unit cost required")
        gain = float(self.predict([features])[0])
        return {"model_schema": SCHEMA, "model_sha256": self.model_sha256,
                "expected_signed_gain": gain, "expected_net_voi": gain - cost_units,
                "cost_units": float(cost_units), **PROVENANCE}

    def rank_shadow(self, candidates):
        if not candidates or len({c["action_id"] for c in candidates}) != len(candidates):
            raise ValueError("Nonempty menu with unique action_ids required")
        rows = [{"action_id": c["action_id"], **self.estimate(c["features"], cost_units=c["cost_units"])}
                for c in candidates]
        rows.sort(key=lambda r: (-r["expected_net_voi"], r["action_id"]))
        candidate = rows[0]
        choice = candidate["action_id"] if candidate["expected_net_voi"] > self.payload["stop_margin"] else "STOP"
        return {"schema": "synthetic-g-shared-world-shadow-v2", "candidates": rows,
                "shadow_proposal": choice, "stop_margin": self.payload["stop_margin"],
                "selection_allowed": False, "request_executed": False, **PROVENANCE}


def build_artifact(output, *, seed=20261010, matches=600, states_per_match=3):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a new empty artifact directory; frozen outputs are never overwritten")
    dataset = generate_dataset(seed=seed, matches=matches, states_per_match=states_per_match)
    split = group_split(dataset["rows"], seed=seed + 1)
    partitions = {k: [r for r in dataset["rows"] if r["physical_match_id"] in set(v)] for k, v in split.items()}
    protocol = {"schema": SCHEMA, "seed": seed, "split": split,
        "training_target": "exact_expected_signed_log_loss_gain_under_shared_synthetic_law",
        "learner_candidates": {"trees": 96, "depth": 14, "min_samples_leaf": [3, 8]},
        "stop_margin_candidates": [0., .002, .005, .01],
        "primary_selection_metric": "validation_mean_net_decision_regret_including_STOP",
        "test_use": "evaluate_once_after_model_and_margin_frozen",
        "provenance": PROVENANCE}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "protocol.json").write_bytes(_canonical(protocol))
    model, fitted = train_model(partitions["train"], partitions["validation"], seed=seed + 2)
    (folder / "model.json").write_bytes(_canonical(model))
    # Test is accessed only after model bytes are committed locally.
    test = partitions["test"]
    prediction = fitted.predict(_matrix(test))
    evaluated = evaluate_decisions(test, prediction, threshold=model["stop_margin"])
    # Every policy comparison must use the same frozen stop margin.  Passing
    # the default zero threshold here would let the constant baseline query
    # under a different stopping rule than the learned policy, making the
    # regret comparison impossible to interpret.
    baseline = evaluate_decisions(test, np.full(len(test), model["constant_baseline"]),
                                  threshold=model["stop_margin"])
    # STOP-only is a relevant baseline when requests can worsen the forecast.
    stop = evaluate_decisions(test, np.zeros(len(test)), threshold=model["stop_margin"])
    portable = SharedWorldGModel.load(folder / "model.json")
    portable_prediction = portable.predict([r["features"] for r in test])
    if not np.allclose(portable_prediction, prediction, rtol=1e-10, atol=1e-12):
        raise AssertionError("Portable and training inference differ")
    per_match_delta = []
    for mid in split["test"]:
        current = [t["net_value"] for t in evaluated["traces"] if t["physical_match_id"] == mid]
        constant = [t["net_value"] for t in baseline["traces"] if t["physical_match_id"] == mid]
        per_match_delta.append(float(np.mean(current) - np.mean(constant)))
    rng = np.random.default_rng(seed + 3)
    boots = np.mean(rng.choice(per_match_delta, size=(2000, len(per_match_delta)), replace=True), axis=1)
    report = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": _digest(model), "protocol_sha256": _digest(protocol),
        "dataset_sha256": _digest(dataset), "counts": {"matches": matches, "rows": len(dataset["rows"]),
            **{k + "_matches": len(v) for k, v in split.items()}},
        "test": evaluated["summary"], "test_constant": baseline["summary"], "test_stop_only": stop["summary"],
        "net_improvement_over_constant": float(np.mean(per_match_delta)),
        "group_bootstrap_95pct_net_improvement": np.quantile(boots, [.025, .975]).tolist(),
        "bootstrap_scope": "synthetic_physical_match_groups_only_not_real_world_confidence",
        "negative_teacher_gross_rows": sum(r["teacher"]["expected_signed_gain"] < -1e-12 for r in dataset["rows"]),
        "max_marginal_coherence_error": max(r["teacher"]["marginal_coherence_error"] for r in dataset["rows"]),
        "portable_inference_max_error": float(np.max(np.abs(portable_prediction - prediction))),
        "formal_runtime_modified": False, "network_requests": 0, "provenance": PROVENANCE,
        "limitations": ["Not trained or calibrated on real football gains",
            "Latent uncertainty and effect vectors are caller-declared mechanism priors",
            "One-step state decisions; no claim of optimal multi-step policy",
            "Real pre-query feature provider and fixed-f contract must match before integration",
            "Existing empirical UNKNOWN/constant gate is unchanged"]}
    for name, content in (("dataset.json", dataset), ("report.json", report),
                          ("test-decisions.json", evaluated), ("test-constant-decisions.json", baseline)):
        (folder / name).write_bytes(_canonical(content))
    files = {p.name: {"sha256": sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in folder.iterdir() if p.is_file()}
    (folder / "manifest.json").write_bytes(_canonical({"schema": SCHEMA, "files": files}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--matches", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    print(json.dumps(build_artifact(args.output, matches=args.matches, seed=args.seed), indent=2))
