"""Small-n Bayesian shadow for prospective frozen-f quality targets.

This module is deliberately separate from :mod:`prospective_quality_g` and
from the live acquisition policy.  It consumes only already admitted,
raw-bound gross log-loss targets.  A weak synthetic teacher may define a wide
coefficient prior, but it is never treated as an observed label and never
changes the empirical UNKNOWN/constant gate.

The estimator is a conjugate Bayesian linear regression over a deterministic
hashed pre-query representation.  Group identities are physical matches;
groups, rather than action rows, are the unit used for train/calibration/
validation/test separation.  The public API always reports
``selection_allowed=False`` and ``production_ready=False``.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .prospective_quality_g import _check_query

VERSION = "prospective-frozen-f-quality-g-bayesian-shadow-v1"
QUERY_VERSION = "prospective-frozen-f-quality-query-v1"
TARGET = "conditional_expected_gross_log_loss_reduction"
FEATURE_SCHEMA = "footballmind-g-bayesian-shadow-features-v1"
PRIOR_SCHEMA = "synthetic-g-bayesian-wide-prior-v1"
SPLIT_SCHEMA = "physical-match-group-split-v1"
MIN_PRODUCTION_MATCHES = 12
MIN_PRODUCTION_DATES = 2
DEFAULT_PRIOR_STD = 10.0
DEFAULT_NOISE_STD = 0.25
DEFAULT_LCB_Z = 1.645
HASH_BUCKETS = 48


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _hash_text(values, prefix):
    result = np.zeros(HASH_BUCKETS, dtype=float)
    for text in values:
        if not isinstance(text, str):
            raise ValueError("Query text values must be strings")
        # The query contract already removed future response data. Hashing is
        # fixed and does not fit a vocabulary on validation/test groups.
        tokens = text.lower().split()
        for token in tokens:
            digest = sha256((prefix + ":" + token).encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % HASH_BUCKETS
            result[index] += 1. if digest[8] & 1 else -1.
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def query_features(query):
    """Build a fixed pre-query vector without reading outcomes or responses."""
    if not isinstance(query, dict):
        raise ValueError("A frozen pre-query mapping is required")
    _check_query(query)
    context = query["context"]
    numeric = context.get("numeric")
    if not isinstance(numeric, list) or not numeric:
        raise ValueError("The pre-query numeric vector is missing")
    numeric = np.asarray([_finite(x, "numeric") for x in numeric], dtype=float)
    if np.any(numeric < 0.) or np.any(numeric > 1.):
        raise ValueError("Pre-query numeric values must be bounded")
    texts = context.get("texts", [])
    labels = context.get("labels", [])
    if not isinstance(texts, list) or not isinstance(labels, list):
        raise ValueError("Pre-query text projections must be lists")
    vector = np.concatenate(([1.], numeric, _hash_text(texts, "text"),
                             _hash_text(labels, "label")))
    if not np.all(np.isfinite(vector)):
        raise ValueError("Nonfinite Bayesian feature vector")
    return vector


def _target_row(row):
    if not isinstance(row, dict):
        raise ValueError("A target row must be an object")
    if row.get("target_status") != "eligible":
        raise ValueError("Only admitted real gross-gain rows may be used")
    if row.get("synthetic") is True or row.get("evidence_origin") not in {None, "prospective_capture"}:
        raise ValueError("Synthetic or non-prospective rows cannot become real Bayesian labels")
    physical = row.get("physical_match_id")
    if not isinstance(physical, str) or not physical:
        raise ValueError("A canonical physical_match_id is required")
    gain = _finite(row.get("gain"), "gain")
    query = row.get("query")
    vector = query_features(query)
    date = (query.get("context") or {}).get("date")
    if not isinstance(date, str) or not date:
        raise ValueError("A UTC decision date is required")
    return {"sample_id": row.get("sample_id"), "physical_match_id": physical,
            "date": date, "gain": gain, "query": deepcopy(query), "vector": vector}


def validate_group_partitions(rows):
    """Validate a four-way physical-match split without repairing it."""
    allowed = {"train", "calibration", "validation", "test"}
    groups = {}
    clean = []
    for raw in rows:
        item = _target_row(raw)
        partition = raw.get("partition")
        if partition not in allowed:
            raise ValueError("Bayesian shadow rows require an explicit four-way group partition")
        previous = groups.setdefault(item["physical_match_id"], partition)
        if previous != partition:
            raise ValueError("A physical match cannot cross Bayesian partitions")
        item["partition"] = partition
        clean.append(item)
    return clean, groups


def group_split(rows, *, seed=20261010):
    """Create a deterministic group split only for unpartitioned raw rows.

    This helper is a research convenience.  It refuses rows that already carry
    train/holdout assignments, so a pre-registered split cannot be silently
    replaced by a new random split.
    """
    if any(row.get("partition") is not None for row in rows):
        raise ValueError("Use the pre-registered four-way split; do not overwrite it")
    clean = [_target_row(row) for row in rows]
    ids = sorted({row["physical_match_id"] for row in clean})
    if not ids:
        raise ValueError("At least one admitted physical match is required")
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    # Small-n shadow allows a diagnostic posterior with sparse groups; this
    # split never implies production readiness.
    n = len(ids)
    cuts = (0, max(1, int(round(.50 * n))), max(1, int(round(.70 * n))),
            max(1, int(round(.85 * n))), n)
    cuts = tuple(sorted(min(n, value) for value in cuts))
    names = ("train", "calibration", "validation", "test")
    groups = {name: set(ids[cuts[i]:cuts[i + 1]]) for i, name in enumerate(names)}
    # Ensure a nonempty group gets at least one id where possible. With fewer
    # than four groups the report remains explicitly non-production.
    for index, physical in enumerate(ids):
        if not any(physical in values for values in groups.values()):
            groups[names[min(index, len(names) - 1)]].add(physical)
    return [{**item, "partition": next(name for name, values in groups.items()
                                      if item["physical_match_id"] in values)} for item in clean], groups


class BayesianQualityGShadow:
    """Conjugate small-n posterior; diagnostic ranking only."""

    def __init__(self, *, prior_mean=None, prior_std=DEFAULT_PRIOR_STD,
                 noise_std=DEFAULT_NOISE_STD, lcb_z=DEFAULT_LCB_Z,
                 synthetic_prior=None):
        self.prior_mean = None if prior_mean is None else np.asarray(prior_mean, dtype=float)
        self.prior_std = _finite(prior_std, "prior_std")
        self.noise_std = _finite(noise_std, "noise_std")
        self.lcb_z = _finite(lcb_z, "lcb_z")
        if self.prior_std <= 0 or self.noise_std <= 0 or self.lcb_z < 0:
            raise ValueError("Bayesian scales must be positive and lcb_z nonnegative")
        if synthetic_prior is not None:
            if not isinstance(synthetic_prior, dict) or synthetic_prior.get("schema") != PRIOR_SCHEMA:
                raise ValueError("Only the explicit wide synthetic prior schema is accepted")
            if synthetic_prior.get("production_ready") is not False:
                raise ValueError("Synthetic prior must remain non-production")
            values = synthetic_prior.get("prior_mean")
            if not isinstance(values, list):
                raise ValueError("Synthetic prior mean must be explicit")
            self.prior_mean = np.asarray(values, dtype=float)
        if self.prior_mean is not None and (self.prior_mean.ndim != 1
                or not np.all(np.isfinite(self.prior_mean))):
            raise ValueError("Prior mean must be a finite vector")
        self.synthetic_prior = deepcopy(synthetic_prior)
        self.mean = None
        self.covariance = None
        self.metadata = {}
        self.model_version = ""
        self._train_vectors = None
        self._train_targets = None

    def _fit_posterior(self, vectors, targets):
        x = np.asarray(vectors, dtype=float)
        y = np.asarray(targets, dtype=float)
        dimension = x.shape[1]
        prior = np.zeros(dimension) if self.prior_mean is None else self.prior_mean
        if len(prior) != dimension:
            raise ValueError("Synthetic prior dimension differs from frozen query features")
        precision = (x.T @ x) / (self.noise_std ** 2)
        precision += np.eye(dimension) / (self.prior_std ** 2)
        covariance = np.linalg.pinv(precision, hermitian=True)
        mean = covariance @ ((x.T @ y) / (self.noise_std ** 2)
                             + prior / (self.prior_std ** 2))
        self.mean, self.covariance = mean, covariance

    def fit(self, rows):
        clean, groups = validate_group_partitions(rows)
        train = [row for row in clean if row["partition"] == "train"]
        if not train:
            raise ValueError("At least one admitted training group is required")
        if len({row["sample_id"] for row in train}) != len(train):
            raise ValueError("Duplicate Bayesian shadow sample identity")
        pipelines = {row["query"].get("prediction_pipeline_version") for row in train}
        if len(pipelines) != 1 or None in pipelines:
            raise ValueError("Bayesian shadow training rows must share one f pipeline")
        self._train_vectors = np.asarray([row["vector"] for row in train], dtype=float)
        self._train_targets = np.asarray([row["gain"] for row in train], dtype=float)
        self._fit_posterior(self._train_vectors, self._train_targets)
        dates = {row["date"] for row in train}
        production_gate = {
            "minimum_physical_matches": MIN_PRODUCTION_MATCHES,
            "minimum_dates": MIN_PRODUCTION_DATES,
            "physical_matches": len({row["physical_match_id"] for row in train}),
            "dates": len(dates),
            "passed": len({row["physical_match_id"] for row in train}) >= MIN_PRODUCTION_MATCHES
                      and len(dates) >= MIN_PRODUCTION_DATES,
        }
        self.metadata = {
            "schema": VERSION, "target": TARGET, "feature_schema": FEATURE_SCHEMA,
            "training_pipeline": next(iter(pipelines)), "training_rows": len(train),
            "training_physical_match_ids": sorted({row["physical_match_id"] for row in train}),
            "training_rows_sha256": _digest([{k: row[k] for k in ("sample_id", "physical_match_id", "date", "gain")}
                                              for row in train]),
            "group_partition_counts": {name: len({r["physical_match_id"] for r in clean if r["partition"] == name})
                                        for name in ("train", "calibration", "validation", "test")},
            "prior": {"kind": "synthetic_wide" if self.synthetic_prior is not None else "zero_mean_wide",
                      "schema": PRIOR_SCHEMA if self.synthetic_prior is not None else None,
                      "std": self.prior_std, "synthetic_is_label": False},
            "noise_std": self.noise_std, "lcb_z": self.lcb_z,
            "production_gate": production_gate,
            "fit_execution_at": datetime.now(timezone.utc).isoformat(),
            "selection_authority": "empirical_g_only", "empirical_g_enabled": False,
            "support_status": "shadow_only", "selection_allowed": False,
            "production_ready": False, "calibration_status": "not_established",
            "conditioning": "admitted_real_gross_labels_only; not_unconditional_action_EV",
        }
        self.model_version = VERSION + ":" + _digest(self.payload())[:24]
        return self

    def calibrate(self, rows):
        """Estimate noise scale from calibration groups without refitting mean."""
        if self.mean is None:
            raise ValueError("Fit the Bayesian shadow before calibration")
        clean, _ = validate_group_partitions(rows)
        calibration = [row for row in clean if row["partition"] == "calibration"]
        if not calibration:
            return self
        if any(row["query"].get("prediction_pipeline_version") != self.metadata["training_pipeline"]
               for row in calibration):
            raise ValueError("Calibration belongs to a different f pipeline")
        residuals = np.asarray([row["gain"] - self.mean @ row["vector"] for row in calibration], dtype=float)
        variance = max(1e-6, float(np.mean(residuals ** 2)))
        self.noise_std = math.sqrt(variance)
        self._fit_posterior(self._train_vectors, self._train_targets)
        self.metadata["calibration"] = {
            "rows": len(calibration), "physical_matches": len({row["physical_match_id"] for row in calibration}),
            "noise_std": self.noise_std, "used_for_threshold": False,
        }
        self.metadata["calibration_status"] = "descriptive_small_n_only"
        self.model_version = VERSION + ":" + _digest(self.payload())[:24]
        return self

    def estimate(self, query):
        if self.mean is None or not self.model_version:
            raise ValueError("Bayesian shadow is not fitted")
        if self.model_version != VERSION + ":" + _digest(self.payload())[:24]:
            raise ValueError("Frozen Bayesian shadow changed")
        if query.get("prediction_pipeline_version") != self.metadata["training_pipeline"]:
            raise ValueError("Query belongs to a different f pipeline")
        vector = query_features(query)
        mean = float(vector @ self.mean)
        variance = max(0., float(vector @ self.covariance @ vector))
        std = math.sqrt(variance)
        return {"schema": VERSION, "model_version": self.model_version,
                "posterior_mean": mean, "posterior_std": std,
                "posterior_lcb": mean - self.lcb_z * std,
                "support_status": "shadow_only", "selection_allowed": False,
                "production_ready": False, "empirical_g_enabled": False,
                "selection_authority": "empirical_g_only",
                "prior_is_synthetic_label": False,
                "conditioning": self.metadata["conditioning"]}

    def rank_shadow(self, candidates, *, stop_lcb=0.):
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("A nonempty candidate menu is required")
        if any(not isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("Every shadow candidate must be an object")
        if len({candidate.get("action_id") for candidate in candidates}) != len(candidates):
            raise ValueError("Shadow candidate action identities must be unique")
        rows = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("action_id"), str):
                raise ValueError("Every shadow candidate needs an action_id")
            query = candidate.get("query")
            estimate = self.estimate(query)
            cost = _finite(candidate.get("cost_units", 0.), "cost_units")
            if cost < 0:
                raise ValueError("cost_units cannot be negative")
            rows.append({"action_id": candidate["action_id"], **estimate,
                         "cost_units": cost,
                         "posterior_net_mean": estimate["posterior_mean"] - cost,
                         "posterior_net_lcb": estimate["posterior_lcb"] - cost})
        rows.sort(key=lambda row: (-row["posterior_net_lcb"], row["action_id"]))
        top = rows[0]
        proposed = top["action_id"] if top["posterior_net_lcb"] > _finite(stop_lcb, "stop_lcb") else "STOP"
        return {"schema": "prospective-frozen-f-quality-g-bayesian-ranking-v1",
                "candidates": rows, "shadow_proposal": proposed,
                "selection_allowed": False, "production_ready": False,
                "empirical_g_enabled": False, "selection_authority": "empirical_g_only",
                "prior_is_diagnostic_only": True}

    def payload(self):
        return {"schema": VERSION, "metadata": self.metadata,
                "prior_mean": None if self.prior_mean is None else self.prior_mean.tolist(),
                "prior_std": self.prior_std, "noise_std": self.noise_std, "lcb_z": self.lcb_z,
                "posterior_mean": None if self.mean is None else self.mean.tolist(),
                "posterior_covariance": None if self.covariance is None else self.covariance.tolist()}

    def save(self, path):
        if not self.model_version or self.model_version != VERSION + ":" + _digest(self.payload())[:24]:
            raise ValueError("Invalid unfrozen Bayesian shadow")
        body = {**self.payload(), "model_version": self.model_version}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(body, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    @classmethod
    def load(cls, path):
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        if body.get("schema") != VERSION:
            raise ValueError("Unsupported Bayesian shadow schema")
        prior = body.get("prior_mean")
        obj = cls(prior_mean=prior, prior_std=body["prior_std"],
                  noise_std=body["noise_std"], lcb_z=body["lcb_z"])
        obj.metadata = body["metadata"]
        obj.mean = np.asarray(body["posterior_mean"], dtype=float)
        obj.covariance = np.asarray(body["posterior_covariance"], dtype=float)
        obj.model_version = body["model_version"]
        if obj.model_version != VERSION + ":" + _digest(obj.payload())[:24]:
            raise ValueError("Bayesian shadow digest differs")
        return obj


def fit_shadow(dataset, *, synthetic_prior=None, seed=20261010,
               prior_std=DEFAULT_PRIOR_STD, noise_std=DEFAULT_NOISE_STD,
               lcb_z=DEFAULT_LCB_Z):
    """Fit/evaluate a shadow from a four-way pre-registered dataset.

    Thresholds are deliberately not selected here.  A caller may use the
    returned validation rows to create a descriptive rule, but this function
    never reads test outcomes to choose a threshold and never emits a live
    model.  If only train/holdout rows are available, it returns a waiting
    report instead of silently re-splitting those groups.
    """
    if not isinstance(dataset, dict):
        raise ValueError("An assembled quality dataset is required")
    raw = dataset.get("records", [])
    if not raw:
        return None, {"schema": VERSION, "fit_status": "no_admitted_labels",
                      "selection_allowed": False, "production_ready": False}
    try:
        rows, groups = validate_group_partitions(raw)
    except ValueError as exc:
        return None, {"schema": VERSION, "fit_status": "waiting_for_four_way_group_split",
                      "reason": str(exc), "selection_allowed": False, "production_ready": False}
    model = BayesianQualityGShadow(prior_std=prior_std, noise_std=noise_std, lcb_z=lcb_z,
                                   synthetic_prior=synthetic_prior)
    train = [dict(row, partition="train") for row in raw if row.get("partition") == "train"]
    if not train:
        return None, {"schema": VERSION, "fit_status": "waiting_for_admitted_training_labels",
                      "groups": {k: len(v) for k, v in groups.items()},
                      "selection_allowed": False, "production_ready": False}
    model.fit(raw)
    calibration = [row for row in raw if row.get("partition") == "calibration"]
    if calibration:
        model.calibrate(raw)
    validation = [row for row in rows if row["partition"] == "validation"]
    test = [row for row in rows if row["partition"] == "test"]
    def predictions(items):
        return [{"sample_id": row["sample_id"], "physical_match_id": row["physical_match_id"],
                 "observed_gain": row["gain"], **model.estimate(row["query"])} for row in items]
    validation_predictions = predictions(validation)
    test_predictions = predictions(test)
    report = {"schema": VERSION, "fit_status": "fitted_shadow_only",
              "train_physical_matches": len({r["physical_match_id"] for r in train}),
              "calibration_physical_matches": len({r["physical_match_id"] for r in calibration}),
              "validation_physical_matches": len({r["physical_match_id"] for r in validation}),
              "test_physical_matches": len({r["physical_match_id"] for r in test}),
              "validation": validation_predictions, "test": test_predictions,
              "threshold_selection": "validation_only_not_selected_by_fit_shadow",
              "production_gate": model.metadata["production_gate"],
              "selection_allowed": False, "production_ready": False,
              "empirical_g_enabled": False, "selection_authority": "empirical_g_only",
              "prior_is_diagnostic_only": True,
              "group_split_schema": SPLIT_SCHEMA, "random_seed": seed}
    return model, report


__all__ = ["BayesianQualityGShadow", "fit_shadow", "group_split", "validate_group_partitions",
           "query_features", "VERSION", "PRIOR_SCHEMA"]
