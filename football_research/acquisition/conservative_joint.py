"""Adapt evidence strength inside a single coherent joint probability model.

q_s(z|y,H) = s q(z|y,H) + (1-s) sum_y' p0(y'|H) q(z|y',H).
The reference return marginal is retained, including correlations between
tools. Every subsequent update conditions this same initial joint; we never
reset p0 to the current posterior or discount each tool independently.

The held-out fitting function is evaluator-owned. Runtime inference receives
only the ordinary public state. A selected strength is not a calibration
guarantee, a confidence interval, or proof of positive real acquisition gain.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .contextual_partial import ContextualPartialPredictor
from .experiment import outcome, state_after
from .joint_returns import ACTION_IDS, JointReturnPredictor, VERSION as REFERENCE_VERSION
from .state import parse_time
from .value import _canonical

VERSION = "coherent-evidence-strength-v1"
ACQUIRED_SETS = (("results_view",), ("xg_view",), ACTION_IDS)


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _restore_reference(payload):
    if payload.get("schema") != REFERENCE_VERSION:
        raise ValueError("A frozen original joint-return model is required")
    model = JointReturnPredictor(ContextualPartialPredictor.from_state(payload["base_predictor"]),
                                 **payload["parameters"])
    model.quantiles = deepcopy(payload["quantiles"])
    model.records = deepcopy(payload["records"])
    model.metadata = deepcopy(payload["metadata"])
    version = REFERENCE_VERSION + ":" + _digest(model._payload())[:20]
    if payload.get("model_version") != version:
        raise ValueError("Reference joint model hash mismatch")
    model.model_version = model.version = version
    return model


def minimize_strength(prior_label_probabilities, posterior_label_probabilities):
    """Convex mean log loss; rows are equally weighted matches x three states.

    The inputs are label probabilities from the same frozen reference model,
    not model EVSI targets. No query policy or future outcome is involved.
    """
    p = np.asarray(prior_label_probabilities, dtype=float)
    q = np.asarray(posterior_label_probabilities, dtype=float)
    if (p.ndim != 2 or p.shape[1] != len(ACQUIRED_SETS) or p.shape != q.shape or not p.size
            or not np.isfinite(p).all() or not np.isfinite(q).all()
            or (p <= 0).any() or (q <= 0).any() or (p > 1).any() or (q > 1).any()):
        raise ValueError("Positive match-by-acquired-set label probabilities required")
    delta = q - p

    def derivative(s):
        return -float(np.mean(delta / ((1-s)*p + s*q)))

    if derivative(0.) >= 0.:
        strength = 0.
    elif derivative(1.) <= 0.:
        strength = 1.
    else:
        lo, hi = 0., 1.
        for _ in range(64):
            mid = (lo+hi)/2
            if derivative(mid) < 0.:
                lo = mid
            else:
                hi = mid
        strength = (lo+hi)/2
    objective = lambda s: -float(np.mean(np.log((1-s)*p + s*q)))
    return {"strength": strength, "objective_at_zero": objective(0.),
            "objective_at_one": objective(1.), "selected_objective": objective(strength),
            "derivative_at_zero": derivative(0.), "derivative_at_one": derivative(1.),
            "derivative_at_selected": derivative(strength), "unique_matches": p.shape[0],
            "acquired_sets_per_match": len(ACQUIRED_SETS), "iterations": 64,
            "objective": "equal_match_equal_acquired_set_log_loss",
            "flat_objective_tie_break": "zero"}


class ConservativeJointReturnPredictor(JointReturnPredictor):
    """Frozen probability adapter accepted by the existing strict VoI policy.

    Constructing with ``strength`` alone is a declared mechanism experiment.
    Use ``fit_strength`` for chronological held-out parameter selection.
    Original artifacts and their caches are never changed.
    """

    def __init__(self, reference, *, strength, selection=None):
        if type(reference) is not JointReturnPredictor or not reference.model_version:
            raise ValueError("A frozen original JointReturnPredictor is required")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Evidence strength must be finite and in [0, 1]")
        payload = deepcopy({**reference._payload(), "model_version": reference.model_version})
        snapshot = _restore_reference(payload)
        super().__init__(snapshot.base_predictor, snapshot.bandwidth, snapshot.dirichlet_mass)
        self._reference_payload = payload
        self._evidence_strength = float(strength)
        self._selection = deepcopy(selection)
        self.quantiles, self.records = snapshot.quantiles, snapshot.records
        self.metadata = snapshot.metadata
        self._adaptation_ids = set()
        if selection is not None:
            if (selection.get("schema") != VERSION or selection.get("reference_model_version") != reference.model_version
                    or selection.get("reference_payload_sha256") != _digest(payload)
                    or selection.get("optimization", {}).get("strength") != self.evidence_strength):
                raise ValueError("Selection is not bound to this reference and strength")
            if parse_time(selection["as_of"]) < parse_time(self.metadata["training_as_of"]):
                raise ValueError("Adaptation cutoff precedes reference training")
            self._adaptation_ids = set(selection["match_ids"])
            trained = set(self.metadata["training_match_ids"]) | set(self.metadata["base_training_match_ids"])
            if self._adaptation_ids & trained:
                raise ValueError("Adaptation reused a reference training match")
            self.metadata["training_as_of"] = self.metadata["trained_as_of"] = selection["as_of"]
        self.metadata["evidence_strength_adaptation"] = {
            "strength": self.evidence_strength, "reference_model_version": reference.model_version,
            "reference_training_as_of": reference.metadata["training_as_of"],
            "selection_kind": "chronological_same_version_heldout" if selection else "declared_mechanism_only",
            "selection_match_ids": sorted(self._adaptation_ids),
            "selected_as_of": selection["as_of"] if selection else None,
            "calibration_status": "not_established", "return_marginal_changed": False}
        self.model_version = self.version = VERSION + ":" + _digest(self._payload())[:20]

    @property
    def evidence_strength(self):
        return self._evidence_strength

    @property
    def selection_report(self):
        return deepcopy(self._selection)

    def fit(self, *args, **kwargs):
        raise TypeError("Use fit_strength with a frozen reference and held-out rows")

    def _visible(self, state, *, hypothetical=False):
        clean, observations = super()._visible(state, hypothetical=hypothetical)
        if clean["match_id"] in self._adaptation_ids:
            raise ValueError("Adaptation match cannot be evaluated as held out")
        return clean, observations

    def _q(self, p0, *, condition_on_p0=True):
        q, audit = super()._q(p0, condition_on_p0=condition_on_p0)
        marginal = np.sum(np.asarray(p0)[:, None, None]*q, axis=0)
        s = self.evidence_strength
        mixed = s*q + (1-s)*marginal[None, :, :]
        return mixed, {**audit, "evidence_strength": s,
                       "adaptation_scope": "same_initial_joint_before_any_observation",
                       "calibration_status": "not_established"}

    def _payload(self):
        return {"schema": VERSION, "reference": self._reference_payload,
                "evidence_strength": self.evidence_strength, "selection": self._selection}

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported evidence-strength artifact schema")
        model = cls(_restore_reference(payload["reference"]), strength=payload["evidence_strength"],
                    selection=payload["selection"])
        if model.model_version != payload.get("model_version"):
            raise ValueError("Evidence-strength artifact hash mismatch")
        return model

    @classmethod
    def fit_strength(cls, reference, rows, actions, *, as_of, min_matches=12):
        """Evaluator-only fitting; later prediction IDs are never an input.

        All selection labels must be known by as_of. Runtime inference then
        rejects earlier cutoffs and every selected match identity, even if a
        caller attempts to relabel that match's cutoff as a later date.
        """
        if type(reference) is not JointReturnPredictor or not reference.model_version:
            raise ValueError("A frozen original JointReturnPredictor is required")
        if isinstance(min_matches, bool) or not isinstance(min_matches, int) or min_matches < 1:
            raise ValueError("A positive integer minimum match count is required")
        if (tuple(a["action_id"] for a in actions) != ACTION_IDS
                or set(actions[0]["group_ids"]) != {"elo", "recent_form"}
                or set(actions[1]["group_ids"]) != {"xg"}):
            raise ValueError("Adaptation must retain the modeled action contracts")
        cutoff, trained_at = parse_time(as_of), parse_time(reference.metadata["training_as_of"])
        if cutoff <= trained_at:
            raise ValueError("Adaptation cutoff must follow reference fitting")
        rows = sorted(rows, key=lambda r: (parse_time(r["prediction_cutoff"]), str(r["match_id"])))
        if len(rows) < min_matches:
            raise ValueError("Insufficient held-out matches for strength selection")
        ids = [str(r["match_id"]) for r in rows]
        trained = set(reference.metadata["training_match_ids"]) | set(reference.metadata["base_training_match_ids"])
        if len(set(ids)) != len(ids) or set(ids) & trained:
            raise ValueError("Duplicate or reference-training adaptation match")
        dataset_versions = {r.get("dataset_version") for r in rows}
        if len(dataset_versions) != 1 or not next(iter(dataset_versions)):
            raise ValueError("One explicit adaptation dataset version is required")
        samples, priors, posteriors = [], [], []
        for row in rows:
            prediction_at, kickoff = parse_time(row["prediction_cutoff"]), parse_time(row["kickoff"])
            result_at = parse_time(row["result_available_at"])
            # training_as_of is the reference's information cutoff, not a
            # measured wall-clock fit completion. Existing purged folds allow
            # inference at that cutoff; target IDs and labels remain excluded.
            if not trained_at <= prediction_at < kickoff < result_at <= cutoff:
                raise ValueError("Adaptation violates prediction/label chronology")
            if (row.get("information_mode") != reference.metadata["information_mode"]
                    or row.get("score_period") != "regulation"):
                raise ValueError("Adaptation information mode or score period differs")
            for field in ("fixture_available_at", "features_available_at"):
                if row.get(field) is None or parse_time(row[field]) > prediction_at:
                    raise ValueError("Adaptation input unavailable at prediction cutoff")
            for field in ("home_goals", "away_goals"):
                if isinstance(row.get(field), bool) or not isinstance(row.get(field), int) or row[field] < 0:
                    raise ValueError("Nonnegative integer regulation result required")
            before = reference.predict_partial(state_after(row, actions, []))
            if before.get("unknown_identity_fields"):
                raise ValueError("Adaptation fixture has an unknown reference identity")
            p0, label = before["probabilities"], outcome(row)
            after = [reference.predict_partial(state_after(row, actions, acquired))["probabilities"]
                     for acquired in ACQUIRED_SETS]
            priors.append([p0[label]]*len(ACQUIRED_SETS))
            posteriors.append([p[label] for p in after])
            samples.append({"match_id": str(row["match_id"]), "prediction_cutoff": row["prediction_cutoff"],
                            "kickoff": row["kickoff"], "result_available_at": row["result_available_at"],
                            "outcome_index": label, "initial_probabilities": p0,
                            "acquired_set_probabilities": after,
                            "input_sha256": _digest(row)})
        optimization = minimize_strength(priors, posteriors)
        selection = {"schema": VERSION, "as_of": cutoff.isoformat(),
                     "reference_model_version": reference.model_version,
                     "reference_payload_sha256": _digest({**reference._payload(), "model_version": reference.model_version}),
                     "dataset_version": next(iter(dataset_versions)), "match_ids": ids,
                     "earliest_prediction_cutoff": rows[0]["prediction_cutoff"],
                     "latest_label_available_at": max(parse_time(r["result_available_at"]) for r in rows).isoformat(),
                     "acquired_sets": [list(a) for a in ACQUIRED_SETS], "minimum_matches": min_matches,
                     "sample_digest": _digest(samples), "samples": samples, "optimization": optimization,
                     "scope": "equal-match held-out probability adaptation; empirical VoI calibration not established"}
        return cls(reference, strength=optimization["strength"], selection=selection)
