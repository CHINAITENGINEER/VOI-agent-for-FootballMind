"""Shrink only the outcome/X association remaining after the R response.

J_eta(Y,R,X|H) = eta J(Y,R,X|H) + (1-eta) J(Y,R|H) J(X|R,H).
The outcome/R and R/X marginals are preserved. Both acquisition orders always
condition this same initial joint; no post-query switch or action veto occurs.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .conservative_joint import ConservativeJointReturnPredictor, _digest, _restore_reference
from .contextual_joint import ContextualJointReturnPredictor
from .joint_returns import JointReturnPredictor, RESULT_CATEGORIES, XG_CATEGORIES, _probabilities
from .state import GROUP_FIELDS, parse_time

VERSION = "conditional-residual-joint-v1"


def _restore_model(payload):
    """Only the three already-supported frozen probability artifacts."""
    schema = payload.get("schema")
    if schema == "learned-joint-returns-v1":
        model = _restore_reference(payload)
    elif schema == "coherent-evidence-strength-v1":
        model = ConservativeJointReturnPredictor(_restore_reference(payload["reference"]),
                                                 strength=payload["evidence_strength"], selection=payload["selection"])
    elif schema == "contextual-joint-returns-v1":
        model = ContextualJointReturnPredictor(_restore_reference(payload["reference"]), payload["fixture_contexts"],
                                               context_weight=payload["context_weight"], strength=payload["evidence_strength"],
                                               selection=payload["selection"])
    else:
        raise ValueError("Unsupported frozen residual reference")
    if model.model_version != payload.get("model_version"):
        raise ValueError("Residual reference artifact hash mismatch")
    return model


def minimize_residual_strength(before_label_probabilities, after_label_probabilities):
    before, after = np.asarray(before_label_probabilities, dtype=float), np.asarray(after_label_probabilities, dtype=float)
    if (before.ndim != 1 or before.shape != after.shape or not before.size or
            not np.isfinite(before).all() or not np.isfinite(after).all() or
            (before <= 0).any() or (after <= 0).any() or (before > 1).any() or (after > 1).any()):
        raise ValueError("Positive paired conditional label probabilities required")
    delta = after-before
    derivative = lambda eta: -float(np.mean(delta/(before+eta*delta)))
    if derivative(0.) >= 0.:
        eta = 0.
    elif derivative(1.) <= 0.:
        eta = 1.
    else:
        lower, upper = 0., 1.
        for _ in range(64):
            middle = (lower+upper)/2
            if derivative(middle) < 0.:
                lower = middle
            else:
                upper = middle
        eta = (lower+upper)/2
    loss = lambda value: -float(np.mean(np.log(before+value*delta)))
    return {"eta": eta, "objective_at_zero": loss(0.), "objective_at_one": loss(1.),
            "selected_objective": loss(eta), "derivative_at_zero": derivative(0.),
            "derivative_at_one": derivative(1.), "derivative_at_selected": derivative(eta),
            "matches": len(before), "iterations": 64, "flat_objective_tie_break": "zero",
            "objective": "past_validation_conditional_outcome_log_loss_after_results_then_xg"}


class ResidualJointReturnPredictor(JointReturnPredictor):
    def __init__(self, reference, *, eta, selection=None):
        if type(reference) not in {JointReturnPredictor, ConservativeJointReturnPredictor, ContextualJointReturnPredictor}:
            raise ValueError("A supported frozen joint reference is required")
        if not reference.model_version:
            raise ValueError("Reference must be fitted")
        if isinstance(eta, bool) or not isinstance(eta, (float, int)) or not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("Residual eta must be finite and in [0, 1]")
        self._reference_payload = deepcopy({**reference._payload(), "model_version": reference.model_version})
        self._reference = _restore_model(self._reference_payload)
        super().__init__(self._reference.base_predictor, self._reference.bandwidth, self._reference.dirichlet_mass)
        self.quantiles, self.records = self._reference.quantiles, self._reference.records
        self.metadata = deepcopy(self._reference.metadata)
        self.residual_strength = float(eta)
        self._selection = deepcopy(selection)
        self._adaptation_ids = set(getattr(self._reference, "_adaptation_ids", ()))
        if selection is not None:
            source_selection = getattr(self._reference, "selection_report", None)
            if (selection.get("schema") != VERSION or selection.get("eta") != self.residual_strength or
                    selection.get("reference_payload_sha256") != _digest(self._reference_payload) or
                    selection.get("source_selection_sha256") != _digest(source_selection) or
                    parse_time(selection["as_of"]) != parse_time(self.metadata["training_as_of"]) or
                    set(selection["match_ids"]) != self._adaptation_ids):
                raise ValueError("Residual selection is not bound to this reference and eta")
        self.metadata["residual_calibration"] = {
            "eta": self.residual_strength, "anchor_response": "results_view", "residual_response": "xg_view",
            "reference_model_version": reference.model_version,
            "preserved_marginals": ["Y,R", "R,X", "Y"],
            "scope": "same initial joint; both acquisition orders supported",
            "selection_kind": "existing_past_validation_conditional_log_score" if selection else "declared_mechanism_only",
            "selection_match_ids": sorted(self._adaptation_ids) if selection else [],
            "selection_reuses_reference_validation": bool(selection),
            "calibration_status": "not_established", "parameter_uncertainty": "not_estimated",
        }
        self.model_version = self.version = VERSION + ":" + _digest(self._payload())[:20]

    @property
    def selection_report(self):
        return deepcopy(self._selection)

    def fit(self, *args, **kwargs):
        raise TypeError("Use fit_residual with an existing past-validation reference")

    def _visible(self, state, *, hypothetical=False):
        clean, observations = super()._visible(state, hypothetical=hypothetical)
        if clean["match_id"] in self._adaptation_ids:
            raise ValueError("Residual/reference adaptation match cannot be evaluated as held out")
        return clean, observations

    def joint_distribution(self, state, *, hypothetical=False, condition_on_p0=True):
        clean, observations = self._visible(state, hypothetical=hypothetical)
        free = {k: deepcopy(clean[k]) for k in ("match_id", "prediction_cutoff", "information_mode", "base")}
        free["groups"] = {g: {"status": "not_acquired", "values": {}} for g in GROUP_FIELDS}
        original = self._reference.joint_distribution(free, condition_on_p0=condition_on_p0)
        joint = np.asarray(original["probabilities"])
        yr, rx = joint.sum(axis=2), joint.sum(axis=0)
        r = rx.sum(axis=1)
        x_given_r = np.divide(rx, r[:, None], out=np.zeros_like(rx), where=r[:, None] > 0)
        independent = yr[:, :, None]*x_given_r[None, :, :]
        eta = self.residual_strength
        joint = eta*joint + (1-eta)*independent
        if "results_view" in observations:
            joint[:, np.arange(len(RESULT_CATEGORIES)) != RESULT_CATEGORIES.index(observations["results_view"]), :] = 0.
        if "xg_view" in observations:
            joint[:, :, np.arange(len(XG_CATEGORIES)) != XG_CATEGORIES.index(observations["xg_view"])] = 0.
        mass = float(joint.sum())
        if mass <= 0:
            raise ValueError("Impossible response under residual joint")
        joint /= mass
        prior = original["prior_probabilities"]
        return {"probabilities": joint.tolist(), "prior_probabilities": prior,
                "current_probabilities": joint.sum(axis=(1, 2)).tolist() if observations else list(prior),
                "observations": observations, "results_categories": list(RESULT_CATEGORIES),
                "xg_categories": list(XG_CATEGORIES), "evidence_probability": mass,
                "q_support": {**original["q_support"], "residual_eta": eta,
                              "residual_calibration_scope": "same_initial_joint_conditional_association_shrinkage"},
                "base_model_version": self.base_predictor.model_version, "model_version": self.model_version}

    def observation_support(self, state):
        self._visible(state)
        support = self._reference.observation_support(state)
        return {**support, "residual_eta": self.residual_strength,
                "residual_support_rule": "reference empirical coverage unchanged; eta adds no observations"}

    def _payload(self):
        return {"schema": VERSION, "reference": self._reference_payload,
                "eta": self.residual_strength, "selection": self._selection}

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != VERSION:
            raise ValueError("Unsupported residual joint artifact schema")
        model = cls(_restore_model(payload["reference"]), eta=payload["eta"], selection=payload["selection"])
        if model.model_version != payload.get("model_version"):
            raise ValueError("Residual joint artifact hash mismatch")
        return model

    @classmethod
    def fit_residual(cls, reference, *, as_of):
        if type(reference) not in {ConservativeJointReturnPredictor, ContextualJointReturnPredictor}:
            raise ValueError("Residual fitting requires an existing past-validation reference")
        source = reference.selection_report
        if source is None or parse_time(as_of) != parse_time(reference.metadata["training_as_of"]):
            raise ValueError("Use the frozen reference's exact past-validation cutoff")
        if parse_time(source["latest_label_available_at"]) > parse_time(as_of):
            raise ValueError("Residual calibration labels are not available at cutoff")
        samples = source["samples"]
        ids = [str(s["match_id"]) for s in samples]
        if not ids or len(set(ids)) != len(ids) or set(ids) != set(source["match_ids"]):
            raise ValueError("One bound past-validation sample per match required")
        before, after, projected = [], [], []
        s = reference.evidence_strength
        for sample in samples:
            y = sample["outcome_index"]
            if type(y) is not int or y not in {0, 1, 2}:
                raise ValueError("Recorded calibration label required")
            p0 = _probabilities(sample["initial_probabilities"])
            acquired = sample["acquired_set_probabilities"]
            if len(acquired) != 3:
                raise ValueError("Recorded results/xg/both validation probabilities required")
            # Existing reports store the reference's pre-strength posteriors.
            # Coherent evidence shrinkage makes every actual posterior this
            # exact mixture. Do not call runtime on its own adaptation rows.
            pr = (1-s)*p0 + s*_probabilities(acquired[0])
            prx = (1-s)*p0 + s*_probabilities(acquired[2])
            before.append(float(pr[y]))
            after.append(float(prx[y]))
            projected.append({"match_id": str(sample["match_id"]), "outcome_index": y,
                              "after_results_probabilities": pr.tolist(),
                              "after_results_xg_probabilities": prx.tolist()})
        optimization = minimize_residual_strength(before, after)
        selection = {"schema": VERSION, "as_of": parse_time(as_of).isoformat(),
                     "eta": optimization["eta"], "optimization": optimization, "match_ids": ids,
                     "reference_model_version": reference.model_version,
                     "reference_payload_sha256": _digest({**reference._payload(), "model_version": reference.model_version}),
                     "source_selection_sha256": _digest(source), "samples_digest": _digest(projected),
                     "latest_label_available_at": source["latest_label_available_at"],
                     "minimum_matches": source["minimum_matches"],
                     "source_validation_reused_for_multiple_parameters": True,
                     "scope": "past validation conditional log score; independent calibration not established",
                     "costs_or_query_thresholds_changed": False}
        return cls(reference, eta=optimization["eta"], selection=selection)
